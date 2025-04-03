from typing import Callable, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Registry
from model.layers_with_custom_init import LinearCustomInit as Linear
try:
    from model.flash_attn_triton import flash_attn_func
except ModuleNotFoundError as e:
    print(f"Failed to import flash_attn_triton: {e}")


class Attention(nn.Module):

    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        n_kv_heads: int,
        max_len: int, 
        dropout: float, 
        qk_encoding: Callable, 
        use_bias: bool, 
        use_qk_norm: bool,
        norm_cls: nn.Module,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.max_len = max_len
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads

        # TODO: should merge K, V into a single linear layer 
        self.W_q = Linear(d_model, d_model, bias=use_bias)
        self.W_v = Linear(d_model, self.n_kv_heads*self.d_head, bias=use_bias)
        self.W_k = Linear(d_model, self.n_kv_heads*self.d_head, bias=use_bias)
        self.W_o = Linear(d_model, d_model, bias=use_bias)

        self.dropout = dropout
        self.qk_encoding = qk_encoding

        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = norm_cls(d_model)
            self.k_norm = norm_cls(self.n_kv_heads*self.d_head)

    def _get_qkv(self, x):
        n, t, _ = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # (N, N-heads, T, D-head)
        q = q.view(n, t, self.n_heads, self.d_head).transpose(1, 2)
        # (N, N-KV-heads, T, D-head)
        k = k.view(n, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(n, t, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = self.qk_encoding(q)
        k = self.qk_encoding(k)

        n_repeats = self.n_heads // self.n_kv_heads
        if n_repeats > 1:
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)
        return q, k, v


class SDPAAttention(Attention):

    def __init__(self, kernel_backend, **kwargs):
        super().__init__(**kwargs)
        self.kernel_backend = kernel_backend
    
    def forward(self, x, kv_cache=None, mask=None) -> torch.Tensor:
        """
        x: (N, T, D)

        if kv_cache is not None, we assume that x has shape (N, 1, D)
        """
        n, t, _ = x.shape
        q, k, v = self._get_qkv(x)

        if kv_cache is not None:
            kv_cache.add(k, v)
            k, v = kv_cache.k, kv_cache.v

        dropout = self.dropout if self.training else 0.0
        # Only enable flash attention backend
        with nn.attention.sdpa_kernel(self.kernel_backend):
            # (N, H, T, E)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout, attn_mask=None)
        # (N, T, d_model)
        o = self.W_o(a.transpose(1, 2).reshape(n, t, self.n_heads * self.d_head))
        return o


@Registry.register("base_attn")
class BaseAttention(SDPAAttention):

    def __init__(self, **kwargs):
        super().__init__(kernel_backend=nn.attention.SDPBackend.MATH, **kwargs)
 

@Registry.register("flash_attn")
class FlashAttention(SDPAAttention):

    def __init__(self, **kwargs):
        super().__init__(kernel_backend=nn.attention.SDPBackend.FLASH_ATTENTION, **kwargs)


@Registry.register("cudnn_attn")
class CuDNNAttention(SDPAAttention):

    def __init__(self, **kwargs):
        super().__init__(kernel_backend=nn.attention.SDPBackend.CUDNN_ATTENTION, **kwargs)
 

@Registry.register("flash_attn_with_mask")
class FlashAttentionWithMask(Attention):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.dropout is not None:
            raise ValueError("triton flash attn does not support dropout. Dropout must be set to None")

    def forward(self, x, kv_cache=None, mask=None) -> torch.Tensor:
        """
        x: (N, T, D)

        if kv_cache is not None, we assume that x has shape (N, 1, D)
        """
        assert mask
        # we don't support kv cache for flash attn with mask yet
        assert kv_cache is None

        n, t, _ = x.shape
        q, k, v = self._get_qkv(x)

        # flash attn requires shape to be (N, T, H, E)
        # TODO: we could get around these transposes if we changed the qk_encoding
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_causal = mask is None
        mask = mask.unsqueeze(1)
        a = self._flash_attn(q, k, v, attn_bias=mask, is_causal=is_causal).view(n, t, -1)
        o = self.W_o(a)
        return o

    def _flash_attn(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).

        Returns:
            out: (batch_size, seq_len, nheads, headdim)

        TODO: when you switch over to GQA you can look at repeating the key & value tensors.
        TODO: they have a "packed" version where kv are together or qkv are together
        """
        assert all([dtype in [torch.bfloat16] for dtype in [query.dtype, key.dtype, value.dtype]])
        out = flash_attn_func(query, key, value, attn_bias, is_causal)
        return out

