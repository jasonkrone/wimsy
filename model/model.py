from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Config, Registry
from model.layers_with_custom_init import (
    LinearCustomInit as Linear,
    EmbeddingCustomInit as Embedding,
)


@Registry.register("mlp_relu")
class MLP(nn.Module):

    def __init__(self, d_model: int, d_hidden: int, use_bias: bool, activation_fn: nn.Module = nn.ReLU()) -> None:
        super().__init__()
        self.lin1 = Linear(d_model, d_hidden, bias=use_bias)
        self.lin2 = Linear(d_hidden, d_model, bias=use_bias)
        self.activation_fn = activation_fn

    def forward(self, x) -> torch.Tensor:
        """
        x: (N, T, D)
        """
        x = self.lin1(x)
        x = self.activation_fn(x)
        x = self.lin2(x)
        return x


@Registry.register("mlp_swiglu")
class MLPSwiGLU(nn.Module):

    def __init__(self, d_model: int, d_hidden: int, use_bias: bool) -> None:
        super().__init__()
        self.lin1a = Linear(d_model, d_hidden, bias=use_bias)
        self.lin1b = Linear(d_model, d_hidden, bias=use_bias)
        self.lin2 = Linear(d_hidden, d_model, bias=use_bias)

    def forward(self, x) -> torch.Tensor:
        """
        x: (N, T, D)
        """
        x = F.silu(self.lin1a(x)) * self.lin1b(x)
        x = self.lin2(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_hidden: int,
        max_len: int,
        dropout: float,
        norm_cls: nn.Module,
        attn_cls: nn.Module,
        mlp_cls: nn.Module,
        qk_encoding: Callable,
        use_bias: bool,
        use_qk_norm: bool,
        layer_id: int,
        # TODO: this is temporary to test dMoE
        mlp_args: Config = None,
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(dropout)
        self.mlp_dropout = nn.Dropout(dropout)
        self.qk_encoding = qk_encoding
        self.attn = attn_cls(
            d_model=d_model, 
            n_heads=n_heads, 
            n_kv_heads=n_kv_heads,
            max_len=max_len,
            dropout=dropout,
            qk_encoding=self.qk_encoding,
            use_bias=use_bias,
            use_qk_norm=use_qk_norm,
            norm_cls=norm_cls,
        )
        if mlp_args is None:
            self.mlp = mlp_cls(d_model=d_model, d_hidden=d_hidden, use_bias=use_bias)
        # TODO: this is temporary to test dMoE
        else:
            self.mlp = mlp_cls(mlp_args)
        self.attn_norm = norm_cls(d_model)
        self.mlp_norm = norm_cls(d_model)
        self.layer_id = layer_id


@Registry.register("pre_norm_transformer_block")
class PreNormTransformerBlock(TransformerBlock):

    def forward(self, x, mask, cache):
        """
        x: (N, T, D)
        """
        kv_cache = None
        if cache is not None:
            kv_cache = cache.get_kv_cache_for_layer(self.layer_id)
        x = x + self.attn_dropout(self.attn(self.attn_norm(x), kv_cache=kv_cache, mask=mask))
        x = x + self.mlp_dropout(self.mlp(self.mlp_norm(x)))
        return x


@Registry.register("res_post_norm_transformer_block")
class ResPostNormTransformerBlock(TransformerBlock):

    def forward(self, x, mask, cache):
        """
        x: (N, T, D)
        """
        kv_cache = None
        if cache is not None:
            kv_cache = cache.get_kv_cache_for_layer(self.layer_id)
        x = x + self.attn_dropout(self.attn_norm(self.attn(x, kv_cache=kv_cache, mask=mask)))
        x = x + self.mlp_dropout(self.mlp_norm(self.mlp(x)))
        return x


class InputEmbedding(nn.Module):

    def __init__(self, pos_encoding, vocab_size, d_model, p_dropout, max_len, precision, layer_id, cache=None):
        super().__init__()
        self.layer_id = layer_id
        self.dropout = nn.Dropout(p_dropout)
        self.tok_embed = Embedding(vocab_size, d_model)
        pos_cls = Registry.get(pos_encoding)
        self.pos_encoding = pos_cls(d_model=d_model, max_len=max_len, precision=precision)
        self.cache = cache

    def forward(self, x, mask=None, cache=None):
        kv_cache = None
        if self.cache is not None:
            kv_cache = self.cache.get_kv_cache_for_layer(self.layer_id)
        x = self.tok_embed(x)
        x = self.pos_encoding(x, kv_cache)
        x = self.dropout(x)
        return x


class LMHead(nn.Module):

    def __init__(self, vocab_size, d_model, norm_cls):
        super().__init__()
        self.norm = norm_cls(d_model)
        self.W_out = Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, x, mask=None, cache=None):
        x = self.norm(x)
        logits = self.W_out(x)
        return logits


class SequentialWithArgs(nn.Sequential):

    def forward(self, input, **kwargs):
        for module in self:
            input = module(input, **kwargs)
        return input


@Registry.register("decoder_lm")
class Decoder(nn.Module):

    def __init__(self, config) -> None:
        # TODO: we should probably switch over to using the torch layer norm; I bet it's faster
        # model must have a pos_encoding or qk_encoding
        assert config.pos_encoding is not None or config.qk_encoding is not None
        super().__init__()
        self.config = config
        self.p_dropout = config.p_dropout
        self.max_len = config.max_len
        self.init_stdev = config.init_stdev
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // config.n_heads
        self.d_hidden = config.d_hidden
        self.precision = config.precision
        self.attn_cls = Registry.get(config.attn)
        self.mlp_cls = Registry.get(config.mlp)
        self.norm_cls = Registry.get(config.norm)
        self.transformer_block_cls = Registry.get(config.transformer_block)
        self.use_cache = config.use_cache
        self.use_bias = config.use_bias
        self.use_qk_norm = config.use_qk_norm

        layers = []
        input_embed = InputEmbedding(
            config.pos_encoding,
            config.vocab_size,
            config.d_model,
            config.p_dropout,
            config.max_len,
            precision=self.precision,
            layer_id=0,
        )
        layers.append(input_embed)

        # TODO: this is temporary to test dMoE
        mlp_args = None if config.mlp != "dmoe" else config

        qk_encoding_cls = Registry.get(config.qk_encoding)
        self.qk_encoding = qk_encoding_cls(
            d_head=self.d_head,
            max_len=self.max_len,
            base=config.get("rope_base"),
            device=input_embed.tok_embed.weight.device,
        )

        for i in range(config.n_layers):
            block = self.transformer_block_cls(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                d_hidden=self.d_hidden,
                max_len=self.max_len,
                dropout=self.p_dropout,
                norm_cls=self.norm_cls,
                attn_cls=self.attn_cls,
                mlp_cls=self.mlp_cls,
                qk_encoding=self.qk_encoding,
                use_bias=self.use_bias,
                use_qk_norm=self.use_qk_norm,
                layer_id=i,
                mlp_args=mlp_args,
            )
            layers.append(block)

        lm_head = LMHead(vocab_size=self.vocab_size, d_model=self.d_model, norm_cls=self.norm_cls)
        layers.append(lm_head)
        self.layers = SequentialWithArgs(*layers)
        # TODO: should allow model to specify custom initialization to the layers

    def as_sequential(self):
        return self.layers

    def get_transformer_layer_cls(self):
        return self.transformer_block_cls

    def forward(self, input_ids, mask=None, cache=None) -> torch.Tensor:
        """
        input_ids: (N, T)
        """
        _, t = input_ids.shape
        if cache is not None:
            assert t == 1 or cache.get_kv_cache_for_layer(0).seqlen() == 0
        logits = self.layers(input_ids, mask=mask, cache=cache)
        return logits
