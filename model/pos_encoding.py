import torch
import torch.nn as nn

from utils import Precision, Registry
from model.kv_cache import MultiLayerCache


@Registry.register("identity")
class IdentityEncoding(nn.Identity):

    def forward(self, x, kv_cache=None) -> torch.Tensor:
        return x


@Registry.register("pos_encoding")
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, precision: str) -> None:
        super().__init__()
        self.dtype = Precision.PRECISION_NAME_TO_DTYPE[precision]
        self.d_model = d_model
        self.max_len = max_len
        encoding = torch.empty((1, max_len, d_model), dtype=self.dtype)
        self.register_buffer("encoding", encoding)
        self.init_params()

    def init_params(self):
        const = 10000 ** (-1.0 * torch.arange(start=0, end=self.d_model - 1, step=2) / self.d_model)
        # (max_len, 1)
        idx = torch.arange(self.max_len).view(-1, 1)
        # set for even idx
        self.encoding[:, :, ::2] = torch.sin(idx * const).type(self.dtype)
        # set for odd idx
        self.encoding[:, :, 1::2] = torch.cos(idx * const).type(self.dtype)

    def forward(self, x, kv_cache=None) -> torch.Tensor:
        """
        x: (N, T, D)
        """
        out = None
        _, t, _ = x.shape
        if kv_cache is None or t > 1:
            # gives you the first t elements
            out = x + self.encoding[:, :t, :]
        else:
            assert t == 1
            t = kv_cache.seqlen()
            # we add 1 b/c the pos encoding is called before seqlen is incremented in the attn layer
            out = x + self.encoding[:, t:t+1, :]
        return out


@Registry.register("rotary_encoding")
class RotaryEncoding(nn.Module):

    def __init__(
        self,
        d_head: int,
        max_len: int,
        base: int,
        device: torch.device,
        layer_id: int = None,
        cache: MultiLayerCache = None
    ) -> None:
        super().__init__()
        self.d_head = d_head
        self.max_len = max_len
        self.base = base
        self.layer_id = layer_id
        self.cache = cache
        self.init_params(device)

    def init_params(self, device):
        theta = 1.0 / (10000 ** (torch.arange(start=0, end=self.d_head - 1, step=2, device=device).float() / self.d_head))
        # theta1, theta1, theta2, theta2, ...
        theta = theta.repeat_interleave(2)
        # (T, E)
        rotation = torch.outer(torch.arange(self.max_len, device=theta.device, dtype=theta.dtype), theta).float()
        self.register_buffer("_cos", torch.cos(rotation))
        self.register_buffer("_sin", torch.sin(rotation))

    def forward(self, x) -> torch.Tensor:
        """
        x: (N, H, T, E)
        """
        kv_cache = None
        if self.cache is not None:
            kv_cache = self.cache.get_kv_cache_for_layer(self.layer_id)

        dtype = x.dtype
        x = x.float()
        n, h, t, e = x.shape

        # (N, H, T, E)
        x_perm = torch.stack([-x[:, :, :, 1::2], x[:, :, :, ::2]], dim=-1).view(n, h, t, e)
        if kv_cache is None or t > 1:
            x_cos = torch.mul(x, self._cos[:t, :])
            x_sin = torch.mul(x_perm, self._sin[:t, :])
        else:
            assert t == 1
            t = kv_cache.seqlen()
            # +1 b/c this is called before kv cache is updated
            x_cos = torch.mul(x, self._cos[t:t+1, :])
            x_sin = torch.mul(x_perm, self._sin[t:t+1, :])

        out = x_cos + x_sin
        out = out.type(dtype)
        return out

