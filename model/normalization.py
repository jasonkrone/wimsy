from typing import Optional
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Registry, Precision

try:
    from model.rms_norm_triton import XFormersRMSNorm
    Registry.register("xformers_rms_norm")(XFormersRMSNorm)
except Exception as e:
    print(f"Failed to import xformers: {e}")

try:
    from apex.normalization.fused_layer_norm import FusedRMSNorm as ApexRMSNorm
    Registry.register("apex_rms_norm")(ApexRMSNorm)
except Exception as e:
    print(f"Failed to import apex: {e}")


@Registry.register("layer_norm")
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, use_bias=False) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.alpha = nn.Parameter(torch.empty(self.normalized_shape))
        self.register_parameter("alpha", self.alpha)
        if use_bias:
            self.beta = nn.Parameter(torch.empty(self.normalized_shape))
            self.register_parameter("beta", self.beta)
        else:
            self.beta = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.alpha)
        if self.beta is not None:
            nn.init.zeros_(self.beta)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.alpha, self.beta, self.eps)


@Registry.register("rms_norm")
class RMSNorm(nn.Module):
    """
    Casts input value to precision (defaults to fp32) for the rms calculation
    then back to input dtype for scale by alpha
    """

    def __init__(self, normalized_shape, eps=1e-5, precision="fp32") -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.alpha = nn.Parameter(torch.empty(self.normalized_shape))
        self.register_parameter("alpha", self.alpha)
        self.dtype = None if precision is None else Precision.PRECISION_NAME_TO_DTYPE[precision]
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.alpha)

    def forward(self, x):
        xdt = x
        og_dt = x.dtype
        if self.dtype is not None:
            xdt = x.type(self.dtype)
        rms = xdt * torch.rsqrt(torch.mean(xdt.pow(2), dim=-1, keepdim=True) + self.eps)
        rms = rms.to(og_dt)
        out = self.alpha * rms
        return out
