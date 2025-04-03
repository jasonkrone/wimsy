# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional
import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import rsqrt


no_torch_dynamo = lambda recursive=True: lambda f: torch._dynamo.disable(f, recursive=recursive)


@triton.jit
def _rms_norm_kernel(
    x_ptr,
    h1_ptr,
    w_ptr,
    eps,
    stride,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INCLUDE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        a = tl.load(
            x_ptr + cols, mask=cols < N_COLS, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        _mean += a * a
    rstd = rsqrt((tl.sum(_mean, axis=0) / N_COLS) + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)


@triton.jit
def _rms_norm_add_kernel(
    x_ptr,
    y_ptr,
    h1_ptr,
    w_ptr,
    eps,
    stride,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INCLUDE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    h1_ptr += row * stride

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        ax = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        ay = tl.load(
            y_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        a = ax + ay
        tl.store(x_ptr + cols, a, mask=mask)
        _mean += a * a
    rstd = rsqrt((tl.sum(_mean, axis=0) / N_COLS) + eps)
    for offset in range(0, N_COLS, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N_COLS
        a = tl.load(
            x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        if INCLUDE_WEIGHT:
            w = tl.load(w_ptr + cols, mask=mask)
            tl.store(h1_ptr + cols, a * rstd * w, mask=mask)
        else:
            tl.store(h1_ptr + cols, a * rstd, mask=mask)


def _rms_norm_forward(x, attn_norm_weights, eps):
    if not x.is_contiguous():
        raise ValueError("data must be contiguous")
    if attn_norm_weights is not None:
        if not attn_norm_weights.is_contiguous():
            raise ValueError("weights must be contiguous")
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device):
        _rms_norm_kernel[(M,)](
            x_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            INCLUDE_WEIGHT=attn_norm_weights is not None,
        )
    return out


def _rms_norm_add_forward(x, y, attn_norm_weights, eps):
    # x, y contiguous of same shape [..., n]
    # output of same shape, normed over the last dim.
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not y.is_contiguous():
        raise ValueError("y must be contiguous")
    if attn_norm_weights is not None:
        if not attn_norm_weights.is_contiguous():
            raise ValueError("weights must be contiguous")
    out = torch.empty_like(x)
    x_arg = x.reshape(-1, x.shape[-1])
    y_arg = y.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    with torch.cuda.device(x.device):
        _rms_norm_add_kernel[(M,)](
            x_arg,
            y_arg,
            out,
            attn_norm_weights,
            eps,
            x_arg.stride(0),
            N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            INCLUDE_WEIGHT=attn_norm_weights is not None,
        )
    return out


def rms_norm(x, weight: Optional[torch.Tensor], eps: float = 1e-6):
    """
    RMS Normalization along the last dimension.

    This is similar to torch.nn.functional.normalize but with eps being added
    instead of max.

    Expects x contiguous of shape (..., dim), and returns normalized data
    of the same shape. For each dim-length vector x, the result has

        x / sqrt( x*x.sum() + eps)

    If weights are included, they are a contiguous parameter of length dim
    which multiplies the result.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad or (weight is not None and weight.requires_grad)
    ):
        raise ValueError("Gradients not supported.")

    return _rms_norm_forward(x, weight, eps)


class XFormersRMSNorm(torch.nn.Module):
    """
    RMS Normalization layer along the last dimension.

    This is similar to torch.nn.functional.normalize but with eps being added
    instead of max.

    Expects contiguous input of shape (..., dim), and returns normalized data
    of the same shape. For each dim-length vector x, the result has

        x / sqrt( x*x.sum() + eps)

    If weights are included, they are a parameter of length dim which multiplies
    the result.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """

    def __init__(self, dim: int, include_weight: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if include_weight:
            self.weight: Optional[nn.Parameter] = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    @no_torch_dynamo()
    def forward(self, x: torch.Tensor):
        return rms_norm(x, self.weight, self.eps)  # type: ignore

    def increment_and_forward_(self, x: torch.Tensor, y: torch.Tensor):
        """
        An addition fused with forward.

            z = layer.increment_and_forward_(x, y)

        is equivalent to

            x += y
            z = layer(x)
        """
        return rms_norm_add(x, y, self.weight, self.eps)  # type: ignore
