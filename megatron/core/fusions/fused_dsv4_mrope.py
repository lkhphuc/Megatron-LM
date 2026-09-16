# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Out-of-place DSv4 adjacent-pair rotary with physical-row-aligned angles.

Unlike the scalar MLA LUT kernel, this kernel accepts independent angles for
every physical row and batch item. CP halos, packed padding, and compressed
groups are mapped by the caller before dispatch.
"""

import os
import shutil
from unittest.mock import MagicMock

# Triton resolves ptxas via TRITON_PTXAS_PATH (not PATH). Prefer that env var
# (set by devenv zshrc); otherwise try CUDA_HOME/bin/ptxas before Triton imports.
_ptxas = os.environ.get('TRITON_PTXAS_PATH') or shutil.which('ptxas')
if not _ptxas:
    _cuda_root = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH') or '/usr/local/cuda'
    _ptxas = os.path.join(_cuda_root, 'bin', 'ptxas')
_PTXAS_UNAVAILABLE_REASON: str | None = (
    "ptxas is not available; set TRITON_PTXAS_PATH to the CUDA ptxas "
    "binary (for example /usr/local/cuda/bin/ptxas)"
)
if _ptxas and os.path.isfile(_ptxas) and os.access(_ptxas, os.X_OK):
    os.environ['TRITON_PTXAS_PATH'] = _ptxas
    _PTXAS_UNAVAILABLE_REASON = None

import torch
from torch import Tensor

from megatron.core.fusions.fused_mrope import get_fused_mrope_unavailable_reason
from megatron.core.utils import null_decorator

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    libdevice = MagicMock()


def get_fused_dsv4_mrope_unavailable_reason(x: Tensor, angles: Tensor, pos_dim: int) -> str | None:
    """Return a launch restriction without synchronizing tensor values to the host."""
    if not HAVE_TRITON:
        return "Triton is not available"
    if _PTXAS_UNAVAILABLE_REASON is not None:
        return _PTXAS_UNAVAILABLE_REASON
    if x.ndim not in (3, 4) or angles.ndim != 3:
        return "DSv4 fused rotary requires SBHD, SBD, or THD input and [S, B, pairs] angles"
    # Reuse only its CUDA/device/dtype/stride checks. This generic capability
    # helper deliberately does not validate the raw mRoPE frequency shape;
    # selected physical-row angles have their own shape checks below.
    reason = get_fused_mrope_unavailable_reason(x, angles)
    if reason is not None:
        return reason
    if angles.requires_grad:
        return "DSv4 fused rotary does not differentiate position angles"
    batch = x.shape[1] if x.ndim == 4 or angles.shape[1] != 1 else 1
    if angles.shape[:2] != (x.shape[0], batch):
        return "DSv4 fused rotary angles must match the physical rows and batch"
    if not 0 < 2 * angles.shape[-1] <= pos_dim <= x.shape[-1]:
        return "DSv4 rotary spectrum must fit inside the positional subspace"
    if x.shape[-1] > 4096:
        return "DSv4 fused rotary supports head dimensions up to 4096"
    return None


@triton.jit
def _rotary_kernel(
    X,
    A,
    Y,
    SX: tl.constexpr,
    BX: tl.constexpr,
    HX: tl.constexpr,
    SA: tl.constexpr,
    BA: tl.constexpr,
    PA: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    START: tl.constexpr,
    PAIRS: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    s = row // (B * H)
    b = row // H % B
    h = row % H
    d = tl.arange(0, BLOCK)
    in_rotary = (d >= START) & (d < START + 2 * PAIRS)
    pair = (d - START) // 2
    even = (d - START) % 2 == 0
    offset = s * SX + b * BX + h * HX
    value = tl.load(X + offset + d, d < D, other=0)
    partner_d = tl.where(even, d + 1, d - 1)
    partner = tl.load(X + offset + partner_d, in_rotary & (d < D), other=0)
    angle = tl.load(A + s * SA + b * BA + pair * PA, in_rotary, other=0)
    # libdevice retains range reduction for large multimodal coordinates.
    cos = libdevice.cos(angle).to(value.dtype)
    sin = libdevice.sin(angle).to(value.dtype)
    if INVERSE:
        sin = -sin
    # Match eager low-precision multiplication rounding before the addition.
    first = (value.to(tl.float32) * cos.to(tl.float32)).to(value.dtype).to(tl.float32)
    second = (partner.to(tl.float32) * sin.to(tl.float32)).to(value.dtype).to(tl.float32)
    result = tl.where(even, first - second, first + second)
    result = tl.where(in_rotary, result, value)
    tl.store(Y + row * D + d, result, d < D)


def _launch(x: Tensor, angles: Tensor, pos_dim: int, inverse: bool) -> Tensor:
    # Normalize views only; the kernel accepts noncontiguous row/head strides.
    if x.ndim == 4:
        view = x
    elif angles.shape[1] == 1:
        view = x.unsqueeze(1)  # THD, including a single-head S1D tensor.
    else:
        view = x.unsqueeze(2)  # SBD shared KV.
    s, b, h, d = view.shape
    output = torch.empty(view.shape, device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        if not output.numel():
            return output.view(x.shape)
        _rotary_kernel[(s * b * h,)](
            view,
            angles,
            output,
            *view.stride()[:3],
            *angles.stride(),
            b,
            h,
            d,
            d - pos_dim,
            angles.shape[-1],
            inverse,
            triton.next_power_of_2(d),
            enable_fp_fusion=False,
        )
    return output.view(x.shape)


class _FusedDSv4Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, angles, pos_dim, inverse):
        """Save immutable angles and rotate into independent output storage."""
        ctx.save_for_backward(angles)
        ctx.pos_dim = pos_dim
        ctx.inverse = inverse
        return _launch(x, angles, pos_dim, inverse)

    @staticmethod
    def backward(ctx, grad_output):
        """Apply the transpose rotation without mutating the upstream gradient."""
        (angles,) = ctx.saved_tensors
        # Expanded or strided upstream gradients need not satisfy forward's
        # contiguous-channel contract. Materialize only when necessary.
        if grad_output.stride(-1) != 1:
            grad_output = grad_output.contiguous()
        return _launch(grad_output, angles, ctx.pos_dim, not ctx.inverse), None, None, None


def fused_dsv4_mrope(x: Tensor, angles: Tensor, pos_dim: int, *, inverse: bool = False) -> Tensor:
    """Rotate adjacent pairs within the positional suffix into private storage."""
    reason = get_fused_dsv4_mrope_unavailable_reason(x, angles, pos_dim)
    if reason is not None:
        raise ValueError(reason)
    return _FusedDSv4Rotary.apply(x, angles, pos_dim, inverse)
