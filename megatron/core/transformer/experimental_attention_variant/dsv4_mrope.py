# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Token-aligned, unit-magnitude DSv4 rotary reference path.

Positions are physical-row aligned: [B, S] or [3, B, S], including packed
padding. Contiguous CP receives the replicated global buffer on every rank.
The explicit path is selected by ``mrope_section``. ``apply_rope_fusion``
selects out-of-place Triton rotation when supported, with a PyTorch fallback.
"""

import warnings

import torch
from torch import Tensor

from megatron.core.fusions.fused_dsv4_mrope import (
    fused_dsv4_mrope,
    get_fused_dsv4_mrope_unavailable_reason,
)

_FUSION_FALLBACK_WARNINGS: set[str] = set()


def validate_positions(position_ids: Tensor, batch: int, rows: int) -> None:
    """Validate the external scalar or T/H/W physical-row contract."""
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("DSv4 position_ids must have int32 or int64 dtype")
    if position_ids.shape not in (torch.Size((batch, rows)), torch.Size((3, batch, rows))):
        raise ValueError(
            f"DSv4 position_ids must be [{batch}, {rows}] or [3, {batch}, {rows}], "
            f"got {tuple(position_ids.shape)}; CP requires replicated global positions"
        )


def select_positions(position_ids: Tensor, rows: Tensor, valid: Tensor | None = None) -> Tensor:
    """Gather physical rows, assigning zero coordinates to invalid capacity/halo rows."""
    selected = position_ids.index_select(-1, rows.clamp(0, position_ids.shape[-1] - 1).long())
    if valid is not None:
        selected = torch.where(valid, selected, 0)
    return selected


def compressed_positions(
    position_ids: Tensor,
    ratio: int,
    capacity: int,
    cu_seqlens: Tensor | None = None,
    cu_compressed: Tensor | None = None,
    group_ids: Tensor | None = None,
) -> Tensor:
    """Use each current compression group's first original token as its coordinate.

    For CP pre-grouped buffers, ``cu_compressed`` contains local compact group
    prefixes, ``cu_seqlens`` contains global token prefixes, and ``group_ids``
    identifies the original group within its sequence. Ratio-4 overlap does
    not change the representative or the causal visibility rule.
    """
    rows = torch.arange(capacity, device=position_ids.device)
    if cu_seqlens is None:
        return select_positions(position_ids, rows * ratio)
    if cu_compressed is None:
        counts = (cu_seqlens[1:] - cu_seqlens[:-1]) // ratio
        cu_compressed = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    batch = torch.bucketize(rows, cu_compressed[1:], right=True).clamp_max(cu_seqlens.numel() - 2)
    groups = rows - cu_compressed[batch] if group_ids is None else group_ids
    valid = (rows < cu_compressed[-1]) & (groups >= 0)
    source = cu_seqlens[batch] + groups * ratio
    return select_positions(position_ids, source, valid)


def rotary_angles(position_ids: Tensor, rope, section: list[int], interleaved: bool) -> Tensor:
    """Build [S, B, pairs] angles using the layer's existing RoPE/YaRN spectrum."""
    if rope.rotary_interleaved:
        raise ValueError("DSv4 explicit positions require rotary_interleaved=False")
    # Position one exposes the exact existing layer spectrum, including YaRN
    # interpolation, without a max(position) host synchronization or a large LUT.
    emb = rope.get_emb(1, offset=1)
    emb = emb[0] if isinstance(emb, tuple) else emb
    spectrum = emb[0, 0, 0, : emb.shape[-1] // 2]
    pairs = spectrum.numel()
    if len(section) != 3 or any(s < 0 for s in section) or sum(section) != pairs:
        raise ValueError(
            f"DSv4 mrope_section must contain three nonnegative counts summing to {pairs}"
        )
    if interleaved and tuple(section) != ((pairs + 2) // 3, (pairs + 1) // 3, pairs // 3):
        raise ValueError("Interleaved DSv4 mrope_section must match stride-3 T/H/W assignment")
    positions = (
        position_ids.unsqueeze(0).expand(3, -1, -1) if position_ids.ndim == 2 else position_ids
    )
    if interleaved:
        axes = torch.arange(pairs, device=positions.device) % 3
    else:
        axes = torch.repeat_interleave(
            torch.arange(3, device=positions.device),
            torch.tensor(section, device=positions.device),
            output_size=pairs,
        )
    selected = positions.permute(1, 2, 0)[..., axes]
    return (selected.float() * spectrum.to(positions.device)).transpose(0, 1)


def apply_rotary(
    x: Tensor, angles: Tensor, pos_dim: int, *, inverse: bool = False, fused: bool = False
) -> Tensor:
    """Rotate adjacent pairs in the positional suffix; preserve all other channels.

    ``x`` is SBHD, SB(D), or THD (angles batch=1); output has private storage,
    including inverse output rotation needed by attention backward.
    The legacy DSv4 MLA path converts adjacent stored pairs into split halves
    before rotation and restores adjacent storage afterwards. This is unrelated
    to T/H/W frequency assignment or the split-half frequency table layout.
    Unsupported fused inputs use the differentiable PyTorch reference below.
    """
    if fused:
        reason = get_fused_dsv4_mrope_unavailable_reason(x, angles, pos_dim)
        if reason is None:
            return fused_dsv4_mrope(x, angles, pos_dim, inverse=inverse)
        if reason not in _FUSION_FALLBACK_WARNINGS:
            _FUSION_FALLBACK_WARNINGS.add(reason)
            warnings.warn(
                f"DSv4 fused mRoPE is unavailable: {reason}. Falling back to PyTorch rotary.",
                stacklevel=2,
            )
    if x.ndim == 3 and angles.shape[1] == 1 and x.shape[1] != 1:
        angles = angles[:, 0, None, :]
    elif x.ndim == 4:
        angles = angles.unsqueeze(-2)
    width = angles.shape[-1] * 2
    if width > pos_dim or pos_dim > x.shape[-1]:
        raise ValueError("DSv4 rotary spectrum exceeds the positional subspace")
    start = x.shape[-1] - pos_dim
    rotary = x[..., start : start + width]
    even, odd = rotary[..., 0::2], rotary[..., 1::2]
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    if inverse:
        sin = -sin
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return torch.cat((x[..., :start], rotated, x[..., start + width :]), dim=-1)
