# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Fused DSv4 rotary parity, dispatch, and retained-storage regression tests."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions.fused_dsv4_mrope import (
    fused_dsv4_mrope,
    fused_dsv4_mrope_raw,
    get_fused_dsv4_mrope_unavailable_reason,
)
from megatron.core.transformer.experimental_attention_variant import dsv4_mrope


@pytest.fixture(scope='session', autouse=True)
def ensure_test_data():
    """Tensor-only tests need no dataset downloads."""
    yield


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('layout', ['sbhd', 'sbd', 'thd'])
@pytest.mark.parametrize('inverse', [False, True])
@pytest.mark.parametrize('partial', [False, True])
def test_fused_rotary_matches_reference(dtype, layout, inverse, partial):
    batch = 2 if layout != 'thd' else 1
    shape = {'sbhd': (19, batch, 3, 512), 'sbd': (19, batch, 512), 'thd': (19, 3, 512)}[layout]
    x = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    # Independent axes/CSA coordinate selection reduce to independent row
    # angles. Include large angles, zero padding, and partial rotary tails.
    pairs = 16 if partial else 32
    angles = torch.randn(19, batch, pairs, device='cuda') * 100000
    angles[0] = 0
    before = x.detach().clone()
    expected = dsv4_mrope.apply_rotary(x, angles, 64, inverse=inverse)
    actual = fused_dsv4_mrope(x, angles, 64, inverse=inverse)
    assert actual.data_ptr() != x.data_ptr()
    assert torch.equal(x, before)
    assert torch.equal(actual[..., :448], x[..., :448])
    assert torch.equal(actual[..., 448 + pairs * 2 :], x[..., 448 + pairs * 2 :])
    tolerance = {torch.float32: 1e-6, torch.float16: 4e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    # The additional retained x use catches mutation during inverse backward.
    grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual + x.square(), x, grad, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected + x.square(), x, grad)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)
    assert torch.equal(x, before)


@pytest.mark.parametrize('inplace', [True, False])
def test_fused_raw_out_buffer(inplace):
    """Raw launch writes into out= without an extra temporary+copy."""
    x = torch.randn(11, 2, 4, 96, device='cuda')
    angles = torch.randn(11, 2, 16, device='cuda') * 1000
    expected = dsv4_mrope.apply_rotary(x, angles, 64)
    if inplace:
        actual = x.clone()
        returned = fused_dsv4_mrope_raw(actual, angles, 64, out=actual)
        assert returned.data_ptr() == actual.data_ptr()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    else:
        before = x.clone()
        dest = torch.empty_like(x)
        returned = fused_dsv4_mrope_raw(x, angles, 64, out=dest)
        assert returned.data_ptr() == dest.data_ptr()
        assert torch.equal(x, before)
        torch.testing.assert_close(dest, expected, atol=1e-6, rtol=1e-6)


def test_fused_raw_out_rejects_bad_buffer():
    x = torch.randn(7, 1, 3, 96, device='cuda')
    angles = torch.randn(7, 1, 16, device='cuda')
    with pytest.raises(ValueError, match='out= must match'):
        fused_dsv4_mrope_raw(x, angles, 32, out=torch.empty(7, 2, 3, 96, device='cuda'))
    strided = torch.randn(14, 1, 3, 96, device='cuda')[::2]
    with pytest.raises(ValueError, match='contiguous'):
        fused_dsv4_mrope_raw(strided.contiguous(), angles, 32, out=strided)


@pytest.mark.parametrize('case', ['cpu', 'dtype', 'stride', 'angles_grad', 'no_triton'])
def test_fused_fallback(case):
    device = 'cpu' if case == 'cpu' else 'cuda'
    dtype = torch.float64 if case == 'dtype' else torch.float32
    x = torch.randn(7, 1, 3, 96, device=device, dtype=dtype, requires_grad=True)
    angles = torch.randn(7, 1, 16, device=device)
    if case == 'stride':
        x = x[..., ::2]
    if case == 'angles_grad':
        angles.requires_grad_()
    with patch('megatron.core.fusions.fused_dsv4_mrope.HAVE_TRITON', case != 'no_triton'):
        assert get_fused_dsv4_mrope_unavailable_reason(x, angles, 32) is not None
        with patch.object(
            dsv4_mrope, 'fused_dsv4_mrope', side_effect=AssertionError('must fall back')
        ):
            with patch.object(dsv4_mrope, '_FUSION_FALLBACK_WARNINGS', set()):
                with pytest.warns(UserWarning, match='Falling back to PyTorch rotary'):
                    actual = dsv4_mrope.apply_rotary(x, angles, 32, fused=True)
                with patch.object(dsv4_mrope.warnings, 'warn') as warn:
                    dsv4_mrope.apply_rotary(x, angles, 32, fused=True)
                warn.assert_not_called()
        expected = dsv4_mrope.apply_rotary(x, angles, 32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        grad = torch.randn_like(actual)
        torch.testing.assert_close(
            torch.autograd.grad(actual, x, grad, retain_graph=True)[0],
            torch.autograd.grad(expected, x, grad)[0],
            atol=0,
            rtol=0,
        )


def test_fused_strided_rows_and_expanded_gradient():
    x = torch.randn(26, 2, 6, 96, device='cuda', requires_grad=True)[::2, :, ::2]
    angles = torch.randn(13, 2, 16, device='cuda').transpose(0, 1).contiguous().transpose(0, 1)
    expected = dsv4_mrope.apply_rotary(x, angles, 64)
    actual = fused_dsv4_mrope(x, angles, 64)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        atol=1e-6,
        rtol=1e-6,
    )


def test_fused_shape_capability():
    x = torch.randn(7, 2, 3, 96, device='cuda')
    for angles in [torch.randn(7, 1, 16, device='cuda'), torch.randn(7, 2, device='cuda')]:
        assert get_fused_dsv4_mrope_unavailable_reason(x, angles, 32) is not None
        with pytest.raises(ValueError):
            fused_dsv4_mrope(x, angles, 32)


def test_fused_noncurrent_device():
    if torch.cuda.device_count() < 2:
        pytest.skip('Requires two CUDA devices')
    with torch.cuda.device(0):
        x = torch.randn(7, 1, 3, 96, device='cuda:1', requires_grad=True)
        angles = torch.randn(7, 1, 16, device='cuda:1')
        actual = fused_dsv4_mrope(x, angles, 32)
        expected = dsv4_mrope.apply_rotary(x, angles, 32)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), x, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), x)[0],
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.cuda.current_device() == 0
        assert get_fused_dsv4_mrope_unavailable_reason(x, angles.to('cuda:0'), 32) is not None
