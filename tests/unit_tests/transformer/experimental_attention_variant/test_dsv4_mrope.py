# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent rotary oracle and DSv4 explicit-position integration tests."""

from unittest.mock import patch

import pytest
import torch

from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant import dsv4_mrope
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _build_attention,
    _make_config,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention_cp import (
    _assert_cp_tensor_match,
    _make_thd_packed_seq_params,
    _ReferenceCPGroup,
)


@pytest.fixture(scope='session', autouse=True)
def ensure_test_data():
    """These tensor-only tests do not require dataset downloads."""
    yield


@pytest.mark.parametrize('yarn', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('inverse', [False, True])
def test_legacy_rotary_oracle(yarn, dtype, inverse):
    """Legacy MLA rearranges stored adjacent pairs into halves and back."""
    rope = (
        YarnRotaryEmbedding(64, scaling_factor=16, original_max_position_embeddings=128)
        if yarn
        else RotaryEmbedding(64, rotary_percent=1.0)
    )
    positions = torch.arange(13, device='cuda').expand(2, -1)
    angles = dsv4_mrope.rotary_angles(positions, rope, [11, 11, 10], True)
    x = torch.randn(13, 2, 3, 96, device='cuda', dtype=dtype, requires_grad=True)
    emb = rope.get_emb(13)
    emb = emb[0] if isinstance(emb, tuple) else emb
    expected = torch.cat(
        (
            x[..., :32],
            _apply_rotary_pos_emb_bshd(
                x[..., 32:],
                emb,
                mla_rotary_interleaved=True,
                mla_output_remove_interleaving=True,
                inverse=inverse,
            ),
        ),
        dim=-1,
    )
    actual = dsv4_mrope.apply_rotary(x, angles, 64, inverse=inverse)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    grad = torch.randn_like(actual)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, grad, retain_graph=True)[0],
        torch.autograd.grad(expected, x, grad)[0],
        atol=0,
        rtol=0,
    )
    assert torch.equal(actual[..., :32], x[..., :32])


def test_thw_assignment_inverse_and_scalar():
    rope = RotaryEmbedding(64, rotary_percent=1.0)
    positions = torch.tensor([[[2, 5]], [[13, 17]], [[23, 31]]], device='cuda')
    angles = dsv4_mrope.rotary_angles(positions, rope, [11, 11, 10], True)
    expected = torch.empty_like(angles)
    for pair in range(32):
        expected[:, 0, pair] = positions[pair % 3, 0] * rope.inv_freq[pair]
    torch.testing.assert_close(angles, expected, atol=0, rtol=0)
    x = torch.randn(2, 1, 3, 96, device='cuda')
    rotated = dsv4_mrope.apply_rotary(x, angles, 64)
    restored = dsv4_mrope.apply_rotary(rotated, angles, 64, inverse=True)
    torch.testing.assert_close(restored, x, atol=5e-7, rtol=5e-7)
    scalar = positions[0]
    torch.testing.assert_close(
        dsv4_mrope.rotary_angles(scalar, rope, [11, 11, 10], True),
        dsv4_mrope.rotary_angles(scalar.expand(3, -1, -1), rope, [11, 11, 10], True),
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError):
        dsv4_mrope.rotary_angles(positions, rope, [10, 11, 11], True)
    with pytest.raises(ValueError):
        dsv4_mrope.validate_positions(positions, 1, 4)


def test_contiguous_mrope_section_assignment():
    """Non-interleaved sections take contiguous T then H then W frequency pairs."""
    rope = RotaryEmbedding(64, rotary_percent=1.0)
    section = [11, 11, 10]
    positions = torch.tensor([[[2, 5]], [[13, 17]], [[23, 31]]], device='cuda')
    angles = dsv4_mrope.rotary_angles(positions, rope, section, interleaved=False)
    expected = torch.empty_like(angles)
    axis = 0
    offset = 0
    for pair in range(32):
        if pair - offset >= section[axis]:
            offset += section[axis]
            axis += 1
        expected[:, 0, pair] = positions[axis, 0] * rope.inv_freq[pair]
    torch.testing.assert_close(angles, expected, atol=0, rtol=0)
    interleaved = dsv4_mrope.rotary_angles(positions, rope, section, interleaved=True)
    assert not torch.equal(angles, interleaved)
    # Equal axes still match interleaved / scalar RoPE.
    equal = positions[0].expand(3, -1, -1)
    torch.testing.assert_close(
        dsv4_mrope.rotary_angles(equal, rope, section, False),
        dsv4_mrope.rotary_angles(equal, rope, section, True),
        atol=0,
        rtol=0,
    )


def test_partial_rotary_preserves_both_content_and_tail():
    rope = RotaryEmbedding(64, rotary_percent=0.5)
    positions = torch.arange(7, device='cuda').view(1, -1)
    angles = dsv4_mrope.rotary_angles(positions, rope, [6, 5, 5], True)
    x = torch.randn(7, 1, 2, 96, device='cuda')
    actual = dsv4_mrope.apply_rotary(x, angles, 64)
    expected = _apply_rotary_pos_emb_bshd(
        x[..., 32:],
        rope.get_emb(7),
        mla_rotary_interleaved=True,
        mla_output_remove_interleaving=True,
    )
    assert torch.equal(actual[..., :32], x[..., :32])
    assert torch.equal(actual[..., 64:], x[..., 64:])
    torch.testing.assert_close(actual[..., 32:], expected, atol=0, rtol=0)


@pytest.mark.parametrize('ratio', [4, 128])
def test_compressed_first_token_packed_capacity_and_cp(ratio):
    # Two sequences, incomplete tails, and axes that deliberately differ.
    lengths = [ratio * 3 + 1, ratio * 2 + 3]
    cu = torch.tensor([0, lengths[0], sum(lengths)], device='cuda')
    ids = torch.arange(sum(lengths), device='cuda').view(1, 1, -1).expand(3, -1, -1).clone()
    ids[1] *= 3
    ids[2] += 91
    expected_rows = torch.tensor(
        [0, ratio, ratio * 2, lengths[0], lengths[0] + ratio], device='cuda'
    )
    out = dsv4_mrope.compressed_positions(ids, ratio, 7, cu)
    assert torch.equal(out[..., :5], ids[..., expected_rows])
    assert torch.count_nonzero(out[..., 5:]) == 0
    # CP compact order includes an overlapping predecessor group (group 1)
    # and two groups from sequence 1. Every row keeps original coordinates.
    local_cu = torch.tensor([0, 2, 4], device='cuda')
    groups = torch.tensor([1, 2, 0, 1, -1, -1], device='cuda')
    cp = dsv4_mrope.compressed_positions(ids, ratio, 6, cu, local_cu, groups)
    assert torch.equal(cp[..., :4], ids[..., expected_rows[1:]])
    assert torch.count_nonzero(cp[..., 4:]) == 0
    halo = dsv4_mrope.select_positions(
        ids,
        torch.tensor([-2, -1, 0, 1], device='cuda'),
        torch.tensor([False, False, True, True], device='cuda'),
    )
    assert torch.count_nonzero(halo[..., :2]) == 0


def test_attention_none_position_ids_keeps_scalar_path():
    """mrope_section with position_ids=None retains the legacy scalar rotary path."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(42)
        config = _make_config(
            csa_compress_ratios=[0] * 4, apply_rope_fusion=False, dsa_kernel_backend='none'
        )
        attn = _build_attention(config, 1, ProcessGroupCollection.use_mpu_process_groups()).cuda()
        attn.eval()
        x = torch.randn(16, 1, config.hidden_size, device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            expected = attn(x, None)[0]
            config.mrope_section = [6, 5, 5]
            config.mrope_interleaved = True
            actual = attn(x, None, position_ids=None)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        Utils.destroy_model_parallel()


def test_explicit_positions_reject_sequence_parallel_and_cuda_graph():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(7)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        x = torch.randn(8, 1, 256, device='cuda', dtype=torch.bfloat16)
        ids = torch.arange(8, device='cuda').view(1, 1, 8).expand(3, -1, -1)
        # Mutate after construction: MLATransformerConfig forbids SP without TP>1.
        sp_config = _make_config(
            csa_compress_ratios=[0] * 4,
            apply_rope_fusion=False,
            dsa_kernel_backend='none',
            mrope_section=[6, 5, 5],
            mrope_interleaved=True,
        )
        sp_attn = _build_attention(sp_config, 1, pg).cuda().eval()
        sp_attn.config.sequence_parallel = True
        with pytest.raises(ValueError, match='sequence parallelism'):
            sp_attn(x, None, position_ids=ids)

        graph_config = _make_config(
            csa_compress_ratios=[0] * 4,
            apply_rope_fusion=False,
            dsa_kernel_backend='none',
            cuda_graph_impl='local',
            mrope_section=[6, 5, 5],
            mrope_interleaved=True,
        )
        graph_attn = _build_attention(graph_config, 1, pg).cuda().eval()
        with pytest.raises(ValueError, match='eager execution'):
            graph_attn(x, None, position_ids=ids)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('ratio', [0, 4, 128])
@pytest.mark.parametrize('packed', [False, True])
def test_attention_scalar_regression_and_multimodal_backward(ratio, packed):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(42)
        config = _make_config(
            csa_compress_ratios=[ratio] * 4, apply_rope_fusion=False, dsa_kernel_backend='none'
        )
        attn = _build_attention(config, 1, ProcessGroupCollection.use_mpu_process_groups()).cuda()
        attn.eval()
        length = 256 if ratio == 128 else 16
        x = torch.randn(length, 1, config.hidden_size, device='cuda', dtype=torch.bfloat16)
        params = None
        scalar = torch.arange(length, device='cuda').unsqueeze(0)
        if packed:
            cu = torch.tensor([0, length // 2, length], device='cuda', dtype=torch.int32)
            params = PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=length // 2,
                max_seqlen_kv=length // 2,
            )
            scalar = torch.arange(length // 2, device='cuda').repeat(2).unsqueeze(0)
        with torch.no_grad():
            expected = attn(x, None, packed_seq_params=params)[0]
            config.mrope_section = [6, 5, 5]
            config.mrope_interleaved = True
            actual = attn(x, None, packed_seq_params=params, position_ids=scalar)[0]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        ids = scalar.expand(3, -1, -1).clone()
        ids[1] = ids[1] * 2 + 3
        ids[2] = ids[2] * 3 + 7
        x.requires_grad_(True)
        eager = attn(x, None, packed_seq_params=params, position_ids=ids)[0]
        eager_grad = torch.autograd.grad(eager.float().square().mean(), x)[0]
        config.apply_rope_fusion = True
        with (
            patch(
                'megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention.fused_mla_rope_inplace',
                side_effect=AssertionError(
                    'scalar fused kernel must not consume multimodal positions'
                ),
            ),
            patch.object(
                dsv4_mrope, 'fused_dsv4_mrope', wraps=dsv4_mrope.fused_dsv4_mrope
            ) as fused,
        ):
            output = attn(x, None, packed_seq_params=params, position_ids=ids)[0]
        # Q, KV, and inverse output always dispatch; compressed KV adds another
        # launch for CSA, and ratio 4 also rotates the indexer Q/KV.
        assert fused.call_count >= (3 if ratio == 0 else 4)
        torch.testing.assert_close(output, eager, atol=2e-2, rtol=2e-2)
        output.float().square().mean().backward()
        torch.testing.assert_close(x.grad, eager_grad, atol=2e-3, rtol=3e-2)
        assert torch.isfinite(output).all()
        assert torch.isfinite(x.grad).all()
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('ratio', [0, 4, 128])
@pytest.mark.parametrize('fused', [False, True])
def test_cp2_multimodal_matches_cp1(ratio, fused):
    if Utils.world_size != 2:
        pytest.skip('Requires exactly two ranks')
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    try:
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg = ProcessGroupCollection.use_mpu_process_groups()
        ref_pg.cp = _ReferenceCPGroup()
        common = dict(
            csa_compress_ratios=[ratio] * 4,
            apply_rope_fusion=False,
            dsa_kernel_backend='none',
            mrope_section=[6, 5, 5],
            mrope_interleaved=True,
        )
        cp_config = _make_config(
            **common,
            context_parallel_size=2,
            cp_partition_mode='contiguous',
            sequence_packing_scheduler='dp_balanced',
        )
        ref_config = _make_config(**common)
        cp_config.apply_rope_fusion = fused
        cp_attn = _build_attention(cp_config, 1, pg).cuda().eval()
        ref_attn = _build_attention(ref_config, 1, ref_pg).cuda().eval()
        ref_attn.load_state_dict(cp_attn.state_dict())
        lengths = (133, 379) if ratio == 128 else (13, 51)
        params = _make_thd_packed_seq_params(lengths)
        rows = sum(lengths)
        x = torch.randn(rows, 1, cp_config.hidden_size, device='cuda', dtype=torch.bfloat16)
        torch.distributed.broadcast(x, 0)
        ids = (
            torch.cat([torch.arange(n, device='cuda') for n in lengths])
            .view(1, 1, -1)
            .expand(3, -1, -1)
            .clone()
        )
        ids[1] = ids[1] * 2 + 17
        ids[2] = ids[2] * 3 + 31
        local_rows = rows // 2
        start = pg.cp.rank() * local_rows
        local = x[start : start + local_rows].clone().requires_grad_(True)
        reference = x.clone().requires_grad_(True)
        expected = ref_attn(reference, None, packed_seq_params=params, position_ids=ids)[0]
        actual = cp_attn(local, None, packed_seq_params=params, position_ids=ids)[0]
        _assert_cp_tensor_match(
            actual, expected[start : start + local_rows], 'multimodal CP2 output'
        )
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
        _assert_cp_tensor_match(
            local.grad, reference.grad[start : start + local_rows], 'multimodal CP2 input grad'
        )
    finally:
        Utils.destroy_model_parallel()


def test_cp2_rejects_zigzag_and_rank_local_positions():
    """Contiguous CP owns a global position buffer; zigzag and rank-local ids fail."""
    if Utils.world_size != 2:
        pytest.skip('Requires exactly two ranks')
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
    )
    try:
        from dataclasses import replace

        model_parallel_cuda_manual_seed(11)
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = _make_config(
            csa_compress_ratios=[0] * 4,
            apply_rope_fusion=False,
            dsa_kernel_backend='none',
            mrope_section=[6, 5, 5],
            mrope_interleaved=True,
            context_parallel_size=2,
            cp_partition_mode='contiguous',
            sequence_packing_scheduler='dp_balanced',
        )
        attn = _build_attention(config, 1, pg).cuda().eval()
        lengths = (13, 51)
        params = _make_thd_packed_seq_params(lengths)
        rows = sum(lengths)
        local_rows = rows // 2
        start = pg.cp.rank() * local_rows
        x = torch.randn(local_rows, 1, config.hidden_size, device='cuda', dtype=torch.bfloat16)
        global_ids = (
            torch.cat([torch.arange(n, device='cuda') for n in lengths])
            .view(1, 1, -1)
            .expand(3, -1, -1)
            .clone()
        )
        # Rank-local buffers are the wrong length for the global ownership contract.
        local_ids = global_ids[..., start : start + local_rows]
        with pytest.raises(ValueError, match='replicated global positions'):
            attn(x, None, packed_seq_params=params, position_ids=local_ids)

        zigzag = replace(params, cp_partition_mode='zigzag')
        with pytest.raises(ValueError, match='contiguous'):
            attn(x, None, packed_seq_params=zigzag, position_ids=global_ids)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('recompute', [False, True])
@pytest.mark.parametrize('pre_process', [False, True])
def test_hybrid_position_plumbing(wrapped, recompute, pre_process):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        extra = dict(
            enable_hyper_connections=wrapped,
            hidden_dropout=0.0,
            mrope_section=[6, 5, 5],
            mrope_interleaved=True,
            dsa_kernel_backend='none',
        )
        if recompute:
            extra.update(
                recompute_granularity='full', recompute_method='uniform', recompute_num_layers=1
            )
        config = _make_config(num_layers=1, csa_compress_ratios=[0], **extra)
        block = HybridStack(
            config,
            hybrid_dsv4_stack_spec(config).submodules,
            layer_type_list=['W'],
            pp_layer_offset=0,
            pre_process=pre_process,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        attention = (
            block.layers[0].inner_layer.self_attention
            if wrapped
            else block.layers[0].self_attention
        )
        seen = []
        attention.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs.get('position_ids')), with_kwargs=True
        )
        x = torch.randn(
            8, 1, config.hidden_size, device='cuda', dtype=torch.bfloat16, requires_grad=True
        )
        if wrapped and not pre_process:
            from megatron.core.transformer.hyper_connection import HyperConnectionModule

            x_input = HyperConnectionModule.input_expand(x, config.num_residual_streams)
        else:
            x_input = x
        ids = torch.arange(8, device='cuda').view(1, 1, 8).expand(3, -1, -1)
        if not pre_process:
            block.set_input_tensor(x_input)
        output = block(x_input, attention_mask=None, position_ids=ids)
        output.float().square().mean().backward()
        assert seen and all(torch.equal(value, ids) for value in seen)
        assert torch.isfinite(x.grad).all()
        if recompute:
            assert len(seen) >= 2
    finally:
        Utils.destroy_model_parallel()
