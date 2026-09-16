# DSv4 explicit rotary positions

The optional `position_ids` argument is carried from `HybridModel` through
`HybridStack`, ordinary and mHC transformer layers, and full recomputation.
DSv4 interprets it when `config.mrope_section` is set. Without that configuration
or without positions, the existing scalar path is retained. No parameters or
checkpoint layouts change.

## Position and layout contract

- SBHD: integer `[B, S]` scalar or `[3, B, S]` temporal/height/width coordinates.
- THD: `[1, T]` or `[3, 1, T]`, aligned to physical packed rows, including padding.
- Contiguous THD CP: every rank receives the same **global** positions buffer.
  Local Q/output rows and preceding boundary KV rows select their original
  physical rows. Before-zero halo rows use zero coordinates. Rank-local position
  buffers and zigzag CP layouts are rejected.
- Packed boundaries remain in `PackedSeqParams`; coordinates never define masks
  or sample boundaries. Callers must supply coordinates in its padded physical
  row order. Padding is excluded by existing attention/indexer masks.

`mrope_section` contains three nonnegative frequency-pair counts summing to the
actual rotary width divided by two. `mrope_interleaved=True` uses stride-three
T/H/W assignment (64 rotary channels require `[11, 11, 10]`); False uses contiguous
sections. The section is caller configuration, not an inferred Falcon default.

## Geometry

The positional suffix is `qk_pos_emb_head_dim`, independent of the content
channels. The DSv4 attention layer's existing `RotaryEmbedding` or
`YarnRotaryEmbedding` supplies the spectrum. Consequently main/sliding layers
retain their ordinary base, compressed layers retain their compressed base and
YaRN parameters, and concentration scaling remains exactly one.

DSv4's legacy `mla_rotary_interleaved=True` converts adjacent stored pairs into
halves, rotates them, and restores adjacent storage with
`mla_output_remove_interleaving=True`. The explicit implementation performs this
same adjacent-pair transform directly. This storage convention is distinct from
`rotary_interleaved=False` (the frequency table layout) and T/H/W interleaving.
Tests compare the exact legacy transform and its gradients in FP32/BF16/FP16.
The attention output uses the identical local coordinates with negated angles;
it is out of place to preserve output tensors retained by attention backward.

## CSA semantics

A compressed row uses the coordinates of the **first token of its current
group**, preserving scalar `group_id * ratio`. Ratio-four overlapping compression
still uses the current group, even though learned pooling also consumes the
preceding group. Groups reset at packed sequence boundaries; incomplete tails
are discarded as before. Image rows/modality boundaries do not reset groups.
Input alignment at such boundaries remains a model/data policy for WP6.

CP compression uses the existing compact group IDs and local compact prefixes
to recover each group's global source row. Invalid capacity rows use coordinates
zero and retain existing visibility masks. The same coordinate gather runs after
both fused and unfused pooling. Position assignment never changes causal
visibility: a group must be complete before it becomes visible.

## Supported surface and validation

The initial path is eager training/forward with SBHD or THD, CP1 or contiguous
THD CP, ratios 0/4/128, and `rotary_interleaved=False`. Full hybrid recomputation
propagates coordinates. `apply_rope_fusion=True` selects the DSv4 Triton rotary
kernel for Q, shared KV, compressed/indexer KV, indexer Q, and inverse output.
Sequence parallelism and configured CUDA graphs are explicitly rejected pending
their validation.
CP metadata must be supplied on every pipeline stage; the pipeline activation
transport does not communicate it automatically. MTP position shifting and Falcon
position construction/integration remain later work.

Focused test entry point:

```bash
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_mrope.py --experimental
uv run python -m torch.distributed.run --nproc-per-node 2 -m pytest -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_mrope.py \
  -k cp2 --experimental
```

Scalar comparisons require bitwise identity; FP32 inverse round-trip uses
`atol=rtol=5e-7`. CP2 output/input-gradient comparison uses the existing DSv4 CP
cosine and tensor similarity gates (>0.999), separately from scalar parity.

## Fused rotary

`fused_dsv4_mrope(x, angles, pos_dim, inverse=False)` accepts FP32 angles of
shape `[physical_rows, batch, frequency_pairs]`. Coordinate selection and
RoPE/YaRN spectrum construction are shared with the reference implementation.
The kernel rotates adjacent stored pairs starting at `head_dim - pos_dim`,
copies the content prefix and any unrotated tail, and always returns private
storage. Backward applies the transpose rotation into another private tensor;
attention output saved for backward is never mutated.

The existing general fused mRoPE kernel rotates split halves at the front of
each head, and the scalar MLA kernel uses an in-place scalar frequency LUT.
The DSv4 kernel therefore uses a separate physical-row-aligned entry point,
reusing generic mRoPE device/dtype capability checks. Both forward and backward
fuse trigonometry, adjacent-pair rotation, and copying untouched channels.

Supported inputs are CUDA FP32/FP16/BF16 SBHD, SBD, and THD tensors with a
contiguous last dimension and head width up to 4096. Other row/head strides are
supported. FP32/FP16 require SM70 or later; BF16 requires SM80 or later. Missing
Triton, unsupported device/dtype/strides/shape, or differentiable angles cause
the dispatcher to warn once per reason and use the PyTorch implementation.
Invalid inputs still fail reference validation where applicable; direct calls
to the fused API raise a descriptive `ValueError` for unsupported inputs.
Kernel compilation/runtime errors propagate rather than being hidden.

Fused tests use `atol=rtol=1e-6` for FP32, `4e-3` for FP16, and `3e-2` for
BF16, covering forward/inverse and input gradients, full/partial rotation,
large coordinates, strided rows, expanded gradients, and retained-storage
safety. Attention tests compare eager/fused output and gradients for SBHD/THD
ratios 0/4/128 and assert dispatch. CP2 compares fused local results against
the unfused CP1 reference using the same similarity gates above.

```bash
uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/transformer/experimental_attention_variant/test_dsv4_mrope_fused.py
```

An indicative B200 regression screen (2026-09-16, BF16, 64 rotary channels)
measured forward apply with prebuilt FP32 angles. After 30 warmups and a device
synchronization, CUDA events bracketed 200 calls (Q/KV) or 500 calls (compressed
KV). The node also ran training, so these are contention-sensitive microbenchmarks,
not end-to-end throughput claims; coordinate construction and backward are excluded.

| Input shape | PyTorch apply | Fused apply |
| --- | ---: | ---: |
| Q `[1024, 1, 16, 512]` | 0.146 ms | 0.038 ms |
| KV `[1024, 1, 512]` | 0.076 ms | 0.039 ms |
| Compressed KV `[8, 1, 512]` | 0.072 ms | 0.040 ms |
