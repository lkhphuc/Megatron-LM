# DSv4 explicit rotary positions (WP1–WP4)

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
propagates coordinates. Explicit positions fall back to PyTorch rotary even when
`apply_rope_fusion=True`; WP5 owns multimodal rotary fusion. Sequence parallelism
and configured CUDA graphs are explicitly rejected pending their validation.
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
