# qwen35/

Qwen3.5 hybrid model (transformer attention + gated delta-net linear attention). Split from a single `qwen35.cpp` to isolate the three delta-net execution paths by responsibility.

## Files

- **`qwen35.cpp`** — Model constructor (graph build loop), transformer attention layers, FFN, input projections (`build_qkvz`), gated normalization, and the linear attention dispatcher (`build_layer_attn_linear`). Also contains `build_delta_net_autoregressive`, a thin function that does shared preprocessing (l2-norm, scale, sigmoid) then dispatches to fused or unfused.

- **`delta-net-fused.cpp`** — `build_delta_net_fused`: single-sequence decode path using the `ggml_gated_delta_net` Metal kernel. Requires `n_seqs == 1`. Receives preprocessed tensors (already normed/scaled/sigmoided). The kernel itself is fast (~2-3% of decode time); the cost historically lived in the plumbing around it.

- **`delta-net-unfused.cpp`** — `build_delta_net_unfused`: multi-sequence decode fallback using elementwise ggml ops. Used when `n_seqs > 1` (e.g., parallel sequence decoding). Same interface as the fused path.

- **`delta-net-chunking.cpp`** — `build_delta_net_chunking`: prompt-path chunked linear recurrence with triangular solve. Processes multiple tokens per sequence in chunks of 64. Operates in ggml state convention (S), with boundary transposes in the caller to convert to/from the kernel convention (S^T) used by the fused decode path.

## Execution paths

The dispatcher in `build_layer_attn_linear` selects the path:
- `n_seq_tokens == 1` → `build_delta_net_autoregressive` → fused (if `n_seqs == 1`) or unfused
- `n_seq_tokens > 1` → `build_delta_net_chunking` (prompt)

State in the persistent cache is stored in kernel convention (S^T). The fused path reads/writes S^T directly. The chunking path transposes at the boundary.

## Performance notes

The fused kernel accounts for a small fraction of decode time. Most of the Qwen decode overhead comes from recurrent-state management — layout materialization (CONT) and cache read/write traffic (CPY). Two optimizations were applied:

1. **CONT elimination**: For single-token decode, all tensors are effectively 1D (higher dimensions have size 1, exempt from contiguity checks). Replaced `ggml_cont_Nd` with `ggml_reshape_Nd` in the decode path, removing 9 unnecessary copy-and-repack ops per SSM layer.

2. **Zero-dispatch removal**: Guarded `build_rs` in `llama-graph.cpp` to skip extra-state copies and zero-scale ops when `n_rs == n_seqs`, eliminating empty GPU kernel dispatches.

The remaining bottleneck is the 49 real CPY ops per decode graph — two per SSM layer for conv and SSM state writeback. These require an architectural change (direct cache writes from the fused kernel) to eliminate.

## When editing this code

- The fused and unfused paths receive **preprocessed** tensors from the dispatcher. Do not duplicate l2-norm, scale, or sigmoid in the sub-functions.
- For decode (`n_seq_tokens == 1`), prefer `ggml_reshape_Nd` over `ggml_cont_Nd` — the data is already contiguous when all higher dimensions are 1.
- The chunking path uses ggml convention (S); the fused path uses kernel convention (S^T). Boundary transposes live in `build_layer_attn_linear`, not in the delta-net functions.
- Changes to `build_rs` in `llama-graph.cpp` affect all recurrent models (Mamba, RWKV, etc.), not just Qwen.
- Always validate with actual text generation, not just throughput numbers. Use `llama-simple` with greedy sampling for deterministic comparison.
