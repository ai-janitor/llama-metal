# qwen35/

Qwen3.5 hybrid model (transformer attention + gated delta-net linear attention). Split from a single `qwen35.cpp` to isolate the three delta-net execution paths by responsibility.

## Files

- **`qwen35.cpp`** — Model constructor (graph build loop), transformer attention layers, FFN, input projections (`build_qkvz`), gated normalization, and the linear attention dispatcher (`build_layer_attn_linear`). Also contains `build_delta_net_autoregressive`, a thin function that does shared preprocessing (l2-norm, scale, sigmoid) then dispatches to fused or unfused.

- **`delta-net-fused.cpp`** — `build_delta_net_fused`: single-sequence decode path using the `ggml_gated_delta_net` Metal kernel. Requires `n_seqs == 1`. Receives preprocessed tensors (already normed/scaled/sigmoided). Accepts an optional `state_dst` parameter; when set, uses `ggml_gated_delta_net_ext` to write the new SSM state directly to the persistent cache, bypassing `ggml_cpy`.

- **`delta-net-unfused.cpp`** — `build_delta_net_unfused`: multi-sequence decode fallback using elementwise ggml ops. Used when `n_seqs > 1` (e.g., parallel sequence decoding). Same interface as the fused path.

- **`delta-net-chunking.cpp`** — `build_delta_net_chunking`: prompt-path chunked linear recurrence with triangular solve. Processes multiple tokens per sequence in chunks of 64. The chunking math operates in ggml state convention (S). The boundary conversion to and from the fused kernel's cache layout (S^T) happens in the caller, not inside this file.

## Execution paths

The dispatcher in `build_layer_attn_linear` selects the path:
- `n_seq_tokens == 1` → `build_delta_net_autoregressive` → fused (if `n_seqs == 1`) or unfused
- `n_seq_tokens > 1` → `build_delta_net_chunking` (prompt)

State in the persistent cache is stored in kernel convention (S^T). The fused path reads/writes S^T directly. The chunking path transposes at the boundary.

## Performance notes

The original Qwen 4B decode profile was overhead-bound: CPY and CONT consumed 42% of decode time while the fused kernel was only 1.6%. Four optimizations moved the profile to compute-bound (MUL_MAT dominant at 55.5%):

1. **CONT elimination**: For single-token decode, tensors have unit extent in higher dimensions, so `ggml_reshape_Nd` preserves contiguous layout without materialization. Replaced `ggml_cont_Nd` with `ggml_reshape_Nd`, removing 9 unnecessary copy ops per SSM layer.

2. **Zero-dispatch removal**: Guarded `build_rs` in `llama-graph.cpp` to skip extra-state copies and zero-scale ops when `n_rs == n_seqs`, eliminating empty GPU kernel dispatches.

3. **Direct SSM state writes**: `ggml_gated_delta_net_ext` writes the new SSM state directly to the persistent cache buffer via a `state_dst` parameter (src[6]), eliminating one `ggml_cpy` per SSM layer.

4. **Direct conv state writes**: `ggml_ssm_conv_ext` writes the last (d_conv-1) elements per channel directly to the persistent conv cache buffer via a `state_dst` parameter (src[2]), fused into the convolution compute. Eliminates one `ggml_cpy` per SSM layer.

Result: CPY dropped from 97 ops / 35 ms to 1 op / 0.03 ms. Decode graph: 149 ms to 82 ms (-45%). Qwen 4B: 20.5 to 31.7 t/s (+55%). Qwen 0.8B: 43.5 to 69.2 t/s (+59%).

## When editing this code

- The fused and unfused paths receive **preprocessed** tensors from the dispatcher. Do not duplicate l2-norm, scale, or sigmoid in the sub-functions.
- For decode (`n_seq_tokens == 1`), prefer `ggml_reshape_Nd` over `ggml_cont_Nd` — the data is already contiguous when all higher dimensions are 1.
- The chunking path uses ggml convention (S); the fused path uses kernel convention (S^T). Keep the boundary transposes in `build_layer_attn_linear`; do not hide them inside the delta-net sub-functions.
- The `_ext` API variants (`ggml_gated_delta_net_ext`, `ggml_ssm_conv_ext`) are backward compatible. When `state_dst` is NULL, they behave identically to the non-ext versions. The `_ext` path is only taken for single-seq decode (`n_seq_tokens == 1 && n_seqs == 1`).
- Changes to `build_rs` in `llama-graph.cpp` affect all recurrent models (Mamba, RWKV, etc.), not just Qwen.
- Always validate with actual text generation, not just throughput numbers. Use `llama-simple` with greedy sampling for deterministic comparison.
