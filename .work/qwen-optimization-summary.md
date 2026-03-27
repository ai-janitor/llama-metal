## Qwen3.5 decode optimization: layout and state-traffic overhead elimination

- `src/models/qwen35.cpp` was split into `src/models/qwen35/{qwen35,delta-net-chunking,delta-net-fused,delta-net-unfused}.cpp` to isolate prompt chunking, fused single-seq decode, and unfused multi-seq fallback. This separation was load-bearing: it stopped profiling and path-specific optimization from contaminating each other, and directly enabled the four phases below.

- The initial Qwen 4B decode profile was overhead-bound: 148.96 ms total, with CPY at 35.01 ms (97 ops, 23.5%) and CONT at 28.46 ms (224 ops, 19.1%). The fused delta-net kernel itself was only 2.42 ms (1.6%). The regression relative to comparable transformer models came from layout materialization and recurrent-state copy traffic around the kernel, not from the kernel's math.

- **Phase 1 — CONT elimination:** In the fused decode path, 9 `ggml_cont_Nd` calls per SSM layer were replaced with `ggml_reshape_Nd`. Under single-token decode, all higher dimensions have unit extent, so `ggml_is_contiguous` passes without materialization. CONT dropped from 28.46 ms (224 ops) to 0.26 ms (8 ops). Propagated to `qwen3next.cpp` and `qwen35moe.cpp`.

- **Phase 2 — Zero-dispatch removal:** `build_rs` in `src/llama-graph.cpp` was guarded to skip extra-state copies and zero-scale ops when `n_rs == n_seqs` (verified to be the single-seq decode case). This removed 48 empty CPY, 48 empty GET_ROWS, and 48 empty SCALE dispatches from the graph. Wall-time savings were modest since zero-sized dispatches were cheap, but it simplified the graph and removed profiling noise. This change applies to all recurrent models (Mamba, RWKV, etc.), not just Qwen.

- **Phase 3 — Direct SSM state writes:** `ggml_gated_delta_net_ext` was added, taking an optional `state_dst` (src[6]) parameter. When set, the Metal kernel writes the new state directly to the persistent cache buffer instead of packing it into the output tensor for a subsequent `ggml_cpy`. This eliminated 24 SSM state-writeback CPY ops. Propagated to `qwen3next.cpp`.

- **Phase 4 — Direct conv state writes:** `ggml_ssm_conv_ext` was added, taking an optional `state_dst` (src[2]) parameter. The ssm_conv Metal kernel was extended to write the last (d_conv-1) elements per channel directly to the persistent conv cache buffer, fused into the convolution compute. This eliminated 24 conv state-writeback CPY ops. All four Metal ssm_conv kernel variants were updated (f32, f32_4, batched, batched_4). Propagated to `qwen3next.cpp` and `qwen35moe.cpp`.

- Final Qwen 4B decode profile: 81.66 ms total, 1035 nodes. MUL_MAT is now dominant at 55.5%. CPY is effectively zero (1 op / 0.03 ms — just KV attention). CONT is 0.25 ms (8 ops). The profile is compute-bound.

- End-to-end throughput: Qwen 4B went from ~20.5 t/s to 31.7 t/s (+55%). Qwen 0.8B went from 43.5 t/s to 69.2 t/s (+59%). Llama 3.2 3B at 49.3 t/s for reference. All measurements are short cold-run `llama-simple` on AMD Radeon Pro 5500M; `llama-bench` tg128 on this GPU showed ±4 t/s variance from thermal throttling and is not trusted for absolute numbers.

- Correctness was validated throughout via deterministic greedy decoding on short prompts. An earlier lesson from the preceding session: mixing prompt-path and fused-path regressions in a single file caused false conclusions during debugging. The split-by-responsibility refactor eliminated that class of confusion.

- A zero-copy `build_rs` approach (replacing `ggml_get_rows` with a view for single-seq decode) was attempted but hit graph-allocator/reuse invariants — `s_copy`/`s_copy_extra` tensors become zero-sized when the view path is taken, and the graph scheduler does not assign buffers to them, causing an assert in `set_input`. This is valid in principle but requires framework-level work to preserve the allocator contract across prompt-to-decode graph shape changes.

## Files changed

### New ggml APIs (backward compatible)

| File | Change |
|------|--------|
| `ggml/include/ggml.h` | Declare `ggml_gated_delta_net_ext`, `ggml_ssm_conv_ext` |
| `ggml/src/ggml.c` | Implement both `_ext` variants; same op enum, src[6]/src[2] for state_dst |

### Metal kernel changes

| File | Change |
|------|--------|
| `ggml/src/ggml-metal/ggml-metal-impl.h` | Add `has_state_dst` to `ggml_metal_kargs_gated_delta_net` and `ggml_metal_kargs_ssm_conv` |
| `ggml/src/ggml-metal/ssm/ssm.metal` | All ssm_conv kernels (f32, f32_4, batched, batched_4) gain `state_dst` buffer param; `kernel_gated_delta_net` gains `state_dst` buffer param; both write state to `state_dst` when `has_state_dst == 1` |
| `ggml/src/ggml-metal/ssm/ops-ssm.cpp` | Bind src[6]/src[2] buffers conditionally; pass `has_state_dst` in kernel args |

### Framework change

| File | Change |
|------|--------|
| `src/llama-graph.cpp` | Guard `build_rs` extra-state copy when `n_rs == n_seqs`; guard zero-scale when `rs_zero < 0` |

### Model code

| File | Change |
|------|--------|
| `src/models/qwen35/qwen35.cpp` | Use `ggml_ssm_conv_ext` for conv writeback; pass `state_dst` to fused path; skip SSM/conv CPY for single-seq decode; replace `ggml_cont_Nd` with `ggml_reshape_Nd` in decode path; `alpha` cont_3d → reshape_3d; `cur` cont_2d → reshape_2d |
| `src/models/qwen35/delta-net-fused.cpp` | Accept `state_dst` param; use `ggml_gated_delta_net_ext` when set; return `state_dst` as new_state; remove 4 cont ops (g_t, beta_t, core_attn_out, new_state) |
| `src/models/models.h` | Update signatures for `build_delta_net_autoregressive`, `build_delta_net_fused` (qwen35); update `build_delta_net_autoregressive` (qwen3next) |
| `src/models/qwen3next.cpp` | Same CONT + direct-write optimizations as qwen35 (both SSM and conv) |
| `src/models/qwen35moe.cpp` | CONT optimization + direct conv-write (no fused SSM path in this variant, so SSM CPY stays) |

### Documentation

| File | Change |
|------|--------|
| `src/models/qwen35/README.md` | Documents folder split, execution paths, performance notes, maintenance invariants |
| `.work/qwen-optimization-summary.md` | This file |

## Commits

1. `314046a` — qwen35 folder refactor + CONT elimination + zero-dispatch removal
2. `ca92196` — CONT propagation to qwen3next/qwen35moe
3. `8a7ec77` — Direct SSM state writes via `ggml_gated_delta_net_ext`
4. `9f1b657` — Direct conv state writes via `ggml_ssm_conv_ext`
5. `a789076` — Propagate direct writes to qwen3next/qwen35moe
