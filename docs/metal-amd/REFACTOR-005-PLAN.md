# Implementation Plan: REFACTOR-005 Metal Backend Capability-Based Architecture

Spec: ./SPEC.md
Architecture: ../BACKLOG-architecture-notes.md
Generated: 2026-02-14, Updated: 2026-02-14

## Status

- [x] Phase 1: Device profile header + Metal population (done in llama-compare/llama-metal-amd)
- [x] Phase 2: Dispatch rename (done in llama-compare/llama-metal-amd)
- [x] Phase 3: Build, test, verify (done in llama-compare/llama-metal-amd)
- [ ] Phase 4: Port to main repo (llama.cpp)
- [ ] Phase 5: Shader file split
- [ ] Phase 6: Intel iGPU validation
- [ ] Phase 7: Final verification

## Core Architecture

Two memory paths — every GPU falls into one:

1. **Shared memory** — CPU and GPU see the same RAM. Zero copy.
2. **Separate memory** — GPU has its own RAM. Must copy data to it.

Everything else (PCIe, NVLink, Thunderbolt, RPC) is just transfer speed — not a different code path.

Compute tier is orthogonal: **Matrix HW** (simdgroup_mm / tensor cores) vs **Scalar**.

## Completed Phases (in llama-compare/llama-metal-amd)

### Phase 1: Device Profile Header + Metal Population [DONE]
- Created `ggml/include/ggml-device-profile.h` — backend-agnostic struct
- Added profile field to `ggml_metal_device` struct
- Added `ggml_metal_device_get_profile()` accessor
- Fixed multi-device init with `MTLCopyAllDevices()`
- Populated profile from existing Metal API queries
- Logged profile at init — confirmed AMD and Intel iGPU both appear

### Phase 2: Dispatch Rename [DONE]
- `has_simdgroup_mm` → `profile->has_matrix_hw` (4 refs in ops-matmul.cpp, 1 in ops-flash-attn.cpp)
- `has_unified_memory` → `profile->shared_memory` (1 ref in ggml-metal-context.m)
- Pipeline gate in ggml-metal-device.m sourced from profile
- Added include to ops-internal.h

### Phase 3: Build, Test, Verify [DONE]
- Full build succeeds
- test-backend-ops MUL_MAT passes on AMD
- Bench: 64.95 ± 0.64 t/s (baseline ~64.8, within ±2%)
- Profile log confirmed: `vendor=2 name=AMD Radeon Pro 5300M shared=false matrix_hw=false`
- Intel iGPU profile: `vendor=3 name=Intel(R) UHD Graphics 630 shared=true matrix_hw=false`

---

## Remaining Phases (all in llama-metal-amd/)

- **Phase 5:** [PHASE-5-shader-split.md](./PHASE-5-shader-split.md) — Reorganize kernels/ by capability tier (6 steps)
- **Phase 6:** [PHASE-6-intel-igpu.md](./PHASE-6-intel-igpu.md) — Validate Intel UHD 630 shared-memory path + profile values (3 steps)
- **Phase 7:** [PHASE-7-final-verify.md](./PHASE-7-final-verify.md) — Full test suite, commit (3 steps)

~~Phase 4 (port to llama.cpp) — DEFERRED. Develop in fork first, port last.~~

### Phase Status

- [x] Phase 5: Shader capability-tier split (already done — kernels/ has 21 files, matrixhw in 2)
- [ ] Phase 6: Intel iGPU validation
- [ ] Phase 7: Final verification

---

## Agent Work Package Template

Each phase is a self-contained work package for a Sonnet agent:

```
You are implementing Phase N of REFACTOR-005 for the llama.cpp Metal backend.

Working directory: /Users/hung/projects/llama.cpp

## What This Refactor Is
- Move, don't rewrite. No new logic, no new kernels, no behavior changes.
- Every line traces back to existing upstream code.
- Two memory paths: shared vs separate. Compute tier: matrix HW vs scalar.

## Reference Implementation
Phase 1-3 already done in /Users/hung/projects/llama-compare/llama-metal-amd
Diff those files to see exactly what was changed.

## Your Work Package
[paste phase details]

## Verification
After making changes:
1. Build: cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON && cmake --build build -j8
2. Test: ./build/bin/test-backend-ops -b MTL0 -o MUL_MAT 2>&1 | tail -5
3. Bench: ./build/bin/llama-bench -m ~/Library/Caches/llama.cpp/Qwen_Qwen2.5-1.5B-Instruct-GGUF_qwen2.5-1.5b-instruct-q4_k_m.gguf -t 1 -p 0 -n 128

## Agent Output Protocol
Write ALL results directly to files. Return ONLY a receipt (5 lines max).
Do NOT narrate. Do NOT summarize code. The files ARE the deliverable.
```

## Verification Checklist

- [x] `ggml-device-profile.h` exists at `ggml/include/` (in llama-compare)
- [x] Profile populated in `ggml_metal_device_init` (in llama-compare)
- [x] Multi-device init uses `MTLCopyAllDevices` (in llama-compare)
- [x] `has_simdgroup_mm` → `has_matrix_hw` in dispatch (in llama-compare)
- [x] `has_unified_memory` → `shared_memory` in dispatch (in llama-compare)
- [x] Profile logged at device init (in llama-compare)
- [ ] All above ported to main llama.cpp repo
- [ ] Shader monolith split into common + universal + matrixhw
- [ ] Intel iGPU tested on shared memory path
- [ ] `test-backend-ops` passes on both devices
- [ ] `llama-bench` within ±2% of baseline
- [ ] No new compiler warnings
