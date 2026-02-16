# llama.cpp Metal — Work Backlog

**Last updated**: 2026-02-14
**Hardware**: 2019 MacBook Pro — AMD Radeon Pro 5300M (discrete), Intel UHD 630 (shared)
**Fork**: `ai-janitor/llama.cpp`

## Completed

### Bugs
- [x] **BUG-001/003**: Concurrent dispatch NaN on non-shared-memory GPUs
  - Branch: `fix/metal-amd-concurrency`
- [x] **BUG-002**: `simd_max(half)` broken with -INF on AMD RDNA
  - Branch: `fix/metal-amd-fa-vec`

### Features
- [x] **FEAT-001**: Scalar FA kernel for non-matrix-hw GPUs
  - Branch: `feat/metal-scalar-fa`

### Performance
- [x] **PERF-001 pp512**: Metal 127.90 vs Vulkan 103.73 (+23.3%)
  - Fix: mul_mv_ext for large n on non-matrix-hw GPUs
  - PR: #19600
- [x] **PERF-002**: 12x mmap slowdown on separate-memory devices
  - Fix: auto-disable mmap, weights go to private VRAM
  - Commit: fccb4d93f

## Active

- [ ] **PERF-003**: tg128 gap — Metal 60 vs Vulkan 69 (13.5%)
  - Kernel tuning: thread mapping, adaptive dispatch
  - Branch: `perf/PERF-003-metal-amd-tg128`
  - Details: [PERF-003-metal-amd-tg128/](PERF-003-metal-amd-tg128/README.md)

- [ ] **FEAT-002**: Metal Adaptive Dispatch for Discrete GPUs
  - Device profile, function constants, per-device kernel tuning
  - Details: [FEAT-002-metal-adaptive-dispatch/](FEAT-002-metal-adaptive-dispatch/README.md)

- [ ] **REFACTOR-005**: Metal Backend Reorg
  - Split 10K shader monolith by capability tier
  - Device profile header at ggml/ level (backend-agnostic)
  - Fix device enumeration (expose Intel iGPU as test device)
  - Details: [REFACTOR-005-metal-shader-reorg/](REFACTOR-005-metal-shader-reorg/README.md)

## Backlog

- [ ] **FEAT-001b**: Fix `has_simdgroup_reduction` gate for non-Apple GPUs
  - Gate: `Apple7 || Metal3` — AMD passes (Metal3), Intel iGPU fails BOTH
  - Reality: `simd_sum`/`simd_max` are MSL 2.1 intrinsics, available on ALL Metal GPUs since macOS 10.14
  - Impact: Intel iGPU gets `has_simdgroup_reduction=false` → SOFT_MAX, NORM, MUL_MAT, FA all fall to CPU
  - Fix options: (a) detect from Common2+ instead, or (b) threadgroup fallback kernels
  - AMD works by accident (supports Metal3), Intel is locked out
  - Depends on: REFACTOR-005 (device enumeration fix to test Intel iGPU)

- [ ] **Multi-device**: Test both GPUs on this MacBook simultaneously
  - AMD (separate path) + Intel iGPU (shared path)
  - Scheduler already supports multi-backend in ggml-backend.cpp
  - Requires REFACTOR-005 device enumeration fix

- [ ] **NVIDIA Kepler on Metal**: Test on patched macOS (OpenCore Legacy Patcher)
  - GT 650M/750M etc. — separate memory, scalar, no matrix_hw
  - Same code path as AMD discrete, different driver
  - Profile struct already has `GGML_GPU_VENDOR_NVIDIA`
  - Real users exist — OCLP runs modern macOS on old Macs

- [ ] **FEAT-003**: RPC Device Profile Exchange
  - Server sends `ggml_device_profile` in HELLO, real `supports_op()`, bandwidth-weighted scheduling
  - Depends on: REFACTOR-005 (profile struct), FEAT-002 (adaptive dispatch)
  - Details: [FEAT-003-rpc-profile-exchange/](FEAT-003-rpc-profile-exchange/README.md)

## Architecture

See [BACKLOG-architecture-notes.md](BACKLOG-architecture-notes.md) for the full discussion:
- Two memory paths (shared vs separate)
- Device profile struct design
- Multi-GPU, RPC, NVLink analysis
- Why this MacBook tests both paths

## Benchmarks (2026-02-13, Qwen2.5-1.5B Q4_K_M)

| Config | pp512 (t/s) | tg128 (t/s) |
|--------|------------|------------|
| Metal FA=1 | 127.90 | 60.51 |
| Vulkan FA=1 | 103.73 | 68.69 |
| Metal real inference (after PERF-002) | ~127 | ~60 |
| Metal real inference (before PERF-002) | ~10 | ~5.5 |

## Upstream Issues

| Issue | Title | Status |
|-------|-------|--------|
| [#19601](https://github.com/ggml-org/llama.cpp/issues/19601) | Metal correctness fixes for AMD discrete | Posted, ball in their court |
| [#19600](https://github.com/ggml-org/llama.cpp/pull/19600) | mul_mv_ext for large n | PR open |
| [#19563](https://github.com/ggml-org/llama.cpp/issues/19563) | Metal garbage on AMD | Our issue |
| [#19431](https://github.com/ggml-org/llama.cpp/issues/19431) | FA on Intel/AMD Metal | All fixes posted as comment |
