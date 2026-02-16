# MSL Portability Tiers

Reference for writing Metal kernels that work across all GPU vendors.

**Based on real-world testing:**
- Apple Silicon (M1/M2/M3 family)
- AMD RDNA (Radeon Pro 5300M)
- Intel Gen 9.5 (UHD 630)

**Last updated:** 2026-02-14

---

## Tier 1: PROVEN (tested on all 3 GPU families, identical output)

Tested on Apple Silicon, AMD RDNA, Intel Gen 9.5. Same input → same output. Ship one kernel.

| Feature | Notes |
|---------|-------|
| Basic arithmetic | `+`, `-`, `*`, `/`, `fma`, `min`, `max`, `clamp`, `abs`, `floor`, `ceil`, `round` |
| Type casts | `float()`, `half()`, `int()`, `uint()`, `short()`, `uchar()` |
| Reinterpret casts | `as_type<T>()` for bitwise reinterpretation |
| Comparison ops | `<`, `>`, `==`, `!=`, `<=`, `>=`, `select()` |
| Math functions | `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `erf` |
| Buffer access | `device T*`, `constant T*` pointers |
| Threadgroup memory | `threadgroup T[]` declaration and access |
| Memory barriers | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Thread builtins | `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`, `threads_per_threadgroup`, `simdgroup_index_in_threadgroup`, `thread_index_in_simdgroup` |
| Atomic ops (int/uint) | `atomic_fetch_add`, `atomic_fetch_max`, `atomic_fetch_min`, `atomic_compare_exchange_weak` |
| Texture operations | Read, write, sample |
| SIMD ops (float) | `simd_sum(float)`, `simd_max(float)`, `simd_min(float)`, `simd_shuffle`, `simd_shuffle_xor`, `simd_broadcast` |

---

## Tier 2: VENDOR-SPECIFIC (works but behavior differs per GPU)

Behavior varies by vendor. Each vendor gets its own code path or explicit handling.
Don't write one "portable" version — write per-vendor copies, each proven for that GPU.

| Feature | Caveat | Affected GPU | Mitigation |
|---------|--------|-------------|------------|
| Threadgroup memory initialization | NOT guaranteed zero on allocation | Intel (no zero), AMD/Apple (zero) | Always explicitly initialize before use |
| `threadExecutionWidth` | Always 32 in Metal, but native HW width varies | AMD native 64, Intel native 8-16, Apple 32 | Don't assume native width matches Metal width |
| MTLResourceStorageModeShared | 12x slower on discrete GPUs (PCIe vs VRAM bandwidth) | AMD, Intel (discrete) | Use StorageModePrivate on non-UMA (`has_unified_memory=false`) |
| SIMD half ops (tested) | `simd_shuffle`, `simd_broadcast` work but less tested | All | Test thoroughly, prefer float casts for critical ops |
| Float precision edge cases | Denormals, NaN handling may vary slightly | All | Avoid relying on exact bit patterns |

---

## Tier 3: GATED (requires capability check, not all GPUs have it)

Only available on some GPUs. MUST gate with `supports_op` or device property.
If the gate fails: crash loud or provide a PROVEN Tier 1 fallback. Never silently degrade.

| Feature | Gate | Available on | Missing on | Fallback |
|---------|------|-------------|-----------|----------|
| simdgroup_matrix (simdgroup_mm) | `has_simdgroup_mm` | Apple7+ (Apple Silicon) | AMD, Intel | Use standard matmul kernel |
| simdgroup_reduction intrinsics | `has_simdgroup_reduction` | Common2+ (all current Metal GPUs) | Legacy GPUs | Manual reduction loops |
| BFloat16 (bfloat) | `has_bfloat` | Metal3+ (M3+) | M1, M2, AMD, Intel | Use FP16 or FP32 |
| MTLGPUFamilyApple7+ features | `MTLGPUFamilyApple7` check | Apple Silicon | AMD, Intel | Feature-specific fallback |
| MTLGPUFamilyMetal3 features | `MTLGPUFamilyMetal3` check | M3+ | M1, M2, AMD, Intel | Feature-specific fallback |

---

## Tier 4: BROKEN (compiles, runs, produces wrong output)

**BANNED.** Compiles without error. Runs without error. Returns garbage.
No warning, no crash — just wrong answers. These are the most dangerous bugs in GPU programming.

| Feature | Bug | Affected GPU | Workaround | Work Item |
|---------|-----|-------------|------------|-----------|
| `simd_max(half)` with -INF | Returns -INF instead of finite max when operands include `half(-INFINITY)` (0xFC00) | AMD RDNA | Cast to float first: `simd_max(float(x))` | BUG-002 |
| `simd_min(half)` with +INF | Likely returns +INF instead of finite min (untested but assume broken by symmetry) | AMD RDNA (suspected) | Cast to float first: `simd_min(float(x))` | BUG-002 (inferred) |
| MTLDispatchTypeConcurrent on non-UMA | Produces NaN output due to race conditions in discrete VRAM | AMD, Intel (discrete GPUs) | Auto-disable when `has_unified_memory=false` | BUG-003 |

---

## Rules for Kernel Authors

### 1. Tier 1 (PROVEN): Use freely
Tested and verified across all GPU families. One kernel, ship it.

### 2. Tier 2 (VENDOR-SPECIFIC): Separate code paths per vendor
Don't write one "clever" version that tries to handle all vendors.
Write per-vendor copies. Each copy is proven for that GPU.
- **Threadgroup memory:** Apple/AMD path can skip init. Intel path MUST zero explicitly.
- **Storage mode:** UMA path uses Shared. Non-UMA path uses Private. Two paths.
- **SIMD half:** Apple path uses half directly. AMD path casts to float. Two paths.

### 3. Tier 3 (GATED): Gate AND provide fallback or crash
Every Tier 3 feature MUST have:
1. A `supports_op` gate or device property check in `ggml-metal-device.m`
2. A PROVEN Tier 1 fallback kernel, OR
3. `GGML_ABORT` with descriptive error message — never silent CPU fallback

**Example:**
```cpp
if (ctx->device->has_simdgroup_mm) {
    [encoder setComputePipelineState:ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MM].pipeline];
} else {
    [encoder setComputePipelineState:ctx->kernels[GGML_METAL_KERNEL_TYPE_MUL_MV].pipeline];
}
```

### 4. Tier 4 (BROKEN): BANNED — use workaround only
NEVER use the raw intrinsic. Always use the proven workaround.

**Examples:**
```metal
// BANNED on AMD:
half max_val = simd_max(sm[tiisg]);  // Wrong results with -INF

// REQUIRED workaround:
half max_val = simd_max(float(sm[tiisg]));  // Correct
```

```cpp
// BANNED on non-UMA:
encoder.dispatchType = MTLDispatchTypeConcurrent;  // NaN output

// REQUIRED check:
if (ctx->device->has_unified_memory) {
    encoder.dispatchType = MTLDispatchTypeConcurrent;
} else {
    // Use serial dispatch
}
```

### 5. Testing Requirements
Every new kernel MUST be tested on at least 2 GPU families before merge:
- **Minimum:** Apple Silicon + AMD (or Intel)
- **Preferred:** Apple Silicon + AMD + Intel
- **Test cases must include:** edge cases (±INF, NaN, zero, large values)

### 6. When In Doubt: Cast Half to Float
SIMD operations on half precision are less battle-tested on AMD/Intel.

**Safe pattern:**
```metal
float safe_result = simd_max(float(half_value));
half result = safe_result;  // Cast back if needed
```

The float path is Tier 1 (universal safe). The half path is Tier 2 (less tested) or Tier 4 (broken).

### 7. Document GPU-Specific Behavior
When you encounter new vendor-specific behavior:
1. Add it to this document (correct tier)
2. Add a comment in the kernel code
3. Update the work item (if part of active investigation)
4. Update `~/.claude/projects/-Users-hung-projects-llama-cpp/memory/MEMORY.md`

---

## Testing Checklist for New Kernels

Before submitting a kernel that uses any Tier 2/3/4 features:

- [ ] Tested on Apple Silicon
- [ ] Tested on AMD RDNA (Radeon Pro 5300M or similar)
- [ ] Tested on Intel (if available)
- [ ] Edge cases tested: ±INF, NaN, zero, max values
- [ ] Tier 3 features have `supports_op` gates
- [ ] Tier 4 features are not used (workarounds only)
- [ ] Threadgroup memory explicitly initialized (if used)
- [ ] Storage mode appropriate for UMA vs non-UMA

---

## Version History

| Date | Change | Author |
|------|--------|--------|
| 2026-02-14 | Initial version based on BUG-002, BUG-003, PERF-002 findings | Claude (Opus 4.6) |

---

## References

- **BUG-002:** `~/.knowledge/llama-cpp/work/BUG-002-metal-amd-fa-vec/` — simd_max(half) -INF bug
- **BUG-003:** `~/.knowledge/llama-cpp/work/BUG-003-metal-amd-inference-garbage/` — MTLDispatchTypeConcurrent NaN bug
- **PERF-002:** `~/.knowledge/llama-cpp/work/PERF-002-metal-amd-graph-splits/T4-real-inference-slowdown.md` — mmap + StorageModeShared slowdown
- **MEMORY.md:** `~/.claude/projects/-Users-hung-projects-llama-cpp/memory/MEMORY.md` — consolidated Metal AMD learnings
