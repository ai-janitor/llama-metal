# Architecture Pattern: No Silent Fallback

## Rule

**Every op either runs on the target device or fails loudly. No silent CPU fallback.**

**Fallback = unfinished code.** If a device can't run an op, that means the kernel hasn't been written yet. Either write it or disable the feature. Don't ship incomplete work and let the scheduler hide it.

## Why

Silent fallback is silent failure. When `supports_op` returns false and the scheduler quietly moves the op to CPU:

1. **User thinks they have GPU acceleration.** They don't.
2. **Performance degrades with no indication.** 9 t/s instead of 35 t/s, no warning.
3. **GPU↔CPU bouncing kills throughput.** 56 command buffer syncs per inference pass.
4. **Bugs hide.** Nobody notices the missing kernel because "it still runs."
5. **Code never gets fixed.** The fallback removes the pressure to implement the real path.

## Pattern

### At device level: `supports_op`
- Returns `true` = this device WILL run this op correctly and efficiently
- Returns `false` = this device CANNOT run this op

### At model level: feature gating
- Before enabling a feature (FA, quantization, etc.), check if the device supports ALL ops it requires
- If not → **disable the feature entirely**, don't enable it and let individual ops scatter to CPU
- Log once: `"Flash Attention disabled on Intel UHD 630 — no GPU kernel available"`

### At scheduler level: no cross-device op migration
- If an op is assigned to a device, it runs there or the graph build fails
- No silent promotion to CPU
- No "well the CPU backend can do it" escape hatch

## Anti-patterns

### BAD: Silent CPU fallback
```
supports_op(FA) → false on iGPU
scheduler: "CPU can do FA, I'll just put it there"
result: GPU→CPU→GPU→CPU→GPU 56 times per pass, 4x slower, user has no idea
```

### BAD: Partial feature support
```
supports_op(FA_VEC) → true   (small batch)
supports_op(FA_SCALAR) → false  (large batch)
result: FA works for tg, silently degrades for pp, inconsistent behavior
```

### GOOD: Feature-level gate
```
device_supports_feature(FA) → false on iGPU
→ FA disabled for this device
→ standard attention used (all on GPU, 35 t/s)
→ log: "FA disabled on Intel UHD 630"
→ user knows, performance is consistent
```

### GOOD: Fail loud
```
supports_op(FA) → false
no CPU backend registered for this op
→ build fails: "FLASH_ATTN_EXT not supported on Intel UHD 630, disable FA with -fa 0"
→ user knows exactly what to do
```

## Implementation Checklist

For every new kernel/feature:
- [ ] `supports_op` returns true ONLY if the device has a working, tested kernel
- [ ] Feature gate checks ALL required ops before enabling
- [ ] Missing support → feature disabled + one-time log message
- [ ] No op ever silently migrates to a different device
- [ ] `GGML_SCHED_DEBUG=1` confirms zero unexpected splits
