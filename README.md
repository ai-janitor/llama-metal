# llama-metal

Metal GPU backend fixes for [llama.cpp](https://github.com/ggml-org/llama.cpp) on non-Apple-Silicon hardware.

## What This Is

llama.cpp's Metal backend assumes Apple Silicon — `simd_shuffle`, `quad_group` intrinsics, unified memory, SIMD width 32. This fork makes Metal work correctly on:

- **AMD Radeon** (discrete GPUs over PCIe, e.g. Radeon Pro 5300M)
- **Intel UHD/Arc** (integrated GPUs with SIMD width 16)

All changes are in the Metal backend (`ggml/src/ggml-metal/`). The rest of llama.cpp is unmodified upstream.

## Key Changes

### Managed Buffers
Discrete GPUs use `MTLResourceStorageModeManaged` for cached PCIe reads instead of Shared mode, which is slow without unified memory.

### SIMD Reduction Guards
AMD GPUs lack Apple's `simd_shuffle` / `quad_group` intrinsics. Kernels detect hardware at runtime and use portable fallback reduction paths.

### Dynamic SIMD Width
Intel iGPUs have SIMD width 16 (vs 32 on Apple/AMD). Kernels use `threads_per_simdgroup` instead of hardcoded constants for threadgroup sizing, shared memory allocation, and reduction loops.

### Vendor Verification
Every pipeline carries a `verified_vendors` bitmask. Unverified GPU+kernel combinations log a warning and fall back to safe paths. Vendor bits are added after passing `test-backend-ops` on that GPU.

### Modular Kernel Architecture
Monolithic shader files split into per-op directories (`matmul/`, `flash-attn/`, `norm/`, `softmax/`, etc.) with separate pipeline setup and dispatch files.

## Status

| GPU | test-backend-ops | Inference |
|-----|-----------------|-----------|
| AMD Radeon Pro 5300M | MUL_MAT 1009/1009, SSM_SCAN 3/3 | Working |
| Intel UHD 630 | In progress | In progress |

Flash attention on AMD uses a scalar fallback path (no `simd_matrix` support). Correct but slower than Apple Silicon's SIMD group path.

## Building

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j
```

For full build options, model setup, and usage, see the [llama.cpp documentation](https://github.com/ggml-org/llama.cpp).

## License

MIT — same as upstream llama.cpp.
