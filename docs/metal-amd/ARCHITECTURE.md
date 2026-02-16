# Architecture Notes — Device Model & Multi-GPU Strategy

**Date**: 2026-02-14
**Context**: Deep discussion on how to structure the Metal backend for all GPU types

## Core Insight: Two Memory Paths

Every GPU in every computer falls into one of two paths:

1. **Shared memory** — CPU and GPU see the same RAM. Zero copy.
2. **Separate memory** — GPU has its own RAM. Must copy data to it.

**Everything else is just transfer speed.** PCIe (8 GB/s), NVLink (900 GB/s), Thunderbolt (5 GB/s), ethernet (1 GB/s), TCP, RDMA — all different speeds of the copy in path 2. Not different code paths.

## Inference Flow (Universal)

Every inference, every machine:

1. CPU loads model from disk into CPU RAM — always
2. **Shared path**: Done. GPU already sees the weights.
3. **Separate path**: Copy weights from CPU RAM → GPU VRAM. Once at load time.

The mmap fix (PERF-002) was about this — weights were staying in CPU RAM (SharedMode), and the GPU was reading over PCIe every token instead of copying once to VRAM.

## Compute Tier (Orthogonal to Memory Path)

Independent of whether memory is shared or separate:

- **Matrix HW** — hardware matrix multiply (Apple simdgroup_mm, NVIDIA Tensor Cores, AMD WMMA via Vulkan)
- **Scalar** — general ALU, no matrix accelerator

Shared + Matrix HW = Apple M1+. Shared + Scalar = Intel iGPU. Separate + Scalar = AMD discrete (and NVIDIA Kepler via OpenCore Legacy Patcher). Separate + Matrix HW = NVIDIA discrete (future/non-Mac). All four quadrants exist.

## Multi-Device: Just More Rows in the Pool

Adding a device (local or remote) is adding a row to the device pool. The scheduler reads profiles and splits layers by effective bandwidth and compute capacity.

This MacBook has both paths on one machine:
- Device 0: AMD 5300M → separate memory (test separate path)
- Device 1: Intel UHD 630 → shared memory (test shared path)

This makes it a complete test rig for both code paths without needing Apple Silicon.

## Remote Devices (ggml-rpc)

`ggml-rpc` already exists in llama.cpp — TCP-based, proof of concept.

### How it works today
- **Slave**: `rpc-server` on 2nd machine, exposes its GPU over TCP port 50052
- **Master**: `llama-cli --rpc ip:50052` connects, scheduler sees remote GPU
- Splits layers by VRAM proportion (dumb — no bandwidth awareness)
- Has local cache (`-c`) to avoid re-transferring model weights

### What it's missing
- **No profile exchange** — master doesn't know remote device's capabilities
- **No bandwidth awareness** — splits by VRAM only, ignores transfer cost
- **TCP only** — no RDMA option (macOS doesn't support RDMA anyway)
- **No smart scheduling** — doesn't consider shared vs separate on remote end

### Data flow for remote inference
```
Your GPU → your CPU RAM → wire → their CPU RAM → their GPU
```
Even with RDMA, it's CPU RAM to CPU RAM. The remote GPU still needs a local copy unless it's shared memory (Apple Silicon). A remote M-series Mac is the ideal slave — data lands in CPU RAM and the GPU already sees it (one hop, not two).

### Connection types
| Transport | Bandwidth | Protocol | macOS Support |
|-----------|-----------|----------|--------------|
| TCP over ethernet | ~1 GB/s | ggml-rpc (today) | Yes |
| TCP over Thunderbolt | ~3-4 GB/s | ggml-rpc (today) | Yes |
| RDMA over Thunderbolt | ~5 GB/s | Would need new transport | No (Apple doesn't expose RDMA) |
| NVLink | ~900 GB/s | NVIDIA proprietary | N/A (not Mac) |

### TCP overhead for activations
Activations between layers are small (~4MB per token). At 3 GB/s TCP over Thunderbolt, that's 1.3ms transfer per layer boundary. Token generation is ~15ms. So ~8% overhead — workable if the remote GPU is fast enough to justify it.

## NVIDIA Fabric / NVLink

NVLink is hardware — dedicated silicon-to-silicon interconnect. Not a protocol, not software. Soldered on the board.

- **NVLink**: point-to-point between 2 GPUs (900 GB/s)
- **NVSwitch**: hub connecting 8+ GPUs, any-to-any
- **NVLink-C2C**: connects GPU to CPU (Grace Hopper) — makes separate memory look shared
- **NVFabric**: extends NVSwitch across racks

NVIDIA is spending billions moving everything toward shared memory. NVLink-C2C (Grace Hopper) is "fake shared" — separate chips, separate memory, but hardware makes it look unified to software.

From the profile struct perspective, NVLink is just `shared_memory=false, transfer_bandwidth=900GB/s`. Not a new path — just a fat pipe.

## Exo (Distributed Inference)

Exo splits model layers across multiple machines. Each node runs the same code, same two memory paths locally. The "innovation" is just the orchestration layer on top.

`ggml-rpc` already has the same plumbing. The difference is ggml-rpc is dumb (splits by VRAM proportion) while exo has a smarter scheduler. With the device profile struct, ggml-rpc could be equally smart.

## Device Profile Struct (Final Design)

```c
struct ggml_device_profile {
    // Identity
    enum ggml_gpu_vendor     vendor;     // Apple, AMD, Intel, NVIDIA, Unknown
    char                     name[64];

    // Memory path (the only code path decision)
    bool     shared_memory;              // true = zero copy, false = must transfer
    uint64_t vram_size;                  // device-local memory capacity
    uint64_t local_bandwidth;            // device-local memory speed
    uint64_t transfer_bandwidth;         // cost to move data here (PCIe, TB, NVLink, network)

    // Compute tier (which kernels can run)
    bool     has_matrix_hw;              // simdgroup_mm / tensor cores / cooperative matrix
    bool     has_simd_reduction;         // simd_sum/max vs threadgroup fallback
    uint32_t compute_units;              // CUs, SMs, EUs
    uint32_t simd_width;                 // threads per simdgroup/warp
    uint32_t max_threads_per_threadgroup;
    uint32_t shared_mem_size;            // threadgroup/shared memory bytes
};
```

Properties:
- **Backend-agnostic** — no Metal/Vulkan/CUDA vocabulary
- **Append-only** — new fields never break old code
- **Lives at `ggml/include/`** — any backend can populate it
- **`shared_memory`** is the only code path decision. Everything else is tuning.

## Key Decisions Made

1. **Two paths, not three or four.** Shared vs separate. RPC is just separate with a longer pipe.
2. **Profile struct at ggml/ level**, not inside ggml-metal/. Backend-agnostic from day one.
3. **`shared_memory` not `unified_memory`** — clearer intent, no vendor jargon.
4. **Transfer bandwidth as a single number** — no hops enum, no connection type enum. Just bytes/sec. The scheduler does math.
5. **Move, don't rewrite** — reorganize GG's code into the right files. Every line traces back to upstream.
6. **This MacBook is both test rigs** — Intel iGPU (shared) + AMD discrete (separate) tests both paths.

## Related Documents

| Document | Relation |
|----------|----------|
| [REFACTOR-005 README](REFACTOR-005-metal-shader-reorg/README.md) | Implementation plan for the reorg |
| [FEAT-002 README](FEAT-002-metal-adaptive-dispatch/README.md) | Adaptive dispatch uses the profile |
| [Project README](../README.md) | Architecture overview |
| [BACKLOG.md](BACKLOG.md) | Work item status |
