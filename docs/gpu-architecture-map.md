# GPU Architecture & Data Path Map

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CPU (Host)                                      │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐   │
│  │ ggml-metal-  │  │  dispatch.cpp │  │  ggml-metal-context.m            │   │
│  │ device.m     │  │  pipeline.cpp │  │  (set_tensor / get_tensor)       │   │
│  │              │  │              │  │                                    │   │
│  │ • probe simd │  │ • nsg calc   │  │  • buffer transfers               │   │
│  │ • profile    │  │ • smem size  │  │  • newBufferWithBytes/NoCopy      │   │
│  │ • library    │  │ • kernel pick│  │  • StorageMode selection          │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────────┘   │
│         │                 │                          │                        │
└─────────┼─────────────────┼──────────────────────────┼───────────────────────┘
          │                 │                          │
          ▼                 ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM RAM (DDR4)                                  │
│                                                                              │
│  ┌─────────────────────────────────────┐                                    │
│  │  Model weights, KV cache, tensors   │                                    │
│  └──────┬──────────────┬───────────┬───┘                                    │
│         │              │           │                                         │
└─────────┼──────────────┼───────────┼────────────────────────────────────────┘
          │              │           │
          │              │           │
  ┌───────┴───┐  ┌───────┴───┐  ┌───┴──────────────┐
  │  SHARED   │  │   PCIe    │  │      PCIe         │
  │ (zero-copy│  │   copy    │  │      copy         │
  │  UMA)     │  │           │  │                   │
  ▼           │  ▼           │  ▼                   │
              │              │                      │
┌─────────────┴──┐ ┌────────┴────────┐ ┌───────────┴──────────┐
│ INTEL iGPU     │ │ AMD 5500M/5600M │ │ AMD VEGA 56 / D700   │
│ (UHD 630)      │ │ (RDNA)          │ │ (GCN)                │
│                │ │                 │ │                      │
│ UMA ✓          │ │ UMA ✗           │ │ UMA ✗                │
│ Discrete ✗     │ │ Discrete ✓      │ │ Discrete ✓           │
│                │ │                 │ │                      │
│ ┌────────────┐ │ │ ┌─────────────┐ │ │ ┌──────────────────┐ │
│ │  NO VRAM   │ │ │ │  8GB VRAM   │ │ │ │  8GB VRAM (Vega) │ │
│ │ uses DDR4  │ │ │ │  128-bit bus│ │ │ │  6GB VRAM (D700)  │ │
│ │ directly   │ │ │ │  GDDR6     │ │ │ │  384-bit bus (D700)│ │
│ └────────────┘ │ │ └─────────────┘ │ │ │  HBM2 (Vega)     │ │
│                │ │                 │ │ └──────────────────┘ │
│ SIMD: 8-16     │ │ SIMD: 32       │ │ SIMD: 64             │
│ Wave16         │ │ Wave32         │ │ Wave64               │
│                │ │                 │ │                      │
│ ┌────────────┐ │ │ ┌─────────────┐ │ │ ┌──────────────────┐ │
│ │  KERNELS   │ │ │ │  KERNELS    │ │ │ │  KERNELS         │ │
│ │            │ │ │ │             │ │ │ │                  │ │
│ │ NW=16      │ │ │ │ NW=32      │ │ │ │ NW=64            │ │
│ │ nsg=8      │ │ │ │ nsg=4      │ │ │ │ nsg=2            │ │
│ │ simd_sum() │ │ │ │ simd_sum() │ │ │ │ simd_sum()       │ │
│ │ 16 lanes   │ │ │ │ 32 lanes   │ │ │ │ 64 lanes         │ │
│ └────────────┘ │ │ └─────────────┘ │ │ └──────────────────┘ │
│                │ │                 │ │                      │
│ BUFFER MODE:   │ │ BUFFER MODE:    │ │ BUFFER MODE:         │
│ StorageShared  │ │ StoragePrivate  │ │ StoragePrivate       │
│ (zero-copy OK) │ │ (PCIe copy)    │ │ (PCIe copy)          │
│                │ │                 │ │                      │
│ get_tensor:    │ │ get_tensor:     │ │ get_tensor:          │
│ NoCopy ✓       │ │ NoCopy ✓*      │ │ NoCopy ✗ CRASHES     │
│                │ │ (RDNA driver   │ │ (D700 returns nil)   │
│                │ │  allows it)    │ │ ← OPEN BUG           │
└────────────────┘ └─────────────────┘ └──────────────────────┘
```

## Code Module Map

| Module | Responsibility | Affected by SIMD width? | Affected by memory model? |
|--------|---------------|------------------------|--------------------------|
| `kernels/00-common.metal` | `N_SIMDWIDTH` definition | YES — function_constant per device | No |
| `ggml-metal-device.m` | Probe `threadExecutionWidth`, compile pipelines | YES — injects FC_SIMD_WIDTH | YES — `use_shared_buffers` |
| `matmul/dispatch.cpp` | `nsg` calculation, kernel selection | YES — `nsg = tg / simd_width` | No |
| `matmul/pipeline.cpp` | smem sizing, pipeline compilation | YES — `smem = simd_width * sizeof(float)` | No |
| `norm/pipeline-norm.cpp` | Norm kernel smem sizing | YES — same smem fix | No |
| `reduction/pipeline-reduction.cpp` | Reduction kernel smem sizing | YES — same smem fix | No |
| `flash-attn/ops-flash-attn.cpp` | Vec vs scalar kernel routing | YES — vec assumes NW=32 | No |
| `ggml-metal-context.m` | Buffer transfers (set/get tensor) | No | YES — `newBufferWithBytesNoCopy` crash on D700 |

## Three Separate Concerns

1. **SIMD width** (dispatch fix) — affects all three GPU types, same code path
2. **Memory model** (buffer allocation) — UMA vs discrete, D700 `newBufferWithBytesNoCopy` crash is a separate bug
3. **Flash-attn routing** — vec kernel only safe on Wave32 until made NW-parametric

## Device Summary

| GPU | Arch | SIMD | UMA | Memory | Bus | `shmem_reduce` | Native dispatch (target) |
|-----|------|------|-----|--------|-----|----------------|-------------------------|
| Intel UHD 630 | Gen9 | **8/16/32** | Yes | DDR4 shared | N/A | Yes (required) | N/A — shmem_reduce permanent |
| AMD 5500M/5600M | RDNA 1 | 32 | No | 8GB GDDR6 | 128-bit | No (native) | NW=32, nsg=4 (unchanged) |
| AMD Vega 56 | GCN 5.0 | 64 | No | 8GB HBM2 | 2048-bit | No (native) | NW=64, nsg=2 |
| AMD D700 | GCN 1.0 | 64 | No | 6GB GDDR5 | 384-bit | No (native) | NW=64, nsg=2 (+ buffer fix needed) |
| Apple M1/M2/M3 | Apple | 32 | Yes | Unified | N/A | No (native) | NW=32, nsg=4 (unchanged) |

### Intel iGPU: Why shmem_reduce is permanent

Intel UHD 630 `threadExecutionWidth` varies **per pipeline** based on register pressure:

| Kernel category | th_width | th_max |
|----------------|----------|--------|
| Simple (concat, short variants) | 32 | 1024 |
| Matmul f32/f16/bf16, q4_0/q4_1/q4_K/q6_K/q8_0, iq4_*, mxfp4, tiled | 16 | 896 |
| Heavy quants q2_K/q3_K/q5_0/q5_1/q5_K, iq1_*/iq2_*/iq3_* | 8 | 448 |

A single global `FC_SIMD_WIDTH` cannot capture this — NW would need to be set per-pipeline, not per-device. Additionally, Intel's `simd_sum()` produces wrong results (BUG-005), so `shmem_reduce` is required for correctness regardless of SIMD width.

Native dispatch for Intel would require per-pipeline SIMD width detection and verified `simd_sum()` correctness — not planned.

## Memory Architecture: The Three Towers

### Tower 1: Intel iGPU (UMA — Unified Memory Access)

```
┌──────────────┐
│   GPU cores  │
│   Wave16     │
└──────┬───────┘
       │ direct access (zero-copy)
       ▼
┌──────────────────────┐
│    SYSTEM RAM        │
│    (shared)          │
│    model weights     │
│    KV cache          │
└──────────────────────┘
```

**Code path:**
```
ggml-metal-device.m:
  has_unified_memory = true
  use_shared_buffers = true

Buffer allocation:
  newBufferWithBytesNoCopy()   ← GPU reads CPU pointer directly
  MTLResourceStorageModeShared ← both sides see same physical memory

get_tensor (read back to CPU):
  newBufferWithBytesNoCopy()   ← works, same shared memory
```

No PCIe transfer. No VRAM. GPU reads system RAM directly. Lowest latency for small models that fit in RAM.

### Tower 2: AMD RDNA (Discrete — Separate VRAM)

```
┌──────────────┐
│   GPU cores  │
│   Wave32     │
└──────┬───────┘
       │ direct access
┌──────┴───────┐
│   8GB VRAM   │          ▲
│   GDDR6      │          │ PCIe copy (once at model load)
│   128-bit    │          │
└──────────────┘          │
                   ┌──────┴───────────┐
                   │   SYSTEM RAM     │
                   │   (host side)    │
                   └──────────────────┘
```

**Code path:**
```
ggml-metal-device.m:
  has_unified_memory = false
  use_shared_buffers = false

Buffer allocation (model load):
  newBufferWithBytes()          ← copies data from RAM → VRAM (once)
  MTLResourceStorageModePrivate ← lives in VRAM only

During inference:
  Everything stays in VRAM     ← no transfers
  Only input tokens in (tiny), logits out (tiny)

get_tensor (read back to CPU):
  newBufferWithBytesNoCopy()   ← RDNA driver allows this
  Blit encoder: VRAM → temp shared buffer → CPU
```

One-time PCIe copy at load. During inference, GPU works entirely from VRAM. The 128-bit GDDR6 bus feeds the cores.

### Tower 3: AMD GCN (Discrete — Separate VRAM, Legacy)

```
┌──────────────┐
│   GPU cores  │
│   Wave64     │
└──────┬───────┘
       │ direct access
┌──────┴───────┐
│   VRAM       │          ▲
│   HBM2/GDDR5│          │ PCIe copy (once at model load)
│   384-bit+   │          │
└──────────────┘          │
                   ┌──────┴───────────┐
                   │   SYSTEM RAM     │
                   │   (host side)    │
                   └──────────────────┘
```

**Code path:**
```
ggml-metal-device.m:
  has_unified_memory = false
  use_shared_buffers = false

Buffer allocation (model load):
  newBufferWithBytes()          ← copies data from RAM → VRAM (once)
  MTLResourceStorageModePrivate ← lives in VRAM only

During inference:
  Same as Tower 2              ← should work the same

get_tensor (read back to CPU):
  newBufferWithBytesNoCopy()   ← RETURNS NIL on D700 ← BUG
  GGML_ASSERT(buf_dst)        ← CRASH

  FIX NEEDED (ggml-metal-context.m:350):
    if (!buf_dst) {
        buf_dst = [device newBufferWithLength:size
                           options:MTLResourceStorageModeShared];
        // blit from VRAM to this buffer, then memcpy to CPU
    }
```

Same as RDNA for loading weights, but the `get_tensor` read-back path crashes because the D700's older Metal driver doesn't support `newBufferWithBytesNoCopy` with `StorageModeShared` on non-UMA hardware. Needs an explicit allocation + blit fallback.

### When does PCIe transfer happen?

| Phase | UMA (Intel) | Discrete (RDNA/GCN) |
|-------|-------------|---------------------|
| Model load | No transfer (shared) | One-time copy RAM → VRAM |
| Prompt input | No transfer (shared) | Tiny copy (tokens) |
| Inference compute | No transfer | No transfer (all in VRAM) |
| Read output logits | No transfer (shared) | Tiny copy VRAM → RAM |
| KV cache growth | No transfer (shared) | Allocated in VRAM directly |

The PCIe bus is only a bottleneck at model load time. During inference, discrete GPUs work entirely from VRAM — the wider bus (384-bit D700, 2048-bit Vega HBM2) is what feeds the compute cores.

## Commit History

| Commit | Description | Status |
|--------|------------|--------|
| `deaceab` | float-cast simd_max/min(half) for GCN | Merged to main |
| `bd29936` | NW=64 function constant (broke dispatch) | Reverted |
| `54baab7` | shmem_reduce workaround for non-32 SIMD | Current on main |
| `83d1d88` | Route non-32 SIMD to scalar flash-attn | Current on main |
| Branch: `native-wave-dispatch` | Native NW per device, fix dispatch | In progress |
