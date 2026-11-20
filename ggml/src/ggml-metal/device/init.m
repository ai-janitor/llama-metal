#import "device/ggml-metal-device-internal.h"
#include <pthread.h>
#include "ggml-metal-probe.h"
#include "lanes/select.h"
#include "matmul/batch-route.h"
#include "matmul/tiled-plan.h"
GGML_METAL_FAMILY(device) // overload of MTLGPUFamilyMetalX (not available in some environments)

// The wave width of the card, read from a compiled pipeline of lib (kernel_concat_f32 has no function constants;
// every kernel of a device has the same width).
static void ggml_metal_device_probe_wave(ggml_metal_device_t dev, ggml_metal_library_t lib) {
    struct ggml_metal_pipeline_with_params probe =
        ggml_metal_library_compile_pipeline(lib, "kernel_concat_f32", "kernel_concat_f32", NULL);
    if (probe.pipeline) {
        const int tw = ggml_metal_pipeline_thread_execution_width(probe);
        if (tw > 0) {
            dev->profile.simd_width = (uint32_t) tw;
            dev->card.wave          = (uint32_t) tw;
            dev->card.wave_probed   = 1;
            lib->simd_width         = tw;
            GGML_LOG_INFO("%s: probed simd_width = %d\n", __func__, tw);
        }
    }
}

// One lock for every device: a library is built and probed by one caller at a time.
static pthread_mutex_t ggml_metal_device_library_mutex = PTHREAD_MUTEX_INITIALIZER;

ggml_metal_library_t ggml_metal_device_ensure_library(ggml_metal_device_t dev) {
    pthread_mutex_lock(&ggml_metal_device_library_mutex);
    if (dev->library == NULL) {
        ggml_metal_library_t lib = ggml_metal_library_init(dev);
        if (lib != NULL) {
            // probed first, published after: a reader of dev->library never sees a library without its width
            ggml_metal_device_probe_wave(dev, lib);
            // tuning-load-after-probe: the card's tuning rows are loaded once its probed facts (the wave width) are
            // in the object, so a row may depend on them; before this the object holds no rows.
            ggml_metal_tiled_tune_load(&dev->card, dev->profile.name);
            dev->library = lib;
        }
    }
    ggml_metal_library_t res = dev->library;
    pthread_mutex_unlock(&ggml_metal_device_library_mutex);
    return res;
}
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

// The PCI vendor id of a Metal device, from its IORegistry parents; 0 when not found. It tells the maker only: a card's
// table rows are keyed by its Metal name.
static uint32_t ggml_metal_pci_prop(io_registry_entry_t entry, CFStringRef key) {
    CFTypeRef data = IORegistryEntrySearchCFProperty(entry, kIOServicePlane, key, kCFAllocatorDefault,
                                                     kIORegistryIterateRecursively | kIORegistryIterateParents);
    uint32_t value = 0;
    if (data && CFGetTypeID(data) == CFDataGetTypeID() && CFDataGetLength((CFDataRef) data) >= 4) {
        const UInt8 * b = CFDataGetBytePtr((CFDataRef) data);
        value = (uint32_t) b[0] | ((uint32_t) b[1] << 8);
    }
    if (data) {
        CFRelease(data);
    }
    return value;
}

static uint32_t ggml_metal_pci_vendor(id<MTLDevice> mtl_device) {
    uint32_t vendor = 0;
    io_registry_entry_t entry = IOServiceGetMatchingService(MACH_PORT_NULL, IORegistryEntryIDMatching(mtl_device.registryID));
    if (entry) {
        vendor = ggml_metal_pci_prop(entry, CFSTR("vendor-id"));
        IOObjectRelease(entry);
    }
    return vendor;
}

ggml_metal_device_t ggml_metal_device_init(int device) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    if (dev->mtl_device == nil) {
        // Enumerate GPUs with system default first (high-perf GPU).
        // device 0 = system default, device 1+ = remaining GPUs in enumeration order.
        {
            id<MTLDevice> defaultDevice = MTLCreateSystemDefaultDevice();
            NSArray<id<MTLDevice>> *allDevices = MTLCopyAllDevices();

            // Build ordered list: default device first, then others
            NSMutableArray<id<MTLDevice>> *ordered = [NSMutableArray arrayWithCapacity:allDevices.count];
            [ordered addObject:defaultDevice];
            for (id<MTLDevice> d in allDevices) {
                if (d.registryID != defaultDevice.registryID) {
                    [ordered addObject:d];
                }
            }

            if (device < (int)ordered.count) {
                dev->mtl_device = [ordered[device] retain];
            } else {
                GGML_LOG_WARN("%s: device index %d out of range (%d devices), using default\n",
                    __func__, device, (int)ordered.count);
                dev->mtl_device = [defaultDevice retain];
            }

            [allDevices release];
            [defaultDevice release];
        }

        if (dev->mtl_device) {
            dev->mtl_queue = [dev->mtl_device newCommandQueue];
            if (dev->mtl_queue == nil) {
                GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            }

            dev->addr_virt = 0x000000400ULL;

            dev->props.device = device;
            // simd_sum/simd_max are MSL 2.1 intrinsics, available on all Common2+ devices (macOS 10.14+)
            // This includes Intel iGPU, AMD discrete, and Apple Silicon
            dev->props.has_simdgroup_reduction = [dev->mtl_device supportsFamily:MTLGPUFamilyCommon2];

            dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;
            dev->props.is_low_power       = dev->mtl_device.isLowPower;

            dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
            if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                dev->props.has_bfloat = false;
            }
            // the 16-bit storage kernels cover the scalar, tiled and vec paths only (no simdgroup matrix kernels)
            dev->props.has_bf16_u16 = !dev->props.has_bfloat && !dev->props.has_simdgroup_mm &&
                getenv("GGML_METAL_BF16_U16_DISABLE") == NULL;

            dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
            if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                dev->props.has_tensor = false;
            }

            // note: disable the tensor API by default for old chips because with the current implementation it is not useful
            // - M2 Ultra:   ~5% slower
            // - M4, M4 Max: no significant difference
            //
            // TODO: try to update the tensor API kernels to at least match the simdgroup performance
            if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                ![[dev->mtl_device name] containsString:@"M5"] &&
                ![[dev->mtl_device name] containsString:@"M6"] &&
                ![[dev->mtl_device name] containsString:@"A19"] &&
                ![[dev->mtl_device name] containsString:@"A20"]) {
                GGML_LOG_WARN("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                dev->props.has_tensor = false;
            }

            // double-check that the tensor API compiles
            if (dev->props.has_tensor) {
                const char * src_tensor_f16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(8, 8, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                    dev->props.has_tensor = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
            if (dev->props.has_tensor && dev->props.has_bfloat) {
                const char * src_tensor_bf16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(8, 8, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                    dev->props.has_bfloat = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            ggml_metal_device_placement_init(dev);

            // Profile: identity
            const uint32_t pci_vendor = ggml_metal_pci_vendor(dev->mtl_device);
            dev->profile.vendor = GGML_GPU_VENDOR_UNKNOWN;
            if (pci_vendor == 0x1002) dev->profile.vendor = GGML_GPU_VENDOR_AMD;
            if (pci_vendor == 0x8086) dev->profile.vendor = GGML_GPU_VENDOR_INTEL;
            if (pci_vendor == 0) {
                if ([[dev->mtl_device name] containsString:@"AMD"])   dev->profile.vendor = GGML_GPU_VENDOR_AMD;
                if ([[dev->mtl_device name] containsString:@"Intel"]) dev->profile.vendor = GGML_GPU_VENDOR_INTEL;
                if ([[dev->mtl_device name] containsString:@"Apple"]) dev->profile.vendor = GGML_GPU_VENDOR_APPLE;
            }
            strlcpy(dev->profile.name, [[dev->mtl_device name] UTF8String], sizeof(dev->profile.name));

            // Profile: memory path
            dev->profile.shared_memory      = dev->props.has_unified_memory;
            dev->profile.vram_size           = dev->props.max_working_set_size;
            dev->profile.local_bandwidth     = 0;
            dev->profile.transfer_bandwidth  = 0;

            // Profile: compute tier
            dev->profile.has_matrix_hw               = dev->props.has_simdgroup_mm;
            dev->profile.has_simd_reduction          = dev->props.has_simdgroup_reduction;
            dev->profile.compute_units               = 0;
            dev->profile.simd_width                  = 32; // default; probed below after library init
            dev->profile.max_threads_per_threadgroup = (uint32_t)dev->mtl_device.maxThreadsPerThreadgroup.width;
            dev->profile.shared_mem_size             = (uint32_t)dev->mtl_device.maxThreadgroupMemoryLength;
            dev->card.wave      = 32; // default; probed below after library init
            ggml_metal_tiled_tune_load(&dev->card, ""); // no entries until the library is built and the card probed
            ggml_metal_tiled_test_arm();
            dev->card.group_mem = (uint32_t)dev->mtl_device.maxThreadgroupMemoryLength;
            dev->profile.max_threadgroups_per_dispatch = (dev->profile.vendor == GGML_GPU_VENDOR_INTEL) ? 512 : 0;

            // Profile: cache hierarchy (defaults to unknown)
            dev->profile.l1_cache_per_cu     = 0;
            dev->profile.l2_cache_total      = 0;
            dev->profile.mv_reduce           = 0;
            dev->profile.f_scalar_w64        = 0;
            dev->profile.mv_group            = 0;
            dev->profile.attn_gate_fuse      = 0;
            dev->profile.conv_l2_fold        = 0;
            dev->profile.add_norm_fuse       = 0;
            dev->profile.cb_seq              = 0;
            dev->profile.q8_load_bytes       = 1;
            dev->profile.q8_rows8_min        = 0;
            dev->profile.q8_act_load_bytes   = 4;
            dev->profile.set_nowait          = 0;
            dev->profile.staging_keep        = 0;
            dev->profile.mv_group_red        = 0;
            dev->profile.mv_rows             = NULL;
            dev->profile.n_mv_rows           = 0;
            dev->profile.kq_prep             = 0;
            dev->profile.kq_cols             = 0;

            // =========================================================================
            // Hardware constants lookup table.
            // =========================================================================
            // Metal doesn't expose cache sizes, bandwidth, or CU counts through its
            // API. We maintain a static table of known hardware indexed by the device
            // name string returned by [MTLDevice name].
            //
            // Apple's Metal GPU ecosystem is finite — every Mac/iPhone/iPad GPU that
            // supports Metal is enumerable. When adding a new GPU, add an entry here
            // with specs from the vendor's documentation or hardware databases.
            //
            // Fields:
            //   name_match:      substring to match against [MTLDevice name]
            //   compute_units:   CUs/SMs/EUs on this GPU
            //   l1_cache_per_cu: L1 data cache per CU in bytes
            //   l2_cache_total:  shared L2 cache in bytes
            //   local_bw:        VRAM bandwidth in bytes/sec
            //   transfer_bw:     host<->GPU bandwidth in bytes/sec (PCIe/TB)
            //
            // Sources:
            //   GCN: AMD GCN Architecture whitepaper, techpowerup.com
            //   RDNA: AMD RDNA Architecture whitepaper
            //   Apple: Apple GPU programming guide, anandtech reviews
            {
                struct hw_entry {
                    const char * name_match;       // substring to match in device name
                    uint32_t     compute_units;
                    uint32_t     l1_cache_per_cu;  // bytes
                    uint32_t     l2_cache_total;   // bytes
                    uint64_t     local_bw;         // bytes/sec
                    uint64_t     transfer_bw;      // bytes/sec
                };

                // Table of known Metal GPUs.
                // Ordered most-specific first — first match wins.
                static const struct hw_entry hw_table[] = {
                    // =====================================================================
                    // AMD GCN (Mac Pro 2013, Retina iMac, MacBook Pro 2015)
                    // Architecture: GCN 1.0 (Tahiti), wavefront=64, 16 KB L1/CU, 768 KB L2
                    // =====================================================================
                    { "D700",       32, 16384,  768*1024,  264000000000ULL, 0 },  // 6 GB GDDR5, 264 GB/s
                    { "D500",       24, 16384,  768*1024,  176000000000ULL, 0 },  // 3 GB GDDR5, 176 GB/s
                    { "D300",       12, 16384,  512*1024,  132000000000ULL, 0 },  // 2 GB GDDR5, 132 GB/s

                    // =====================================================================
                    // AMD Polaris (eGPU, iMac 2017-2019, MacBook Pro 2017)
                    // Architecture: GCN 4.0, wavefront=64, 16 KB L1/CU, 2 MB L2
                    // =====================================================================
                    { "RX 580",     36, 16384,  2048*1024, 256000000000ULL, 0 },  // 8 GB GDDR5, 256 GB/s
                    { "RX 570",     32, 16384,  2048*1024, 224000000000ULL, 0 },  // 4 GB GDDR5, 224 GB/s
                    { "560",        16, 16384,  2048*1024, 112000000000ULL, 0 },  // Polaris 21
                    { "555",        12, 16384,  1024*1024,  80000000000ULL, 0 },
                    { "Pro 580",    36, 16384,  2048*1024, 256000000000ULL, 0 },
                    { "Pro 575",    36, 16384,  2048*1024, 256000000000ULL, 0 },
                    { "Pro 570",    32, 16384,  2048*1024, 224000000000ULL, 0 },
                    { "Pro 560",    16, 16384,  2048*1024, 112000000000ULL, 0 },

                    // =====================================================================
                    // AMD Vega (iMac Pro 2017, eGPU)
                    // Architecture: GCN 5.0 (Vega), wavefront=64, 16 KB L1/CU, 4 MB L2
                    // =====================================================================
                    { "Vega 64",    64, 16384,  4096*1024, 484000000000ULL, 0 },  // 16 GB HBM2
                    { "Vega 56",    56, 16384,  4096*1024, 410000000000ULL, 0 },  // 8 GB HBM2
                    { "Vega 48",    48, 16384,  4096*1024, 394000000000ULL, 0 },  // Vega Pro in iMac Pro
                    { "Vega II",    64, 16384,  4096*1024, 1024000000000ULL,0 },  // 32 GB HBM2 (MPX)
                    { "Vega 20",    20, 16384,  4096*1024, 394000000000ULL, 0 },  // MacBook Pro 2018

                    // =====================================================================
                    // AMD RDNA 1 (MacBook Pro 2019, iMac 2019-2020)
                    // Architecture: RDNA 1, wavefront=32, 16 KB L1/CU, 4 MB L2
                    // =====================================================================
                    { "5500M",      22, 16384,  4096*1024, 192000000000ULL, 0 },  // 8 GB GDDR6
                    { "5500 XT",    22, 16384,  4096*1024, 192000000000ULL, 0 },
                    { "5300M",      20, 16384,  4096*1024, 192000000000ULL, 0 },  // 4 GB GDDR6
                    { "5300",       20, 16384,  4096*1024, 192000000000ULL, 0 },
                    { "5700 XT",    40, 16384,  4096*1024, 448000000000ULL, 0 },  // 16 GB GDDR6 (MPX)
                    { "5700",       36, 16384,  4096*1024, 448000000000ULL, 0 },

                    // =====================================================================
                    // AMD RDNA 2 (Mac Pro 2023 MPX modules, eGPU)
                    // Architecture: RDNA 2, wavefront=32, 16 KB L1/CU, 4 MB L2
                    // Note: W6000 series used in Mac Pro tower
                    // =====================================================================
                    { "W6900X",     80, 16384,  16384*1024, 512000000000ULL, 0 },  // 32 GB GDDR6
                    { "W6800X",     60, 16384,  16384*1024, 512000000000ULL, 0 },  // 32 GB GDDR6
                    { "W6600X",     28, 16384,  8192*1024,  256000000000ULL, 0 },  // 8 GB GDDR6

                    // =====================================================================
                    // Apple Silicon — unified memory, cache sizes from die analysis
                    // L1: 128 KB per GPU core (Apple's "core" ≠ SM; one core ≈ 128 ALUs)
                    // SLC (System-Level Cache) acts as L2 for GPU, shared with CPU/NPU
                    // =====================================================================
                    { "M1 Ultra",   64, 131072, 48*1024*1024, 800000000000ULL, 0 },
                    { "M1 Max",     32, 131072, 24*1024*1024, 400000000000ULL, 0 },
                    { "M1 Pro",     16, 131072, 24*1024*1024, 200000000000ULL, 0 },
                    { "M1",          8, 131072, 16*1024*1024, 68000000000ULL,  0 },  // base M1

                    { "M2 Ultra",   76, 131072, 48*1024*1024, 800000000000ULL, 0 },
                    { "M2 Max",     38, 131072, 24*1024*1024, 400000000000ULL, 0 },
                    { "M2 Pro",     19, 131072, 24*1024*1024, 200000000000ULL, 0 },
                    { "M2",         10, 131072, 16*1024*1024, 100000000000ULL, 0 },

                    { "M3 Max",     40, 131072, 36*1024*1024, 400000000000ULL, 0 },
                    { "M3 Pro",     18, 131072, 24*1024*1024, 150000000000ULL, 0 },
                    { "M3",         10, 131072, 24*1024*1024, 100000000000ULL, 0 },

                    { "M4 Max",     40, 131072, 48*1024*1024, 546000000000ULL, 0 },
                    { "M4 Pro",     20, 131072, 24*1024*1024, 273000000000ULL, 0 },
                    { "M4",         10, 131072, 16*1024*1024, 120000000000ULL, 0 },

                    // =====================================================================
                    // Intel (MacBook Air/Pro 2012-2020, iMac, Mac Mini, Mac Pro)
                    // Integrated GPUs with shared system memory.
                    // L1/L2 not meaningful for dispatch tuning on these low-perf parts.
                    // =====================================================================
                    { "UHD 630",    24, 0, 0, 42000000000ULL,  0 },
                    { "Iris Plus",  48, 0, 0, 34000000000ULL,  0 },
                    { "Iris Pro",   48, 0, 0, 25000000000ULL,  0 },
                    { "HD 6000",    48, 0, 0, 25000000000ULL,  0 },
                    { "HD 5000",    40, 0, 0, 25000000000ULL,  0 },
                    { "HD 4000",    16, 0, 0, 21000000000ULL,  0 },
                };

                const int n_hw_entries = sizeof(hw_table) / sizeof(hw_table[0]);
                NSString * dev_name = [dev->mtl_device name];

                for (int i = 0; i < n_hw_entries; i++) {
                    if ([dev_name containsString:[NSString stringWithUTF8String:hw_table[i].name_match]]) {
                        dev->profile.compute_units   = hw_table[i].compute_units;
                        dev->profile.l1_cache_per_cu = hw_table[i].l1_cache_per_cu;
                        dev->profile.l2_cache_total  = hw_table[i].l2_cache_total;
                        dev->profile.local_bandwidth = hw_table[i].local_bw;
                        dev->profile.transfer_bandwidth = hw_table[i].transfer_bw;
                        GGML_LOG_INFO("%s: hw profile: CUs=%u, L1=%u KB/CU, L2=%u KB, BW=%.0f GB/s\n",
                            __func__,
                            hw_table[i].compute_units,
                            hw_table[i].l1_cache_per_cu / 1024,
                            hw_table[i].l2_cache_total / 1024,
                            hw_table[i].local_bw / 1e9);
                        break;
                    }
                }
            }

            // The card's rows, by its Metal name: data/device.tsv [devices] and data/matmul.tsv [matvec_defaults]
            // (generated headers); a -1 field keeps the value set above.
            {
                const struct ggml_metal_device_row * row = ggml_metal_device_row_of(dev->profile.name);
                const struct ggml_metal_matvec_default_row * mv = ggml_metal_matvec_default_row_of(dev->profile.name);
                if (row != NULL) {
                    GGML_METAL_PROBE_ONLY(ggml_metal_probe_choose_mode(dev, row);)
                    if (row->compute_units   >= 0) dev->profile.compute_units   = (uint32_t) row->compute_units;
                    if (row->l1_cache_per_cu >= 0) dev->profile.l1_cache_per_cu = (uint32_t) row->l1_cache_per_cu;
                    if (row->l2_cache_total  >= 0) dev->profile.l2_cache_total  = (uint32_t) row->l2_cache_total;
                    if (row->local_bw        >= 0) dev->profile.local_bandwidth = (uint64_t) row->local_bw;
                    if (row->transfer_bw     >= 0) dev->profile.transfer_bandwidth = (uint64_t) row->transfer_bw;
                    if (row->attn_gate_fuse  >= 0) dev->profile.attn_gate_fuse  = (uint32_t) row->attn_gate_fuse;
                    if (row->conv_l2_fold    >= 0) dev->profile.conv_l2_fold    = (uint32_t) row->conv_l2_fold;
                    if (row->add_norm_fuse   >= 0) dev->profile.add_norm_fuse   = (uint32_t) row->add_norm_fuse;
                    if (row->cb_seq          >= 0) dev->profile.cb_seq          = (uint32_t) row->cb_seq;
                    if (row->set_nowait      >= 0) dev->profile.set_nowait      = (uint32_t) row->set_nowait;
                    if (row->staging_keep    >= 0) dev->profile.staging_keep    = (uint32_t) row->staging_keep;
                }
                if (mv != NULL) {
                    if (mv->mv_reduce         >= 0) dev->profile.mv_reduce         = (uint32_t) mv->mv_reduce;
                    if (mv->f_scalar_w64      >= 0) dev->profile.f_scalar_w64      = (uint32_t) mv->f_scalar_w64;
                    if (mv->mv_group          >= 0) dev->profile.mv_group          = (uint32_t) mv->mv_group;
                    if (mv->q8_load_bytes     >= 0) dev->profile.q8_load_bytes     = (uint32_t) mv->q8_load_bytes;
                    if (mv->q8_rows8_min      >= 0) dev->profile.q8_rows8_min      = (uint32_t) mv->q8_rows8_min;
                    if (mv->q8_act_load_bytes >= 0) dev->profile.q8_act_load_bytes = (uint32_t) mv->q8_act_load_bytes;
                    if (mv->mv_group_red      >= 0) dev->profile.mv_group_red      = (uint32_t) mv->mv_group_red;
                    // GGML_METAL_GROUP_RED=0 keeps the grouped K-quant matvec on the row-at-a-time sums.
                    const char * red_env = getenv("GGML_METAL_GROUP_RED");
                    if (red_env && strcmp(red_env, "0") == 0) {
                        dev->profile.mv_group_red = 0;
                    }
                }
                if (row != NULL || mv != NULL) {
                    GGML_LOG_INFO("%s: hw row %s (%s): CUs=%u, L1=%u KB/CU, L2=%u KB, BW=%.0f GB/s, mv_reduce=%u, f_scalar_w64=%u, mv_group=%u, attn_gate_fuse=%u, conv_l2_fold=%u, add_norm_fuse=%u, cb_seq=%u, q8_load_bytes=%u, q8_rows8_min=%u, q8_act_load_bytes=%u, set_nowait=%u, staging_keep=%u, mv_group_red=%u\n",
                        __func__, dev->profile.name, row != NULL ? row->chip : "",
                        dev->profile.compute_units, dev->profile.l1_cache_per_cu / 1024,
                        dev->profile.l2_cache_total / 1024, dev->profile.local_bandwidth / 1e9, dev->profile.mv_reduce, dev->profile.f_scalar_w64, dev->profile.mv_group, dev->profile.attn_gate_fuse, dev->profile.conv_l2_fold, dev->profile.add_norm_fuse, dev->profile.cb_seq, dev->profile.q8_load_bytes, dev->profile.q8_rows8_min, dev->profile.q8_act_load_bytes, dev->profile.set_nowait, dev->profile.staging_keep, dev->profile.mv_group_red);
                }
            }

            dev->profile.lane_ok = ggml_metal_lane_ok_mask(dev->profile.name);
            ggml_metal_tiled_test_arm();

            // Per-shape matvec rows of this card: data/matmul.tsv [matvec_shapes] (generated header; the rows of one card stand together).
            // GGML_METAL_KQ_HW_ROWS=0 ignores them; GGML_METAL_KQ_PREP=0, GGML_METAL_KQ_COLS=0, GGML_METAL_KQ_EXT_ROWS=0 (the rows of data/matmul.tsv [tiles]) switch one part off.
            {
                const char * rows_env = getenv("GGML_METAL_KQ_HW_ROWS");
                const char * prep_env = getenv("GGML_METAL_KQ_PREP");
                const char * cols_env = getenv("GGML_METAL_KQ_COLS");
                const char * ext_env  = getenv("GGML_METAL_KQ_EXT_ROWS");
                if (!(rows_env && strcmp(rows_env, "0") == 0)) {
                    uint32_t n_rows = 0;
                    const struct ggml_device_mv_row * rows = ggml_metal_mv_rows_of(dev->profile.name, &n_rows);
                    dev->profile.mv_rows   = rows;
                    dev->profile.n_mv_rows = n_rows;
                    for (uint32_t i = 0; i < n_rows; i++) {
                        const struct ggml_device_mv_row * row = &rows[i];
                        if (row->prep == 1 && !(prep_env && strcmp(prep_env, "0") == 0)) {
                            dev->profile.kq_prep = 1;
                        }
                        if (row->cols > 1 && !(cols_env && strcmp(cols_env, "0") == 0)) {
                            dev->profile.kq_cols = 1;
                        }
                    }
                }
                if (dev->profile.n_mv_rows > 0) {
                    GGML_LOG_INFO("%s: hw mv rows %s: %u rows, kq_prep=%u, kq_cols=%u, ext_min rows=%u\n", __func__, dev->profile.name, dev->profile.n_mv_rows, dev->profile.kq_prep, dev->profile.kq_cols, (unsigned) ((ext_env && strcmp(ext_env, "0") == 0) ? 0 : ggml_metal_batch_rows_count(dev->profile.name)));
                }
            }

            // Model profile: initialized to defaults (populated=false).
            // Will be populated later via ggml_backend_metal_set_model_profile()
            // after GGUF metadata is loaded.
            dev->model_profile = ggml_model_profile_default();

            dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            dev->props.op_offload_min_batch_size  = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
            dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
            if (@available(macOS 10.12, iOS 16.0, *)) {
                dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
            } else {
                dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
            }

            snprintf(dev->props.name, sizeof(dev->props.name), "MTL%d:%s", device, [[dev->mtl_device name] UTF8String]);
            // desc set after family probing below

            // built and probed for the wave width in one step (see ggml_metal_device_ensure_library)
            ggml_metal_device_ensure_library(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
            }

            // Probe actual SIMD group width from a compiled pipeline.
            // AMD GCN/Vega reports threadExecutionWidth=64 while Apple Silicon and RDNA report 32.
            // The hardcoded value of 32 causes wrong results on GCN: simd_sum reduces across 64
            // threads but shared memory slots only cover 32, and tiisg ranges 0-63 not 0-31.
            //
            // We compile a simple kernel (kernel_concat_f32 has no function constants) and read
            // threadExecutionWidth from the resulting pipeline state. All kernels on a given
            // device share the same SIMD group width, so one probe is sufficient.

            if (dev->props.use_residency_sets) {
                dev->rsets = ggml_metal_rsets_init();
            } else {
                dev->rsets = nil;
            }

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, dev->props.name);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                dev->props.max_apple_family  = 0;
                dev->props.max_common_family = 0;
                dev->props.max_metal_family  = 0;

                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        dev->props.max_apple_family = i;
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        dev->props.max_common_family = i;
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        dev->props.max_metal_family = i;
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            // Build desc from family probing results + capabilities
            {
                char desc[128];
                int pos = 0;

                if (dev->props.max_common_family) {
                    pos += snprintf(desc + pos, sizeof(desc) - pos, "Common%d",
                        dev->props.max_common_family - (int)MTLGPUFamilyCommon1 + 1);
                }
                if (dev->props.max_apple_family) {
                    pos += snprintf(desc + pos, sizeof(desc) - pos, "%sApple%d",
                        pos > 0 ? " " : "",
                        dev->props.max_apple_family - (int)MTLGPUFamilyApple1 + 1);
                }
                if (dev->props.max_metal_family) {
                    pos += snprintf(desc + pos, sizeof(desc) - pos, "%sMetal%d",
                        pos > 0 ? " " : "",
                        dev->props.max_metal_family - (int)MTLGPUFamilyMetal3_GGML + 3);
                }
                snprintf(desc + pos, sizeof(desc) - pos, " shared=%s matrix_hw=%s simd_w=%u",
                    dev->props.has_unified_memory ? "true" : "false",
                    dev->props.has_simdgroup_mm ? "true" : "false",
                    dev->profile.simd_width);

                strlcpy(dev->props.desc, desc, sizeof(dev->props.desc));
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: has bf16 u16 kernels  = %s\n", __func__, dev->props.has_bf16_u16            ? "true" : "false");
            GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

            GGML_LOG_INFO("%s: profile: vendor=%d name=%s shared=%s matrix_hw=%s simd_w=%u max_tg=%u smem=%u\n",
                __func__,
                dev->profile.vendor, dev->profile.name,
                dev->profile.shared_memory ? "true" : "false",
                dev->profile.has_matrix_hw ? "true" : "false",
                dev->profile.simd_width,
                dev->profile.max_threads_per_threadgroup,
                dev->profile.shared_mem_size);

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
            }
#endif
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    ggml_metal_rsets_free(dev->rsets);

    ggml_metal_library_free(dev->library);
    dev->library = NULL;

    if (dev->mtl_queue) {
        [dev->mtl_queue release];
        dev->mtl_queue = nil;
    }

    if (dev->norm_means) {
        [dev->norm_means release];
        dev->norm_means = nil;
    }

    if (dev->mtl_device) {
        [dev->mtl_device release];
        dev->mtl_device = nil;
    }

    free(dev);
}

struct ggml_metal_buffer_id ggml_metal_device_get_norm_means(ggml_metal_device_t dev, int slot) {
    dispatch_once(&dev->norm_means_once, ^{
        dev->norm_means = [dev->mtl_device newBufferWithLength:GGML_METAL_NORM_MEANS_SLOTS*sizeof(float) options:MTLResourceStorageModePrivate];
    });
    struct ggml_metal_buffer_id res = { (__bridge void *) dev->norm_means, (size_t) slot*sizeof(float) };
    return res;
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}


struct ggml_metal_event {
    void * obj; // id<MTLSharedEvent> — shared events support cross-device signaling

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLEvent> event = (id<MTLEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLEvent> event = (id<MTLEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLEvent> event = ev->obj;
    [event release];

    free(ev);

    GGML_UNUSED(dev);
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLEvent> event = ev->obj;

        ggml_metal_upload_flush(dev->mtl_queue);
        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];
        [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = dev->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - dev->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

const struct ggml_device_profile * ggml_metal_device_get_profile(ggml_metal_device_t dev) {
    return &dev->profile;
}

const struct ggml_metal_card * ggml_metal_device_get_card(ggml_metal_device_t dev) {
    return &dev->card;
}

void ggml_metal_device_set_model_profile(ggml_metal_device_t dev,
        uint32_t n_embd, uint32_t n_ff, uint32_t n_layer,
        uint32_t n_head, uint32_t n_head_kv) {
    ggml_model_profile_set(&dev->model_profile,
        n_embd, n_ff, n_layer, n_head, n_head_kv,
        dev->profile.l1_cache_per_cu);
    GGML_LOG_INFO("%s: model profile: n_embd=%u, n_ff=%u, n_layer=%u, act=%u B, L1 pressure=%.2f\n",
        __func__, n_embd, n_ff, n_layer,
        dev->model_profile.activation_bytes_f32,
        dev->model_profile.l1_pressure_ratio);
}

const struct ggml_model_profile * ggml_metal_device_get_model_profile(ggml_metal_device_t dev) {
    return &dev->model_profile;
}


#ifdef GGML_METAL_PROBES
// The sampling mode of a probe build, from the card's row of data/device.tsv [devices]: only where the timestamp counter worked.
void ggml_metal_probe_choose_mode(ggml_metal_device_t dev, const struct ggml_metal_device_row * row) {
    if (!ggml_metal_probe_enabled() || row->probe_timestamp != 1) {
        return;
    }
    id<MTLDevice> mtl = dev->mtl_device;
    if (row->probe_dispatch == 1 && [mtl supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        dev->probe_gpu_mode = GGML_METAL_PROBE_DISPATCH;
    } else if (row->probe_stage == 1 && [mtl supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        dev->probe_gpu_mode = GGML_METAL_PROBE_ENCODER;
    }
}
#endif
