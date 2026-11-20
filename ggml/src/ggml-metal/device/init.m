#import "device/ggml-metal-device-internal.h"
#include <pthread.h>
#include "ggml-metal-probe.h"
#include "lanes/select.h"
#include "matmul/batch-route.h"
#include "matmul/tiled-plan.h"
#include "matmul/matvec-plan.h"
#include "device/device-sizes.h"
GGML_METAL_FAMILY(device) // overload of MTLGPUFamilyMetalX (not available in some environments)

// The card object's matvec settings and per-shape rows (data/matmul.tsv [matvec_defaults], [matvec_shapes]) and its family
// (data/device.tsv [families]). Called when the device is set up, before the wave is probed (the default row only),
// and again once the wave is probed: a card's own rows are used only while it reports the facts written in them.
// GGML_METAL_GROUP_RED=0 keeps the grouped K-quant matvec on the row-at-a-time sums. GGML_METAL_KQ_HW_ROWS=0 ignores
// the per-shape rows; GGML_METAL_KQ_PREP=0 and GGML_METAL_KQ_COLS=0 switch one part of them off.
static void ggml_metal_device_matvec_load(ggml_metal_device_t dev) {
    const char * red_env  = getenv("GGML_METAL_GROUP_RED");
    const char * rows_env = getenv("GGML_METAL_KQ_HW_ROWS");
    const char * prep_env = getenv("GGML_METAL_KQ_PREP");
    const char * cols_env = getenv("GGML_METAL_KQ_COLS");
    (void) ggml_metal_matvec_forced_env(); // the test hook GGML_METAL_TEST_MV_PLAN: a value that cannot be read aborts here
    dev->card.l1_per_cu = dev->profile.l1_cache_per_cu;
    ggml_metal_matvec_tune_load(&dev->card, dev->profile.name, !(rows_env && strcmp(rows_env, "0") == 0), !(prep_env && strcmp(prep_env, "0") == 0), !(cols_env && strcmp(cols_env, "0") == 0), !(red_env && strcmp(red_env, "0") == 0));

    dev->card.family[0] = 0;
    if (dev->card.wave_probed == 0) {
        return;
    }
    char metal_name[64] = "";
#if defined(__MAC_14_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_14_0
    if (@available(macOS 14.0, *)) {
        NSString * arch = dev->mtl_device.architecture.name;
        if (arch != nil) {
            snprintf(metal_name, sizeof(metal_name), "%s", [arch UTF8String]);
        }
    }
#endif
    const char * vendor = dev->profile.vendor == GGML_GPU_VENDOR_AMD    ? "amd"
                        : dev->profile.vendor == GGML_GPU_VENDOR_APPLE  ? "apple"
                        : dev->profile.vendor == GGML_GPU_VENDOR_INTEL  ? "intel"
                        : dev->profile.vendor == GGML_GPU_VENDOR_NVIDIA ? "nvidia" : "";
    const struct ggml_metal_family_row * fam = ggml_metal_family_row_of(dev->profile.name, vendor, metal_name, dev->card.wave, dev->card.group_mem);
    if (fam != NULL) {
        snprintf(dev->card.family, sizeof(dev->card.family), "%s", fam->name);
    }
    uint32_t n_candidates = 0;
    ggml_metal_family_mv_rows_of(dev->card.family, &n_candidates);
    const char * ext_env = getenv("GGML_METAL_KQ_EXT_ROWS");
    GGML_LOG_INFO("%s: matvec settings %s: mv_sum=%u, mv_reduce=%u, f_scalar_w64=%u, mv_group=%u, q8_load_bytes=%u, q8_rows8_min=%u, q8_act_load_bytes=%u, mv_group_red=%u, l1_share_cut=%.2f, rows=%u/%u/%u, f_nsg_max=%u, f_k_per_sg=%u\n", __func__, dev->profile.name,
        dev->card.mv_sum, dev->card.mv_reduce, dev->card.f_scalar_w64, dev->card.mv_group, dev->card.q8_load_bytes, dev->card.q8_rows8_min, dev->card.q8_act_load_bytes, dev->card.mv_group_red,
        (double) dev->card.l1_share_cut, dev->card.mv_rows_below, dev->card.mv_rows_from, dev->card.mv_rows_unknown, dev->card.f_nsg_max, dev->card.f_k_per_sg);
    if (dev->card.n_mv_rows > 0) {
        GGML_LOG_INFO("%s: hw mv rows %s: %u rows, kq_prep=%u, kq_cols=%u, ext_min rows=%u\n", __func__, dev->profile.name, dev->card.n_mv_rows, dev->card.kq_prep, dev->card.kq_cols, (unsigned) ((ext_env && strcmp(ext_env, "0") == 0) ? 0 : ggml_metal_batch_rows_count(dev->profile.name)));
    }
    if (dev->card.family[0] != 0) {
        GGML_LOG_INFO("%s: family %s (Metal architecture name '%s'): %u candidate shape rows for a sweep, not applied\n", __func__, dev->card.family, metal_name, n_candidates);
    }
}

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
            ggml_metal_device_matvec_load(dev);
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

            // Profile: compute tier
            dev->profile.has_matrix_hw               = dev->props.has_simdgroup_mm;
            dev->profile.has_simd_reduction          = dev->props.has_simdgroup_reduction;
            dev->profile.simd_width                  = 32; // default; probed below after library init
            dev->profile.max_threads_per_threadgroup = (uint32_t)dev->mtl_device.maxThreadsPerThreadgroup.width;
            dev->profile.shared_mem_size             = (uint32_t)dev->mtl_device.maxThreadgroupMemoryLength;
            dev->card.wave      = 32; // default; probed below after library init
            ggml_metal_tiled_tune_load(&dev->card, ""); // no entries until the library is built and the card probed
            ggml_metal_tiled_test_arm();
            dev->card.group_mem = (uint32_t)dev->mtl_device.maxThreadgroupMemoryLength;
            dev->profile.max_threadgroups_per_dispatch = (dev->profile.vendor == GGML_GPU_VENDOR_INTEL) ? 512 : 0;

            // The card's row of data/device.tsv [devices], by its Metal name: sizes and scheduling settings
            // (device/device-sizes.h). A card without a row leaves them not known (0).
            {
                const struct ggml_metal_device_row * row = ggml_metal_device_sizes_load(&dev->profile, dev->profile.name);
                if (row != NULL) {
                    GGML_METAL_PROBE_ONLY(ggml_metal_probe_choose_mode(dev, row);)
                }
                if (row != NULL) {
                    GGML_LOG_INFO("%s: hw row %s (%s): CUs=%u, L1=%u KB/CU, L2=%u KB, BW=%.0f GB/s, attn_gate_fuse=%u, conv_l2_fold=%u, add_norm_fuse=%u, cb_seq=%u, set_nowait=%u, staging_keep=%u\n",
                        __func__, dev->profile.name, row->chip,
                        dev->profile.compute_units, dev->profile.l1_cache_per_cu / 1024,
                        dev->profile.l2_cache_total / 1024, dev->profile.local_bandwidth / 1e9, dev->profile.attn_gate_fuse, dev->profile.conv_l2_fold, dev->profile.add_norm_fuse, dev->profile.cb_seq, dev->profile.set_nowait, dev->profile.staging_keep);
                }
            }

            dev->profile.lane_ok = ggml_metal_lane_ok_mask(dev->profile.name);
            ggml_metal_tiled_test_arm();

            // The matvec settings and rows before the wave is probed: the default row. They are loaded again when the
            // library is built and the wave known (ggml_metal_device_ensure_library).
            dev->card.wave_probed = 0;
            ggml_metal_device_matvec_load(dev);

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
