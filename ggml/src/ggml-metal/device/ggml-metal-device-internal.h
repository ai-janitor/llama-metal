#pragma once
#include "ggml-metal-probe.h"

#import "ggml-metal-device.h"
#import "ggml-metal-command-buffer-error.h"
#import "ggml-metal-impl.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-test-override.h"

#ifdef GGML_METAL_TEST_OBSERVER
#include "ggml-metal-test-observer-helpers.h"
#endif

#import "ggml-impl.h"
#import "ggml-device-profile.h"
#import "device-rows.h"  // generated from data/device.tsv [devices]
#import "matvec-rows.h"  // generated from data/matmul.tsv [matvec_defaults] and [matvec_shapes]

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>
#include <IOKit/IOKitLib.h>

#include <stdatomic.h>
#include <pthread.h>
#include <limits.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

struct ggml_metal_cv_entry {
    int32_t        idx;
    MTLDataType    type;
};

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;

    // Presence/type of every constant set through the setters below. MTLFunctionConstantValues
    // cannot be read back, and the candidate loader (P12.18) must check a candidate's required
    // function constants against what the caller supplied before creating a pipeline.
    struct ggml_metal_cv_entry * set;
    int n_set;
    int cap_set;
};

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;

    int device_index;     // which device compiled this pipeline
    int min_family;       // minimum MTLGPUFamily required (0 = Common1, any device)
    uint8_t verified_vendors; // bitmask of GGML_METAL_VERIFIED_* flags (default: 0 = unverified)
    char name[128];       // kernel function name (for profiler dispatch labels)

    // P12.34: identity of the binary this PSO was actually compiled from, for the dispatch receipt.
    // "builtin" for ordinary pipelines; the candidate metallib SHA-256 (64 hex) for a pipeline created
    // through a selected candidate. Fixed at creation and never re-derived from the environment.
    char binary_sha[65];
};

// One loaded candidate metallib, owned by the library (and so the device) that created it and
// cached by full SHA, never by path. Entries live as long as the library, which outlives every
// pipeline compiled from them.
enum ggml_metal_test_override_abi {
    GGML_METAL_TEST_OVERRIDE_ABI_UNCHECKED = 0,
    GGML_METAL_TEST_OVERRIDE_ABI_OK,
    GGML_METAL_TEST_OVERRIDE_ABI_REJECTED,
};

struct ggml_metal_test_override_entry {
    char sha[65];
    id<MTLLibrary> obj;
    enum ggml_metal_test_override_abi abi;
    char abi_kind[24];    // "compile-reject" | "unsupported", valid when abi == REJECTED
    char abi_detail[256]; // valid when abi == REJECTED
    struct ggml_metal_test_override_entry * next;
};

struct ggml_metal_library {
    id<MTLLibrary> obj;
    id<MTLDevice> device;
    int device_index;

    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;

    int simd_width; // actual threadExecutionWidth for this device (set after probe)
    const struct ggml_device_profile * dev_profile; // parent device's hardware profile (read-only)

    struct ggml_metal_test_override_entry * test_overrides; // candidate binaries by SHA (P12.18)
};

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
    // This is the dispatch mode of the encoder actually created, not the caller policy.
    // P11.65: now present in every build because the dispatch receipt keys on it; the observer API still
    // only exposes it under GGML_METAL_TEST_OBSERVER.
    bool                         concurrent;

    // per-dispatch profiler state (NULL when profiling is off)
    ggml_metal_profiler_t profiler;
    int                   profiler_cb_idx;

    // last pipeline name set via set_pipeline (for profiler labels)
    const char *          profiler_pipeline_name;

    // current ggml op name (set before dispatch by ggml_metal_op_encode)
    const char *          profiler_op_name;

    // tensor shape for shape-attributed profiling (set before dispatch)
    int32_t               profiler_ne00;   // src0 inner dimension (K)
    int32_t               profiler_ne01;   // src0 row count

    // P12.34: identity of the pipeline last bound by set_pipeline, copied by value (empty name = none bound).
    // Selects the receipt key only; binding never counts, only an actual dispatch does.
    char                  receipt_name[128];
    char                  receipt_sha[65];

    // Bytes of threadgroup memory set since the pipeline was bound (index 0). Read by the launch print only.
    size_t                launch_mem;
};

struct ggml_metal_device {
#ifdef GGML_METAL_PROBES
    enum ggml_metal_probe_gpu_mode probe_gpu_mode;
#endif
    id<MTLDevice> mtl_device;

    // GGML_METAL_NORM_MATVEC: one float per graph node index, written by a residual add, read by the fused norm
    id<MTLBuffer> norm_means;
    dispatch_once_t norm_means_once;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    struct ggml_device_profile profile;
    struct ggml_metal_card       card;             // card object (card-object.h), filled at init
    struct ggml_model_profile  model_profile;  // set via ggml_backend_metal_set_model_profile()

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};


// shared across the split files
void ggml_metal_test_override_fail(const char * kind, const char * detail);
struct ggml_metal_test_override ggml_metal_test_override_current(const char * base, const char * name);
struct ggml_metal_test_override_entry * ggml_metal_test_override_library(
        ggml_metal_library_t lib, const struct ggml_metal_test_override * sel);
bool ggml_metal_test_override_check_specialization(
        id<MTLLibrary> func_lib, NSString * base_func, ggml_metal_cv_t cv);
void ggml_metal_dispatch_receipt_record(const char * name, const char * sha, bool concurrent);
bool ggml_metal_has_matvec_type(enum ggml_type type);
// One "can the GPU run this" predicate per op: a row names the op, how many inputs a predicate may read, and the predicate.
typedef bool (*ggml_metal_supports_fn)(ggml_metal_device_t dev, const struct ggml_tensor * op);
struct ggml_metal_supports_row {
    enum ggml_op op;
    int n_src;
    ggml_metal_supports_fn fn;
};
struct ggml_metal_supports_table {
    const struct ggml_metal_supports_row * rows;
    int n;
};
const struct ggml_tensor * ggml_metal_src(const struct ggml_tensor * op, int i);
struct ggml_metal_supports_table ggml_metal_supports_table_conv(void);
struct ggml_metal_supports_table ggml_metal_supports_table_copy(void);
struct ggml_metal_supports_table ggml_metal_supports_table_flash_attn(void);
struct ggml_metal_supports_table ggml_metal_supports_table_matmul(void);
struct ggml_metal_supports_table ggml_metal_supports_table_misc(void);
struct ggml_metal_supports_table ggml_metal_supports_table_norm(void);
struct ggml_metal_supports_table ggml_metal_supports_table_reduction(void);
struct ggml_metal_supports_table ggml_metal_supports_table_rope(void);
struct ggml_metal_supports_table ggml_metal_supports_table_spatial(void);
struct ggml_metal_supports_table ggml_metal_supports_table_ssm(void);
struct ggml_metal_supports_table ggml_metal_supports_table_unary(void);
void ggml_metal_device_placement_init(ggml_metal_device_t dev);

#ifdef GGML_METAL_PROBES
void ggml_metal_probe_choose_mode(ggml_metal_device_t dev, const struct ggml_metal_device_row * row);
ggml_metal_profiler_t ggml_metal_probe_profiler_init(ggml_metal_device_t dev);
bool ggml_metal_probe_profiler_begin(ggml_metal_profiler_t prof, int n_nodes);
bool ggml_metal_probe_profiler_finish(ggml_metal_profiler_t prof);
void ggml_metal_probe_transfer(id<MTLCommandBuffer> cb, const char * kind);
#endif
