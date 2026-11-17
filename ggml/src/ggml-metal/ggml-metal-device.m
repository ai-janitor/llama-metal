#import "ggml-metal-device.h"
#import "ggml-metal-impl.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-test-override.h"

#ifdef GGML_METAL_TEST_OBSERVER
#include "ggml-metal-test-observer-helpers.h"
#endif

#import "ggml-impl.h"
#import "ggml-device-profile.h"
#import "ggml-metal-hw-rows.h"

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

// bandwidth tracking counters
struct ggml_metal_transfer_stats ggml_metal_xfer_stats = {0};

void ggml_metal_transfer_stats_report(void) {
    int64_t h2d_bytes   = ggml_metal_xfer_stats.h2d_bytes;
    int64_t h2d_time_us = ggml_metal_xfer_stats.h2d_time_us;
    int64_t h2d_count   = ggml_metal_xfer_stats.h2d_count;
    int64_t d2h_bytes   = ggml_metal_xfer_stats.d2h_bytes;
    int64_t d2h_time_us = ggml_metal_xfer_stats.d2h_time_us;
    int64_t d2h_count   = ggml_metal_xfer_stats.d2h_count;

    if (h2d_count == 0 && d2h_count == 0) {
        return;
    }

    GGML_LOG_INFO("ggml_metal: transfer stats:\n");
    if (h2d_count > 0) {
        double h2d_mb = h2d_bytes / (1024.0 * 1024.0);
        double h2d_ms = h2d_time_us / 1000.0;
        double h2d_bw = (h2d_time_us > 0) ? (h2d_bytes / (h2d_time_us / 1e6)) / (1024.0 * 1024.0) : 0;
        GGML_LOG_INFO("ggml_metal:   H2D: %8.2f MiB in %8.2f ms (%6.0f MiB/s, %" PRId64 " calls)\n",
            h2d_mb, h2d_ms, h2d_bw, h2d_count);
    }
    if (d2h_count > 0) {
        double d2h_mb = d2h_bytes / (1024.0 * 1024.0);
        double d2h_ms = d2h_time_us / 1000.0;
        double d2h_bw = (d2h_time_us > 0) ? (d2h_bytes / (d2h_time_us / 1e6)) / (1024.0 * 1024.0) : 0;
        GGML_LOG_INFO("ggml_metal:   D2H: %8.2f MiB in %8.2f ms (%6.0f MiB/s, %" PRId64 " calls)\n",
            d2h_mb, d2h_ms, d2h_bw, d2h_count);
    }
    int64_t total_us = h2d_time_us + d2h_time_us;
    if (total_us > 0) {
        GGML_LOG_INFO("ggml_metal:   total transfer time: %.2f ms\n", total_us / 1000.0);
    }
}

void ggml_metal_transfer_stats_reset(void) {
    memset(&ggml_metal_xfer_stats, 0, sizeof(ggml_metal_xfer_stats));
}

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

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

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv->set);
    free(cv);
}

// Record (idx, type); a later set of the same index replaces the earlier type, as Metal does.
static void ggml_metal_cv_track(ggml_metal_cv_t cv, int32_t idx, MTLDataType type) {
    for (int i = 0; i < cv->n_set; ++i) {
        if (cv->set[i].idx == idx) {
            cv->set[i].type = type;
            return;
        }
    }
    if (cv->n_set == cv->cap_set) {
        const int cap = cv->cap_set ? cv->cap_set * 2 : 8;
        struct ggml_metal_cv_entry * grown = realloc(cv->set, (size_t) cap * sizeof(*grown));
        if (!grown) {
            // Out of memory: leave the entry untracked. A candidate selection then reports a
            // named specialization-missing-constant rather than silently passing the check.
            GGML_LOG_ERROR("%s: cannot track function constant index=%d\n", __func__, (int) idx);
            return;
        }
        cv->set     = grown;
        cv->cap_set = cap;
    }
    cv->set[cv->n_set].idx  = idx;
    cv->set[cv->n_set].type = type;
    cv->n_set++;
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
    ggml_metal_cv_track(cv, idx, MTLDataTypeShort);
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
    ggml_metal_cv_track(cv, idx, MTLDataTypeInt);
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
    ggml_metal_cv_track(cv, idx, MTLDataTypeBool);
}

//
// MTLComputePipelineState wrapper
//

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

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj               =*/ nil,
        /*.device_index      =*/ -1,
        /*.min_family        =*/ 0,
        /*.verified_vendors  =*/ 0,
    };

    snprintf(res->binary_sha, sizeof(res->binary_sha), "%s", "builtin");

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

void ggml_metal_pipeline_set_min_family(ggml_metal_pipeline_t pipeline, int min_family) {
    pipeline->min_family = min_family;
}

int ggml_metal_pipeline_get_min_family(ggml_metal_pipeline_t pipeline) {
    return pipeline->min_family;
}

int ggml_metal_pipeline_get_device_index(ggml_metal_pipeline_t pipeline) {
    return pipeline->device_index;
}

void ggml_metal_pipeline_set_verified_vendors(ggml_metal_pipeline_t pipeline, uint8_t vendors) {
    pipeline->verified_vendors = vendors;
}

uint8_t ggml_metal_pipeline_get_verified_vendors(ggml_metal_pipeline_t pipeline) {
    return pipeline->verified_vendors;
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

int ggml_metal_pipeline_thread_execution_width(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.threadExecutionWidth;
}

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

//
// Candidate loader (P12.18): lets a host compile one registered kernel from a candidate metallib
// through the same compile_pipeline, with the route's own function constants. All of it is inert
// unless the GGML_METAL_TEST_* selector env vars are set.
//

static pthread_mutex_t ggml_metal_test_override_err_mutex = PTHREAD_MUTEX_INITIALIZER;

// Sticky diagnostic error: the host clears GGML_METAL_TEST_OVERRIDE_ERROR per job and rejects the
// timing if it is set, so a fallback cannot pass as a candidate. The stderr/log line is printed
// once per distinct message so a rejected selection does not flood per request.
static void ggml_metal_test_override_fail(const char * kind, const char * detail) {
    char msg[512];
    snprintf(msg, sizeof(msg), "kind=%s %s", kind, detail ? detail : "");

    pthread_mutex_lock(&ggml_metal_test_override_err_mutex);
    const char * cur = getenv("GGML_METAL_TEST_OVERRIDE_ERROR");
    const bool dup = cur && strcmp(cur, msg) == 0;
    if (!dup) {
        setenv("GGML_METAL_TEST_OVERRIDE_ERROR", msg, 1);
    }
    pthread_mutex_unlock(&ggml_metal_test_override_err_mutex);

    if (!dup) {
        fprintf(stderr, "TEST_OVERRIDE_ERROR %s\n", msg);
        GGML_LOG_ERROR("%s: TEST_OVERRIDE_ERROR %s\n", __func__, msg);
    }
}

static struct ggml_metal_test_override ggml_metal_test_override_current(const char * base, const char * name) {
    return ggml_metal_test_override_resolve(
        getenv("GGML_METAL_TEST_SLOT"),
        getenv("GGML_METAL_TEST_VARIANT"),
        getenv("GGML_METAL_TEST_METALLIB"),
        getenv("GGML_METAL_TEST_METALLIB_SHA"), base, name);
}

// ABI probe (lane_contract_version_probe): dispatch exactly one thread on an owned queue and
// require the row's lane contract version. Returns 0 = ok, 1 = unsupported, 2 = compile-reject.
// Every temporary resource is released on every exit. Caller holds lib->lock.
static int ggml_metal_test_override_check_abi(ggml_metal_library_t lib,
        const struct ggml_metal_test_override_row * row,
        struct ggml_metal_test_override_entry * entry, char * detail, size_t detail_size) {
    int rc = 0;

    NSError * error = nil;
    id<MTLFunction> fn = [entry->obj newFunctionWithName:@"lane_contract_version_probe"];
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLBuffer> buf = nil;
    id<MTLCommandQueue> queue = nil;

    if (!fn) {
        snprintf(detail, detail_size, "candidate binary has no lane_contract_version_probe");
        rc = 1;
        goto done;
    }

    // constant-free specialization: the probe must not need any function constant
    for (NSString * key in [fn functionConstantsDictionary]) {
        MTLFunctionConstant * c = [fn functionConstantsDictionary][key];
        if (c.required) {
            snprintf(detail, detail_size, "lane_contract_version_probe requires function constant index=%d", (int) c.index);
            rc = 1;
            goto done;
        }
    }

    pipeline = [lib->device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        snprintf(detail, detail_size, "lane_contract_version_probe pipeline: %s",
                 error ? [[error description] UTF8String] : "unknown error");
        rc = 2;
        goto done;
    }

    buf   = [lib->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    queue = [lib->device newCommandQueue];
    if (!buf || !queue) {
        snprintf(detail, detail_size, "could not create ABI probe buffer/queue");
        rc = 1;
        goto done;
    }

    memset([buf contents], 0, sizeof(uint32_t));

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = cmd ? [cmd computeCommandEncoder] : nil;
    if (!cmd || !enc) {
        snprintf(detail, detail_size, "could not create ABI probe command buffer/encoder");
        rc = 1;
        goto done;
    }

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) {
        snprintf(detail, detail_size, "ABI probe command buffer did not complete (status %d)", (int) [cmd status]);
        rc = 1;
        goto done;
    }

    {
        const uint32_t value = *(const uint32_t *) [buf contents];
        if (value != (uint32_t) row->lane_contract_version) {
            snprintf(detail, detail_size, "lane_contract_version %u != %d", value, row->lane_contract_version);
            rc = 1;
        }
    }

done:
    [fn release];
    [pipeline release];
    [buf release];
    [queue release];

    return rc;
}

// Get or load the candidate library for a validated selection. Caller holds lib->lock. A cache
// hit by SHA never re-reads or re-hashes the file. Returns the entry only when its bytes matched
// the declared SHA and its ABI probe passed; otherwise records the sticky error and returns NULL.
static struct ggml_metal_test_override_entry * ggml_metal_test_override_library(
        ggml_metal_library_t lib, const struct ggml_metal_test_override * sel) {
    struct ggml_metal_test_override_entry * entry = NULL;

    for (struct ggml_metal_test_override_entry * e = lib->test_overrides; e; e = e->next) {
        if (strcmp(e->sha, sel->sha) == 0) {
            entry = e;
            break;
        }
    }

    if (!entry) {
        char detail[512];

        NSData * data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:sel->path]];
        if (!data || [data length] == 0) {
            snprintf(detail, sizeof(detail), "cannot read metallib at %s", sel->path);
            ggml_metal_test_override_fail("compile-reject", detail);
            return NULL;
        }

        // hash exactly the bytes handed to newLibraryWithData
        char actual[65];
        if (!ggml_metal_test_override_sha256_hex((const unsigned char *) [data bytes], [data length], actual)) {
            ggml_metal_test_override_fail("compile-reject", "could not hash metallib bytes");
            return NULL;
        }
        if (strcmp(actual, sel->sha) != 0) {
            snprintf(detail, sizeof(detail), "metallib SHA mismatch declared=%s actual=%s", sel->sha, actual);
            ggml_metal_test_override_fail("compile-reject", detail);
            return NULL;
        }

        // DISPATCH_DATA_DESTRUCTOR_DEFAULT copies the bytes, so `data` need not outlive this call,
        // but it is retained through library creation anyway.
        [data retain];
        dispatch_data_t dispatch_data = dispatch_data_create(
            [data bytes], [data length], NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        if (!dispatch_data) {
            [data release];
            ggml_metal_test_override_fail("compile-reject", "could not wrap metallib bytes for Metal");
            return NULL;
        }
        NSError * error = nil;
        id<MTLLibrary> obj = [lib->device newLibraryWithData:dispatch_data error:&error];
        dispatch_release(dispatch_data);
        [data release];
        if (!obj) {
            snprintf(detail, sizeof(detail), "newLibraryWithData: %s",
                     error ? [[error description] UTF8String] : "unknown error");
            ggml_metal_test_override_fail("compile-reject", detail);
            return NULL;
        }

        entry = (struct ggml_metal_test_override_entry *) calloc(1, sizeof(*entry));
        if (!entry) {
            [obj release];
            ggml_metal_test_override_fail("compile-reject", "could not allocate candidate library entry");
            return NULL;
        }
        memcpy(entry->sha, sel->sha, sizeof(entry->sha));
        entry->obj  = obj;
        entry->next = lib->test_overrides;
        lib->test_overrides = entry;
    }

    // ABI probe: once per loaded candidate; the outcome (pass or reject) is remembered.
    if (entry->abi == GGML_METAL_TEST_OVERRIDE_ABI_UNCHECKED) {
        char detail[256];
        const int abi = ggml_metal_test_override_check_abi(lib, sel->row, entry, detail, sizeof(detail));
        if (abi == 0) {
            entry->abi = GGML_METAL_TEST_OVERRIDE_ABI_OK;
        } else {
            entry->abi = GGML_METAL_TEST_OVERRIDE_ABI_REJECTED;
            snprintf(entry->abi_kind, sizeof(entry->abi_kind), "%s", abi == 2 ? "compile-reject" : "unsupported");
            snprintf(entry->abi_detail, sizeof(entry->abi_detail), "%s", detail);
        }
    }

    if (entry->abi == GGML_METAL_TEST_OVERRIDE_ABI_REJECTED) {
        ggml_metal_test_override_fail(entry->abi_kind, entry->abi_detail);
        return NULL;
    }

    return entry;
}

// Check the candidate's required function constants against what the caller supplied (tracked in
// cv) plus the SIMD-width constant compile_pipeline injects itself. A required constant that is
// absent, or present with another type, is a named rejection BEFORE any pipeline is created.
// Returns true when every required constant is satisfied.
static bool ggml_metal_test_override_check_specialization(
        id<MTLLibrary> func_lib, NSString * base_func, ggml_metal_cv_t cv) {
    id<MTLFunction> fn = [func_lib newFunctionWithName:base_func];
    if (!fn) {
        char detail[512];
        snprintf(detail, sizeof(detail), "missing function %s in candidate binary", [base_func UTF8String]);
        ggml_metal_test_override_fail("compile-reject", detail);
        return false;
    }

    bool ok = true;
    NSDictionary<NSString *, MTLFunctionConstant *> * consts = [fn functionConstantsDictionary];
    for (NSString * key in consts) {
        MTLFunctionConstant * c = consts[key];
        if (!c.required) {
            continue;
        }

        bool found = false;
        MTLDataType type = MTLDataTypeNone;
        if ((int32_t) c.index == FC_SIMD_WIDTH) {
            found = true;
            type  = MTLDataTypeShort;
        } else if (cv) {
            for (int i = 0; i < cv->n_set; ++i) {
                if (cv->set[i].idx == (int32_t) c.index) {
                    found = true;
                    type  = cv->set[i].type;
                    break;
                }
            }
        }

        char detail[160];
        if (!found) {
            snprintf(detail, sizeof(detail), "specialization-missing-constant index=%d", (int) c.index);
            ggml_metal_test_override_fail("compile-reject", detail);
            ok = false;
            break;
        }
        if (type != c.type) {
            snprintf(detail, sizeof(detail), "specialization-type-mismatch index=%d", (int) c.index);
            ggml_metal_test_override_fail("compile-reject", detail);
            ok = false;
            break;
        }
    }

    [fn release];
    return ok;
}

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLLibrary> library = nil;
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    //
    // TODO: move to a function
    {
        const int64_t t_start = ggml_time_us();

        NSError * error = nil;
        NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
        if (path_lib == nil) {
            // Try to find the resource in the directory where the current binary located.
            NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

            NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                    if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                    }
                    if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                        // Link to the resource could not be resolved.
                        path_lib_default = nil;
                    } else {
                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                path_lib_default = nil;
            }

            path_lib = path_lib_default;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        } else {
            GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        }
#endif

        if (!library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ggml_metal_device_get_props(dev)->has_bfloat) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                }

                if (ggml_metal_device_get_props(dev)->has_tensor) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
                }

#if GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                [options setFastMathEnabled:false];

                library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return nil;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }

#if GGML_METAL_EMBED_LIBRARY
        [src release];
#endif // GGML_METAL_EMBED_LIBRARY

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));

    res->obj          = library;
    res->device       = device;
    res->device_index = ggml_metal_device_get_props(dev)->device;
    res->pipelines    = ggml_metal_pipelines_init();
    res->lock         = [NSLock new];
    res->simd_width   = 32; // safe default; probed and updated after init by caller
    res->dev_profile  = ggml_metal_device_get_profile(dev);

    return res;
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->obj          = library;
    res->device       = device;
    res->device_index = ggml_metal_device_get_props(dev)->device;
    res->pipelines    = ggml_metal_pipelines_init();
    res->lock         = [NSLock new];
    res->simd_width   = 32; // safe default; caller may update after probing
    res->dev_profile  = ggml_metal_device_get_profile(dev);

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    if (lib->obj) {
        [lib->obj release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    // candidate binaries are released after the pipelines compiled from them
    while (lib->test_overrides) {
        struct ggml_metal_test_override_entry * entry = lib->test_overrides;
        lib->test_overrides = entry->next;
        [entry->obj release];
        free(entry);
    }

    [lib->lock release];

    free(lib);
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    // Selection is resolved before the cache lookup so a hit cannot bypass validation, and the
    // lookup key carries the candidate identity so a candidate PSO is never found under the
    // baseline name. With no selector env set this is the original lookup.
    const struct ggml_metal_test_override sel = ggml_metal_test_override_current(NULL, name);
    if (sel.status == GGML_METAL_TEST_OVERRIDE_ERROR) {
        ggml_metal_test_override_fail("compile-reject", sel.error);
    } else if (sel.status == GGML_METAL_TEST_OVERRIDE_SELECTED) {
        char * key = ggml_metal_test_override_cache_key(name, sel.sha, sel.row->slot, sel.row->variant);
        if (key) {
            res.pipeline = ggml_metal_pipelines_get(lib->pipelines, key);
            free(key);
        }
    } else {
        res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    }

    [lib->lock unlock];

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    const struct ggml_metal_test_override sel = ggml_metal_test_override_current(base, name);
    if (sel.status == GGML_METAL_TEST_OVERRIDE_ERROR) {
        ggml_metal_test_override_fail("compile-reject", sel.error);
        return res;
    }
    const bool override_selected = (sel.status == GGML_METAL_TEST_OVERRIDE_SELECTED);
    char * effective_key = override_selected
        ? ggml_metal_test_override_cache_key(name, sel.sha, sel.row->slot, sel.row->variant)
        : NULL;
    if (override_selected && !effective_key) {
        ggml_metal_test_override_fail("compile-reject", "could not allocate candidate cache key");
        return res;
    }
    const char * cache_name = effective_key ? effective_key : name;

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, cache_name);
    if (res.pipeline) {
        [lib->lock unlock];
        free(effective_key);

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        // P11.65: a selected candidate is specialized from the row's candidate_entry (equal to base for the
        // grouped Q4 row); the baseline path keeps base_func.
        NSString * func_name = override_selected ? [NSString stringWithUTF8String:sel.row->candidate_entry] : base_func;

        // A selected request compiles from the candidate library with the caller's own cv and the
        // same SIMD-width loop below; nothing about the specialization is taken from the selector.
        id<MTLLibrary> func_lib = lib->obj;
        if (override_selected) {
            struct ggml_metal_test_override_entry * entry = ggml_metal_test_override_library(lib, &sel);
            if (!entry || !ggml_metal_test_override_check_specialization(entry->obj, func_name, cv)) {
                [lib->lock unlock];
                free(effective_key);
                return res;
            }
            func_lib = entry->obj;
        }

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        // Compile pipeline with adaptive FC_SIMD_WIDTH.
        // First pass: inject the device's probed simd_width.
        // After compilation, check the pipeline's actual threadExecutionWidth.
        // If it differs (e.g., Intel compiles heavy kernels to th_width=8 or 16
        // due to register pressure), recompile with the correct value.
        // This eliminates all hardcoded SIMD width assumptions.
        short simd_w = (short) lib->simd_width;
        id<MTLComputePipelineState> obj = nil;

        for (int attempt = 0; attempt < 2; ++attempt) {
            MTLFunctionConstantValues * fcv = cv ? [cv->obj copy] : [[MTLFunctionConstantValues alloc] init];
            [fcv setConstantValue:&simd_w type:MTLDataTypeShort atIndex:FC_SIMD_WIDTH];

            id<MTLFunction> mtl_function = [func_lib newFunctionWithName:func_name constantValues:fcv error:&error];
            [fcv release];
            if (!mtl_function) {
                [lib->lock unlock];

                if (override_selected) {
                    char detail[512];
                    snprintf(detail, sizeof(detail), "function %s in candidate binary: %s",
                             sel.row->candidate_entry, error ? [[error description] UTF8String] : "unknown error");
                    ggml_metal_test_override_fail("compile-reject", detail);
                }
                GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
                if (error) {
                    GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
                }
                free(effective_key);

                return res;
            }

            obj = [lib->device newComputePipelineStateWithFunction:mtl_function error:&error];

            [mtl_function release];

            if (!obj) {
                [lib->lock unlock];

                if (override_selected) {
                    char detail[512];
                    snprintf(detail, sizeof(detail), "pipeline state for %s: %s",
                             sel.row->candidate_entry, error ? [[error description] UTF8String] : "unknown error");
                    ggml_metal_test_override_fail("compile-reject", detail);
                }
                GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
                if (error) {
                    GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
                }
                free(effective_key);

                return res;
            }

            const int actual_width = (int) obj.threadExecutionWidth;

            if (actual_width != simd_w && attempt == 0) {
                // Pipeline compiled to a different SIMD width than expected.
                // Recompile with the actual value so NW matches the hardware.
                // Upward (Vega 64>32) or downward (Intel 16<32) — kernel inner
                // loops handle variable NW via chunks_per_thread iteration.
                GGML_LOG_DEBUG("%s: %-40s simd mismatch: expected %d, got %d — recompiling\n",
                        __func__, name, (int)simd_w, actual_width);
                [obj release];
                obj = nil;
                simd_w = (short) actual_width;
                continue;
            }

            break;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            if (override_selected) {
                ggml_metal_test_override_fail("unsupported", "incompatible pipeline (max threads or width is 0)");
            }
            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);
            free(effective_key);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;
        res.pipeline->device_index = lib->device_index;

        // store kernel function name for profiler dispatch labels
        strncpy(res.pipeline->name, name, sizeof(res.pipeline->name) - 1);
        res.pipeline->name[sizeof(res.pipeline->name) - 1] = '\0';

        // P12.34: tag the actual pipeline with the candidate identity before it becomes visible in the
        // cache; a later cache hit keeps this tag and dispatch never consults the selector env.
        if (override_selected) {
            snprintf(res.pipeline->binary_sha, sizeof(res.pipeline->binary_sha), "%s", sel.sha);
        }

        ggml_metal_pipelines_add(lib->pipelines, cache_name, res.pipeline);
        if (override_selected) {
            // one receipt per newly compiled selected PSO (cache hits above do not repeat it)
            GGML_LOG_INFO("%s: TEST_OVERRIDE binary_sha=%s slot=%s variant=%d base=%s name=%s\n",
                          __func__, sel.sha, sel.row->slot, sel.row->variant, base, name);
        }
        free(effective_key);
    }

    [lib->lock unlock];

    return res;
}

int ggml_metal_library_get_simd_width(ggml_metal_library_t lib) {
    return lib ? lib->simd_width : 32;
}

const struct ggml_device_profile * ggml_metal_library_get_device_profile(ggml_metal_library_t lib) {
    return lib ? lib->dev_profile : NULL;
}

//
// Dispatch receipt (P12.34): process-local count of encoded dispatches per (pipeline name, binary sha).
// Inactive until ggml_backend_metal_test_dispatch_receipt(..., reset != 0) is called, so a host that never
// calls it sees no records and pays one relaxed atomic load per dispatch. Compiling or binding a pipeline
// never records; only ggml_metal_encoder_dispatch_threadgroups does. No GPU call is made here.
//

static pthread_mutex_t ggml_metal_dispatch_receipt_mutex = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int     ggml_metal_dispatch_receipt_active = 0;
// key "<mode>|<sha>|<name>" -> NSMutableDictionary { pipeline, binary_sha, encoder_mode, count }; guarded by the mutex
static NSMutableDictionary * ggml_metal_dispatch_receipt_records = nil;

static void ggml_metal_dispatch_receipt_record(const char * name, const char * sha, bool concurrent) {
    if (!atomic_load_explicit(&ggml_metal_dispatch_receipt_active, memory_order_relaxed)) {
        return;
    }
    if (!name || name[0] == '\0' || !sha) {
        return; // no pipeline bound on this encoder: nothing to attribute
    }

    @autoreleasepool {
        NSString * n   = [NSString stringWithUTF8String:name];
        NSString * h   = [NSString stringWithUTF8String:sha];
        if (!n || !h) {
            return;
        }
        // P11.65: the encoder's actual dispatch mode is part of the identity (never inferred from flags)
        NSString * m   = concurrent ? @"concurrent" : @"serial";
        NSString * key = [NSString stringWithFormat:@"%@|%@|%@", m, h, n];

        pthread_mutex_lock(&ggml_metal_dispatch_receipt_mutex);
        if (ggml_metal_dispatch_receipt_records) {
            NSMutableDictionary * rec = [ggml_metal_dispatch_receipt_records objectForKey:key];
            if (rec) {
                [rec setObject:[NSNumber numberWithUnsignedLongLong:[[rec objectForKey:@"count"] unsignedLongLongValue] + 1]
                        forKey:@"count"];
            } else {
                rec = [NSMutableDictionary dictionaryWithObjectsAndKeys:
                        n, @"pipeline", h, @"binary_sha", m, @"encoder_mode", [NSNumber numberWithUnsignedLongLong:1], @"count", nil];
                [ggml_metal_dispatch_receipt_records setObject:rec forKey:key];
            }
        }
        pthread_mutex_unlock(&ggml_metal_dispatch_receipt_mutex);
    }
}

int ggml_backend_metal_test_dispatch_receipt(char * buf, size_t len, int reset) {
    int res = -1;

    @autoreleasepool {
        NSArray * snapshot = nil;

        pthread_mutex_lock(&ggml_metal_dispatch_receipt_mutex);
        if (reset != 0) {
            // new recording window: drop old records, then record from here on
            [ggml_metal_dispatch_receipt_records release];
            ggml_metal_dispatch_receipt_records = [[NSMutableDictionary alloc] init];
            atomic_store_explicit(&ggml_metal_dispatch_receipt_active, 1, memory_order_relaxed);
            snapshot = [NSArray array];
        } else if (ggml_metal_dispatch_receipt_records) {
            // copy the records so serialization runs outside the lock
            NSMutableArray * copy = [NSMutableArray arrayWithCapacity:[ggml_metal_dispatch_receipt_records count]];
            for (NSMutableDictionary * rec in [ggml_metal_dispatch_receipt_records allValues]) {
                [copy addObject:[NSDictionary dictionaryWithDictionary:rec]];
            }
            snapshot = copy;
        } else {
            snapshot = [NSArray array];
        }
        pthread_mutex_unlock(&ggml_metal_dispatch_receipt_mutex);

        // stable order so equal windows serialize equal
        snapshot = [snapshot sortedArrayUsingComparator:^NSComparisonResult(NSDictionary * a, NSDictionary * b) {
            NSComparisonResult r = [[a objectForKey:@"pipeline"] compare:[b objectForKey:@"pipeline"]];
            if (r != NSOrderedSame) {
                return r;
            }
            r = [[a objectForKey:@"binary_sha"] compare:[b objectForKey:@"binary_sha"]];
            return r != NSOrderedSame ? r : [[a objectForKey:@"encoder_mode"] compare:[b objectForKey:@"encoder_mode"]];
        }];

        NSError * err = nil;
        NSData * json = [NSJSONSerialization dataWithJSONObject:snapshot options:0 error:&err];
        if (json && [json length] < (NSUInteger) INT_MAX) {
            const size_t need = (size_t) [json length] + 1; // includes terminating NUL
            res = (int) need;
            if (buf && len > 0 && len >= need) {
                memcpy(buf, [json bytes], need - 1);
                buf[need - 1] = '\0';
            }
            // buf == NULL, len == 0 or len < need: size query, nothing written
        }
    }

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

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
};

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));
    res->concurrent = concurrent;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    // no profiling in standard path
    res->profiler = NULL;
    res->profiler_cb_idx = -1;
    res->profiler_pipeline_name = NULL;

    return res;
}

#ifdef GGML_METAL_TEST_OBSERVER
bool ggml_metal_encoder_is_concurrent(ggml_metal_encoder_t encoder) {
    return encoder->concurrent;
}
#endif

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];

    // P12.34: remember which pipeline (and which binary) the next dispatch will use; does not count
    memcpy(encoder->receipt_name, pipeline.pipeline->name,       sizeof(encoder->receipt_name));
    memcpy(encoder->receipt_sha,  pipeline.pipeline->binary_sha, sizeof(encoder->receipt_sha));

    // capture pipeline name for profiler dispatch labels (if active)
    // uses the name stored on the pipeline struct at compile time (not Metal label,
    // which is unsupported on older AMD drivers like BronzeMtl on D700)
    if (encoder->profiler) {
        encoder->profiler_pipeline_name = pipeline.pipeline->name;
    }
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

size_t ggml_metal_buffer_id_length(struct ggml_metal_buffer_id buffer) {
    return buffer.metal ? (size_t) [(id<MTLBuffer>) buffer.metal length] : 0;
}

bool ggml_metal_buffer_get_gpu_address(struct ggml_metal_buffer_id buffer, uint64_t * address) {
    if (!buffer.metal || !address) {
        return false;
    }
    if (@available(macOS 13.0, *)) {
        const uint64_t base_address = [(id<MTLBuffer>) buffer.metal gpuAddress];
        if (base_address == 0 || buffer.offs > UINT64_MAX - base_address) {
            return false;
        }
        *address = base_address + buffer.offs;
        return true;
    }
    return false;
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_set_op_name(ggml_metal_encoder_t encoder, const char * op_name) {
    // stores op name for the next profiled dispatch
    // no-op when profiling is off (profiler field is NULL)
    encoder->profiler_op_name = op_name;
}

void ggml_metal_encoder_set_dispatch_shape(ggml_metal_encoder_t encoder, int32_t ne00, int32_t ne01) {
    // stores tensor shape for shape-attributed profiling
    // allows profile=3 to bucket dispatches by (op, kernel, ne00) instead of just (op, kernel)
    encoder->profiler_ne00 = ne00;
    encoder->profiler_ne01 = ne01;
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    // P12.34: count this encoded dispatch exactly once, before the profiled/normal split below
    // (the profiled wrapper is only reached from here, so it must not record again)
    if (tg0 > 0 && tg1 > 0 && tg2 > 0) {
        ggml_metal_dispatch_receipt_record(encoder->receipt_name, encoder->receipt_sha, encoder->concurrent);
    }

    // when profiler is active on this encoder, route through the profiled dispatch path
    if (encoder->profiler) {
        ggml_metal_encoder_dispatch_profiled(
            encoder, tg0, tg1, tg2, tptg0, tptg1, tptg2,
            encoder->profiler, encoder->profiler_cb_idx,
            encoder->profiler_op_name ? encoder->profiler_op_name : "UNKNOWN_OP",
            encoder->profiler_pipeline_name ? encoder->profiler_pipeline_name : "unknown_kernel");
        return;
    }

    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

//
// Per-dispatch GPU profiler (GGML_METAL_PROFILE=3)
//

#define PROFILER_MAX_CB          9   // 8 worker CBs + 1 main thread CB (matches GGML_METAL_MAX_COMMAND_BUFFERS + 1)
#define PROFILER_MAX_DISPATCHES  1024
#define PROFILER_SAMPLES_PER_DISPATCH 2

struct profiler_dispatch_record {
    int          cb_idx;
    int          sample_start;   // index into this CB's sample buffer
    int          sample_end;
    const char * op_name;        // ggml op (e.g. "MUL_MAT") — pointer to static string
    const char * pipeline_name;  // Metal kernel name — interned copy
    int          tg0, tg1, tg2;
    int          tptg0, tptg1, tptg2;
    int32_t      ne00, ne01;     // tensor shape for shape-attributed profiling
    uint64_t     gpu_ns;         // filled at resolve time
};

struct profiler_cb_state {
    id<MTLCounterSampleBuffer> sample_buffer;
    int next_sample_idx;
    int max_samples;

    // per-CB dispatch records (thread-local to each encoding thread — no synchronization needed)
    struct profiler_dispatch_record records[PROFILER_MAX_DISPATCHES];
    int n_records;

    // per-CB string intern table for pipeline names (thread-local)
    char label_pool[256][128];
    int  n_labels;
};

struct ggml_metal_profiler {
    bool active;

    // counter set reference (cached at init)
    id<MTLCounterSet> timestamp_set;
    id<MTLDevice>     device;

    // per-command-buffer sample state (each CB is encoded by a single thread)
    struct profiler_cb_state cb_states[PROFILER_MAX_CB];
    int n_cb_states;  // set atomically from begin_graph, read-only during encoding

    // overflow tracking (printed in report footer if nonzero)
    int dropped_dispatches; // dispatches that couldn't be sampled (per-CB record array full)
};

// Intern a pipeline name string into per-CB label pool (thread-safe: each CB has its own pool)
static const char * profiler_intern_label(struct profiler_cb_state * cbs, const char * str) {
    if (!str) return "unknown";
    for (int i = 0; i < cbs->n_labels; i++) {
        if (strcmp(cbs->label_pool[i], str) == 0) return cbs->label_pool[i];
    }
    if (cbs->n_labels < 256) {
        strncpy(cbs->label_pool[cbs->n_labels], str, 127);
        cbs->label_pool[cbs->n_labels][127] = '\0';
        return cbs->label_pool[cbs->n_labels++];
    }
    return "label_pool_full"; // stable sentinel
}

ggml_metal_profiler_t ggml_metal_profiler_init(ggml_metal_device_t dev, int profile_level) {
    if (profile_level != 3) return NULL;

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // find timestamp counter set
    id<MTLCounterSet> ts_set = nil;
    for (id<MTLCounterSet> cs in [device counterSets]) {
        if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            ts_set = cs;
            break;
        }
    }
    if (!ts_set) {
        fprintf(stderr, "[METAL_PROFILE] profile=3: no timestamp counter set on %s, falling back to profile=1\n",
                device.name.UTF8String);
        return NULL;
    }

    if (![device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        fprintf(stderr, "[METAL_PROFILE] profile=3: dispatch boundary sampling unsupported on %s, falling back to profile=1\n",
                device.name.UTF8String);
        return NULL;
    }

    ggml_metal_profiler_t prof = calloc(1, sizeof(struct ggml_metal_profiler));
    prof->active = true;
    prof->timestamp_set = ts_set;
    prof->device = device;

    fprintf(stderr, "[METAL_PROFILE] profile=3: per-dispatch timestamp profiling enabled on %s\n",
            device.name.UTF8String);

    return prof;
}

void ggml_metal_profiler_free(ggml_metal_profiler_t prof) {
    if (!prof) return;
    // MRC: release sample buffers (created via newCounterSampleBufferWithDescriptor: → +1 ownership)
    for (int i = 0; i < PROFILER_MAX_CB; i++) {
        if (prof->cb_states[i].sample_buffer) {
            [prof->cb_states[i].sample_buffer release];
            prof->cb_states[i].sample_buffer = nil;
        }
    }
    free(prof);
}

void ggml_metal_profiler_begin_graph(ggml_metal_profiler_t prof, int n_nodes) {
    if (!prof || !prof->active) return;
    (void)n_nodes;
    prof->dropped_dispatches = 0;
    for (int i = 0; i < PROFILER_MAX_CB; i++) {
        prof->cb_states[i].next_sample_idx = 0;
        prof->cb_states[i].n_records = 0;
        prof->cb_states[i].n_labels = 0;
    }
}

ggml_metal_encoder_t ggml_metal_encoder_init_profiled(
    ggml_metal_cmd_buf_t  cmd_buf_raw,
    bool                  concurrent,
    ggml_metal_profiler_t prof,
    int                   cb_idx
) {
    // if no profiler, fall through to standard encoder creation
    if (!prof || !prof->active || cb_idx < 0 || cb_idx >= PROFILER_MAX_CB) {
        return ggml_metal_encoder_init(cmd_buf_raw, concurrent);
    }

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    // allocate or reuse counter sample buffer for this CB
    struct profiler_cb_state * cbs = &prof->cb_states[cb_idx];
    if (cbs->sample_buffer == nil) {
        MTLCounterSampleBufferDescriptor * desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet  = prof->timestamp_set;
        desc.sampleCount = PROFILER_MAX_DISPATCHES * PROFILER_SAMPLES_PER_DISPATCH;
        desc.storageMode = MTLStorageModeShared;
        desc.label       = @"ggml_profile3";

        NSError * err = nil;
        cbs->sample_buffer = [prof->device newCounterSampleBufferWithDescriptor:desc error:&err];
        [desc release]; // MRC: descriptor no longer needed after buffer creation

        if (!cbs->sample_buffer) {
            fprintf(stderr, "[METAL_PROFILE] failed to allocate sample buffer for cb %d: %s\n",
                    cb_idx, err.localizedDescription.UTF8String);
            return ggml_metal_encoder_init(cmd_buf_raw, concurrent);
        }
        cbs->max_samples = PROFILER_MAX_DISPATCHES * PROFILER_SAMPLES_PER_DISPATCH;
    }
    cbs->next_sample_idx = 0;

    // create encoder with sample buffer attachment via compute pass descriptor
    MTLComputePassDescriptor * pass_desc = [MTLComputePassDescriptor computePassDescriptor];
    pass_desc.sampleBufferAttachments[0].sampleBuffer = cbs->sample_buffer;
    pass_desc.sampleBufferAttachments[0].startOfEncoderSampleIndex = MTLCounterDontSample;
    pass_desc.sampleBufferAttachments[0].endOfEncoderSampleIndex   = MTLCounterDontSample;

    // note: computeCommandEncoderWithDescriptor does not support MTLDispatchTypeConcurrent,
    // so when profiling is active we always use serial dispatch. This is acceptable because
    // profile=3 is a debug mode and concurrency is already disabled on discrete AMD GPUs.
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));
    // Profiled encoders are always serial (see note above), so the actual mode is false.
    res->concurrent = false;
    res->obj = [cmd_buf computeCommandEncoderWithDescriptor:pass_desc];
    [res->obj retain];

    // CRITICAL: store profiler state on the encoder so dispatch can route to profiled path
    res->profiler              = prof;
    res->profiler_cb_idx       = cb_idx;
    res->profiler_pipeline_name = NULL;
    res->profiler_op_name      = NULL;

    // track highest CB index seen — safe because main thread (highest cb_idx = n_cb)
    // runs first, before dispatch_apply launches worker threads (cb_idx = 0..n_cb-1)
    if (prof->n_cb_states <= cb_idx) {
        prof->n_cb_states = cb_idx + 1;
    }

    return res;
}

void ggml_metal_encoder_dispatch_profiled(
    ggml_metal_encoder_t  encoder,
    int tg0, int tg1, int tg2,
    int tptg0, int tptg1, int tptg2,
    ggml_metal_profiler_t prof,
    int                   cb_idx,
    const char *          op_name,
    const char *          pipeline_name
) {
    // fast path: no profiler → standard dispatch
    if (!prof || !prof->active || cb_idx < 0 || cb_idx >= PROFILER_MAX_CB) {
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                     threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        return;
    }

    struct profiler_cb_state * cbs = &prof->cb_states[cb_idx];
    if (!cbs->sample_buffer || cbs->next_sample_idx + 2 > cbs->max_samples) {
        // no sample buffer or overflow — dispatch without sampling
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                     threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        return;
    }

    if (cbs->n_records >= PROFILER_MAX_DISPATCHES) {
        // per-CB dispatch record array full — dispatch without sampling, count the drop
        __atomic_fetch_add(&prof->dropped_dispatches, 1, __ATOMIC_RELAXED);
        [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                     threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
        return;
    }

    int start_idx = cbs->next_sample_idx++;
    int end_idx   = cbs->next_sample_idx++;

    // pre-dispatch sample
    [encoder->obj sampleCountersInBuffer:cbs->sample_buffer
                           atSampleIndex:(NSUInteger)start_idx
                             withBarrier:NO];

    // the actual dispatch
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2)
                 threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];

    // post-dispatch sample
    [encoder->obj sampleCountersInBuffer:cbs->sample_buffer
                           atSampleIndex:(NSUInteger)end_idx
                             withBarrier:NO];

    // record metadata (per-CB — no thread contention)
    struct profiler_dispatch_record * rec = &cbs->records[cbs->n_records++];
    rec->cb_idx         = cb_idx;
    rec->sample_start   = start_idx;
    rec->sample_end     = end_idx;
    rec->op_name        = op_name;  // pointer to static ggml_op_name() string
    rec->pipeline_name  = profiler_intern_label(cbs, pipeline_name);
    rec->tg0 = tg0; rec->tg1 = tg1; rec->tg2 = tg2;
    rec->tptg0 = tptg0; rec->tptg1 = tptg1; rec->tptg2 = tptg2;
    rec->ne00 = encoder->profiler_ne00;
    rec->ne01 = encoder->profiler_ne01;
    rec->gpu_ns = 0;
}

void ggml_metal_profiler_end_graph(ggml_metal_profiler_t prof) {
    if (!prof || !prof->active) return;

    // check if any CB has records
    int total_records = 0;
    for (int cb = 0; cb < prof->n_cb_states; cb++) {
        total_records += prof->cb_states[cb].n_records;
    }
    if (total_records == 0) return;

    typedef struct { uint64_t timestamp; } ts_result_t;

    // RESOLVE PHASE: decode timestamps from each CB's sample buffer
    // @autoreleasepool: NSData from resolveCounterRange: is autoreleased — drain here
    @autoreleasepool {
    for (int cb = 0; cb < prof->n_cb_states; cb++) {
        struct profiler_cb_state * cbs = &prof->cb_states[cb];
        if (!cbs->sample_buffer || cbs->next_sample_idx == 0 || cbs->n_records == 0) continue;

        NSData * resolved = nil;
        @try {
            resolved = [cbs->sample_buffer resolveCounterRange:NSMakeRange(0, cbs->next_sample_idx)];
        } @catch (NSException * e) {
            fprintf(stderr, "[METAL_PROFILE] resolve failed for cb %d: %s\n", cb, e.reason.UTF8String);
            continue;
        }
        if (!resolved) continue;

        const ts_result_t * ts = (const ts_result_t *)resolved.bytes;

        for (int i = 0; i < cbs->n_records; i++) {
            struct profiler_dispatch_record * rec = &cbs->records[i];
            uint64_t t0 = ts[rec->sample_start].timestamp;
            uint64_t t1 = ts[rec->sample_end].timestamp;
            if (t0 == ~0ULL || t1 == ~0ULL || t1 <= t0) {
                rec->gpu_ns = 0;
            } else {
                rec->gpu_ns = t1 - t0;
            }
        }
    }
    } // @autoreleasepool — drains NSData from resolveCounterRange

    // AGGREGATE PHASE: merge per-CB records and group by (op_name, pipeline_name, ne00)
    // ne00 bucketing allows shape-attributed profiling for matmul ops — different matrix
    // dimensions often want different kernel configurations on the same GPU.
    struct { const char * op; const char * pipe; int32_t ne00; int32_t ne01; uint64_t total_ns; int count; } aggs[256];
    int n_aggs = 0;
    uint64_t graph_total_ns = 0;
    int valid_dispatches = 0;
    int n_agg_overflow = 0;

    for (int cb = 0; cb < prof->n_cb_states; cb++) {
        struct profiler_cb_state * cbs = &prof->cb_states[cb];
        for (int i = 0; i < cbs->n_records; i++) {
            struct profiler_dispatch_record * rec = &cbs->records[i];
            if (rec->gpu_ns == 0) continue;

            graph_total_ns += rec->gpu_ns;
            valid_dispatches++;

            // bucket by (op, pipeline, ne00, ne01) — shape-attributed profiling
            // for non-matmul ops, ne00/ne01 are 0 so they aggregate as before
            int found = -1;
            for (int j = 0; j < n_aggs; j++) {
                if (aggs[j].op == rec->op_name &&
                    strcmp(aggs[j].pipe, rec->pipeline_name) == 0 &&
                    aggs[j].ne00 == rec->ne00 &&
                    aggs[j].ne01 == rec->ne01) {
                    found = j;
                    break;
                }
            }
            if (found >= 0) {
                aggs[found].total_ns += rec->gpu_ns;
                aggs[found].count++;
            } else if (n_aggs < 256) {
                aggs[n_aggs].op = rec->op_name;
                aggs[n_aggs].pipe = rec->pipeline_name;
                aggs[n_aggs].ne00 = rec->ne00;
                aggs[n_aggs].ne01 = rec->ne01;
                aggs[n_aggs].total_ns = rec->gpu_ns;
                aggs[n_aggs].count = 1;
                n_aggs++;
            } else {
                n_agg_overflow++;
            }
        }
    }

    // SORT descending by total_ns
    for (int i = 0; i < n_aggs - 1; i++) {
        for (int j = i + 1; j < n_aggs; j++) {
            if (aggs[j].total_ns > aggs[i].total_ns) {
                __typeof__(aggs[0]) tmp = aggs[i]; aggs[i] = aggs[j]; aggs[j] = tmp;
            }
        }
    }

    // REPORT
    int invalid = total_records - valid_dispatches;
    fprintf(stderr, "[METAL_PROFILE] dispatch breakdown (graph %.2f ms, %d dispatches", graph_total_ns / 1e6, total_records);
    if (invalid > 0) fprintf(stderr, ", %d invalid", invalid);
    if (prof->dropped_dispatches > 0) fprintf(stderr, ", %d dropped", prof->dropped_dispatches);
    if (n_agg_overflow > 0) fprintf(stderr, ", %d agg overflow", n_agg_overflow);
    fprintf(stderr, "):\n");

    for (int i = 0; i < n_aggs; i++) {
        double ms  = aggs[i].total_ns / 1e6;
        double pct = graph_total_ns > 0 ? (aggs[i].total_ns * 100.0 / graph_total_ns) : 0;
        double avg = aggs[i].count > 0 ? (aggs[i].total_ns / (double)aggs[i].count / 1e3) : 0;
        // show ne00xne01 shape when available (nonzero) — shape-attributed profiling
        if (aggs[i].ne00 > 0) {
            fprintf(stderr, "  %-24s %-32s [%5dx%-5d] %8.2f ms (%5.1f%%) x%-5d avg=%.1f us\n",
                    aggs[i].op, aggs[i].pipe, aggs[i].ne00, aggs[i].ne01, ms, pct, aggs[i].count, avg);
        } else {
            fprintf(stderr, "  %-24s %-32s              %8.2f ms (%5.1f%%) x%-5d avg=%.1f us\n",
                    aggs[i].op, aggs[i].pipe, ms, pct, aggs[i].count, avg);
        }
    }
}

struct ggml_metal_device {
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
    struct ggml_model_profile  model_profile;  // set via ggml_backend_metal_set_model_profile()

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

ggml_metal_rsets_t ggml_metal_rsets_init(void) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, 2*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  // half a second
                  usleep(500 * 1000);
              }
        }
#endif
    });

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

// PCI vendor and device id of a Metal device, from its IORegistry parents; 0 when not found.
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

static void ggml_metal_pci_id(id<MTLDevice> mtl_device, uint32_t * vendor, uint32_t * device_id) {
    *vendor    = 0;
    *device_id = 0;
    io_registry_entry_t entry = IOServiceGetMatchingService(MACH_PORT_NULL, IORegistryEntryIDMatching(mtl_device.registryID));
    if (entry) {
        *vendor    = ggml_metal_pci_prop(entry, CFSTR("vendor-id"));
        *device_id = ggml_metal_pci_prop(entry, CFSTR("device-id"));
        IOObjectRelease(entry);
    }
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

            dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX
            // In case of eGPU, shared memory may be preferable.
            dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }
            if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                dev->props.use_shared_buffers = true;
            }

            // Profile: identity
            uint32_t pci_vendor = 0;
            uint32_t pci_device = 0;
            ggml_metal_pci_id(dev->mtl_device, &pci_vendor, &pci_device);
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

            // PCI-keyed hardware row (generated header); a -1 field keeps the value set above.
            for (int i = 0; i < ggml_metal_hw_rows_n; i++) {
                const struct ggml_metal_hw_row * row = &ggml_metal_hw_rows[i];
                if (row->pci_vendor != pci_vendor || row->pci_device != pci_device) {
                    continue;
                }
                if (row->compute_units   >= 0) dev->profile.compute_units   = (uint32_t) row->compute_units;
                if (row->l1_cache_per_cu >= 0) dev->profile.l1_cache_per_cu = (uint32_t) row->l1_cache_per_cu;
                if (row->l2_cache_total  >= 0) dev->profile.l2_cache_total  = (uint32_t) row->l2_cache_total;
                if (row->local_bw        >= 0) dev->profile.local_bandwidth = (uint64_t) row->local_bw;
                if (row->mv_reduce       >= 0) dev->profile.mv_reduce       = (uint32_t) row->mv_reduce;
                if (row->f_scalar_w64    >= 0) dev->profile.f_scalar_w64    = (uint32_t) row->f_scalar_w64;
                if (row->mv_group        >= 0) dev->profile.mv_group        = (uint32_t) row->mv_group;
                if (row->attn_gate_fuse  >= 0) dev->profile.attn_gate_fuse  = (uint32_t) row->attn_gate_fuse;
                if (row->conv_l2_fold    >= 0) dev->profile.conv_l2_fold    = (uint32_t) row->conv_l2_fold;
                if (row->add_norm_fuse   >= 0) dev->profile.add_norm_fuse   = (uint32_t) row->add_norm_fuse;
                if (row->cb_seq          >= 0) dev->profile.cb_seq          = (uint32_t) row->cb_seq;
                if (row->q8_load_bytes   >= 0) dev->profile.q8_load_bytes   = (uint32_t) row->q8_load_bytes;
                if (row->q8_rows8_min    >= 0) dev->profile.q8_rows8_min    = (uint32_t) row->q8_rows8_min;
                GGML_LOG_INFO("%s: hw row 0x%04x:0x%04x %s: CUs=%u, L1=%u KB/CU, L2=%u KB, BW=%.0f GB/s, mv_reduce=%u, f_scalar_w64=%u, mv_group=%u, attn_gate_fuse=%u, conv_l2_fold=%u, add_norm_fuse=%u, cb_seq=%u, q8_load_bytes=%u, q8_rows8_min=%u\n",
                    __func__, pci_vendor, pci_device, row->chip,
                    dev->profile.compute_units, dev->profile.l1_cache_per_cu / 1024,
                    dev->profile.l2_cache_total / 1024, dev->profile.local_bandwidth / 1e9, dev->profile.mv_reduce, dev->profile.f_scalar_w64, dev->profile.mv_group, dev->profile.attn_gate_fuse, dev->profile.conv_l2_fold, dev->profile.add_norm_fuse, dev->profile.cb_seq, dev->profile.q8_load_bytes, dev->profile.q8_rows8_min);
                break;
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

            dev->library = ggml_metal_library_init(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
            }

            // Probe actual SIMD group width from a compiled pipeline.
            // AMD GCN/Vega reports threadExecutionWidth=64 while Apple Silicon and RDNA report 32.
            // The hardcoded value of 32 causes wrong results on GCN: simd_sum reduces across 64
            // threads but shared memory slots only cover 32, and tiisg ranges 0-63 not 0-31.
            //
            // We compile a simple kernel (kernel_concat has no function constants) and read
            // threadExecutionWidth from the resulting pipeline state. All kernels on a given
            // device share the same SIMD group width, so one probe is sufficient.
            if (dev->library) {
                struct ggml_metal_pipeline_with_params probe =
                    ggml_metal_library_compile_pipeline(dev->library, "kernel_concat", "kernel_concat", NULL);
                if (probe.pipeline) {
                    const int tw = ggml_metal_pipeline_thread_execution_width(probe);
                    if (tw > 0) {
                        dev->profile.simd_width  = (uint32_t) tw;
                        dev->library->simd_width = tw;
                        GGML_LOG_INFO("%s: probed simd_width = %d\n", __func__, tw);
                    }
                }
            }

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

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, 2*dev->rsets->keep_alive_s, memory_order_relaxed);
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

// Admit only weight types with existing scalar matvec and get_rows entries.
// INTENT/PDL: reject missing kernels before dispatch, preserving CPU fallback.
static bool ggml_metal_has_matvec_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_MXFP4:
            return true;
        default:
            return false;
    }
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->profile.has_matrix_hw;
    const bool has_simdgroup_reduction = dev->profile.has_simd_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ACC:
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            return has_simdgroup_reduction && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
            // SUM_ROWS now handles non-contiguous (transposed) src0 via nb00 stride.
            // This eliminates ggml_cont(ggml_transpose(...)) before sum_rows in SSM
            // delta-net autoregressive path — the "sum_cols" workaround pattern.
            return has_simdgroup_reduction;
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_ROPE:
            return true;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32 && op->op_params[0] == GGML_SCALE_MODE_NEAREST && !(op->op_params[0] & GGML_SCALE_FLAG_ANTIALIAS);
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            // TODO: add circular padding support for metal, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }

            return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_IRFFT:
        case GGML_OP_FOLD:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 576) {
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            // vec kernels only need simdgroup_reduction, not simdgroup_mm
            if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
                return has_simdgroup_reduction;
            }
            if (has_simdgroup_mm) {
                return true;
            }
            // scalar FA fallback for GPUs without simdgroup matrix multiply
            // F16 K/V: all head sizes {32, 40, 48, 64, 72, 80, 96, 112, 128, 192, 256, 576}
            // Q8_0/Q4_0 K/V: subset {64, 128, 256}
            if (has_simdgroup_reduction) {
                const enum ggml_type kv_type = op->src[1]->type;
                const int head_size = op->src[0]->ne[0];

                // Check if K/V type is supported
                if (kv_type == GGML_TYPE_F16) {
                    // F16: all head sizes supported
                    // (already checked in the outer gate above)
                } else if (kv_type == GGML_TYPE_Q8_0 || kv_type == GGML_TYPE_Q4_0) {
                    // Q8_0/Q4_0: only dk={64, 128, 256} have kernels
                    if (head_size != 64 && head_size != 128 && head_size != 256) {
                        GGML_LOG_WARN("%s: SCALAR FA with %s K/V requires head size 64/128/256, got %d — falling back to CPU\n",
                            __func__, ggml_type_name(kv_type), head_size);
                        return false;
                    }
                } else {
                    GGML_LOG_WARN("%s: SCALAR FA got unsupported K/V type %s — falling back to CPU\n",
                        __func__, ggml_type_name(kv_type));
                    return false;
                }

                return true;
            }
            return false;
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_GATED_DELTA_NET: {
            const int width = dev->profile.simd_width;
            const int size = op->src[2]->ne[0];
            const int threads = size < width ? size : width;
            if (!has_simdgroup_reduction || threads <= 0 || size % threads != 0) {
                return false;
            }
            const int nsg = size / threads;
            return (nsg == 1 || nsg == 2 || nsg == 4) &&
                threads * nsg <= (int) dev->profile.max_threads_per_threadgroup;
        }
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_SOLVE_TRI:
            return has_simdgroup_reduction;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            // no kernel takes an F32 weight with a BF16 activation (only bf16 x bf16 reads BF16 src1)
            if (op->op == GGML_OP_MUL_MAT && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_BF16) {
                return false;
            }
            return has_simdgroup_reduction && ggml_metal_has_matvec_type(op->src[0]->type);
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            return op->src[0]->type == GGML_TYPE_I32 || ggml_metal_has_matvec_type(op->src[0]->type);
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

const struct ggml_device_profile * ggml_metal_device_get_profile(ggml_metal_device_t dev) {
    return &dev->profile;
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

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicity "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    ggml_metal_device_rsets_rm(buf->dev, buf->rset);

    for (int i = 0; i < buf->n_buffers; i++) {
        [buf->buffers[i].metal release];
    }

    ggml_metal_buffer_rset_free(buf);

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        // src - try zero-copy first (requires page-aligned pointer and size),
        // fall back to a copy if the source data isn't suitable
        void * data_ptr = (void *)(uintptr_t) data;
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        if (!buf_src) {
            buf_src = [buf->dev->mtl_device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
        }

        GGML_ASSERT(buf_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];
    }

    ggml_metal_xfer_stats.h2d_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.h2d_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.h2d_count   += 1;
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst - try zero-copy first (requires page-aligned pointer and size),
        // fall back to a copy if the destination data isn't suitable
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        const bool dst_needs_copy = (buf_dst == nil);
        if (dst_needs_copy) {
            buf_dst = [buf->dev->mtl_device newBufferWithLength:size options:MTLResourceStorageModeShared];
        }

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (dst_needs_copy) {
            memcpy(data, [buf_dst contents], size);
        }
    }

    ggml_metal_xfer_stats.d2h_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.d2h_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.d2h_count   += 1;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

// Shared view lookup with an explicit byte size, so a caller that already
// checked the span never recomputes it through unchecked arithmetic.
static struct ggml_metal_buffer_id ggml_metal_buffer_get_id_sized(ggml_metal_buffer_t buf, const struct ggml_tensor * t, int64_t tsize) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    return ggml_metal_buffer_get_id_sized(buf, t, ggml_nbytes(t));
}

#ifdef GGML_METAL_TEST_OBSERVER
struct ggml_metal_buffer_id ggml_metal_buffer_get_id_checked(ggml_metal_buffer_t buf, const struct ggml_tensor * t, size_t span) {
    // Keep the observer ABI's historic signed-size limit before mapping the
    // checked span through the unsigned view predicate below.
    if (span > (size_t) INT64_MAX) {
        struct ggml_metal_buffer_id res = { nil, 0 };
        return res;
    }

    // The production mapper above keeps its legacy signed arithmetic. The
    // observer query instead checks each candidate view without signed pointer
    // casts, signed end-offset addition, or narrowing before validation.
    for (int i = 0; i < buf->n_buffers; ++i) {
        size_t view_offset = 0;
        if (ggml_metal_test_observer_checked_view_lookup(
                    (uintptr_t) t->data, (uintptr_t) buf->buffers[i].data,
                    buf->buffers[i].size, span, &view_offset)) {
            struct ggml_metal_buffer_id res = {
                buf->buffers[i].metal,
                view_offset,
            };
            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);
    struct ggml_metal_buffer_id res = { nil, 0 };
    return res;
}
#endif
