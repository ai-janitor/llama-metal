#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(device) //
// Candidate loader (P12.18): lets a host compile one registered kernel from a candidate metallib
// through the same compile_pipeline, with the route's own function constants. All of it is inert
// unless the GGML_METAL_TEST_* selector env vars are set.
//

static pthread_mutex_t ggml_metal_test_override_err_mutex = PTHREAD_MUTEX_INITIALIZER;

// Sticky diagnostic error: the host clears GGML_METAL_TEST_OVERRIDE_ERROR per job and rejects the
// timing if it is set, so a fallback cannot pass as a candidate. The stderr/log line is printed
// once per distinct message so a rejected selection does not flood per request.
void ggml_metal_test_override_fail(const char * kind, const char * detail) {
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

struct ggml_metal_test_override ggml_metal_test_override_current(const char * base, const char * name) {
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
struct ggml_metal_test_override_entry * ggml_metal_test_override_library(
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
bool ggml_metal_test_override_check_specialization(
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

void ggml_metal_dispatch_receipt_record(const char * name, const char * sha, bool concurrent) {
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

