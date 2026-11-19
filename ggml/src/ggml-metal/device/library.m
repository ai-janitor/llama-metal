#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
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

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

GGML_METAL_FAMILY(device) void ggml_metal_cv_free(ggml_metal_cv_t cv) {
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
                } else if (ggml_metal_device_get_props(dev)->has_bf16_u16) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_BF16_U16"];
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

