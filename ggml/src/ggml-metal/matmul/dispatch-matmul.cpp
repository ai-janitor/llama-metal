// Matrix-matrix multiplication dispatch — simdgroup_mm (Apple Silicon) and tiled (AMD/Intel).
// Extracted from dispatch.cpp to keep each dispatch strategy in its own file.
#include "dispatch.h"
#include "formats.h"
#include "tiled-plan.h"
#include "act-copy/act-copy.h"
#include "lanes/select.h"
#include "ggml-metal-probe.h"
#include "ggml-metal.h"
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
GGML_METAL_FAMILY(matmul)
// The suffix of a type's kernel_mul_mat_tiled_<suffix> entry in kernel-tiled.metal, or nullptr when it has none.
// A new tiled type is the tiled cell of its row in data/matmul.tsv [formats] and one template instantiation there.
static const char * tiled_type_suffix(enum ggml_type type) {
    const struct mv_format_row * row = ggml_metal_mv_format(type);
    return row != nullptr && row->tiled ? ggml_type_name(type) : nullptr;
}

int ggml_backend_metal_tiled_plan_count(void) {
    return ggml_metal_tiled_compiled_count();
}

const char * ggml_backend_metal_tiled_plan_id(int i) {
    return ggml_metal_tiled_compiled_id(i);
}

int ggml_backend_metal_tiled_plan_values(int i, int32_t * values, int n) {
    const int32_t * row = ggml_metal_tiled_compiled_row(i);
    if (row == nullptr || values == nullptr || n < GGML_METAL_TILED_PLAN_N) {
        return 0;
    }
    memcpy(values, row, sizeof(int32_t) * GGML_METAL_TILED_PLAN_N);
    return GGML_METAL_TILED_PLAN_N;
}

int ggml_backend_metal_tiled_has_weight(enum ggml_type type) {
    return type >= 0 && type < GGML_TYPE_COUNT && tiled_type_suffix(type) != nullptr ? 1 : 0;
}

// The tiled pipeline of one kernel entry and one plan: the kernel is compiled once per plan with the plan as
// compile-time constants, and the entry of a plan is named <entry>_lay_<the plan's values in field order>.

// Reasons already reported, so each is logged once. Only a call that failed somewhere takes this lock.
static std::mutex tiled_told_lock;
static std::vector<std::string> tiled_told_list;

static bool tiled_tell_once(const char * what) {
    std::lock_guard<std::mutex> guard(tiled_told_lock);
    for (const std::string & s : tiled_told_list) {
        if (s == what) {
            return false;
        }
    }
    tiled_told_list.push_back(what);
    return true;
}

// The test hook GGML_METAL_TEST_TILED_PLAN=<weight type or any>:<plan id>, inert unless set: this process runs that
// compiled plan for that weight type instead of the card's selection (a sweep from one binary). row is the plan's row
// (-1 when unset), weight its ggml_type value (-1 = every type). A value that does not parse aborts. An op the hook
// applies to runs on that plan or the process aborts: it never runs another plan, kernel or route. A process that
// ends without one node dispatched on the plan aborts too, after its receipt: one TILED_TEST_PLAN line (the plan's exact
// entry suffix, the mul_mat nodes of that weight dispatched on the plan and on any other route, whole process) and one
// TILED_TEST_SHAPE line per weight shape, token count and route (k = weight columns, m = weight rows, n = tokens).
struct tiled_test_hook {
    int32_t row;
    int32_t weight;
};

static struct tiled_test_hook tiled_test_parse(void) {
    struct tiled_test_hook hook = { -1, -1 };
    const char * env = getenv("GGML_METAL_TEST_TILED_PLAN");
    if (env == nullptr || env[0] == '\0') {
        return hook;
    }
    const char * colon = strchr(env, ':');
    if (colon == nullptr) {
        GGML_ABORT("GGML_METAL_TEST_TILED_PLAN must be <weight type or any>:<plan id>, got '%s'", env);
    }
    hook.row = ggml_metal_tiled_compiled_find(colon + 1);
    if (hook.row < 0) {
        GGML_ABORT("GGML_METAL_TEST_TILED_PLAN: '%s' is not a plan id of data/matmul.tsv [tiles]", colon + 1);
    }
    const std::string weight(env, colon - env);
    if (weight != "any") {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            const char * name = ggml_type_name((enum ggml_type) i);
            if (name != nullptr && weight == name && tiled_type_suffix((enum ggml_type) i) != nullptr) {
                hook.weight = i;
            }
        }
        if (hook.weight < 0) {
            GGML_ABORT("GGML_METAL_TEST_TILED_PLAN: '%s' is neither any nor a weight type with a tiled kernel", weight.c_str());
        }
    }
    return hook;
}

static const struct tiled_test_hook & tiled_test(void) {
    static const struct tiled_test_hook hook = tiled_test_parse();
    return hook;
}

// GGML_METAL_TEST_TILED_LIMIT=<threads>, inert unless set: the tiled route treats every pipeline as allowing at most
// that many threads per group (never more than it really allows), so the low-limit paths can be run on a card whose
// pipelines allow every compiled plan.
static int tiled_test_limit(void) {
    static const int limit = [] {
        const char * env = getenv("GGML_METAL_TEST_TILED_LIMIT");
        if (env == nullptr) {
            return 0;
        }
        // Set means a value: digits only (an empty value, a sign or a space is refused like any other junk).
        bool digits = env[0] != '\0' && strlen(env) <= 4;
        for (const char * c = env; *c != '\0'; ++c) {
            digits = digits && *c >= '0' && *c <= '9';
        }
        const long value = digits ? strtol(env, nullptr, 10) : 0;
        if (!digits || value < 1 || value > 1024) {
            GGML_ABORT("GGML_METAL_TEST_TILED_LIMIT must be a whole number of threads from 1 to 1024, got '%s'", env);
        }
        return (int) value;
    }();
    return limit;
}

// The mul_mat nodes the hook applies to, by weight type, shape and the route that took them. Kept only while the hook
// is set.
struct tiled_test_shape {
    int32_t type, k, m, n;
    std::string route;
    long nodes;
};
static std::mutex tiled_test_lock;
static std::vector<struct tiled_test_shape> tiled_test_shapes;
static std::atomic<long> tiled_test_on_plan{0};
static std::atomic<long> tiled_test_other{0};

static bool tiled_test_applies(const struct tiled_test_hook & hook, enum ggml_type type) {
    return hook.row >= 0 && (hook.weight < 0 ? tiled_type_suffix(type) != nullptr : hook.weight == (int32_t) type);
}

// A route took nodes idx .. idx + n - 1 (a fused or grouped call takes several): each mul_mat node among them that the
// hook applies to is counted, under its own shape. on_plan: the route is the tiled kernel on the hook's plan.
static void tiled_test_took(ggml_metal_op_t ctx, int idx, int n, const char * route, bool on_plan) {
    const struct tiled_test_hook & hook = tiled_test();
    if (hook.row < 0) {
        return;
    }
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * op = ctx->node(idx + i);
        if (op == nullptr || op->op != GGML_OP_MUL_MAT || op->src[0] == nullptr || op->src[1] == nullptr || !tiled_test_applies(hook, op->src[0]->type)) {
            continue;
        }
        (on_plan ? tiled_test_on_plan : tiled_test_other).fetch_add(1);
        const struct tiled_test_shape s = { (int32_t) op->src[0]->type, (int32_t) op->src[0]->ne[0], (int32_t) op->src[0]->ne[1], (int32_t) op->src[1]->ne[1], route, 1 };
        std::lock_guard<std::mutex> guard(tiled_test_lock);
        bool found = false;
        for (struct tiled_test_shape & have : tiled_test_shapes) {
            if (have.type == s.type && have.k == s.k && have.m == s.m && have.n == s.n && have.route == s.route) {
                have.nodes++;
                found = true;
                break;
            }
        }
        if (!found) {
            tiled_test_shapes.push_back(s);
        }
    }
}

void ggml_metal_tiled_test_route(ggml_metal_op_t ctx, int idx, int n, const char * route) {
    tiled_test_took(ctx, idx, n, route, false);
}

// The receipt of a process run with the hook, printed when it ends.
static void tiled_test_at_exit(void) {
    const struct tiled_test_hook & hook = tiled_test();
    const int32_t * v = ggml_metal_tiled_compiled_row(hook.row);
    char suffix[160];
    int len = snprintf(suffix, sizeof(suffix), "_lay");
    for (int i = 0; i < GGML_METAL_TILED_PLAN_N; ++i) {
        len += snprintf(suffix + len, sizeof(suffix) - len, "_%d", v[i]);
    }
    const long on_plan = tiled_test_on_plan.load();
    fprintf(stderr, "TILED_TEST_PLAN weight=%s plan=%s suffix=%s on_plan=%ld other_route=%ld\n",
            hook.weight < 0 ? "any" : ggml_type_name((enum ggml_type) hook.weight), ggml_metal_tiled_compiled_id(hook.row), suffix, on_plan, tiled_test_other.load());
    {
        std::lock_guard<std::mutex> guard(tiled_test_lock);
        for (const struct tiled_test_shape & s : tiled_test_shapes) {
            fprintf(stderr, "TILED_TEST_SHAPE weight=%s k=%d m=%d n=%d route=%s nodes=%ld\n", ggml_type_name((enum ggml_type) s.type), s.k, s.m, s.n, s.route.c_str(), s.nodes);
        }
    }
    if (on_plan == 0) {
        fprintf(stderr, "GGML_METAL_TEST_TILED_PLAN: no mul_mat node was dispatched on the plan in this process\n");
        fflush(stderr);
        abort();
    }
}

void ggml_metal_tiled_test_arm(void) {
    static std::once_flag once;
    tiled_test_limit();
    if (tiled_test().row >= 0) {
        std::call_once(once, [] { atexit(tiled_test_at_exit); });
    }
}

// The pipeline of entry <entry>_lay_<values>, found or built; the name is written to name.
static ggml_metal_pipeline_with_params tiled_compile(ggml_metal_library_t lib, const char * entry, const struct ggml_metal_tiled_plan * lay, char * name, size_t name_size) {
    int32_t v[GGML_METAL_TILED_PLAN_N];
    ggml_metal_tiled_plan_values(lay, v);
    int len = snprintf(name, name_size, "%s_lay", entry);
    for (int i = 0; i < GGML_METAL_TILED_PLAN_N; ++i) {
        len += snprintf(name + len, name_size - len, "_%d", v[i]);
    }
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (res.pipeline) {
        return res;
    }
    return ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
}

int dispatch_mul_mat_matmul(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const struct ggml_device_profile * profile = ggml_metal_device_get_profile(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // =========================================================================
    // simdgroup_mm path — Apple Silicon (M1+/A14+) with hardware matrix support.
    // =========================================================================
    // for now the matrix-matrix multiplication kernel only works on A14+/M1+ SoCs
    // AMD GPU and older A-chips will reuse matrix-vector multiplication kernel
    if (ggml_metal_mm_route_takes(profile->has_matrix_hw, op)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.tg_x_offset =*/ 0,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + 31)/32), ((ne01 + 63)/64), ne12*ne13, 128, 1, 1);
        tiled_test_took(ctx, idx, 1, "mm", false);
        return 1;  // dispatched
    }

    // =========================================================================
    // Tiled matmul path — for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD).
    // =========================================================================
    if (ggml_metal_batch_route(ctx, idx) == MV_BATCH_TILED) {

        // Tiled matmul for non-simdgroup_mm GPUs (AMD Radeon, Intel UHD). The layout comes from the card's hardware
        // object through one rule (matmul/tiled-plan.cpp); the kernel is compiled per plan (kernel-tiled.metal).
        //
        // INTEL: f16 tiled is MANDATORY — scalar mul_mv fallback causes GPU watchdog timeout (BUG-011).
        // Do NOT remove GGML_METAL_VERIFIED_INTEL for f16. The f16 tiled kernel is faster than
        // scalar and stays within the ~5s Metal watchdog budget. Scalar does not.
        // mxfp4 tiled times out on Intel — stays AMD-only, falls back to scalar.
        const struct ggml_metal_card * card = ggml_metal_device_get_card(ctx->dev);
        const enum ggml_type wtype = op->src[0]->type;
        const struct tiled_test_hook & hook = tiled_test();
        const int32_t forced = tiled_test_applies(hook, wtype) ? hook.row : -1;
        // The plan asked for: the hook's, else the card's selection for this weight type.
        const int32_t wanted = forced >= 0 ? forced : ggml_metal_tiled_plan_for(card, (int32_t) wtype, 0).row;

        // Kernel name built from type suffix — matches template instantiations in kernel-tiled.metal
        char kernel_name[64];
        snprintf(kernel_name, sizeof(kernel_name), "kernel_mul_mat_tiled_%s", tiled_type_suffix(wtype));

        struct ggml_metal_tiled_plan lay = {};
        ggml_metal_pipeline_with_params pipeline = {};
        struct ggml_metal_act_plan act = {};
        char entry[192] = ""; // the entry of the last attempt
        char why[384]   = ""; // the first thing that failed, kept through the retry
        bool ran = false;
        // Attempt 0: the plan asked for. Attempt 1, only when that plan is not the default and could not run and the
        // hook does not apply: the default plan.
        for (int attempt = 0; attempt < 2 && !ran; ++attempt) {
            if (attempt > 0 && forced >= 0) {
                break;
            }
            lay = ggml_metal_tiled_plan_to_run(card, (int32_t) wtype, 0, attempt == 0 ? forced : 0);
            if (lay.from_default || !lay.ok) {
                if (why[0] == '\0') {
                    snprintf(why, sizeof(why), "plan %s needs a wave width or group memory this card does not have (wave %u, group memory %u B)",
                            ggml_metal_tiled_compiled_id(wanted), card->wave, card->group_mem);
                }
                if (!lay.ok || forced >= 0) {
                    break;
                }
            }

            // Activations read from a copy at the model's precision when this card's row says so. A missing entry for
            // the weight type leaves the op on the f32 kernel.
            pipeline = {};
            act = ggml_metal_act_copy_plan(ctx, idx, GGML_METAL_LANE_FAMILY_MATMUL, GGML_METAL_LANE_ROUTE_MM_TILED, op->src[1], (ne01 + lay.bm - 1)/lay.bm);
            if (act.applies) {
                char act_name[64];
                snprintf(act_name, sizeof(act_name), "%s_a%s", kernel_name, ggml_type_name(act.type));
                pipeline = tiled_compile(lib, act_name, &lay, entry, sizeof(entry));
                if (!pipeline.pipeline) {
                    act.applies = 0;
                }
            }
            if (!pipeline.pipeline) {
                pipeline = tiled_compile(lib, kernel_name, &lay, entry, sizeof(entry));
            }
            if (!pipeline.pipeline) {
                if (why[0] == '\0') {
                    snprintf(why, sizeof(why), "the library gave no pipeline for entry %s", entry);
                }
                if (lay.row == 0) {
                    break;
                }
                continue;
            }
            ggml_metal_pipeline_set_min_family(pipeline.pipeline, GGML_METAL_FAMILY_COMMON1); // no simd intrinsics needed
            ggml_metal_pipeline_set_verified_vendors(pipeline.pipeline, ggml_metal_tiled_verified_vendors(wtype));

            // Vendor verification: per weight type, the same for every plan, so no retry.
            const uint8_t verified = ggml_metal_pipeline_get_verified_vendors(pipeline.pipeline);
            const uint8_t vendor_bit = ggml_metal_vendor_to_verified_bit(profile->vendor);
            if (!(verified & vendor_bit)) {
                if (why[0] == '\0') {
                    snprintf(why, sizeof(why), "entry %s is not verified on vendor %d", entry, (int) profile->vendor);
                }
                break;
            }

            // Second step of the rule, on every call (a found pipeline as much as a built one): the pipeline must
            // allow the plan's threads per group.
            int limit = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
            if (tiled_test_limit() > 0 && tiled_test_limit() < limit) {
                limit = tiled_test_limit();
            }
            if (lay.threads > limit) {
                if (why[0] == '\0') {
                    snprintf(why, sizeof(why), "entry %s allows %d threads per group, plan %s needs %d", entry, limit, ggml_metal_tiled_compiled_id(lay.row), lay.threads);
                }
                if (lay.row == 0) {
                    break;
                }
                continue;
            }
            ran = true;
        }

        if (forced >= 0 && (!ran || lay.row != forced || lay.from_default)) {
            GGML_ABORT("GGML_METAL_TEST_TILED_PLAN: plan %s did not run for a %s weight: %s", ggml_metal_tiled_compiled_id(forced), ggml_type_name(wtype),
                    why[0] != '\0' ? why : "no reason recorded");
        }
        if (why[0] != '\0' && tiled_tell_once(why)) {
            if (ran) {
                GGML_LOG_WARN("%s: mul_mat_tiled: %s; entry %s runs instead\n", __func__, why, entry);
            } else {
                GGML_LOG_WARN("%s: mul_mat_tiled: %s; the op leaves the tiled route\n", __func__, why);
            }
        }

        if (ran) {
        if (hook.row >= 0) {
            const std::string route = std::string("tiled:") + ggml_metal_tiled_compiled_id(lay.row) + (act.applies ? "+act" : "");
            tiled_test_took(ctx, idx, 1, route.c_str(), forced >= 0);
        }

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.tg_x_offset =*/ 0,
        };

        if (act.applies) {
            ggml_metal_act_copy_encode(ctx, idx, &act, op->src[1]);
            args.nb10 = act.nb[0];
            args.nb11 = act.nb[1];
            args.nb12 = act.nb[2];
            args.nb13 = act.nb[3];
        }

        const size_t smem = (size_t) lay.group_mem_bytes;

        const int32_t tiled_grid_x = ((ne01 + lay.bm - 1)/lay.bm);
        const int32_t tiled_grid_y = ((ne11 + lay.bn - 1)/lay.bn);
        const int32_t tiled_grid_z = ne12*ne13;
        const int32_t tiled_total_tg = tiled_grid_x * tiled_grid_y * tiled_grid_z;
        // Tiled TGs are heavy (64×64 output tile with full K reduction) — use 1/64 of FA limit
        const int32_t tiled_chunk = (int32_t)profile->max_threadgroups_per_dispatch / 64;

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, act.applies ? act.bid : ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        if (tiled_chunk > 0 && tiled_total_tg > tiled_chunk) {
            const int32_t chunk_x = tiled_chunk / (tiled_grid_y * tiled_grid_z);
            if (chunk_x > 0) {
                for (int32_t x_off = 0; x_off < tiled_grid_x; x_off += chunk_x) {
                    const int32_t x_count = ((x_off + chunk_x) < tiled_grid_x) ? chunk_x : (tiled_grid_x - x_off);

                    args.tg_x_offset = x_off;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, x_count, tiled_grid_y, tiled_grid_z, lay.threads, 1, 1);
                }
            } else {
                for (int32_t x_off = 0; x_off < tiled_grid_x; x_off++) {

                    args.tg_x_offset = x_off;
                    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
                    ggml_metal_encoder_dispatch_threadgroups(enc, 1, tiled_grid_y, tiled_grid_z, lay.threads, 1, 1);
                }
            }
        } else {

            ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, tiled_grid_x, tiled_grid_y, tiled_grid_z, lay.threads, 1, 1);
        }

            return 1;  // dispatched
        }
    }

    return 0;  // neither path handled it
}
