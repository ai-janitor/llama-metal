// Device rows and matvec rows of the Metal backend, host only: no GPU, no model.
// The values every shipped row must hold are written here, apart from the tables: the three cards' device rows, their
// matvec defaults, and the 38 per-shape rows of the D700 in their order (the first row that fits an op wins).
// usage: test_metal_device_rows
#include "ggml.h"
#include "ggml-device-profile.h"
#include "device-rows.h"
#include "matvec-rows.h"

#include <cstdio>
#include <cstring>
#include <string>

static int failures = 0;

#define D700 "AMD Radeon HD - FirePro D700"
#define M5500 "AMD Radeon Pro 5500M"
#define VEGA "AMD Radeon Pro Vega 64"

struct device_want {
    const char * card; const char * chip; const char * arch;
    int wave, group_bytes, cu, l1, l2; long long local_bw, transfer_bw;
    int attn_gate_fuse, conv_l2_fold, add_norm_fuse, cb_seq, set_nowait, staging_keep, probe_dispatch, probe_stage, probe_blit, probe_timestamp;
};
struct default_want {
    const char * card;
    int simd_sum_w64_ok, mv_reduce, f_scalar_w64, mv_group, q8_load_bytes, q8_rows8_min, q8_act_load_bytes, mv_group_red;
};
struct shape_want {
    int kind, type; long long ne00, ne01; int nr0, prep, cols, form;
};

int main() {
    static const device_want devices[] = {
        { D700,  "FirePro D700 (GCN 1.0, Tahiti)",      "gcn1",  64, 32768, 32, 16384,  786432, 263000000000LL, -1, 1, 1, 1, 4, 1, 1, 1, 0, 1, 1 },
        { M5500, "Radeon Pro 5500M (RDNA 1, Navi 14)",  "rdna1", 32, 65536, 24, 16384, 4194304, 192000000000LL, -1, 1, 0, 0, 4, 1, 1, 1, 0, 1, 1 },
        { VEGA,  "Radeon Pro Vega 64 (GCN 5, Vega 10)", "gcn5",  64, 65536, 64, 16384, 4194304, 484000000000LL, -1, 1, 1, 1, 4, 1, 1, 1, 0, 1, 1 },
    };
    static const default_want defaults[] = {
        { D700,   1,  1, 1, 1, 4, 16384, 16,  1 },
        { M5500, -1, -1, 0, 1, 1,    -1, -1, -1 },
        { VEGA,   1,  0, 1, 1, 4,    -1, -1, -1 },
    };
    // kind (0 single, 1 group), type, input columns, output rows, rows per simdgroup, prep, activation columns, form
    static const shape_want shapes[] = {
        { 0, GGML_TYPE_Q6_K, 2560, 248320, 8, 0, 1, 0 },
        { 0, GGML_TYPE_Q4_K, 2560, 4096, 8, 1, 1, 0 },
        { 0, GGML_TYPE_Q4_K, 2560, 4096, 8, 0, 1, 0 },
        { 0, GGML_TYPE_Q4_K, 9216, 2560, 4, 1, 1, 0 },
        { 0, GGML_TYPE_Q4_K, 4096, 2560, 2, 1, 1, 0 },
        { 1, GGML_TYPE_Q4_K, 2560, 9216, 8, 1, 1, 0 },
        { 0, GGML_TYPE_Q5_K, 2560, 8192, 8, 0, 1, 0 },
        { 0, GGML_TYPE_Q5_K, 4096, 2560, 8, 0, 1, 0 },
        { 0, GGML_TYPE_Q4_K, 2560, 9216, 4, 1, 2, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 8192, 4, 0, 2, 4 },
        { 0, GGML_TYPE_Q4_K, 4096, 2560, 2, 1, 2, 0 },
        { 0, GGML_TYPE_Q4_K, 9216, 2560, 2, 0, 2, 2 },
        { 0, GGML_TYPE_Q4_K, 2560, 4096, 4, 1, 2, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 1024, 2, 0, 2, 2 },
        { 0, GGML_TYPE_Q6_K, 2560, 8192, 4, 0, 2, 4 },
        { 0, GGML_TYPE_Q6_K, 9216, 2560, 2, 0, 2, 2 },
        { 0, GGML_TYPE_Q6_K, 2560, 1024, 2, 0, 2, 0 },
        { 0, GGML_TYPE_Q6_K, 2560, 248320, 4, 0, 2, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 9216, 4, 1, 3, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 4096, 4, 1, 3, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 1024, 2, 0, 3, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 8192, 4, 0, 3, 4 },
        { 0, GGML_TYPE_Q4_K, 4096, 2560, 4, 1, 3, 2 },
        { 0, GGML_TYPE_Q4_K, 9216, 2560, 4, 1, 3, 2 },
        { 0, GGML_TYPE_Q6_K, 2560, 8192, 2, 0, 3, 4 },
        { 0, GGML_TYPE_Q6_K, 9216, 2560, 2, 0, 3, 2 },
        { 0, GGML_TYPE_Q6_K, 2560, 1024, 2, 0, 3, 2 },
        { 0, GGML_TYPE_Q6_K, 2560, 248320, 4, 0, 3, 4 },
        { 0, GGML_TYPE_Q4_K, 2560, 9216, 2, 1, 4, 0 },
        { 0, GGML_TYPE_Q4_K, 2560, 8192, 1, 0, 4, 1 },
        { 0, GGML_TYPE_Q4_K, 4096, 2560, 2, 1, 4, 2 },
        { 0, GGML_TYPE_Q4_K, 9216, 2560, 2, 1, 4, 2 },
        { 0, GGML_TYPE_Q4_K, 2560, 4096, 2, 1, 4, 2 },
        { 0, GGML_TYPE_Q4_K, 2560, 1024, 1, 0, 4, 1 },
        { 0, GGML_TYPE_Q6_K, 2560, 8192, 2, 0, 4, 3 },
        { 0, GGML_TYPE_Q6_K, 9216, 2560, 2, 0, 4, 3 },
        { 0, GGML_TYPE_Q6_K, 2560, 1024, 1, 0, 4, 1 },
        { 0, GGML_TYPE_Q6_K, 2560, 248320, 1, 0, 4, 2 },
    };
    const int n_shapes = (int) (sizeof(shapes) / sizeof(shapes[0]));

    // 1. the device rows
    if (ggml_metal_device_rows_n != 3) {
        std::printf("FAIL the table has %d device rows, 3 are written here\n", ggml_metal_device_rows_n);
        failures++;
    }
    for (const auto & w : devices) {
        const struct ggml_metal_device_row * r = ggml_metal_device_row_of(w.card);
        if (r == nullptr) {
            std::printf("FAIL device row '%s': no such row\n", w.card);
            failures++;
            continue;
        }
        if (std::strcmp(r->chip, w.chip) != 0 || std::strcmp(r->arch_family, w.arch) != 0 || r->wave != w.wave || r->group_bytes != w.group_bytes ||
            r->compute_units != w.cu || r->l1_cache_per_cu != w.l1 || r->l2_cache_total != w.l2 || r->local_bw != w.local_bw || r->transfer_bw != w.transfer_bw ||
            r->attn_gate_fuse != w.attn_gate_fuse || r->conv_l2_fold != w.conv_l2_fold || r->add_norm_fuse != w.add_norm_fuse || r->cb_seq != w.cb_seq ||
            r->set_nowait != w.set_nowait || r->staging_keep != w.staging_keep || r->probe_dispatch != w.probe_dispatch || r->probe_stage != w.probe_stage ||
            r->probe_blit != w.probe_blit || r->probe_timestamp != w.probe_timestamp) {
            std::printf("FAIL device row '%s': a cell differs from the value written here\n", w.card);
            failures++;
        }
    }
    for (int i = 0; i < ggml_metal_device_rows_n; ++i) {
        bool known = false;
        for (const auto & w : devices) known = known || std::strcmp(ggml_metal_device_rows[i].card, w.card) == 0;
        if (!known) {
            std::printf("FAIL the device row of card '%s' is not a card of this test\n", ggml_metal_device_rows[i].card);
            failures++;
        }
    }

    // 2. the matvec defaults
    if (ggml_metal_matvec_default_rows_n != 3) {
        std::printf("FAIL the table has %d matvec default rows, 3 are written here\n", ggml_metal_matvec_default_rows_n);
        failures++;
    }
    for (const auto & w : defaults) {
        const struct ggml_metal_matvec_default_row * r = ggml_metal_matvec_default_row_of(w.card);
        if (r == nullptr) {
            std::printf("FAIL matvec defaults '%s': no such row\n", w.card);
            failures++;
            continue;
        }
        if (r->simd_sum_w64_ok != w.simd_sum_w64_ok || r->mv_reduce != w.mv_reduce || r->f_scalar_w64 != w.f_scalar_w64 || r->mv_group != w.mv_group ||
            r->q8_load_bytes != w.q8_load_bytes || r->q8_rows8_min != w.q8_rows8_min || r->q8_act_load_bytes != w.q8_act_load_bytes || r->mv_group_red != w.mv_group_red) {
            std::printf("FAIL matvec defaults '%s': a cell differs from the value written here\n", w.card);
            failures++;
        }
    }

    // 3. the per-shape rows: all are the D700's, in this order
    uint32_t n = 0;
    const struct ggml_device_mv_row * rows = ggml_metal_mv_rows_of(D700, &n);
    if ((int) n != n_shapes || ggml_metal_hw_mv_rows_n != n_shapes) {
        std::printf("FAIL shapes of '%s': %u rows of %d in the table, expected %d\n", D700, n, ggml_metal_hw_mv_rows_n, n_shapes);
        failures++;
    }
    for (int i = 0; i < (int) n && i < n_shapes; ++i) {
        const shape_want & w = shapes[i];
        const struct ggml_device_mv_row & r = rows[i];
        if (r.kind != w.kind || r.type != w.type || r.ne00 != w.ne00 || r.ne01 != w.ne01 || r.nr0 != w.nr0 || r.prep != w.prep || r.cols != w.cols || r.form != w.form) {
            std::printf("FAIL shape row %d: %d %d %lld x %lld rows %d prep %d cols %d form %d differs from the value written here\n", i + 1,
                    r.kind, r.type, (long long) r.ne00, (long long) r.ne01, r.nr0, r.prep, r.cols, r.form);
            failures++;
        }
    }
    for (const char * other : { M5500, VEGA }) {
        ggml_metal_mv_rows_of(other, &n);
        if (n != 0) {
            std::printf("FAIL shapes of '%s': %u rows, expected 0\n", other, n);
            failures++;
        }
    }

    // 4. a name that is not a card's whole name has no row: one character short, another case, a space, nothing, and
    // 63 bytes (what the device holds of a longer name)
    const std::string cut = std::string(D700) + " " + std::string(63 - std::strlen(D700) - 1, 'x');
    const char * strangers[] = { "AMD Radeon HD - FirePro D70", "amd radeon pro vega 64", "AMD Radeon Pro 5500M ", " AMD Radeon Pro 5500M", "Apple M1", "", cut.c_str(), nullptr };
    for (const char * name : strangers) {
        ggml_metal_mv_rows_of(name, &n);
        if (ggml_metal_device_row_of(name) != nullptr || ggml_metal_matvec_default_row_of(name) != nullptr || n != 0) {
            std::printf("FAIL card '%s' has a row\n", name ? name : "(null)");
            failures++;
        }
    }

    std::printf("device rows: %d devices, %d matvec defaults, %d shape rows, %d failures\n",
            ggml_metal_device_rows_n, ggml_metal_matvec_default_rows_n, ggml_metal_hw_mv_rows_n, failures);
    return failures == 0 ? 0 : 1;
}
