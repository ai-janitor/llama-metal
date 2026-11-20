// Host-only checks of the convert-once table (ggml/src/ggml-metal/act-copy) and of the lane picked for a copy.
#include "act-copy/decide.h"
#include "lanes/select.h"

#include <cstdio>
#include <vector>

static int failures = 0;

static void expect(bool ok, const char * what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        failures++;
    }
}

int main() {
    int n = 0;
    const struct ggml_metal_act_copy_row * table = ggml_metal_act_copy_rows_get(&n);
    const int fam = GGML_METAL_LANE_FAMILY_MATMUL, rt = GGML_METAL_LANE_ROUTE_MM_TILED;
    int why = -1;

    // 1. the table as shipped: no op on any listed card gets a copy, whatever the model asks
    int shipped_on = 0;
    for (int i = 0; i < n; ++i) {
        for (enum ggml_type prec : { GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_COUNT }) {
            ggml_metal_act_copy_decide_in(table, n, table[i].card, table[i].family, table[i].route,
                    prec, GGML_TYPE_F32, 1000, 1024, &why);
            if (why == GGML_METAL_ACT_COPY_APPLIES) shipped_on++;
        }
    }
    std::printf("act-copy table: %d rows, %d of them on\n", n, shipped_on);
    // the shipped rows, written here apart from the table: one bf16 row of the tiled matmul per card, by the name the card reports
    for (const char * card : { "AMD Radeon Pro 5500M", "AMD Radeon Pro Vega 64", "AMD Radeon HD - FirePro D700" }) {
        const struct ggml_metal_act_copy_row * shipped = ggml_metal_act_copy_decide_in(table, n, card, fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 1000, 1024, &why);
        expect(shipped != nullptr && why == GGML_METAL_ACT_COPY_ROW_OFF, "shipped table: the card has its bf16 row of the tiled matmul, and it is off");
    }
    expect(n == 3, "shipped table: three rows");

    // 2. a table with one row on
    std::vector<ggml_metal_act_copy_row> rows = {
        { fam, rt, "AMD Radeon Pro 5500M", GGML_TYPE_BF16, 1, 4, 1000000, "control", "control" },
        { fam, rt, "AMD Radeon Pro Vega 64", GGML_TYPE_BF16, 0, 4, 1000000, "control", "control" },
    };
    const int nr = (int) rows.size();
    const struct ggml_metal_act_copy_row * row;

    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == &rows[0] && why == GGML_METAL_ACT_COPY_APPLIES, "row on, bf16 model, 16 re-reads: applies");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_COUNT, GGML_TYPE_F32, 16, 4096, &why);
    expect(why == GGML_METAL_ACT_COPY_NO_PRECISION, "precision not known: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_F32, GGML_TYPE_F32, 16, 4096, &why);
    expect(why == GGML_METAL_ACT_COPY_NO_PRECISION, "f32 model: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_F16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == nullptr && why == GGML_METAL_ACT_COPY_NO_ROW, "f16 model, only a bf16 row: no copy (never another precision)");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F16, 16, 4096, &why);
    expect(why == GGML_METAL_ACT_COPY_NOT_F32, "activation tensor not f32: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "Apple M1", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == nullptr && why == GGML_METAL_ACT_COPY_NO_ROW, "card with no row (Apple Silicon): no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == nullptr && why == GGML_METAL_ACT_COPY_NO_ROW, "card name one character off: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, nullptr, fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == nullptr && why == GGML_METAL_ACT_COPY_NO_ROW, "no card name: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro Vega 64", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == &rows[1] && why == GGML_METAL_ACT_COPY_ROW_OFF, "row off: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 3, 4096, &why);
    expect(why == GGML_METAL_ACT_COPY_FEW_REREADS, "3 re-reads: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, rt, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 2000000, &why);
    expect(why == GGML_METAL_ACT_COPY_TOO_BIG, "copy over the row's size: no copy");
    row = ggml_metal_act_copy_decide_in(rows.data(), nr, "AMD Radeon Pro 5500M", fam, GGML_METAL_LANE_ROUTE_MV_SCALAR, GGML_TYPE_BF16, GGML_TYPE_F32, 16, 4096, &why);
    expect(row == nullptr && why == GGML_METAL_ACT_COPY_NO_ROW, "another route: no copy");

    // 3. the lane for a copy: with the shipped lane table a bf16 or f16 copy lands on f32 multiply-add (wider, exact)
    int nl = 0;
    const struct ggml_metal_lane_row * lanes = ggml_metal_lane_rows_get(&nl);
    for (enum ggml_type copy : { GGML_TYPE_BF16, GGML_TYPE_F16 }) {
        int step = 0;
        const struct ggml_metal_lane_row * lane = ggml_metal_negotiate_lane_in(lanes, nl, ~0u, fam, rt, GGML_TYPE_Q4_K, copy, &step);
        expect(lane != nullptr && lane->operand == GGML_TYPE_F32 && step == GGML_METAL_LANE_HOLDS_WIDER, "shipped lanes: a copy computes on f32 multiply-add");
    }
    // an fp16 lane that holds f16 natively is taken for an f16 copy and never for a bf16 copy
    std::vector<ggml_metal_lane_row> forced(lanes, lanes + nl);
    forced.insert(forced.begin(), { "control-f16", GGML_METAL_LANE_SOURCE_METAL, "", fam, rt, GGML_TYPE_Q4_K, GGML_TYPE_COUNT, GGML_TYPE_F16,
            GGML_METAL_LANE_HOLDS_NATIVE, "_v3", -1, GGML_METAL_LANE_CLASS_EXACT, GGML_METAL_LANE_ORIGIN_OURS, 1, 0, 1, "control", "control" });
    {
        int step = 0;
        const struct ggml_metal_lane_row * lane = ggml_metal_negotiate_lane_in(forced.data(), (int) forced.size(), ~0u, fam, rt, GGML_TYPE_Q4_K, GGML_TYPE_F16, &step);
        expect(lane == &forced[0] && step == GGML_METAL_LANE_HOLDS_NATIVE, "an fp16 lane takes an f16 copy");
        lane = ggml_metal_negotiate_lane_in(forced.data(), (int) forced.size(), ~0u, fam, rt, GGML_TYPE_Q4_K, GGML_TYPE_BF16, &step);
        expect(lane != &forced[0] && lane != nullptr && lane->operand == GGML_TYPE_F32, "an fp16 lane never takes a bf16 copy");
    }

    std::printf("MAIN act-copy: %d rows, %d on as shipped, %d failures\n", n, shipped_on, failures + (shipped_on != 0 ? 1 : 0));
    return failures == 0 && shipped_on == 0 ? 0 : 1;
}
