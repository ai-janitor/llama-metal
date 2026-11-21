// Host check of the tiled half of llama-metal-tiled-sweep (tools/metal-tiled-sweep/tiled-rows.h): the judging step
// over fixed times, and the row it leads to. No Metal, no build of ggml.
//
// usage: test_metal_tiled_sweep_rows TABLE CARD WAVE GROUP_BYTES WINNER row|flat
//   TABLE: a matmul.tsv; its [tiles] rows give the compiled plans and their values (the default plan first).
//   The card's own run is on the default plan. Fixed times, three rounds, three column counts:
//     64   WINNER is 20 % faster with an equal digest; one other plan fails in round 2  -> a row for 64 columns
//     128  the card's own run is not on the tiled kernel in round 2                    -> NOROW
//     512  every plan within 2 %; one plan is twice as fast but its digest differs       -> KEEP
//   and one count where the card's own plan is NOT the default: nothing forced 10.0 ms, its forced copy 9.4 ms, the
//   default plan 9.5 ms -> a row for the default plan (the forced copy of the card's plan decides nothing).
//   row:  prints the row for 64 columns on stdout, exit 0 when the three verdicts are these.
//   flat: the same with WINNER only 2 % faster at 64: the check must fail (exit 1), the control of this test.
#include "tiled-rows.h"

#include <cstdlib>
#include <fstream>
#include <map>

static std::vector<std::string> cells_of(const std::string & line) {
    std::vector<std::string> out;
    size_t a = 0;
    for (;;) {
        const size_t b = line.find('\t', a);
        out.push_back(line.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) return out;
        a = b + 1;
    }
}

static int failures = 0;
static void want(bool ok, const std::string & what) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", what.c_str()); ++failures; }
}

int main(int argc, char ** argv) {
    if (argc != 7) { fprintf(stderr, "usage: test_metal_tiled_sweep_rows TABLE CARD WAVE GROUP_BYTES WINNER row|flat\n"); return 2; }
    const std::string card = argv[2], wave = argv[3], group_bytes = argv[4], winner = argv[5], mode = argv[6];
    if (mode != "row" && mode != "flat") { fprintf(stderr, "the mode is row or flat\n"); return 2; }

    std::vector<std::string> plans;                    // the default plan first
    std::map<std::string, std::vector<int32_t>> values;
    std::string default_plan;
    std::ifstream in(argv[1]);
    if (!in) { fprintf(stderr, "cannot read %s\n", argv[1]); return 2; }
    bool on = false;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line[0] == '[') { on = line == "[tiles]"; continue; }
        if (!on || line.empty() || line[0] == '#') continue;
        const std::vector<std::string> c = cells_of(line);
        if (c.size() != 24 || c[0] == "card" || c[2] == "-") continue;
        if (c[4] == "default") default_plan = c[2]; // by its role, wherever the id was first seen
        if (values.count(c[2])) continue;
        std::vector<int32_t> v;
        for (int i = 9; i < 20; ++i) v.push_back(atoi(c[(size_t) i].c_str()));
        values[c[2]] = v;
        plans.push_back(c[2]);
    }
    const auto at_default = std::find(plans.begin(), plans.end(), default_plan);
    if (default_plan.empty() || at_default == plans.end()) { fprintf(stderr, "FAIL the table has no row with the role default\n"); return 1; }
    std::rotate(plans.begin(), at_default, at_default + 1);
    if (plans.size() < 4 || !values.count(winner) || plans[0] == winner) {
        fprintf(stderr, "FAIL the table gives %zu plans; the check needs four with %s among them and not the default\n", plans.size(), winner.c_str());
        return 1;
    }
    // two more plans, neither the default nor the winner
    std::string fails_at_64, differs_at_512;
    for (size_t p = 1; p < plans.size(); ++p) {
        if (plans[p] == winner) continue;
        if (fails_at_64.empty()) fails_at_64 = plans[p]; else if (differs_at_512.empty()) differs_at_512 = plans[p];
    }

    const double own_ms[3] = { 10.0, 10.2, 9.9 };
    const auto rounds_at = [&](int64_t n) {
        std::vector<tiled_round> rounds;
        for (int r = 0; r < 3; ++r) {
            tiled_round one;
            one.own.ok     = true;
            one.own.plan   = plans[0];
            one.own.ms     = own_ms[r] * (double) n / 64.0;
            one.own.digest = "d" + std::to_string(n);
            if (n == 128 && r == 1) { one.own.plan = ""; one.own.why = "dispatched kernel_mul_mm_control"; }
            for (const std::string & plan : plans) {
                tiled_arm a;
                a.ok     = true;
                a.plan   = plan;
                a.digest = one.own.digest;
                double f = plan == plans[0] ? 1.01 : 1.02;
                if (n == 64 && plan == winner)          f = mode == "flat" ? 0.98 : 0.80;
                if (n == 512 && plan == winner)         f = 0.98;
                if (n == 512 && plan == differs_at_512) { f = 0.50; a.digest = "other"; }
                a.ms = one.own.ms * f;
                if (n == 64 && plan == fails_at_64 && r == 1) { a = tiled_arm(); a.why = "compute"; }
                one.forced.push_back(a);
            }
            rounds.push_back(one);
        }
        return rounds;
    };

    const tiled_verdict v64 = tiled_judge_count(plans, rounds_at(64), 0.03);
    want(v64.kind == tiled_verdict::ROW && v64.plan == winner && v64.own_plan == plans[0], "64 columns: wanted a row for " + winner + ", got " +
         (v64.kind == tiled_verdict::ROW ? "a row for " + v64.plan : v64.kind == tiled_verdict::KEEP ? "KEEP (fastest " + v64.plan + ")" : "NOROW: " + v64.reason));
    want(v64.not_judged == 1, "64 columns: wanted one plan not judged (" + fails_at_64 + "), got " + std::to_string(v64.not_judged));
    want(v64.lines.size() == plans.size(), "64 columns: wanted one line per plan");

    const tiled_verdict v128 = tiled_judge_count(plans, rounds_at(128), 0.03);
    want(v128.kind == tiled_verdict::NOROW && v128.reason.find("not on the tiled kernel") != std::string::npos && v128.reason.find("round 2") != std::string::npos,
         "128 columns: wanted NOROW for round 2 not on the tiled kernel, got kind " + std::to_string((int) v128.kind) + " " + v128.reason);

    const tiled_verdict v512 = tiled_judge_count(plans, rounds_at(512), 0.03);
    want(v512.kind == tiled_verdict::KEEP && v512.own_plan == plans[0] && v512.plan == winner, "512 columns: wanted KEEP with " + winner + " the fastest other plan, got kind " +
         std::to_string((int) v512.kind) + " plan " + v512.plan + " " + v512.reason);
    want(v512.not_judged == 1, "512 columns: wanted one plan not judged (" + differs_at_512 + ", its digest differs), got " + std::to_string(v512.not_judged));
    // the plan that failed at 64 is judged at 512: a failure ends one plan at one count
    bool judged = false;
    for (const std::string & line : v512.lines) if (line.rfind(fails_at_64 + " ", 0) == 0 && line.find("not judged") == std::string::npos) judged = true;
    want(judged, "512 columns: " + fails_at_64 + ", which failed at 64, must be judged here");

    // the card's own plan is not the default; its forced copy reads faster than the default plan, which clears the bar
    std::vector<tiled_round> veto;
    for (int r = 0; r < 3; ++r) {
        tiled_round one;
        one.own.ok = true; one.own.plan = winner; one.own.ms = 10.0; one.own.digest = "d";
        for (const std::string & plan : plans) {
            tiled_arm a;
            a.ok = true; a.plan = plan; a.digest = "d";
            a.ms = plan == winner ? 9.4 : plan == plans[0] ? 9.5 : 10.2;
            one.forced.push_back(a);
        }
        veto.push_back(one);
    }
    const tiled_verdict vv = tiled_judge_count(plans, veto, 0.03);
    want(vv.kind == tiled_verdict::ROW && vv.plan == plans[0] && vv.own_plan == winner, "incumbent " + winner + ": wanted a row for the default plan " + plans[0] +
         " (5 % under the unforced bar), got kind " + std::to_string((int) vv.kind) + " plan " + vv.plan);

    // the exit code: a sweep in which every allocation failed measured nothing (1), whatever NOROW lines it has; a
    // sweep measured on another kernel at every count finished without an eligible count (3); the three counts above (3)
    int exits[3] = { -1, -1, -1 };
    for (int kind = 0; kind < 2; ++kind) {
        int measured = 0, no_verdict = 0, not_judged = 0;
        for (const int64_t n : { (int64_t) 64, (int64_t) 128, (int64_t) 512 }) {
            std::vector<tiled_round> rounds;
            for (int r = 0; r < 3; ++r) {
                tiled_round one;
                one.own.why = "graph allocation";
                if (kind == 1) { one.own.ok = true; one.own.ms = 5.0; one.own.digest = "d" + std::to_string(n); one.own.why = "dispatched kernel_mul_mm_control"; }
                for (size_t p = 0; p < plans.size(); ++p) { tiled_arm a; a.why = "graph allocation"; one.forced.push_back(a); }
                rounds.push_back(one);
            }
            const tiled_verdict v = tiled_judge_count(plans, rounds, 0.03);
            want(v.kind == tiled_verdict::NOROW, "exit case " + std::to_string(kind) + ": wanted NOROW at " + std::to_string(n));
            if (v.kind == tiled_verdict::NOROW) ++no_verdict;
            not_judged += v.not_judged;
            measured   += tiled_measured(rounds);
        }
        exits[kind] = tiled_sweep_exit(measured, no_verdict, not_judged);
    }
    exits[2] = tiled_sweep_exit(tiled_measured(rounds_at(64)) + tiled_measured(rounds_at(128)) + tiled_measured(rounds_at(512)), 1, v64.not_judged + v512.not_judged);
    want(exits[0] == 1, "every allocation fails: wanted exit 1 (nothing measured), got " + std::to_string(exits[0]));
    want(exits[1] == 3, "every count measured on another kernel: wanted exit 3, got " + std::to_string(exits[1]));
    want(exits[2] == 3, "the three counts above: wanted exit 3, got " + std::to_string(exits[2]));
    want(tiled_sweep_exit(5, 0, 0) == 0, "measured, every count judged: wanted exit 0");
    fprintf(stderr, "tiled sweep exit: every allocation fails -> %d · measured on another kernel -> %d · row, NOROW and KEEP -> %d\n", exits[0], exits[1], exits[2]);

    fprintf(stderr, "tiled sweep rows (%s): %zu plans, default %s · 64 -> %s · 128 -> %s · 512 -> %s · non-default incumbent -> %s · %d failures\n", mode.c_str(), plans.size(), plans[0].c_str(),
            v64.kind == tiled_verdict::ROW ? "row" : v64.kind == tiled_verdict::KEEP ? "KEEP" : "NOROW", v128.kind == tiled_verdict::NOROW ? "NOROW" : "a verdict",
            v512.kind == tiled_verdict::KEEP ? "KEEP" : v512.kind == tiled_verdict::ROW ? "row" : "NOROW", vv.kind == tiled_verdict::ROW ? ("row for " + vv.plan).c_str() : "no row", failures);
    if (failures == 0) {
        printf("%s\n", tiled_row_text(card, "q4_K", v64.plan, 64, wave, group_bytes, values[v64.plan].data(), "control row of the check: fixed times", "none").c_str());
    }
    return failures == 0 ? 0 : 1;
}
