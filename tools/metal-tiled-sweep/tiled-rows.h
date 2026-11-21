// The tiled half of llama-metal-tiled-sweep as text and arithmetic, with no Metal in it, so a host check can hold it:
//   - one row of ggml/src/ggml-metal/data/matmul.tsv [tiles] for one column count, the 24 cells of the section's
//     header in its order, separated by one tab;
//   - the judging step for ONE column count over the times the tool measured there.
// tests/Makefile check-tiled-sweep-rows feeds the judging step fixed times and holds the row to the table's generator
// (matmul/gen-tiles.cmake).
//
// Judging one count. Each round holds the card's own run (nothing forced) and one forced run per compiled plan, the
// default plan included. The bar is the plan the card dispatched with nothing forced. A plan is a candidate when every
// round ran it on its own entry with the output digest of the card's own run. Its ratio is the median over the rounds
// of (its time / the card's own time in that round). The replacements are the candidates other than the card's own
// plan: the fastest of them gets a row when its ratio is at most 1 - gain; otherwise the card's plan stays (KEEP).
// The forced run of the card's own plan is printed and decides nothing: the bar is the run with nothing forced.
// Without a bar or without a replacement there is no verdict on the count (NOROW, with the reason). A failure ends
// one plan at this count only.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// values: the plan's eleven values in the table's column order (warp to pad). wave and group_bytes: the card's shape
// as it reported them. plan_test: the command that measured the row.
static inline std::string tiled_row_text(const std::string & card, const std::string & weight, const std::string & plan, int64_t cols, const std::string & wave,
                                         const std::string & group_bytes, const int32_t * values, const std::string & plan_source, const std::string & plan_test) {
    std::string row = card + "\t" + weight + "\t" + plan + "\t" + std::to_string(cols) + "\ttuned\t-\t-\t" + wave + "\t" + group_bytes;
    for (int i = 0; i < 11; ++i) row += "\t" + std::to_string(values[i]);
    return row + "\t-\t-\t" + plan_source + "\t" + plan_test;
}

// One run at one column count.
struct tiled_arm {
    bool        ok = false;  // the run gave a time and a digest at this count
    std::string why;         // when !ok
    std::string plan;        // the id of the plan whose entry was dispatched; "" = not on the tiled kernel (why says what ran)
    double      ms = 0.0;
    std::string digest;
};

struct tiled_round {
    tiled_arm              own;    // nothing forced
    std::vector<tiled_arm> forced; // one per compiled plan, in the plans' order
};

struct tiled_verdict {
    enum kind_t { ROW, KEEP, NOROW } kind = NOROW;
    std::string own_plan;                // the plan the card dispatched (ROW, KEEP)
    std::string plan;                    // ROW: the winner. KEEP: the fastest replacement
    double      own_ms = 0.0, plan_ms = 0.0, ratio = 1.0; // medians over the rounds; ratio as said above
    std::string reason;                  // NOROW
    std::vector<std::string> lines;      // one per plan: its time and ratio, or why it was not judged at this count
    int         not_judged = 0;          // plans (the forced copy of the card's own among them) that could not be judged at this count
};

static inline double tiled_median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}

static inline tiled_verdict tiled_judge_count(const std::vector<std::string> & plans, const std::vector<tiled_round> & rounds, double gain) {
    tiled_verdict v;
    if (rounds.empty()) { v.reason = "no round was run"; return v; }
    std::vector<double> own_ms;
    for (size_t r = 0; r < rounds.size(); ++r) {
        const tiled_arm & own = rounds[r].own;
        const std::string at = "round " + std::to_string(r + 1) + ": ";
        if (!own.ok)                              { v.reason = at + "the card's own run failed: " + own.why; return v; }
        if (own.plan.empty())                     { v.reason = at + "the card's own run is not on the tiled kernel at this count" + (own.why.empty() ? "" : " (" + own.why + ")"); return v; }
        if (own.plan != rounds[0].own.plan)       { v.reason = at + "the card's own run dispatched plan " + own.plan + ", in round 1 plan " + rounds[0].own.plan; return v; }
        if (own.digest != rounds[0].own.digest)   { v.reason = at + "the digest of the card's own run differs from round 1"; return v; }
        if (!(own.ms > 0.0))                      { v.reason = at + "the card's own run has no time"; return v; }
        own_ms.push_back(own.ms);
    }
    v.own_plan = rounds[0].own.plan;
    v.own_ms   = tiled_median(own_ms);

    int    best = -1, differ = 0;
    double best_ratio = 0.0, best_ms = 0.0;
    for (size_t p = 0; p < plans.size(); ++p) {
        std::string bad;
        bool digest_differs = false;
        std::vector<double> ms, ratios;
        for (size_t r = 0; r < rounds.size() && bad.empty(); ++r) {
            const std::string at = "round " + std::to_string(r + 1) + ": ";
            if (p >= rounds[r].forced.size()) { bad = at + "no forced run"; break; }
            const tiled_arm & a = rounds[r].forced[p];
            if (!a.ok)                              bad = at + (a.why.empty() ? "the forced run failed" : a.why);
            else if (a.plan != plans[p])            bad = at + "the forced run was not dispatched on this plan (" + (a.plan.empty() ? (a.why.empty() ? "not on the tiled kernel" : a.why) : "plan " + a.plan) + ")";
            else if (a.digest != rounds[r].own.digest) { bad = at + "digest DIFFERS from the card's own run"; digest_differs = true; }
            else if (!(a.ms > 0.0))                 bad = at + "no time";
            else { ms.push_back(a.ms); ratios.push_back(a.ms / rounds[r].own.ms); }
        }
        char line[512];
        if (!bad.empty()) {
            snprintf(line, sizeof(line), "%-18s not judged at this count: %s", plans[p].c_str(), bad.c_str());
            v.lines.push_back(line);
            ++v.not_judged;
            if (digest_differs) ++differ;
            continue;
        }
        const double ratio = tiled_median(ratios), m = tiled_median(ms);
        snprintf(line, sizeof(line), "%-18s %9.3f ms · %+6.1f %% time against the card's own run · digest equal%s", plans[p].c_str(), m, (ratio - 1.0) * 100.0,
                 plans[p] == v.own_plan ? " · the card's plan, forced (not ranked)" : "");
        v.lines.push_back(line);
        if (plans[p] == v.own_plan) continue;
        if (best < 0 || ratio < best_ratio) { best = (int) p; best_ratio = ratio; best_ms = m; }
    }
    if (best < 0) {
        v.reason = differ == (int) plans.size() ? "the digest of every plan differs from the card's own run" : "no other plan could be compared with the card's own run at this count";
        return v;
    }
    v.plan    = plans[(size_t) best];
    v.plan_ms = best_ms;
    v.ratio   = best_ratio;
    v.kind    = best_ratio <= 1.0 - gain ? tiled_verdict::ROW : tiled_verdict::KEEP;
    return v;
}

// The arms of one count that sent a time and a digest (on the tiled kernel or not).
static inline int tiled_measured(const std::vector<tiled_round> & rounds) {
    int n = 0;
    for (const tiled_round & r : rounds) {
        if (r.own.ok) ++n;
        for (const tiled_arm & a : r.forced) if (a.ok) ++n;
    }
    return n;
}

// The tool's exit code once the sweep ran: 1 = no run sent a time and a digest at any count (nothing was measured,
// whatever lines were printed), 3 = it measured and some count has no verdict or some plan was not judged at a count,
// 0 = every count has a row or a KEEP and every plan was judged.
static inline int tiled_sweep_exit(int measured, int no_verdict, int not_judged) {
    if (measured < 1) return 1;
    return (no_verdict > 0 || not_judged > 0) ? 3 : 0;
}
