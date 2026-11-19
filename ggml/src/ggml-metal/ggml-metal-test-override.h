#pragma once

// Private candidate-loader helpers (P12.18, adapted from the retained P10.51
// component, archive commit 8fd12e8b ggml-metal-test-override.h).
// Portable selection / cache-key / hash logic used by ggml-metal-device.m.
// No Metal types here; this header never opens Metal and never runs the ABI probe.
//
// Selector inputs (all four absent = baseline, nothing here runs):
//   GGML_METAL_TEST_SLOT          registry slot name
//   GGML_METAL_TEST_VARIANT       decimal EXTERNAL variant id (never a function-constant / lane value)
//   GGML_METAL_TEST_METALLIB      absolute path of the candidate metallib
//   GGML_METAL_TEST_METALLIB_SHA  full lowercase 64-hex SHA-256 of that file's bytes
//
// Which (slot, variant) pairs exist, and which kernel requests they apply to, is DATA in
// ggml_metal_test_override_rows below. The generic code never names a slot or a kernel.

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

// Interface version returned by ggml_backend_metal_test_candidate_loader_version().
#define GGML_METAL_TEST_CANDIDATE_LOADER_VERSION 1

// One supported (slot, external variant) selection.
struct ggml_metal_test_override_row {
    const char * slot;                 // GGML_METAL_TEST_SLOT value
    int          variant;              // GGML_METAL_TEST_VARIANT value (external id)
    const char * base;                 // exact base kernel function name the candidate replaces
    int          lane_contract_version; // value the candidate's lane_contract_version_probe must return
    const char * candidate_entry;      // P11.65: function the candidate metallib is specialized from (== base when it keeps the name)
};

// Registry. Add a row only after its candidate family has a verified record.
static const struct ggml_metal_test_override_row ggml_metal_test_override_rows[] = {
    { "q4_k_grouped", 3, "kernel_mul_mv_grouped_kquant_f32", 3, "kernel_mul_mv_grouped_kquant_f32" },
};

enum ggml_metal_test_override_status {
    GGML_METAL_TEST_OVERRIDE_NOT_SELECTED = 0,
    GGML_METAL_TEST_OVERRIDE_SELECTED     = 1,
    GGML_METAL_TEST_OVERRIDE_ERROR        = 2,
};

struct ggml_metal_test_override {
    enum ggml_metal_test_override_status status;
    const struct ggml_metal_test_override_row * row; // valid when SELECTED
    char sha[65];       // valid only when status == SELECTED
    const char * path;  // borrowed env value, valid only when SELECTED
    char error[192];    // valid only when status == ERROR
};

// Full 64-char lowercase hex SHA-256 only.
static inline bool ggml_metal_test_override_sha_valid(const char * sha) {
    if (!sha || strlen(sha) != 64) {
        return false;
    }
    for (int i = 0; i < 64; ++i) {
        const char c = sha[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            return false;
        }
    }
    return true;
}

// Complete decimal variant id: 1..9 digits, no sign, no spaces. Returns false otherwise.
static inline bool ggml_metal_test_override_parse_variant(const char * s, int * out) {
    if (!s || s[0] == '\0') {
        return false;
    }
    long v = 0;
    size_t n = 0;
    for (; s[n]; ++n) {
        if (s[n] < '0' || s[n] > '9' || n >= 9) {
            return false;
        }
        v = v * 10 + (s[n] - '0');
    }
    *out = (int) v;
    return true;
}

// Does this pipeline request belong to the row's kernel? base may be NULL (get_pipeline path),
// in which case name carries the specialization: the base followed by '_' or end-of-string.
static inline bool ggml_metal_test_override_row_matches(
        const struct ggml_metal_test_override_row * row, const char * base, const char * name) {
    if (base) {
        return strcmp(base, row->base) == 0;
    }
    if (!name) {
        return false;
    }
    const size_t n = strlen(row->base);
    return strncmp(name, row->base, n) == 0 && (name[n] == '\0' || name[n] == '_');
}

static inline struct ggml_metal_test_override ggml_metal_test_override_fail_result(const char * detail) {
    struct ggml_metal_test_override res;
    memset(&res, 0, sizeof(res));
    res.status = GGML_METAL_TEST_OVERRIDE_ERROR;
    snprintf(res.error, sizeof(res.error), "%s", detail);
    return res;
}

// Resolve the env selection for one pipeline request (base may be NULL, see above).
//   all four absent                      -> NOT_SELECTED (baseline, original path)
//   registered slot, other kernel        -> NOT_SELECTED (baseline)
//   anything partial/unknown/invalid     -> ERROR (never silently baseline on a matching route)
// An invalid selection whose slot is not in the registry cannot be classified per kernel, so it
// is an ERROR for every request: the caller records the sticky error and does not proceed as
// if a candidate were selected.
static inline struct ggml_metal_test_override ggml_metal_test_override_resolve(
        const char * slot, const char * variant, const char * path, const char * sha,
        const char * base, const char * name) {
    struct ggml_metal_test_override res;
    memset(&res, 0, sizeof(res));
    res.status = GGML_METAL_TEST_OVERRIDE_NOT_SELECTED;

    if (!slot && !variant && !path && !sha) {
        return res;
    }
    if (!slot) {
        return ggml_metal_test_override_fail_result("unknown-selection: GGML_METAL_TEST_SLOT missing");
    }

    // slot must name at least one registry row
    bool slot_known = false;
    const size_t n_rows = sizeof(ggml_metal_test_override_rows) / sizeof(ggml_metal_test_override_rows[0]);
    for (size_t i = 0; i < n_rows; ++i) {
        if (strcmp(ggml_metal_test_override_rows[i].slot, slot) == 0) {
            slot_known = true;
            break;
        }
    }
    if (!slot_known) {
        return ggml_metal_test_override_fail_result("unknown-selection: GGML_METAL_TEST_SLOT is not a registered slot");
    }

    int variant_id = 0;
    const bool variant_ok = ggml_metal_test_override_parse_variant(variant, &variant_id);

    // Find the registered row for (slot, variant); a known slot with an unknown/invalid variant
    // is matched by the slot's rows so non-matching kernels stay baseline.
    const struct ggml_metal_test_override_row * row = NULL;
    for (size_t i = 0; i < n_rows; ++i) {
        const struct ggml_metal_test_override_row * r = &ggml_metal_test_override_rows[i];
        if (strcmp(r->slot, slot) == 0 && variant_ok && r->variant == variant_id) {
            row = r;
            break;
        }
    }
    if (!row) {
        // Is this request on a kernel of this slot at all? Use the slot's first row for matching.
        for (size_t i = 0; i < n_rows; ++i) {
            const struct ggml_metal_test_override_row * r = &ggml_metal_test_override_rows[i];
            if (strcmp(r->slot, slot) == 0) {
                if (!ggml_metal_test_override_row_matches(r, base, name)) {
                    return res; // other pipelines always original
                }
                break;
            }
        }
        return ggml_metal_test_override_fail_result(variant_ok
            ? "unknown-selection: GGML_METAL_TEST_VARIANT is not registered for this slot"
            : "malformed GGML_METAL_TEST_VARIANT: need complete decimal id");
    }
    if (!ggml_metal_test_override_row_matches(row, base, name)) {
        return res; // other pipelines always original
    }
    if (!path || path[0] != '/' || path[1] == '\0') {
        return ggml_metal_test_override_fail_result("malformed GGML_METAL_TEST_METALLIB: need nonempty absolute path");
    }
    if (!ggml_metal_test_override_sha_valid(sha)) {
        return ggml_metal_test_override_fail_result("malformed GGML_METAL_TEST_METALLIB_SHA: need full 64 lowercase hex");
    }
    res.status = GGML_METAL_TEST_OVERRIDE_SELECTED;
    res.row    = row;
    memcpy(res.sha, sha, 65);
    res.path   = path;
    return res;
}

// Owned cache key: original specialization name plus full SHA, slot and external variant.
// The original name already carries the route's NR0/lane. Caller frees. NULL on allocation
// failure; never truncated.
static inline char * ggml_metal_test_override_cache_key(
        const char * name, const char * sha, const char * slot, int variant) {
    if (!name || !sha || !slot) {
        return NULL;
    }
    const char * fmt = "%s|test-override-sha256=%s|slot=%s|variant=%d";
    const int len = snprintf(NULL, 0, fmt, name, sha, slot, variant);
    if (len < 0) {
        return NULL;
    }
    char * key = (char *) malloc((size_t) len + 1);
    if (!key) {
        return NULL;
    }
    snprintf(key, (size_t) len + 1, fmt, name, sha, slot, variant);
    return key;
}

// SHA-256 (hex, lowercase) of the actual bytes via CommonCrypto, fed in chunks so the
// 32-bit CC_LONG length can never truncate a large input.
static inline bool ggml_metal_test_override_sha256_hex(
        const unsigned char * bytes, size_t len, char out[65]) {
    if (!bytes || !out) {
        return false;
    }
#if defined(__APPLE__)
    CC_SHA256_CTX ctx;
    if (!CC_SHA256_Init(&ctx)) {
        return false;
    }
    const size_t chunk = (size_t) 1 << 30;
    for (size_t off = 0; off < len; off += chunk) {
        const size_t n = (len - off < chunk) ? (len - off) : chunk;
        if (!CC_SHA256_Update(&ctx, bytes + off, (CC_LONG) n)) {
            return false;
        }
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    if (!CC_SHA256_Final(digest, &ctx)) {
        return false;
    }
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
        out[2 * i]     = hex[digest[i] >> 4];
        out[2 * i + 1] = hex[digest[i] & 0xf];
    }
    out[64] = '\0';
    return true;
#else
    (void) len;
    out[0] = '\0';
    return false;
#endif
}
