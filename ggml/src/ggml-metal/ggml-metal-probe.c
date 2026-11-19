#include "ggml-metal-probe.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static void ggml_metal_probe_dump_names(FILE * file);

#define ZONE_RECORDER_IMPLEMENTATION
#define ZR_DUMP_EXTRA ggml_metal_probe_dump_names
#include "zone_recorder.h"

enum { PROBE_LABELS = 2048, PROBE_LABEL_SIZE = 128 };
static const uint32_t probe_end = UINT32_C(1) << 31;

static _Thread_local uint16_t current_op;
static _Thread_local uint8_t current_family;
static pthread_mutex_t labels_lock = PTHREAD_MUTEX_INITIALIZER;

static struct {
    uint64_t device;
    char name[PROBE_LABEL_SIZE];
} labels[PROBE_LABELS];
static uint32_t label_count;

static uint32_t event_id(uint8_t family, uint8_t zone, uint16_t op) {
    return (uint32_t) family << 24 | (uint32_t) zone << 16 | op;
}

static uint32_t label_id(uint64_t device, const char * name) {
    if (!name || !name[0] || strlen(name) >= PROBE_LABEL_SIZE) {
        return 0;
    }

    pthread_mutex_lock(&labels_lock);
    uint32_t i = 0;
    for (; i < label_count; ++i) {
        if (labels[i].device == device && strcmp(labels[i].name, name) == 0) {
            break;
        }
    }
    uint32_t result = 0;
    if (i < PROBE_LABELS) {
        if (i == label_count) {
            labels[i].device = device;
            strcpy(labels[i].name, name);
            ++label_count;
        }
        result = i + 1;
    }
    pthread_mutex_unlock(&labels_lock);
    return result;
}

static void ggml_metal_probe_dump_names(FILE * file) {
    #define PROBE_NAME(name) #name,
    static const char * families[] = { GGML_METAL_PROBE_FAMILIES(PROBE_NAME) };
    static const char * zones[] = { GGML_METAL_PROBE_ZONES(PROBE_NAME) };
    #undef PROBE_NAME
    fprintf(file, "# ggml_metal_probe v1\n");
    for (unsigned i = 0; i < sizeof(families)/sizeof(*families); ++i) {
        fprintf(file, "F\t%u\t%s\n", i, families[i]);
    }
    for (unsigned i = 0; i < sizeof(zones)/sizeof(*zones); ++i) {
        fprintf(file, "Z\t%u\t%s\n", i, zones[i]);
    }
    for (unsigned i = 0; i < GGML_OP_COUNT; ++i) {
        fprintf(file, "O\t%u\t%s\n", i, ggml_op_name((enum ggml_op) i));
    }
    pthread_mutex_lock(&labels_lock);
    for (uint32_t i = 0; i < label_count; ++i) {
        fprintf(file, "K\t%u\t%llx\t%s\n", i + 1,
                (unsigned long long) labels[i].device, labels[i].name);
    }
    pthread_mutex_unlock(&labels_lock);
}

bool ggml_metal_probe_enabled(void) {
    return zr_enabled();
}

uint16_t ggml_metal_probe_op(void) {
    return current_op;
}

uint8_t ggml_metal_probe_family(void) {
    return current_family;
}

struct ggml_metal_probe_scope ggml_metal_probe_begin(uint8_t family, uint8_t zone, uint16_t op) {
    struct ggml_metal_probe_scope scope = { 0 };
    if (!ggml_metal_probe_enabled()) {
        return scope;
    }
    scope.previous_op = current_op;
    scope.previous_family = current_family;
    current_family = family;
    current_op = op == GGML_OP_NONE ? current_op : op;
    scope.id = event_id(family, zone, current_op);
    scope.start = zr_now_ns();
    zr_mark_at(scope.id, scope.start, 0);
    return scope;
}

void ggml_metal_probe_end(struct ggml_metal_probe_scope * scope) {
    if (scope->start) {
        zr_mark_at(scope->id | probe_end, zr_now_ns(), 0);
        current_op = scope->previous_op;
        current_family = scope->previous_family;
        scope->start = 0;
    }
}

void ggml_metal_probe_gpu(uint8_t family, uint8_t zone, uint16_t op, uint64_t device,
        const char * kernel, uint64_t start, uint64_t end) {
    if (!start || end <= start) {
        ggml_metal_probe_count(GGML_METAL_PROBE_ZONE_gpu_invalid, 1);
        return;
    }
    const uint32_t id = event_id(family, zone, op);
    const uint32_t label = label_id(device, kernel);
    zr_mark_at(id, start, label);
    zr_mark_at(id | probe_end, end, label);
}

void ggml_metal_probe_count(uint8_t zone, uint64_t count) {
    if (count) {
        zr_mark_v(event_id(GGML_METAL_PROBE_FAMILY_device, zone, GGML_OP_NONE), count);
    }
}
