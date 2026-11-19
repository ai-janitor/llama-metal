// zone_recorder.h — a flight recorder for timing a hot path without disturbing it.
//
// INTENT: record points of interest of a run into a preallocated, pre-touched RAM buffer (one clock read and one store per
//         mark), and write them out ONCE when the process exits. Nothing is printed, formatted, locked or written to a file
//         while the run is in progress. Single header, C99 and C++11, no dependencies; POSIX, macOS and Windows clocks.
//
// USE (on a probe branch of the project, never on its main line):
//   1. Copy this file into the project. In exactly ONE source file:   #define ZONE_RECORDER_IMPLEMENTATION
//                                                                     #include "zone_recorder.h"
//      Every other file that plants a mark:                           #include "zone_recorder.h"
//   2. Build the probe build with -DZR_ENABLE (without it every ZR_* macro compiles to nothing).
//   3. Give each point of interest a small integer id (an enum in the project) and plant:
//        ZR_MARK(id);                 a point in time
//        ZR_MARK_V(id, value);        a point in time with a value (bytes, a count, an index)
//        ZR_MARK_AT(id, t_ns, value); a record whose time came from another clock (a GPU timestamp harvested later)
//      In a loop that runs a few times per repeat: ZR_MARK_V(id, index) on each pass (one record per pass).
//      In a HOT loop (thousands of passes or more) do not log each pass: time it into the id's fixed slot
//        ZR_BEGIN(t);  ... the work ...  ZR_END(id, t);     // slot id keeps count, total, shortest, longest
//      Memory for slots is ZR_MAX_IDS entries however long the run is.
//   4. Run with   ZR_ON=1 ZR_OUT=/abs/path/run.tsv [ZR_CAP=<records, default 1048576>] ./program
//   5. Read the dump with zr_read.py (ids → names, time between marks, per repeat).
//
// PDL: first mark: read ZR_ON; when "1" allocate ZR_CAP records, touch every page, register the dump at exit.
//      mark: take the next index with an atomic add; past the capacity count a drop and return; else store
//            {time ns, value, id, thread tag}.
//      begin/end: read the clock twice; add the difference to the id's fixed slot (count, total, shortest, longest).
//      exit: write one header line, one line per record and one "S" line per used slot to ZR_OUT.
// LITERALS: record 24 bytes · default capacity ZR_DEFAULT_CAP = 1,048,576 records (24 MiB) · ZR_MAX_IDS = 256 slots ·
//           default file "zone-recorder.tsv" · slot lines in the dump start with "S" ·
//           dump columns: id, time (in units of ZR_TIME_UNIT_NS, the PRECISION constant at the top), value, thread. The header line carries clock_step_ns, the smallest step the clock
//           showed on this machine: a difference smaller than a few steps is below the precision of the recording.
//           The thread column tells live threads apart; a thread that ended can hand its tag to a later one.
// NOT COVERED: marks from a signal handler; processes that end by _exit or a kill (call zr_dump() before, it is safe to
//           call more than once); time across machines.
#ifndef ZONE_RECORDER_H
#define ZONE_RECORDER_H

// PRECISION: the unit of every recorded and dumped time, in nanoseconds. 1 = nanoseconds (default), 1000 = microseconds,
// 1000000 = milliseconds. Set it here or with -DZR_TIME_UNIT_NS=<n>; the dump header carries it and zr_read.py follows it.
#ifndef ZR_TIME_UNIT_NS
#define ZR_TIME_UNIT_NS 1
#endif

// SIZE: both stores are fixed, allocated once, and never grow during a run.
// ZR_DEFAULT_CAP: records the event log holds (24 bytes each) when ZR_CAP is not set; a full log stops and counts drops.
// ZR_MAX_IDS:     probe ids that can own a statistics slot (count, total, shortest, longest); ids at or above it are refused.
#ifndef ZR_DEFAULT_CAP
#define ZR_DEFAULT_CAP 1048576
#endif
#ifndef ZR_MAX_IDS
#define ZR_MAX_IDS 256
#endif

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void     zr_mark_at(uint32_t id, uint64_t t_ns, uint64_t value);   // t_ns in nanoseconds on the recorder's clock
void     zr_mark_v (uint32_t id, uint64_t value);
void     zr_mark   (uint32_t id);
uint64_t zr_now_ns (void);   // the recorder's clock: monotonic nanoseconds
int      zr_enabled(void);   // 1 when ZR_ON=1 and the buffer exists
void     zr_dump   (void);   // write the file now; also runs at exit; later calls rewrite it with what is recorded so far
uint64_t zr_begin  (void);                       // the clock now, or 0 when the recorder is off
void     zr_end    (uint32_t id, uint64_t t0);   // add (now - t0) to slot id: count, total, shortest, longest

#ifdef __cplusplus
}
#endif

#ifdef ZR_ENABLE
#define ZR_MARK(id)                 zr_mark((uint32_t) (id))
#define ZR_MARK_V(id, value)        zr_mark_v((uint32_t) (id), (uint64_t) (value))
#define ZR_MARK_AT(id, t_ns, value) zr_mark_at((uint32_t) (id), (uint64_t) (t_ns), (uint64_t) (value))
#define ZR_BEGIN(t)                 uint64_t t = zr_begin()
#define ZR_END(id, t)               zr_end((uint32_t) (id), (t))
#else
#define ZR_MARK(id)                 ((void) 0)
#define ZR_MARK_V(id, value)        ((void) 0)
#define ZR_MARK_AT(id, t_ns, value) ((void) 0)
#define ZR_BEGIN(t)                 ((void) 0)
#define ZR_END(id, t)               ((void) 0)
#endif

#ifdef ZONE_RECORDER_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#else
#include <time.h>
#endif

typedef struct { uint64_t t_ns; uint64_t value; uint32_t id; uint32_t thread; } zr_record;

static zr_record * zr_buf     = NULL;
static uint64_t    zr_cap     = 0;
static uint64_t    zr_n       = 0;   // next free index; may run past zr_cap, the excess is the drop count
static int         zr_state   = -1;  // -1 not looked at, -2 being set up, 0 off, 1 on
typedef struct { uint64_t count; uint64_t total_ns; uint64_t min_ns; uint64_t max_ns; } zr_slot;
static zr_slot     zr_slots[ZR_MAX_IDS];   // fixed: one slot per probe id
static uint64_t    zr_slot_refused = 0;    // ZR_END calls with an id at or above ZR_MAX_IDS
static uint64_t    zr_step_ns = 0;   // smallest step the clock showed at setup: differences below it mean nothing

#if defined(_WIN32)
#define ZR_ADD64(p)        ((uint64_t) _InterlockedExchangeAdd64((volatile long long *) (p), 1))
#define ZR_ADDN(p, v)      ((void) _InterlockedExchangeAdd64((volatile long long *) (p), (long long) (v)))
#define ZR_CAS64(p, o, n)  (_InterlockedCompareExchange64((volatile long long *) (p), (long long) (n), (long long) (o)) == (long long) (o))
#define ZR_CAS(p, old, nw) (_InterlockedCompareExchange((volatile long *) (p), (nw), (old)) == (old))
#define ZR_LOAD(p)         (*(volatile int *) (p))
#define ZR_TLS             __declspec(thread)
#else
#define ZR_ADD64(p)        __atomic_fetch_add((p), 1, __ATOMIC_RELAXED)
#define ZR_ADDN(p, v)      ((void) __atomic_fetch_add((p), (v), __ATOMIC_RELAXED))
static int zr_cas64(uint64_t * p, uint64_t old, uint64_t nw) { return __atomic_compare_exchange_n(p, &old, nw, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED); }
#define ZR_CAS64(p, o, n)  zr_cas64((p), (o), (n))
static int zr_cas(int * p, int old, int nw) { return __atomic_compare_exchange_n(p, &old, nw, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE); }
#define ZR_CAS(p, old, nw) zr_cas((p), (old), (nw))
#define ZR_LOAD(p)         __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define ZR_TLS             __thread
#endif

uint64_t zr_now_ns(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) { QueryPerformanceFrequency(&freq); }
    QueryPerformanceCounter(&now);
    return (uint64_t) ((double) now.QuadPart * 1e9 / (double) freq.QuadPart);
#elif defined(__APPLE__)
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
#endif
}

void zr_dump(void) {
    if (ZR_LOAD(&zr_state) != 1) { return; }
    const char * path = getenv("ZR_OUT");
    FILE * f = fopen(path && path[0] ? path : "zone-recorder.tsv", "w");
    if (!f) { return; }
    const uint64_t n_all = zr_n;
    const uint64_t n     = n_all < zr_cap ? n_all : zr_cap;
    fprintf(f, "# zone_recorder v1\trecords=%llu\tdropped=%llu\tcapacity=%llu\tclock_step_ns=%llu\ttime_unit_ns=%llu\n",
            (unsigned long long) n, (unsigned long long) (n_all - n), (unsigned long long) zr_cap, (unsigned long long) zr_step_ns,
            (unsigned long long) ZR_TIME_UNIT_NS);
    for (uint64_t i = 0; i < n; i++) {
        fprintf(f, "%u\t%llu\t%llu\t%u\n", zr_buf[i].id, (unsigned long long) zr_buf[i].t_ns,
                (unsigned long long) zr_buf[i].value, zr_buf[i].thread);
    }
    for (uint32_t s = 0; s < (uint32_t) ZR_MAX_IDS; s++) {   // the fixed slots: "S", id, count, total, shortest, longest
        if (zr_slots[s].count == 0) { continue; }
        fprintf(f, "S\t%u\t%llu\t%llu\t%llu\t%llu\n", s, (unsigned long long) zr_slots[s].count,
                (unsigned long long) (zr_slots[s].total_ns / (uint64_t) ZR_TIME_UNIT_NS),
                (unsigned long long) (zr_slots[s].min_ns / (uint64_t) ZR_TIME_UNIT_NS),
                (unsigned long long) (zr_slots[s].max_ns / (uint64_t) ZR_TIME_UNIT_NS));
    }
    if (zr_slot_refused) { fprintf(f, "# slot ids refused (at or above ZR_MAX_IDS=%d): %llu calls\n", (int) ZR_MAX_IDS, (unsigned long long) zr_slot_refused); }
#ifdef ZR_DUMP_EXTRA
    ZR_DUMP_EXTRA(f);
#endif
    fclose(f);
}

static int zr_setup(void) {
    for (;;) {
        const int s = ZR_LOAD(&zr_state);
        if (s >= 0) { return s; }
        if (s == -1 && ZR_CAS(&zr_state, -1, -2)) { break; }   // this thread sets up; the others wait in the loop
    }
    int on = 0;
    const char * v = getenv("ZR_ON");
    if (v && strcmp(v, "1") == 0) {
        const char * c = getenv("ZR_CAP");
        uint64_t cap = c ? (uint64_t) strtoull(c, NULL, 10) : 0;
        if (cap == 0) { cap = (uint64_t) ZR_DEFAULT_CAP; }
        zr_buf = (zr_record *) malloc((size_t) cap * sizeof(zr_record));
        if (zr_buf) {
            memset(zr_buf, 0, (size_t) cap * sizeof(zr_record));   // touch every page now, not inside the run
            zr_cap = cap;
            for (int k = 0; k < 200; k++) {   // precision of the clock on this machine
                const uint64_t a = zr_now_ns(); uint64_t b = zr_now_ns();
                while (b == a) { b = zr_now_ns(); }
                if (zr_step_ns == 0 || b - a < zr_step_ns) { zr_step_ns = b - a; }
            }
            atexit(zr_dump);
            on = 1;
        }
    }
#if defined(_WIN32)
    _InterlockedExchange((volatile long *) &zr_state, on);
#else
    __atomic_store_n(&zr_state, on, __ATOMIC_RELEASE);
#endif
    return on;
}

int zr_enabled(void) {
    const int s = ZR_LOAD(&zr_state);
    return (s >= 0 ? s : zr_setup()) == 1;
}

void zr_mark_at(uint32_t id, uint64_t t_ns, uint64_t value) {
    static ZR_TLS char tag;   // its address tells the threads apart
    if (!zr_enabled()) { return; }
    const uint64_t i = ZR_ADD64(&zr_n);
    if (i >= zr_cap) { return; }
    zr_buf[i].t_ns   = t_ns / (uint64_t) ZR_TIME_UNIT_NS;   // stored in the unit set by ZR_TIME_UNIT_NS
    zr_buf[i].value  = value;
    zr_buf[i].id     = id;
    zr_buf[i].thread = (uint32_t) (((uintptr_t) &tag) >> 6);
}

void zr_mark_v(uint32_t id, uint64_t value) {
    if (!zr_enabled()) { return; }
    zr_mark_at(id, zr_now_ns(), value);
}

void zr_mark(uint32_t id) { zr_mark_v(id, 0); }

uint64_t zr_begin(void) { return zr_enabled() ? zr_now_ns() : 0; }

void zr_end(uint32_t id, uint64_t t0) {
    if (t0 == 0 || !zr_enabled()) { return; }
    const uint64_t d = zr_now_ns() - t0;
    if (id >= (uint32_t) ZR_MAX_IDS) { ZR_ADDN(&zr_slot_refused, 1); return; }
    zr_slot * s = &zr_slots[id];
    ZR_ADDN(&s->total_ns, d);
    const uint64_t n = ZR_ADD64(&s->count);
    uint64_t cur = s->min_ns;
    while ((n == 0 || d < cur || cur == 0) && !ZR_CAS64(&s->min_ns, cur, d)) { cur = s->min_ns; if (cur != 0 && d >= cur) { break; } }
    cur = s->max_ns;
    while (d > cur && !ZR_CAS64(&s->max_ns, cur, d)) { cur = s->max_ns; }
}

#endif // ZONE_RECORDER_IMPLEMENTATION
#endif // ZONE_RECORDER_H
