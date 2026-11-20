#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
#include "matmul/matvec-plan.h"
GGML_METAL_FAMILY(memory) // bandwidth tracking counters
struct ggml_metal_transfer_stats ggml_metal_xfer_stats = {0};

void ggml_metal_transfer_stats_report(void) {
    int64_t h2d_bytes   = ggml_metal_xfer_stats.h2d_bytes;
    int64_t h2d_time_us = ggml_metal_xfer_stats.h2d_time_us;
    int64_t h2d_count   = ggml_metal_xfer_stats.h2d_count;
    int64_t d2h_bytes   = ggml_metal_xfer_stats.d2h_bytes;
    int64_t d2h_time_us = ggml_metal_xfer_stats.d2h_time_us;
    int64_t d2h_count   = ggml_metal_xfer_stats.d2h_count;

    if (h2d_count == 0 && d2h_count == 0) {
        return;
    }

    GGML_LOG_INFO("ggml_metal: transfer stats:\n");
    if (h2d_count > 0) {
        double h2d_mb = h2d_bytes / (1024.0 * 1024.0);
        double h2d_ms = h2d_time_us / 1000.0;
        double h2d_bw = (h2d_time_us > 0) ? (h2d_bytes / (h2d_time_us / 1e6)) / (1024.0 * 1024.0) : 0;
        GGML_LOG_INFO("ggml_metal:   H2D: %8.2f MiB in %8.2f ms (%6.0f MiB/s, %" PRId64 " calls)\n",
            h2d_mb, h2d_ms, h2d_bw, h2d_count);
    }
    if (d2h_count > 0) {
        double d2h_mb = d2h_bytes / (1024.0 * 1024.0);
        double d2h_ms = d2h_time_us / 1000.0;
        double d2h_bw = (d2h_time_us > 0) ? (d2h_bytes / (d2h_time_us / 1e6)) / (1024.0 * 1024.0) : 0;
        GGML_LOG_INFO("ggml_metal:   D2H: %8.2f MiB in %8.2f ms (%6.0f MiB/s, %" PRId64 " calls)\n",
            d2h_mb, d2h_ms, d2h_bw, d2h_count);
    }
    int64_t total_us = h2d_time_us + d2h_time_us;
    if (total_us > 0) {
        GGML_LOG_INFO("ggml_metal:   total transfer time: %.2f ms\n", total_us / 1000.0);
    }
}

void ggml_metal_transfer_stats_reset(void) {
    memset(&ggml_metal_xfer_stats, 0, sizeof(ggml_metal_xfer_stats));
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicity "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;

    // Q4_K block headers of this buffer's weight tensors, unpacked once at load (hw mv rows with prep):
    // one record per tensor, all in one private buffer made when the first tensor's bytes arrive
    struct ggml_metal_kq_prep_rec * kq_recs;
    int    kq_n;
    int    kq_cap;
    int    kq_ready;      // records whose every block was written
    size_t kq_total;      // bytes of prepared headers registered
    size_t kq_run_wbase;  // current run: weight offset and prepared offset of its first tensor, weight end of its last
    size_t kq_run_abase;
    size_t kq_run_wend;
    id<MTLBuffer> kq_arena;
};

// Prepared entry: 32 bytes per 144-byte Q4_K block (2 x 8 unpacked scale bytes, d, dmin, 12 unused).
// Inside a run of nearby tensors: prepared offset = (weight offset - run start) / 144 * 32.
#define GGML_METAL_KQ_BLOCK   144
#define GGML_METAL_KQ_ENTRY    32
#define GGML_METAL_KQ_RUN_GAP  65536
#define GGML_METAL_KQ_SLACK    65536 // rows past the last one are read (and not stored) by the kernels, as with the weights

struct ggml_metal_kq_prep_rec {
    const void * data;    // the tensor's data pointer: the key
    size_t woff;          // weight offset in the buffer
    size_t aoff;          // offset of its entries in kq_arena
    size_t n_blocks;
    size_t n_written;
    bool   bad;           // bytes arrived in a form that is not whole blocks: the tensor keeps the unprepared path
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    ggml_metal_device_rsets_rm(buf->dev, buf->rset);

    [buf->kq_arena release];
    free(buf->kq_recs);

    for (int i = 0; i < buf->n_buffers; i++) {
        [buf->buffers[i].metal release];
    }

    ggml_metal_buffer_rset_free(buf);

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

static bool ggml_metal_kq_prep_test(void) {
    const char * v = getenv("GGML_METAL_KQ_PREP_TEST");
    return v != NULL && strcmp(v, "1") == 0;
}

static struct ggml_metal_kq_prep_rec * ggml_metal_buffer_kq_find(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    for (int i = 0; i < buf->kq_n; i++) {
        if (buf->kq_recs[i].data == t->data) {
            return &buf->kq_recs[i];
        }
    }
    return NULL;
}

// Called for every tensor placed in the buffer, before any bytes arrive: a Q4_K weight whose shape has a prep row
// on this card gets a place for its prepared block headers. GGML_METAL_KQ_PREP_TEST=1 takes every Q4_K tensor.
// GGML_METAL_TEST_MV_PLAN with prep on takes the weights of the type and shape it names.
void ggml_metal_buffer_kq_register(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    const struct ggml_metal_card * card = &buf->dev->card;
    // the test hook GGML_METAL_TEST_MV_PLAN with prep on: the weight it names gets its headers too
    const struct ggml_metal_matvec_forced * forced = ggml_metal_matvec_forced_env();
    const bool test = ggml_metal_kq_prep_test() ||
        (forced != NULL && forced->prep == 1 && forced->weight == (int32_t) t->type && forced->k == t->ne[0] && forced->m == t->ne[1]);
    if (buf->is_shared || buf->n_buffers != 1 || buf->kq_arena != nil || (card->kq_prep != 1 && !test)) {
        return;
    }
    if (t->type != GGML_TYPE_Q4_K || t->view_src != NULL || t->data == NULL || t->ne[2] != 1 || t->ne[3] != 1 || t->ne[0] % 256 != 0) {
        return;
    }
    const size_t n_blocks = (size_t) (t->ne[0] / 256) * (size_t) t->ne[1];
    if (ggml_nbytes(t) != n_blocks * GGML_METAL_KQ_BLOCK) {
        return;
    }
    bool wanted = test;
    for (uint32_t i = 0; i < card->n_mv_rows && !wanted; i++) {
        const struct ggml_device_mv_row * row = &card->mv_rows[i];
        wanted = row->prep == 1 && ggml_metal_matvec_row_fits(card, row, -1, (int32_t) t->type, t->ne[0], t->ne[1], -1);
    }
    if (!wanted || ggml_metal_buffer_kq_find(buf, t) != NULL) {
        return;
    }
    if (buf->kq_n == buf->kq_cap) {
        const int cap = buf->kq_cap == 0 ? 64 : 2 * buf->kq_cap;
        struct ggml_metal_kq_prep_rec * recs = realloc(buf->kq_recs, (size_t) cap * sizeof(*recs));
        if (recs == NULL) {
            return;
        }
        buf->kq_recs = recs;
        buf->kq_cap  = cap;
    }
    const size_t woff = (size_t) ((uintptr_t) t->data - (uintptr_t) buf->all_data);
    struct ggml_metal_kq_prep_rec * rec = &buf->kq_recs[buf->kq_n++];
    rec->data      = t->data;
    rec->woff      = woff;
    rec->n_blocks  = n_blocks;
    rec->n_written = 0;
    rec->bad       = false;
    if (buf->kq_n > 1 && woff >= buf->kq_run_wend && woff - buf->kq_run_wend <= GGML_METAL_KQ_RUN_GAP &&
        (woff - buf->kq_run_wbase) % GGML_METAL_KQ_BLOCK == 0) {
        rec->aoff = buf->kq_run_abase + (woff - buf->kq_run_wbase) / GGML_METAL_KQ_BLOCK * GGML_METAL_KQ_ENTRY;
    } else {
        rec->aoff = buf->kq_total;
        buf->kq_run_wbase = woff;
        buf->kq_run_abase = rec->aoff;
    }
    buf->kq_run_wend = woff + n_blocks * GGML_METAL_KQ_BLOCK;
    buf->kq_total    = rec->aoff + n_blocks * GGML_METAL_KQ_ENTRY;
}

// The tensor's bytes are in hand on the host: unpack the header of every whole block in [offset, offset + size) and
// copy the entries up. Same unpack as block_dot_q4_k_swar::accumulate (kmask1 0x3f3f, kmask2 0x0f0f, kmask3 0xc0c0).
static void ggml_metal_buffer_kq_fill(ggml_metal_buffer_t buf, const struct ggml_tensor * t, const void * data, size_t offset, size_t size) {
    if (buf->kq_n == 0) {
        return;
    }
    struct ggml_metal_kq_prep_rec * rec = ggml_metal_buffer_kq_find(buf, t);
    if (rec == NULL || rec->bad) {
        return;
    }
    if (offset % GGML_METAL_KQ_BLOCK != 0 || size % GGML_METAL_KQ_BLOCK != 0 || offset + size > rec->n_blocks * GGML_METAL_KQ_BLOCK) {
        rec->bad = true;
        return;
    }
    @autoreleasepool {
        if (buf->kq_arena == nil) {
            const size_t arena_bytes = buf->kq_total + GGML_METAL_KQ_SLACK;
            buf->kq_arena = [buf->dev->mtl_device newBufferWithLength:arena_bytes options:MTLResourceStorageModePrivate];
            if (buf->kq_arena == nil) {
                GGML_LOG_WARN("%s: no memory for %zu bytes of prepared Q4_K headers; the tensors keep the unprepared path\n", __func__, arena_bytes);
                buf->kq_n = 0;
                return;
            }
            ggml_metal_log_allocated_size(buf->dev->mtl_device, arena_bytes);
        }
        const size_t n = size / GGML_METAL_KQ_BLOCK;
        id<MTLBuffer> stage = [buf->dev->mtl_device newBufferWithLength:n * GGML_METAL_KQ_ENTRY options:MTLResourceStorageModeShared];
        GGML_ASSERT(stage);
        uint8_t * p = (uint8_t *) [stage contents];
        memset(p, 0, n * GGML_METAL_KQ_ENTRY);
        const uint8_t * w = (const uint8_t *) data;
        for (size_t b = 0; b < n; b++) {
            const uint8_t * blk = w + b * GGML_METAL_KQ_BLOCK;
            uint8_t * out = p + b * GGML_METAL_KQ_ENTRY;
            for (int sp = 0; sp < 2; sp++) {
                const uint16_t w0 = (uint16_t) (blk[4 + 2*(sp + 0)] | (blk[5 + 2*(sp + 0)] << 8));
                const uint16_t w2 = (uint16_t) (blk[4 + 2*(sp + 2)] | (blk[5 + 2*(sp + 2)] << 8));
                const uint16_t w4 = (uint16_t) (blk[4 + 2*(sp + 4)] | (blk[5 + 2*(sp + 4)] << 8));
                const uint16_t u[4] = {
                    (uint16_t) (w0 & 0x3f3f), (uint16_t) (w2 & 0x3f3f),
                    (uint16_t) ((w4 & 0x0f0f) | ((w0 & 0xc0c0) >> 2)), (uint16_t) (((w4 >> 4) & 0x0f0f) | ((w2 & 0xc0c0) >> 2)),
                };
                memcpy(out + 8*sp, u, 8);
            }
            memcpy(out + 16, blk, 4);
        }
        ggml_metal_upload_flush(buf->dev->mtl_queue);
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        [encoder copyFromBuffer:stage sourceOffset:0 toBuffer:buf->kq_arena destinationOffset:rec->aoff + offset / GGML_METAL_KQ_BLOCK * GGML_METAL_KQ_ENTRY size:n * GGML_METAL_KQ_ENTRY];
        [encoder endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        [stage release];
        rec->n_written += n;
        if (rec->n_written == rec->n_blocks) {
            buf->kq_ready++;
            if (buf->kq_ready == buf->kq_n) {
                GGML_LOG_INFO("%s: prepared Q4_K headers: %d tensors, %.1f MiB; device allocated %.1f MiB of %.1f MiB recommended\n", __func__,
                    buf->kq_n, buf->kq_total / 1024.0 / 1024.0,
                    buf->dev->mtl_device.currentAllocatedSize / 1024.0 / 1024.0,
                    buf->dev->mtl_device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);
            }
        }
    }
}

// The prepared headers of a weight tensor, or a nil buffer when it has none (not named by a row, or its bytes did not
// all arrive as whole blocks).
struct ggml_metal_buffer_id ggml_metal_buffer_kq_prep_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };
    if (buf->kq_n == 0 || buf->kq_arena == nil) {
        return res;
    }
    const struct ggml_metal_kq_prep_rec * rec = ggml_metal_buffer_kq_find(buf, t);
    if (rec == NULL || rec->bad || rec->n_written != rec->n_blocks) {
        return res;
    }
    res.metal = buf->kq_arena;
    res.offs  = rec->aoff;
    return res;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) { GGML_METAL_ZONE_BEGIN(upload);
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        ggml_metal_upload_flush(buf->dev->mtl_queue);
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, size)
                          value:value];

            [encoder endEncoding];
        }

        GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "upload");) [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

bool ggml_metal_buffer_set_nowait(ggml_metal_buffer_t buf) {
    return !buf->is_shared && buf->dev->profile.set_nowait == 1;
}

// copy the bytes into a new shared buffer and commit the blit without waiting; the graph is committed later on the same
// queue (dev->mtl_queue), so Metal runs it after this copy
void ggml_metal_buffer_set_tensor_nowait(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { GGML_METAL_ZONE_BEGIN(upload);
    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // the copy joins the queue's pending upload batch; false = the batch is switched off
        if (ggml_metal_upload_add(buf->dev->mtl_queue, buf->dev->mtl_device, data, size, bid_dst.metal, bid_dst.offs)) {
            ggml_metal_xfer_stats.h2d_bytes   += (int64_t)size;
            ggml_metal_xfer_stats.h2d_time_us += ggml_time_us() - t0;
            ggml_metal_xfer_stats.h2d_count   += 1;
            ggml_metal_buffer_kq_fill(buf, tensor, data, offset, size);
            return;
        }

        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
        GGML_ASSERT(buf_src);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        [encoder copyFromBuffer:buf_src sourceOffset:0 toBuffer:bid_dst.metal destinationOffset:bid_dst.offs size:size];
        [encoder endEncoding];
        GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "upload");) [cmd_buf commit];
        [buf_src release];
    }

    ggml_metal_xfer_stats.h2d_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.h2d_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.h2d_count   += 1;

    ggml_metal_buffer_kq_fill(buf, tensor, data, offset, size);
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) { GGML_METAL_ZONE_BEGIN(upload);
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        // src - try zero-copy first (requires page-aligned pointer and size),
        // fall back to a copy if the source data isn't suitable
        void * data_ptr = (void *)(uintptr_t) data;
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        if (!buf_src) {
            buf_src = [buf->dev->mtl_device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
        }

        GGML_ASSERT(buf_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        ggml_metal_upload_flush(buf->dev->mtl_queue);
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "upload");) [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];

        // the copy is done: drop the shared wrapper of the host bytes (the device counts it as allocated until then)
        [buf_src release];
    }

    ggml_metal_xfer_stats.h2d_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.h2d_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.h2d_count   += 1;

    ggml_metal_buffer_kq_fill(buf, tensor, data, offset, size);
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) { GGML_METAL_ZONE_BEGIN(readback);
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    const int64_t t0 = ggml_time_us();

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst - try zero-copy first (requires page-aligned pointer and size),
        // fall back to a copy if the destination data isn't suitable
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];
        const bool dst_needs_copy = (buf_dst == nil);
        if (dst_needs_copy) {
            buf_dst = [buf->dev->mtl_device newBufferWithLength:size options:MTLResourceStorageModeShared];
        }

        GGML_ASSERT(buf_dst);

        ggml_metal_upload_flush(buf->dev->mtl_queue);
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        GGML_METAL_PROBE_ONLY(ggml_metal_probe_transfer(cmd_buf, "readback");) [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (dst_needs_copy) {
            memcpy(data, [buf_dst contents], size);
        }
    }

    ggml_metal_xfer_stats.d2h_bytes   += (int64_t)size;
    ggml_metal_xfer_stats.d2h_time_us += ggml_time_us() - t0;
    ggml_metal_xfer_stats.d2h_count   += 1;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        ggml_metal_upload_flush(buf->dev->mtl_queue);
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

// Shared view lookup with an explicit byte size, so a caller that already
// checked the span never recomputes it through unchecked arithmetic.
static struct ggml_metal_buffer_id ggml_metal_buffer_get_id_sized(ggml_metal_buffer_t buf, const struct ggml_tensor * t, int64_t tsize) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}

struct ggml_metal_buffer_id ggml_metal_buffer_base_id(ggml_metal_buffer_t buf) {
    struct ggml_metal_buffer_id res = { nil, 0 };
    res.metal = buf->buffers[0].metal;
    return res;
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    return ggml_metal_buffer_get_id_sized(buf, t, ggml_nbytes(t));
}

#ifdef GGML_METAL_TEST_OBSERVER
struct ggml_metal_buffer_id ggml_metal_buffer_get_id_checked(ggml_metal_buffer_t buf, const struct ggml_tensor * t, size_t span) {
    // Keep the observer ABI's historic signed-size limit before mapping the
    // checked span through the unsigned view predicate below.
    if (span > (size_t) INT64_MAX) {
        struct ggml_metal_buffer_id res = { nil, 0 };
        return res;
    }

    // The production mapper above keeps its legacy signed arithmetic. The
    // observer query instead checks each candidate view without signed pointer
    // casts, signed end-offset addition, or narrowing before validation.
    for (int i = 0; i < buf->n_buffers; ++i) {
        size_t view_offset = 0;
        if (ggml_metal_test_observer_checked_view_lookup(
                    (uintptr_t) t->data, (uintptr_t) buf->buffers[i].data,
                    buf->buffers[i].size, span, &view_offset)) {
            struct ggml_metal_buffer_id res = {
                buf->buffers[i].metal,
                view_offset,
            };
            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);
    struct ggml_metal_buffer_id res = { nil, 0 };
    return res;
}
#endif
