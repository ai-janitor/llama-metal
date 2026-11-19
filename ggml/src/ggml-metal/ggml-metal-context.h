#pragma once

#include "ggml-metal-device.h"
#ifdef GGML_METAL_TEST_OBSERVER
#include "ggml-metal.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

int  ggml_metal_get_profile         (ggml_metal_t ctx);
void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);
void ggml_metal_set_act_precision   (ggml_metal_t ctx, enum ggml_type type);
ggml_metal_device_t ggml_metal_get_dev(ggml_metal_t ctx);
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

// arm an explicit per-dispatch profile window on every eligible context (see ggml_backend_metal_profile_window)
int  ggml_metal_profile_window      (int n_graphs);

#ifdef GGML_METAL_TEST_OBSERVER
void ggml_metal_test_set_observer(ggml_metal_t ctx,
        ggml_backend_metal_test_observer_callback callback, void * user_data);
#endif

#ifdef __cplusplus
}
#endif
