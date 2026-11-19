#pragma once

#include "ggml-metal-fusion.h"
#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"
#include "ggml-metal-test-observer-helpers.h"
#include "ggml-metal-test-override.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// shared across the split files
extern int g_devices;
extern ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i;
bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);
const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(int device);
const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(int device);
const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(int device);
