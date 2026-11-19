#import "device/ggml-metal-device-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(memory) void ggml_metal_device_placement_init(ggml_metal_device_t dev) {
    {
        {
            dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX
            // In case of eGPU, shared memory may be preferable.
            dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }
            if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                dev->props.use_shared_buffers = true;
            }
        }
    }
}
