#include "../ggml-metal-device.h"
#include "../ggml-metal-impl.h"
#include "../ggml-impl.h"

#include <cassert>
#include <memory>
#include <string>

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_irfft(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IRFFT);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_irfft_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fold(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_FOLD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_fold_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}
