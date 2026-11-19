#include "ggml-metal-common-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(norm) int ggml_metal_norm_matvec_mode(void) {
    static const int mode = [] {
        const char * value = std::getenv("GGML_METAL_NORM_MATVEC");
        if (value == nullptr) {
            return 0;
        }
        if (std::strcmp(value, "pack") == 0) {
            return 1;
        }
        if (std::strcmp(value, "nm-self") == 0) {
            return 2;
        }
        return std::strcmp(value, "1") == 0 || std::strcmp(value, "full") == 0 ? 3 : 0;
    }();
    return mode;
}

bool ggml_metal_norm_matvec_enabled(void) {
    return ggml_metal_norm_matvec_mode() >= 2;
}

bool ggml_metal_norm_matvec_add_scale(void) {
    return ggml_metal_norm_matvec_mode() == 3;
}

