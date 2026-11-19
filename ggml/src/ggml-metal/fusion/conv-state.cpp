#include "fusion/ggml-metal-fusion-internal.h"
#include "ggml-metal-probe.h"
GGML_METAL_FAMILY(fusion) // CONCAT(state, input) + CPY(shifted state -> cache cell) + SSM_CONV + silu at one decode token:
// one launch reads the state, computes the conv and writes the shifted state (GGML_METAL_STATE_IN_PLACE).
bool ggml_metal_fusion_check_conv_state(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    if (!ggml_metal_state_in_place()) {
        return false;
    }

    const ggml_tensor * cat  = nodes[0];
    const ggml_tensor * cpy  = nodes[1];
    const ggml_tensor * conv = nodes[2];
    const ggml_tensor * un   = nodes[3];
    const ggml_tensor * st   = cat->src[0];
    const ggml_tensor * in   = cat->src[1];
    const ggml_tensor * w    = conv->src[1];
    const ggml_tensor * tail = cpy->src[0];
    const ggml_tensor * cell = cpy->src[1];

    if (ggml_get_op_params_i32(cat, 0) != 0 || cat->type != GGML_TYPE_F32 || st->type != GGML_TYPE_F32 ||
        in->type != GGML_TYPE_F32 || !ggml_is_contiguous(cat) || (cat->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const int64_t nc = cat->ne[0];
    const int64_t ch = cat->ne[1];
    if (nc < 2 || cat->ne[2] != 1 || cat->ne[3] != 1 || st->ne[0] != nc - 1 || st->ne[1] != ch || st->nb[0] != sizeof(float) ||
        in->ne[0] != 1 || in->ne[1] != ch) {
        return false;
    }
    if (conv->src[0] != cat || !w || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || w->ne[0] != nc || w->ne[1] != ch ||
        conv->type != GGML_TYPE_F32 || (conv->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    if (un->op != GGML_OP_UNARY || ggml_get_unary_op(un) != GGML_UNARY_OP_SILU || un->src[0] != conv || un->src[1] ||
        un->type != GGML_TYPE_F32 || !ggml_is_contiguous(un)) {
        return false;
    }
    // the cpy stores the last nc - 1 columns of the concat into a contiguous cache cell
    if (tail->op != GGML_OP_VIEW || tail->view_src != cat || tail->view_offs != sizeof(float) || tail->ne[0] != nc - 1 ||
        tail->ne[1] != ch || cell->type != GGML_TYPE_F32 || !ggml_is_contiguous(cell) || ggml_nelements(cell) != (nc - 1)*ch) {
        return false;
    }
    if (mode == GGML_METAL_FUSION_FULL && cell->data == nullptr) {
        return false;
    }
    // nothing else reads the concat, its tail view or the raw conv
    int uses_cat = 0, uses_tail = 0, uses_conv = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = gf->nodes[i]->src[s];
            uses_cat  += src == cat;
            uses_tail += src == tail;
            uses_conv += src == conv;
        }
    }
    return uses_cat == 2 && uses_tail == 1 && uses_conv == 1;
}

