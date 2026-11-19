// Generated model requirement rows (input SHA-256 ab843d5aeacd). Do not edit.
#pragma once

#include "ggml.h"

// The activation formats a model asks for, by architecture and general.name. GGML_TYPE_COUNT = not known or none.
struct llama_model_requirement_row {
    const char *   arch;
    const char *   name;
    enum ggml_type hidden;
    enum ggml_type state;
    enum ggml_type kv;
};

static const struct llama_model_requirement_row llama_model_requirement_rows[] = {
    { "qwen35", "Qwen3.5-0.8B", GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_BF16 },
    { "qwen35", "Qwen3.5-2B", GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_BF16 },
    { "qwen35", "Qwen3.5-4B", GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_BF16 },
    { "qwen3", "Qwen3 0.6B", GGML_TYPE_BF16, GGML_TYPE_COUNT, GGML_TYPE_BF16 },
    { "qwen2", "qwen2.5-0.5b-instruct", GGML_TYPE_BF16, GGML_TYPE_COUNT, GGML_TYPE_BF16 },
    { "qwen35", "Qwen3.5-4B-GLB-3D-Coder", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "qwen35", "Qwen3.5 4B GLB 3D Coder", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "qwen35", "Qwen3.5-9B", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "qwen35", "Qwen3.8 27B", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "qwen35", "Qwen3.8 4B", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "qwen3moe", "Qwen3-30B-A3B-Instruct-2507", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "qwen3moe", "Qwen3 30Ba3 Instruct", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "llama", "Llama 3.2 3B Instruct", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
    { "gemma4", "Gemma-4-E4B-It", GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT },
};

static const int llama_model_requirement_rows_n = (int) (sizeof(llama_model_requirement_rows) / sizeof(llama_model_requirement_rows[0]));
