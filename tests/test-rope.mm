// test-rope.mm - RoPE kernel unit test (TEST-001)
// Tests kernel_rope_norm_f32 from kernels/rope.metal

#include "test-kernel-base.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

// Match ggml_metal_kargs_rope struct from ggml-metal-impl.h
struct KargsRope {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_past;
    int32_t  n_dims;
    int32_t  n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    bool     src2;
};

// CPU reference implementation (the oracle)
// RoPE applies rotary position embeddings to pairs of elements
void cpu_rope_norm(
    const float* input,
    const int32_t* positions,
    float* output,
    int ne00,      // elements per row (must be even)
    int ne01,      // number of rows
    int ne02,      // batch dimension
    int n_dims,    // number of dimensions to apply rope to
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor)
{
    const float inv_ndims = -1.0f / n_dims;

    // For each batch
    for (int i2 = 0; i2 < ne02; i2++) {
        const float theta_base = (float) positions[i2];

        // For each row
        for (int i1 = 0; i1 < ne01; i1++) {
            // For each pair of elements
            for (int i0 = 0; i0 < ne00; i0 += 2) {
                const int idx = i2 * (ne01 * ne00) + i1 * ne00 + i0;

                if (i0 < n_dims) {
                    // Compute theta for this pair
                    const float theta = theta_base * powf(freq_base, inv_ndims * i0);

                    // YaRN scaling (simplified - no ext_factor in basic case)
                    const float theta_scaled = theta * freq_scale;
                    const float cos_theta = cosf(theta_scaled) * attn_factor;
                    const float sin_theta = sinf(theta_scaled) * attn_factor;

                    // Apply rotation
                    const float x0 = input[idx];
                    const float x1 = input[idx + 1];

                    output[idx]     = x0 * cos_theta - x1 * sin_theta;
                    output[idx + 1] = x0 * sin_theta + x1 * cos_theta;
                } else {
                    // Beyond n_dims, just copy
                    output[idx]     = input[idx];
                    output[idx + 1] = input[idx + 1];
                }
            }
        }
    }
}

class TestRope : public TestKernelBase {
private:
    id<MTLComputePipelineState> pipeline;
    const char* kernel_source_path;

public:
    TestRope() : pipeline(nil) {
        kernel_source_path = KERNEL_SOURCE_PATH;
    }

    ~TestRope() {
        if (pipeline) {
            [pipeline release];
        }
    }

    bool setup() {
        if (!init_device()) {
            return false;
        }

        // Read and concatenate Metal sources
        // We need: ggml-common.h + ggml-metal-impl.h + 00-common.metal + rope.metal
        NSError* error = nil;
        NSString* kernel_dir = [NSString stringWithUTF8String:KERNEL_DIR];
        NSString* ggml_src_dir = [kernel_dir stringByDeletingLastPathComponent];
        NSString* ggml_dir = [ggml_src_dir stringByDeletingLastPathComponent];

        // Read ggml-common.h
        NSString* common_h_path = [ggml_dir stringByAppendingPathComponent:@"ggml-common.h"];
        NSString* common_h = [NSString stringWithContentsOfFile:common_h_path
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
        if (error) {
            fprintf(stderr, "Failed to read ggml-common.h: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Read ggml-metal-impl.h
        NSString* impl_h_path = [kernel_dir stringByDeletingLastPathComponent];
        impl_h_path = [impl_h_path stringByAppendingPathComponent:@"ggml-metal-impl.h"];
        NSString* impl_h = [NSString stringWithContentsOfFile:impl_h_path
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (error) {
            fprintf(stderr, "Failed to read ggml-metal-impl.h: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Read 00-common.metal
        NSString* common_path = [kernel_dir stringByAppendingPathComponent:@"00-common.metal"];
        NSString* common_src = [NSString stringWithContentsOfFile:common_path
                                                          encoding:NSUTF8StringEncoding
                                                             error:&error];
        if (error) {
            fprintf(stderr, "Failed to read 00-common.metal: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Read rope.metal
        NSString* rope_path = [NSString stringWithUTF8String:KERNEL_SOURCE_PATH];
        NSString* rope_src = [NSString stringWithContentsOfFile:rope_path
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
        if (error) {
            fprintf(stderr, "Failed to read rope.metal: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Replace includes in 00-common.metal with inline content
        common_src = [common_src stringByReplacingOccurrencesOfString:@"#include \"ggml-common.h\""
                                                            withString:common_h];
        common_src = [common_src stringByReplacingOccurrencesOfString:@"#include \"ggml-metal-impl.h\""
                                                            withString:impl_h];

        // Remove the __embed marker line
        common_src = [common_src stringByReplacingOccurrencesOfString:@"__embed_ggml-common.h__"
                                                            withString:@""];

        // Remove the #include "00-common.metal" line from rope_src
        rope_src = [rope_src stringByReplacingOccurrencesOfString:@"#include \"00-common.metal\""
                                                        withString:@""];

        // Concatenate: common + rope
        NSString* combined_src = [common_src stringByAppendingString:@"\n"];
        combined_src = [combined_src stringByAppendingString:rope_src];

        // Compile with function constant for non-imrope mode
        MTLCompileOptions* options = [MTLCompileOptions new];
        library = [device newLibraryWithSource:combined_src options:options error:&error];
        [options release];

        if (error) {
            fprintf(stderr, "Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Get kernel function with function constant
        MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
        bool is_imrope = false;
        [constants setConstantValue:&is_imrope type:MTLDataTypeBool atIndex:0]; // FC_ROPE + 0

        id<MTLFunction> function = [library newFunctionWithName:@"kernel_rope_norm_f32"
                                                 constantValues:constants
                                                          error:&error];
        [constants release];

        if (error || !function) {
            fprintf(stderr, "Failed to get kernel_rope_norm_f32 function: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        // Create pipeline state
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        [function release];

        if (error) {
            fprintf(stderr, "Failed to create pipeline state: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        return true;
    }

    // Test one configuration
    bool test_case(
        const char* name,
        const std::vector<float>& input,
        const std::vector<int32_t>& positions,
        int ne00,
        int ne01,
        int ne02,
        int n_dims,
        float freq_base,
        float freq_scale,
        float attn_factor)
    {
        printf("\n=== Test case: %s (ne00=%d, ne01=%d, ne02=%d, n_dims=%d) ===\n",
               name, ne00, ne01, ne02, n_dims);

        const int total_elements = ne00 * ne01 * ne02;

        // Allocate output buffers
        std::vector<float> cpu_output(total_elements, 0.0f);
        std::vector<float> gpu_output(total_elements, 0.0f);

        // Run CPU reference
        cpu_rope_norm(input.data(), positions.data(), cpu_output.data(),
                      ne00, ne01, ne02, n_dims, freq_base, freq_scale, 0.0f, attn_factor);

        // Setup Metal buffers
        KargsRope kargs = {0};
        kargs.ne00 = ne00;
        kargs.ne01 = ne01;
        kargs.ne02 = ne02;
        kargs.ne03 = 1;
        kargs.nb00 = sizeof(float);
        kargs.nb01 = ne00 * sizeof(float);
        kargs.nb02 = ne00 * ne01 * sizeof(float);
        kargs.nb03 = ne00 * ne01 * ne02 * sizeof(float);
        kargs.ne0 = ne00;
        kargs.ne1 = ne01;
        kargs.ne2 = ne02;
        kargs.ne3 = 1;
        kargs.nb0 = sizeof(float);
        kargs.nb1 = ne00 * sizeof(float);
        kargs.nb2 = ne00 * ne01 * sizeof(float);
        kargs.nb3 = ne00 * ne01 * ne02 * sizeof(float);
        kargs.n_past = 0;
        kargs.n_dims = n_dims;
        kargs.n_ctx_orig = 2048;  // default
        kargs.freq_base = freq_base;
        kargs.freq_scale = freq_scale;
        kargs.ext_factor = 0.0f;  // simplified, no YaRN extrapolation
        kargs.attn_factor = attn_factor;
        kargs.beta_fast = 32.0f;   // default
        kargs.beta_slow = 1.0f;    // default
        kargs.sect_0 = 0;          // mrope params (unused in norm mode)
        kargs.sect_1 = 0;
        kargs.sect_2 = 0;
        kargs.sect_3 = 0;
        kargs.src2 = false;        // no freq_factors

        id<MTLBuffer> kargs_buf = create_buffer(sizeof(KargsRope), &kargs);
        id<MTLBuffer> input_buf = create_buffer(total_elements * sizeof(float), input.data());
        id<MTLBuffer> pos_buf = create_buffer(ne02 * sizeof(int32_t), positions.data());
        id<MTLBuffer> dummy_buf = create_buffer(sizeof(float));  // src2 (unused)
        id<MTLBuffer> output_buf = create_buffer(total_elements * sizeof(float));

        if (!kargs_buf || !input_buf || !pos_buf || !dummy_buf || !output_buf) {
            fprintf(stderr, "Failed to create buffers\n");
            return false;
        }

        // Dispatch kernel
        // Threadgroups: (ne01, ne02, ne03)
        // Threads per threadgroup: (nth, 1, 1) where nth = min(1024, ne00)
        const int nth = std::min(1024, ne00);
        MTLSize threads_per_tg = MTLSizeMake(nth, 1, 1);
        MTLSize threadgroups = MTLSizeMake(ne01, ne02, 1);

        std::vector<id<MTLBuffer>> buffers = {
            kargs_buf,    // index 0: args
            input_buf,    // index 1: src0
            pos_buf,      // index 2: src1 (positions)
            dummy_buf,    // index 3: src2 (freq_factors, unused)
            output_buf    // index 4: dst
        };

        bool dispatch_ok = dispatch(pipeline, buffers, threadgroups, threads_per_tg, 0);

        if (!dispatch_ok) {
            fprintf(stderr, "Dispatch failed\n");
            [kargs_buf release];
            [input_buf release];
            [pos_buf release];
            [dummy_buf release];
            [output_buf release];
            return false;
        }

        // Read back GPU result
        memcpy(gpu_output.data(), [output_buf contents], total_elements * sizeof(float));

        // Compare
        bool match = compare(gpu_output.data(), cpu_output.data(), total_elements, 1e-5f);

        // Cleanup
        [kargs_buf release];
        [input_buf release];
        [pos_buf release];
        [dummy_buf release];
        [output_buf release];

        if (!match) {
            printf("FAILED: %s\n", name);
        } else {
            printf("PASSED: %s\n", name);
        }

        return match;
    }

    int run_tests() override {
        int failed = 0;

        // Standard RoPE parameters
        const float freq_base = 10000.0f;
        const float freq_scale = 1.0f;
        const float attn_factor = 1.0f;

        // EDGE CASES FIRST (per template)

        // Test 1: Single pair (ne00=2, minimal)
        {
            std::vector<float> input = {1.0f, 0.0f};
            std::vector<int32_t> positions = {0};
            if (!test_case("single_pair_pos0", input, positions,
                          2, 1, 1, 2, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 2: Position = 0 (no rotation, output should equal input)
        {
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            std::vector<int32_t> positions = {0};
            if (!test_case("position_zero", input, positions,
                          8, 1, 1, 8, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 3: All zeros input
        {
            std::vector<float> input(64, 0.0f);
            std::vector<int32_t> positions = {42};
            if (!test_case("all_zeros", input, positions,
                          64, 1, 1, 64, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 4: Contains +INF
        {
            std::vector<float> input(32, 1.0f);
            input[10] = std::numeric_limits<float>::infinity();
            std::vector<int32_t> positions = {5};
            if (!test_case("contains_inf", input, positions,
                          32, 1, 1, 32, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 5: Contains -INF
        {
            std::vector<float> input(32, 1.0f);
            input[10] = -std::numeric_limits<float>::infinity();
            std::vector<int32_t> positions = {5};
            if (!test_case("contains_neg_inf", input, positions,
                          32, 1, 1, 32, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 6: Contains NaN
        {
            std::vector<float> input(32, 1.0f);
            input[10] = std::numeric_limits<float>::quiet_NaN();
            std::vector<int32_t> positions = {5};
            if (!test_case("contains_nan", input, positions,
                          32, 1, 1, 32, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 7: Partial RoPE (n_dims < ne00)
        {
            std::vector<float> input(128);
            for (int i = 0; i < 128; i++) input[i] = (float)(i + 1);
            std::vector<int32_t> positions = {10};
            // Only apply RoPE to first 64 dims, rest copied
            if (!test_case("partial_rope", input, positions,
                          128, 1, 1, 64, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 8: Large position (test wraparound behavior)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = sinf((float)i * 0.1f);
            std::vector<int32_t> positions = {100000};
            if (!test_case("large_position", input, positions,
                          64, 1, 1, 64, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 9: Multiple rows (ne01 > 1)
        {
            std::vector<float> input(64 * 3);  // 3 rows of 64 elements
            for (int i = 0; i < 64 * 3; i++) input[i] = (float)(i % 64);
            std::vector<int32_t> positions = {7};
            if (!test_case("multiple_rows", input, positions,
                          64, 3, 1, 64, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 10: Batch dimension (ne02 > 1)
        {
            std::vector<float> input(32 * 2 * 2);  // 2 batches, 2 rows, 32 elements
            for (int i = 0; i < 32 * 2 * 2; i++) input[i] = (float)i;
            std::vector<int32_t> positions = {3, 5};  // different positions per batch
            if (!test_case("batch_dimension", input, positions,
                          32, 2, 2, 32, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 11: Standard transformer head (ne00=64, typical)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = cosf((float)i * 0.05f);
            std::vector<int32_t> positions = {15};
            if (!test_case("standard_head_64", input, positions,
                          64, 1, 1, 64, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 12: Larger head (ne00=128, common in modern models)
        {
            std::vector<float> input(128);
            for (int i = 0; i < 128; i++) input[i] = sinf((float)i * 0.02f);
            std::vector<int32_t> positions = {20};
            if (!test_case("standard_head_128", input, positions,
                          128, 1, 1, 128, freq_base, freq_scale, attn_factor)) failed++;
        }

        // Test 13: freq_scale != 1.0 (YaRN interpolation)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)i / 10.0f;
            std::vector<int32_t> positions = {25};
            if (!test_case("freq_scale_0.5", input, positions,
                          64, 1, 1, 64, freq_base, 0.5f, attn_factor)) failed++;
        }

        // Test 14: Different freq_base
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)(i % 7) - 3.0f;
            std::vector<int32_t> positions = {12};
            if (!test_case("freq_base_100000", input, positions,
                          64, 1, 1, 64, 100000.0f, freq_scale, attn_factor)) failed++;
        }

        // Test 15: attn_factor scaling
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)i;
            std::vector<int32_t> positions = {8};
            if (!test_case("attn_factor_0.5", input, positions,
                          64, 1, 1, 64, freq_base, freq_scale, 0.5f)) failed++;
        }

        printf("\n=== Summary ===\n");
        printf("Total tests: 15\n");
        printf("Failed: %d\n", failed);
        printf("Passed: %d\n", 15 - failed);

        return failed;
    }
};

int main(int argc, char** argv) {
    TestRope test;

    if (!test.setup()) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    int failed = test.run_tests();

    return failed > 0 ? 1 : 0;
}
