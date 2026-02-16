// test-softmax.mm - Softmax kernel unit test (TEST-001)
// Tests kernel_soft_max_f32 from kernels/softmax.metal

#include "test-kernel-base.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

// Match ggml_metal_kargs_soft_max struct from ggml-metal-impl.h
struct KargsSoftMax {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
};

// CPU reference implementation (the oracle)
void cpu_softmax(const float* input, float* output, int ne00, float scale) {
    // Find max (for numerical stability)
    float max_val = -INFINITY;
    for (int i = 0; i < ne00; i++) {
        float val = input[i] * scale;
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < ne00; i++) {
        float exp_val = expf(input[i] * scale - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    // Normalize by sum
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < ne00; i++) {
        output[i] *= inv_sum;
    }
}

class TestSoftMax : public TestKernelBase {
private:
    id<MTLComputePipelineState> pipeline;
    const char* kernel_source_path;

public:
    TestSoftMax() : pipeline(nil) {
        kernel_source_path = KERNEL_SOURCE_PATH;
    }

    ~TestSoftMax() {
        if (pipeline) {
            [pipeline release];
        }
    }

    bool setup() {
        if (!init_device()) {
            return false;
        }

        // Read and concatenate Metal sources
        // We need: ggml-common.h + ggml-metal-impl.h + 00-common.metal + softmax.metal
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

        // Read softmax.metal
        NSString* softmax_path = [NSString stringWithUTF8String:KERNEL_SOURCE_PATH];
        NSString* softmax_src = [NSString stringWithContentsOfFile:softmax_path
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
        if (error) {
            fprintf(stderr, "Failed to read softmax.metal: %s\n",
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

        // Remove the #include "00-common.metal" line from softmax_src
        softmax_src = [softmax_src stringByReplacingOccurrencesOfString:@"#include \"00-common.metal\""
                                                        withString:@""];

        // Concatenate: common + softmax
        NSString* combined_src = [common_src stringByAppendingString:@"\n"];
        combined_src = [combined_src stringByAppendingString:softmax_src];

        // Compile the combined source
        MTLCompileOptions* options = [MTLCompileOptions new];
        library = [device newLibraryWithSource:combined_src options:options error:&error];
        [options release];

        if (error) {
            fprintf(stderr, "Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Get kernel function
        id<MTLFunction> function = [library newFunctionWithName:@"kernel_soft_max_f32"];
        if (!function) {
            fprintf(stderr, "Failed to get kernel_soft_max_f32 function\n");
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
    bool test_case(const char* name, const std::vector<float>& input, int ne00, float scale = 1.0f) {
        printf("\n=== Test case: %s (ne00=%d, scale=%f) ===\n", name, ne00, scale);

        // Allocate output buffers
        std::vector<float> cpu_output(ne00);
        std::vector<float> gpu_output(ne00, 0.0f);

        // Run CPU reference
        cpu_softmax(input.data(), cpu_output.data(), ne00, scale);

        // Setup Metal buffers
        KargsSoftMax kargs = {0};
        kargs.ne00 = ne00;
        kargs.ne01 = 1;
        kargs.ne02 = 1;
        kargs.nb01 = ne00 * sizeof(float);
        kargs.nb02 = kargs.nb01;
        kargs.nb03 = kargs.nb01;
        kargs.ne11 = 1;
        kargs.ne12 = 1;
        kargs.ne13 = 1;
        kargs.nb11 = sizeof(float);
        kargs.nb12 = sizeof(float);
        kargs.nb13 = sizeof(float);
        kargs.nb1 = ne00 * sizeof(float);
        kargs.nb2 = kargs.nb1;
        kargs.nb3 = kargs.nb1;
        kargs.scale = scale;
        kargs.max_bias = 0.0f;  // No ALiBi
        kargs.m0 = 0.0f;
        kargs.m1 = 0.0f;
        kargs.n_head_log2 = 0;

        id<MTLBuffer> kargs_buf = create_buffer(sizeof(KargsSoftMax), &kargs);
        id<MTLBuffer> input_buf = create_buffer(ne00 * sizeof(float), input.data());
        id<MTLBuffer> output_buf = create_buffer(ne00 * sizeof(float));

        if (!kargs_buf || !input_buf || !output_buf) {
            fprintf(stderr, "Failed to create buffers\n");
            return false;
        }

        // For softmax kernel signature:
        // src0 = input, src1 = mask (unused, same as src0), src2 = unused (same as src0)
        std::vector<id<MTLBuffer>> buffers = {
            kargs_buf,
            input_buf,   // src0
            input_buf,   // src1 (mask - will be nullptr when src1==src0)
            input_buf,   // src2 (unused - will be nullptr when src2==src0)
            output_buf   // dst
        };

        // Dispatch kernel
        // For ne00 <= 32: 1 threadgroup of 32 threads
        // For ne00 > 32: use more threads per threadgroup (up to 1024)
        int threads_per_tg = (ne00 <= 32) ? 32 : std::min(1024, ((ne00 + 31) / 32) * 32);
        MTLSize threads_per_tg_size = MTLSizeMake(threads_per_tg, 1, 1);
        MTLSize threadgroups = MTLSizeMake(1, 1, 1);  // One row

        // Threadgroup memory: buf[N_SIMDWIDTH] for reduction
        size_t tg_mem = 32 * sizeof(float);

        bool dispatch_ok = dispatch(pipeline, buffers, threadgroups, threads_per_tg_size, tg_mem);

        if (!dispatch_ok) {
            fprintf(stderr, "Dispatch failed\n");
            [kargs_buf release];
            [input_buf release];
            [output_buf release];
            return false;
        }

        // Read back GPU result
        memcpy(gpu_output.data(), [output_buf contents], ne00 * sizeof(float));

        // Compare
        bool match = compare(gpu_output.data(), cpu_output.data(), ne00, 1e-5f);

        // Cleanup
        [kargs_buf release];
        [input_buf release];
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

        // EDGE CASES FIRST (per template)

        // Test 1: Single element (output should be 1.0)
        {
            std::vector<float> input = {42.0f};
            if (!test_case("single_element", input, 1)) failed++;
        }

        // Test 2: Two elements (smallest non-trivial)
        {
            std::vector<float> input = {1.0f, 2.0f};
            if (!test_case("two_elements", input, 2)) failed++;
        }

        // Test 3: ne00=7 (non-divisible by 4)
        {
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
            if (!test_case("ne00_7_non_divisible", input, 7)) failed++;
        }

        // Test 4: ne00=32 (exactly one simdgroup)
        {
            std::vector<float> input(32);
            for (int i = 0; i < 32; i++) input[i] = (float)(i + 1);
            if (!test_case("ne00_32_one_simdgroup", input, 32)) failed++;
        }

        // Test 5: ne00=64 (vec4 path)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = sinf((float)i * 0.1f);
            if (!test_case("ne00_64_vec4", input, 64)) failed++;
        }

        // Test 6: ne00=100 (non-power-of-2)
        {
            std::vector<float> input(100);
            for (int i = 0; i < 100; i++) input[i] = (float)i / 10.0f - 5.0f;
            if (!test_case("ne00_100_non_power_of_2", input, 100)) failed++;
        }

        // Test 7: ne00=1024 (large)
        {
            std::vector<float> input(1024);
            for (int i = 0; i < 1024; i++) input[i] = cosf((float)i * 0.01f);
            if (!test_case("ne00_1024_large", input, 1024)) failed++;
        }

        // Test 8: All same values (uniform distribution output)
        {
            std::vector<float> input(64, 3.14159f);
            if (!test_case("all_same_values", input, 64)) failed++;
        }

        // Test 9: One very large value (should dominate)
        {
            std::vector<float> input(32, 0.0f);
            input[15] = 100.0f;  // This should dominate
            if (!test_case("one_large_value", input, 32)) failed++;
        }

        // Test 10: Contains -INF (should produce 0 for that element)
        {
            std::vector<float> input(32, 1.0f);
            input[10] = -std::numeric_limits<float>::infinity();
            if (!test_case("contains_neg_inf", input, 32)) failed++;
        }

        // Test 11: All -INF except one (that one should be 1.0)
        {
            std::vector<float> input(32, -std::numeric_limits<float>::infinity());
            input[20] = 0.0f;  // Only finite value
            if (!test_case("all_neg_inf_except_one", input, 32)) failed++;
        }

        // Test 12: Very large positive values (exp overflow risk)
        {
            std::vector<float> input(32);
            for (int i = 0; i < 32; i++) input[i] = 50.0f + (float)i;
            if (!test_case("very_large_values", input, 32)) failed++;
        }

        // Test 13: Scale parameter test
        {
            std::vector<float> input(32);
            for (int i = 0; i < 32; i++) input[i] = (float)i;
            if (!test_case("scale_0.5", input, 32, 0.5f)) failed++;
        }

        // Test 14: Negative values with scale
        {
            std::vector<float> input(32);
            for (int i = 0; i < 32; i++) input[i] = (float)i - 16.0f;
            if (!test_case("negative_values", input, 32, 1.0f)) failed++;
        }

        // Test 15: All zeros (uniform distribution)
        {
            std::vector<float> input(64, 0.0f);
            if (!test_case("all_zeros", input, 64)) failed++;
        }

        // Test 16: Mixed positive and negative
        {
            std::vector<float> input(128);
            for (int i = 0; i < 128; i++) {
                input[i] = (float)(i % 7) - 3.5f;
            }
            if (!test_case("mixed_pos_neg", input, 128)) failed++;
        }

        printf("\n=== Summary ===\n");
        printf("Total tests: 16\n");
        printf("Failed: %d\n", failed);
        printf("Passed: %d\n", 16 - failed);

        return failed;
    }
};

int main(int argc, char** argv) {
    TestSoftMax test;

    if (!test.setup()) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    int failed = test.run_tests();

    return failed > 0 ? 1 : 0;
}
