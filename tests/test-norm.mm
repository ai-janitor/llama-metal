// test-norm.mm - Norm kernel unit test (Phase 1 of TEST-001)
// Tests kernel_norm_f32 from kernels/norm.metal

#include "test-kernel-base.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

// Match ggml_metal_kargs_norm struct from ggml-metal-impl.h
struct KargsNorm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
};

// CPU reference implementation (the oracle)
void cpu_norm(const float* input, float* output, int ne00, float eps) {
    // Compute mean
    float sum = 0;
    for (int i = 0; i < ne00; i++) {
        sum += input[i];
    }
    float mean = sum / ne00;

    // Subtract mean and compute variance
    float var_sum = 0;
    for (int i = 0; i < ne00; i++) {
        output[i] = input[i] - mean;
        var_sum += output[i] * output[i];
    }
    float variance = var_sum / ne00;

    // Scale by 1/sqrt(variance + eps)
    float scale = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < ne00; i++) {
        output[i] *= scale;
    }
}

class TestNorm : public TestKernelBase {
private:
    id<MTLComputePipelineState> pipeline;
    const char* kernel_source_path;

public:
    TestNorm() : pipeline(nil) {
        // Path to norm.metal source - use PROJECT_SOURCE_DIR at compile time
        kernel_source_path = KERNEL_SOURCE_PATH;
    }

    ~TestNorm() {
        if (pipeline) {
            [pipeline release];
        }
    }

    bool setup() {
        if (!init_device()) {
            return false;
        }

        // Read and concatenate Metal sources
        // We need: ggml-common.h + ggml-metal-impl.h + 00-common.metal + norm.metal
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

        // Read norm.metal
        NSString* norm_path = [NSString stringWithUTF8String:KERNEL_SOURCE_PATH];
        NSString* norm_src = [NSString stringWithContentsOfFile:norm_path
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
        if (error) {
            fprintf(stderr, "Failed to read norm.metal: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Replace includes in 00-common.metal with inline content
        common_src = [common_src stringByReplacingOccurrencesOfString:@"#include \"ggml-common.h\""
                                                            withString:common_h];
        common_src = [common_src stringByReplacingOccurrencesOfString:@"#include \"ggml-metal-impl.h\""
                                                            withString:impl_h];

        // Remove the __embed marker line (not used in non-embed build)
        common_src = [common_src stringByReplacingOccurrencesOfString:@"__embed_ggml-common.h__"
                                                            withString:@""];

        // Remove the #include "00-common.metal" line from norm_src
        norm_src = [norm_src stringByReplacingOccurrencesOfString:@"#include \"00-common.metal\""
                                                        withString:@""];

        // Concatenate: common + norm
        NSString* combined_src = [common_src stringByAppendingString:@"\n"];
        combined_src = [combined_src stringByAppendingString:norm_src];

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
        id<MTLFunction> function = [library newFunctionWithName:@"kernel_norm_f32"];
        if (!function) {
            fprintf(stderr, "Failed to get kernel_norm_f32 function\n");
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
    bool test_case(const char* name, const std::vector<float>& input, int ne00, float eps) {
        printf("\n=== Test case: %s (ne00=%d, eps=%e) ===\n", name, ne00, eps);

        // Allocate output buffers
        std::vector<float> cpu_output(ne00);
        std::vector<float> gpu_output(ne00, 0.0f);

        // Run CPU reference
        cpu_norm(input.data(), cpu_output.data(), ne00, eps);

        // Setup Metal buffers
        KargsNorm kargs = {0};
        kargs.ne00 = ne00;
        kargs.ne00_t = ne00;  // ne00_t is for templated type (float vs float4)
        kargs.nb1 = ne00 * sizeof(float);
        kargs.nb2 = kargs.nb1;
        kargs.nb3 = kargs.nb1;
        kargs.eps = eps;

        // For the fused variant (F=1 means no fuse, just norm)
        // We still need to provide dummy src1_0 and src1_1 buffers
        kargs.nef1[0] = kargs.nef1[1] = kargs.nef1[2] = 1;
        kargs.nef2[0] = kargs.nef2[1] = kargs.nef2[2] = 1;
        kargs.nef3[0] = kargs.nef3[1] = kargs.nef3[2] = 1;
        kargs.nbf1[0] = kargs.nbf1[1] = kargs.nbf1[2] = sizeof(float);
        kargs.nbf2[0] = kargs.nbf2[1] = kargs.nbf2[2] = sizeof(float);
        kargs.nbf3[0] = kargs.nbf3[1] = kargs.nbf3[2] = sizeof(float);

        id<MTLBuffer> kargs_buf = create_buffer(sizeof(KargsNorm), &kargs);
        id<MTLBuffer> input_buf = create_buffer(ne00 * sizeof(float), input.data());
        id<MTLBuffer> dummy1_buf = create_buffer(sizeof(float));  // src1_0 (unused)
        id<MTLBuffer> dummy2_buf = create_buffer(sizeof(float));  // src1_1 (unused)
        id<MTLBuffer> output_buf = create_buffer(ne00 * sizeof(float));

        if (!kargs_buf || !input_buf || !dummy1_buf || !dummy2_buf || !output_buf) {
            fprintf(stderr, "Failed to create buffers\n");
            return false;
        }

        // Dispatch kernel
        // threadgroup size: 32 threads (one simdgroup on AMD), 1 simdgroup per threadgroup
        // For small ne00, one threadgroup is enough
        // For large ne00, may need multiple threadgroups, but norm kernel uses grid.x for rows
        MTLSize threads_per_tg = MTLSizeMake(32, 1, 1);
        MTLSize threadgroups = MTLSizeMake(1, 1, 1);  // One row

        // Threadgroup memory: shmem_f32[32] for simdgroup reductions
        size_t tg_mem = 32 * sizeof(float);

        std::vector<id<MTLBuffer>> buffers = {
            kargs_buf,
            input_buf,
            dummy1_buf,
            dummy2_buf,
            output_buf
        };

        bool dispatch_ok = dispatch(pipeline, buffers, threadgroups, threads_per_tg, tg_mem);

        if (!dispatch_ok) {
            fprintf(stderr, "Dispatch failed\n");
            [kargs_buf release];
            [input_buf release];
            [dummy1_buf release];
            [dummy2_buf release];
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
        [dummy1_buf release];
        [dummy2_buf release];
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

        // Test 1: Single element (variance=0, scale=1/sqrt(eps))
        {
            std::vector<float> input = {42.0f};
            if (!test_case("single_element", input, 1, 1e-5f)) failed++;
        }

        // Test 2: All zeros (mean=0, variance=0, scale=1/sqrt(eps))
        {
            std::vector<float> input(64, 0.0f);
            if (!test_case("all_zeros", input, 64, 1e-5f)) failed++;
        }

        // Test 3: All same value (variance=0)
        {
            std::vector<float> input(64, 3.14159f);
            if (!test_case("all_same", input, 64, 1e-5f)) failed++;
        }

        // Test 4: Contains +INF
        {
            std::vector<float> input(32, 1.0f);
            input[15] = std::numeric_limits<float>::infinity();
            if (!test_case("contains_inf", input, 32, 1e-5f)) failed++;
        }

        // Test 5: Contains -INF
        {
            std::vector<float> input(32, 1.0f);
            input[15] = -std::numeric_limits<float>::infinity();
            if (!test_case("contains_neg_inf", input, 32, 1e-5f)) failed++;
        }

        // Test 6: Contains NaN
        {
            std::vector<float> input(32, 1.0f);
            input[15] = std::numeric_limits<float>::quiet_NaN();
            if (!test_case("contains_nan", input, 32, 1e-5f)) failed++;
        }

        // Test 7: ne00=7 (not divisible by 4 - tests non-vec4 path)
        {
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
            if (!test_case("ne00_7_non_divisible", input, 7, 1e-5f)) failed++;
        }

        // Test 8: ne00=32 (exactly one simdgroup)
        {
            std::vector<float> input(32);
            for (int i = 0; i < 32; i++) input[i] = (float)(i + 1);
            if (!test_case("ne00_32_one_simdgroup", input, 32, 1e-5f)) failed++;
        }

        // Test 9: ne00=64 (vec4 path: ne00%4==0)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = sinf((float)i * 0.1f);
            if (!test_case("ne00_64_vec4", input, 64, 1e-5f)) failed++;
        }

        // Test 10: ne00=100 (larger, non-power-of-2)
        {
            std::vector<float> input(100);
            for (int i = 0; i < 100; i++) input[i] = (float)i / 10.0f - 5.0f;
            if (!test_case("ne00_100_non_power_of_2", input, 100, 1e-5f)) failed++;
        }

        // Test 11: ne00=1024 (large, multiple threadgroups needed)
        {
            std::vector<float> input(1024);
            for (int i = 0; i < 1024; i++) input[i] = cosf((float)i * 0.01f);
            if (!test_case("ne00_1024_large", input, 1024, 1e-5f)) failed++;
        }

        // Test 12: Random values with negative numbers
        {
            std::vector<float> input(128);
            for (int i = 0; i < 128; i++) {
                input[i] = (float)(i % 7) - 3.5f;
            }
            if (!test_case("random_with_negatives", input, 128, 1e-5f)) failed++;
        }

        printf("\n=== Summary ===\n");
        printf("Total tests: 12\n");
        printf("Failed: %d\n", failed);
        printf("Passed: %d\n", 12 - failed);

        return failed;
    }
};

int main(int argc, char** argv) {
    TestNorm test;

    if (!test.setup()) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    int failed = test.run_tests();

    return failed > 0 ? 1 : 0;
}
