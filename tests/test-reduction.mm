// test-reduction.mm - Reduction kernel unit test (TEST-001)
// Tests kernel_sum_rows_f32_f32 from kernels/reduction.metal

#include "test-kernel-base.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

// Match ggml_metal_kargs_sum_rows struct from ggml-metal-impl.h
struct KargsSumRows {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
};

// CPU reference implementation (the oracle)
void cpu_sum_rows(const float* input, float* output, int ne00, int ne01) {
    for (int row = 0; row < ne01; row++) {
        float sum = 0.0f;
        for (int col = 0; col < ne00; col++) {
            sum += input[row * ne00 + col];
        }
        output[row] = sum;
    }
}

class TestReduction : public TestKernelBase {
private:
    id<MTLComputePipelineState> pipeline;
    const char* kernel_source_path;

public:
    TestReduction() : pipeline(nil) {
        kernel_source_path = KERNEL_SOURCE_PATH;
    }

    ~TestReduction() {
        if (pipeline) {
            [pipeline release];
        }
    }

    bool setup() {
        if (!init_device()) {
            return false;
        }

        // Read and concatenate Metal sources
        // We need: ggml-common.h + ggml-metal-impl.h + 00-common.metal + reduction.metal
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

        // Read reduction.metal
        NSString* reduction_path = [NSString stringWithUTF8String:KERNEL_SOURCE_PATH];
        NSString* reduction_src = [NSString stringWithContentsOfFile:reduction_path
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
        if (error) {
            fprintf(stderr, "Failed to read reduction.metal: %s\n",
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

        // Remove the #include "00-common.metal" line from reduction_src
        reduction_src = [reduction_src stringByReplacingOccurrencesOfString:@"#include \"00-common.metal\""
                                                        withString:@""];

        // Concatenate: common + reduction
        NSString* combined_src = [common_src stringByAppendingString:@"\n"];
        combined_src = [combined_src stringByAppendingString:reduction_src];

        // Compile the combined source with function constants
        MTLCompileOptions* options = [MTLCompileOptions new];
        library = [device newLibraryWithSource:combined_src options:options error:&error];
        [options release];

        if (error) {
            fprintf(stderr, "Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Set function constants (FC_sum_rows_op = OP_SUM_ROWS = 0)
        MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
        short op_sum_rows = 0;  // OP_SUM_ROWS = 0
        [constantValues setConstantValue:&op_sum_rows type:MTLDataTypeShort atIndex:1400];  // FC_SUM_ROWS = 1400

        // Get kernel function with constants
        id<MTLFunction> function = [library newFunctionWithName:@"kernel_sum_rows_f32_f32"
                                                 constantValues:constantValues
                                                          error:&error];
        [constantValues release];

        if (error || !function) {
            fprintf(stderr, "Failed to get kernel_sum_rows_f32_f32 function: %s\n",
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
    bool test_case(const char* name, const std::vector<float>& input, int ne00, int ne01) {
        printf("\n=== Test case: %s (ne00=%d, ne01=%d) ===\n", name, ne00, ne01);

        // Allocate output buffers
        std::vector<float> cpu_output(ne01);
        std::vector<float> gpu_output(ne01, 0.0f);

        // Run CPU reference
        cpu_sum_rows(input.data(), cpu_output.data(), ne00, ne01);

        // Setup Metal buffers
        KargsSumRows kargs = {0};
        kargs.ne00 = ne00;
        kargs.ne01 = ne01;
        kargs.ne02 = 1;
        kargs.ne03 = 1;
        kargs.nb00 = sizeof(float);
        kargs.nb01 = ne00 * sizeof(float);
        kargs.nb02 = kargs.nb01 * ne01;
        kargs.nb03 = kargs.nb02;

        kargs.ne0 = 1;  // output is [ne01, 1, 1, 1]
        kargs.ne1 = ne01;
        kargs.ne2 = 1;
        kargs.ne3 = 1;
        kargs.nb0 = sizeof(float);
        kargs.nb1 = sizeof(float);
        kargs.nb2 = kargs.nb1 * ne01;
        kargs.nb3 = kargs.nb2;

        id<MTLBuffer> kargs_buf = create_buffer(sizeof(KargsSumRows), &kargs);
        id<MTLBuffer> input_buf = create_buffer(ne00 * ne01 * sizeof(float), input.data());
        id<MTLBuffer> output_buf = create_buffer(ne01 * sizeof(float));

        if (!kargs_buf || !input_buf || !output_buf) {
            fprintf(stderr, "Failed to create buffers\n");
            return false;
        }

        // Dispatch kernel
        // From ops-reduction.cpp:
        // nth = min(max_threads_per_tg, ne00), starting from 32 and doubling
        int nth = 32;
        int max_threads = [pipeline maxTotalThreadsPerThreadgroup];
        while (nth < ne00 && nth < max_threads) {
            nth *= 2;
        }
        nth = std::min(nth, max_threads);
        nth = std::min(nth, ne00);

        MTLSize threads_per_tg = MTLSizeMake(nth, 1, 1);
        MTLSize threadgroups = MTLSizeMake(ne01, 1, 1);  // One threadgroup per row

        // Threadgroup memory: shmem for simdgroup reductions
        // From the kernel: threadgroup float * shmem [[threadgroup(0)]]
        // Need (nth+31)/32 * sizeof(float) for simdgroup partial sums
        int nsg = (nth + 31) / 32;
        size_t tg_mem = nsg * sizeof(float);

        std::vector<id<MTLBuffer>> buffers = {
            kargs_buf,
            input_buf,
            output_buf
        };

        bool dispatch_ok = dispatch(pipeline, buffers, threadgroups, threads_per_tg, tg_mem);

        if (!dispatch_ok) {
            fprintf(stderr, "Dispatch failed\n");
            [kargs_buf release];
            [input_buf release];
            [output_buf release];
            return false;
        }

        // Read back GPU result
        memcpy(gpu_output.data(), [output_buf contents], ne01 * sizeof(float));

        // Compare
        bool match = compare(gpu_output.data(), cpu_output.data(), ne01, 1e-4f);

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

        // Test 1: Single element per row (ne00=1, output=input)
        {
            std::vector<float> input = {42.0f, 3.14f, -5.0f, 0.0f};
            if (!test_case("ne00_1_single_element", input, 1, 4)) failed++;
        }

        // Test 2: Two elements per row
        {
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
            if (!test_case("ne00_2_two_elements", input, 2, 2)) failed++;
        }

        // Test 3: ne00=7 (non-divisible by 4)
        {
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
            if (!test_case("ne00_7_non_divisible", input, 7, 1)) failed++;
        }

        // Test 4: ne00=32 (one simdgroup)
        {
            std::vector<float> input(32);
            for (int i = 0; i < 32; i++) input[i] = (float)(i + 1);
            if (!test_case("ne00_32_one_simdgroup", input, 32, 1)) failed++;
        }

        // Test 5: ne00=64 (vec4 path possible)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)(i + 1);
            if (!test_case("ne00_64_vec4", input, 64, 1)) failed++;
        }

        // Test 6: ne00=1024 (large, multiple simdgroups)
        {
            std::vector<float> input(1024);
            for (int i = 0; i < 1024; i++) input[i] = 1.0f;
            if (!test_case("ne00_1024_large", input, 1024, 1)) failed++;
        }

        // Test 7: All zeros (sum=0)
        {
            std::vector<float> input(128, 0.0f);
            if (!test_case("all_zeros", input, 128, 1)) failed++;
        }

        // Test 8: All ones (sum=ne00)
        {
            std::vector<float> input(64, 1.0f);
            if (!test_case("all_ones", input, 64, 1)) failed++;
        }

        // Test 9: Contains +INF (INF + finite = INF)
        {
            std::vector<float> input(32, 1.0f);
            input[15] = std::numeric_limits<float>::infinity();
            if (!test_case("contains_inf", input, 32, 1)) failed++;
        }

        // Test 10: Contains -INF
        {
            std::vector<float> input(32, 1.0f);
            input[15] = -std::numeric_limits<float>::infinity();
            if (!test_case("contains_neg_inf", input, 32, 1)) failed++;
        }

        // Test 11: Contains NaN (NaN + anything = NaN)
        {
            std::vector<float> input(32, 1.0f);
            input[15] = std::numeric_limits<float>::quiet_NaN();
            if (!test_case("contains_nan", input, 32, 1)) failed++;
        }

        // Test 12: Alternating positive/negative (cancellation test)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) {
                input[i] = (i % 2 == 0) ? 1.0f : -1.0f;
            }
            if (!test_case("alternating_pos_neg", input, 64, 1)) failed++;
        }

        // Test 13: Very large values (overflow risk)
        {
            std::vector<float> input(16, 1e30f);
            if (!test_case("large_values", input, 16, 1)) failed++;
        }

        // Test 14: Multiple rows - ne01=4
        {
            std::vector<float> input(64);  // 16 x 4
            for (int i = 0; i < 64; i++) {
                input[i] = (float)(i % 16 + 1);
            }
            if (!test_case("multiple_rows_ne01_4", input, 16, 4)) failed++;
        }

        // Test 15: Multiple rows - ne01=16
        {
            std::vector<float> input(256);  // 16 x 16
            for (int i = 0; i < 256; i++) {
                input[i] = (float)(i / 16 + 1);  // Each row has same value
            }
            if (!test_case("multiple_rows_ne01_16", input, 16, 16)) failed++;
        }

        // Test 16: Large matrix (128 x 32)
        {
            std::vector<float> input(4096);  // 128 x 32
            for (int i = 0; i < 4096; i++) {
                input[i] = sinf((float)i * 0.01f);
            }
            if (!test_case("large_matrix_128x32", input, 128, 32)) failed++;
        }

        printf("\n=== Summary ===\n");
        printf("Total tests: 16\n");
        printf("Failed: %d\n", failed);
        printf("Passed: %d\n", 16 - failed);

        return failed;
    }
};

int main(int argc, char** argv) {
    TestReduction test;

    if (!test.setup()) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    int failed = test.run_tests();

    return failed > 0 ? 1 : 0;
}
