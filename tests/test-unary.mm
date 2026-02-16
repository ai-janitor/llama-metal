// test-unary.mm - Unary kernel unit test
// Tests kernel_unary_impl from kernels/unary-binary.metal

#include "test-kernel-base.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

// Match ggml_metal_kargs_unary struct from ggml-metal-impl.h
struct KargsUnary {
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
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
};

// Op codes from ggml-metal-impl.h
#define OP_UNARY_NUM_SCALE      10
#define OP_UNARY_NUM_SQR        13
#define OP_UNARY_NUM_SQRT       14
#define OP_UNARY_NUM_SIN        15
#define OP_UNARY_NUM_COS        16
#define OP_UNARY_NUM_LOG        17
#define OP_UNARY_NUM_LEAKY_RELU 18
#define OP_UNARY_NUM_RELU       101
#define OP_UNARY_NUM_TANH       100
#define OP_UNARY_NUM_CLAMP      12

// Function constant indices
#define FC_UNARY 1200

// CPU reference implementations (the oracle)
void cpu_scale(const float* input, float* output, int n, float scale, float bias) {
    for (int i = 0; i < n; i++) {
        output[i] = scale * input[i] + bias;
    }
}

void cpu_sqr(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * input[i];
    }
}

void cpu_sqrt(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = sqrtf(input[i]);
    }
}

void cpu_sin(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = sinf(input[i]);
    }
}

void cpu_cos(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = cosf(input[i]);
    }
}

void cpu_log(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = logf(input[i]);
    }
}

void cpu_relu(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

void cpu_leaky_relu(const float* input, float* output, int n, float slope) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] > 0 ? input[i] : slope * input[i];
    }
}

void cpu_tanh(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = tanhf(input[i]);
    }
}

void cpu_clamp(const float* input, float* output, int n, float min_val, float max_val) {
    for (int i = 0; i < n; i++) {
        output[i] = fminf(fmaxf(input[i], min_val), max_val);
    }
}

class TestUnary : public TestKernelBase {
private:
    id<MTLComputePipelineState> pipeline_scale;
    id<MTLComputePipelineState> pipeline_sqr;
    id<MTLComputePipelineState> pipeline_sqrt;
    id<MTLComputePipelineState> pipeline_sin;
    id<MTLComputePipelineState> pipeline_cos;
    id<MTLComputePipelineState> pipeline_log;
    id<MTLComputePipelineState> pipeline_relu;
    id<MTLComputePipelineState> pipeline_leaky_relu;
    id<MTLComputePipelineState> pipeline_tanh;
    id<MTLComputePipelineState> pipeline_clamp;

public:
    TestUnary() : pipeline_scale(nil), pipeline_sqr(nil), pipeline_sqrt(nil),
                  pipeline_sin(nil), pipeline_cos(nil), pipeline_log(nil),
                  pipeline_relu(nil), pipeline_leaky_relu(nil), pipeline_tanh(nil),
                  pipeline_clamp(nil) {}

    ~TestUnary() {
        if (pipeline_scale) [pipeline_scale release];
        if (pipeline_sqr) [pipeline_sqr release];
        if (pipeline_sqrt) [pipeline_sqrt release];
        if (pipeline_sin) [pipeline_sin release];
        if (pipeline_cos) [pipeline_cos release];
        if (pipeline_log) [pipeline_log release];
        if (pipeline_relu) [pipeline_relu release];
        if (pipeline_leaky_relu) [pipeline_leaky_relu release];
        if (pipeline_tanh) [pipeline_tanh release];
        if (pipeline_clamp) [pipeline_clamp release];
    }

    bool setup() {
        if (!init_device()) {
            return false;
        }

        // Read and concatenate Metal sources
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

        // Read unary-binary.metal
        NSString* unary_path = [kernel_dir stringByAppendingPathComponent:@"unary-binary.metal"];
        NSString* unary_src = [NSString stringWithContentsOfFile:unary_path
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
        if (error) {
            fprintf(stderr, "Failed to read unary-binary.metal: %s\n",
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

        // Remove the #include "00-common.metal" line from unary_src
        unary_src = [unary_src stringByReplacingOccurrencesOfString:@"#include \"00-common.metal\""
                                                          withString:@""];

        // Concatenate: common + unary
        NSString* combined_src = [common_src stringByAppendingString:@"\n"];
        combined_src = [combined_src stringByAppendingString:unary_src];

        // Compile the combined source
        MTLCompileOptions* options = [MTLCompileOptions new];
        library = [device newLibraryWithSource:combined_src options:options error:&error];
        [options release];

        if (error) {
            fprintf(stderr, "Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Create pipelines for each operation
        // Each needs function constants set for FC_unary_op and FC_unary_cnt

        // Helper to create pipeline with constants
        auto create_pipeline = [&](int op_num, bool cnt) -> id<MTLComputePipelineState> {
            MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
            short op = (short)op_num;
            bool cnt_val = cnt;
            [constants setConstantValue:&op type:MTLDataTypeShort atIndex:FC_UNARY + 0];
            [constants setConstantValue:&cnt_val type:MTLDataTypeBool atIndex:FC_UNARY + 1];

            id<MTLFunction> function = [library newFunctionWithName:@"kernel_unary_f32_f32"
                                                     constantValues:constants
                                                              error:&error];
            [constants release];

            if (error || !function) {
                fprintf(stderr, "Failed to get function for op %d: %s\n",
                        op_num, error ? [[error localizedDescription] UTF8String] : "unknown");
                return nil;
            }

            id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:function
                                                                                     error:&error];
            [function release];

            if (error) {
                fprintf(stderr, "Failed to create pipeline for op %d: %s\n",
                        op_num, [[error localizedDescription] UTF8String]);
                return nil;
            }

            return pso;
        };

        // Create pipelines (use cnt=true for contiguous path)
        pipeline_scale = create_pipeline(OP_UNARY_NUM_SCALE, true);
        pipeline_sqr = create_pipeline(OP_UNARY_NUM_SQR, true);
        pipeline_sqrt = create_pipeline(OP_UNARY_NUM_SQRT, true);
        pipeline_sin = create_pipeline(OP_UNARY_NUM_SIN, true);
        pipeline_cos = create_pipeline(OP_UNARY_NUM_COS, true);
        pipeline_log = create_pipeline(OP_UNARY_NUM_LOG, true);
        pipeline_relu = create_pipeline(OP_UNARY_NUM_RELU, true);
        pipeline_leaky_relu = create_pipeline(OP_UNARY_NUM_LEAKY_RELU, true);
        pipeline_tanh = create_pipeline(OP_UNARY_NUM_TANH, true);
        pipeline_clamp = create_pipeline(OP_UNARY_NUM_CLAMP, true);

        if (!pipeline_scale || !pipeline_sqr || !pipeline_sqrt || !pipeline_sin ||
            !pipeline_cos || !pipeline_log || !pipeline_relu || !pipeline_leaky_relu ||
            !pipeline_tanh || !pipeline_clamp) {
            fprintf(stderr, "Failed to create one or more pipelines\n");
            return false;
        }

        return true;
    }

    // Generic test runner
    bool test_unary_op(
        const char* name,
        id<MTLComputePipelineState> pipeline,
        const std::vector<float>& input,
        const std::vector<float>& expected,
        int ne,
        const KargsUnary& kargs,
        float tolerance = 1e-5f)
    {
        printf("\n=== Test case: %s (ne=%d) ===\n", name, ne);

        std::vector<float> gpu_output(ne, 0.0f);

        // Setup Metal buffers
        id<MTLBuffer> kargs_buf = create_buffer(sizeof(KargsUnary), &kargs);
        id<MTLBuffer> input_buf = create_buffer(ne * sizeof(float), input.data());
        id<MTLBuffer> output_buf = create_buffer(ne * sizeof(float));

        if (!kargs_buf || !input_buf || !output_buf) {
            fprintf(stderr, "Failed to create buffers\n");
            return false;
        }

        // Dispatch: contiguous path uses 1D grid over all elements
        // threadgroups = (n, 1, 1), threads_per_tg = (1, 1, 1)
        MTLSize threads_per_tg = MTLSizeMake(1, 1, 1);
        MTLSize threadgroups = MTLSizeMake(ne, 1, 1);

        std::vector<id<MTLBuffer>> buffers = {kargs_buf, input_buf, output_buf};

        bool dispatch_ok = dispatch(pipeline, buffers, threadgroups, threads_per_tg, 0);

        if (!dispatch_ok) {
            fprintf(stderr, "Dispatch failed\n");
            [kargs_buf release];
            [input_buf release];
            [output_buf release];
            return false;
        }

        // Read back GPU result
        memcpy(gpu_output.data(), [output_buf contents], ne * sizeof(float));

        // Compare
        bool match = compare(gpu_output.data(), expected.data(), ne, tolerance);

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

        // ========== SCALE ==========
        printf("\n========== SCALE TESTS ==========\n");

        // Test 1: Single element
        {
            std::vector<float> input = {42.0f};
            std::vector<float> expected(1);
            cpu_scale(input.data(), expected.data(), 1, 2.0f, 3.0f);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;
            kargs.scale = 2.0f;
            kargs.bias = 3.0f;

            if (!test_unary_op("scale_single", pipeline_scale, input, expected, 1, kargs))
                failed++;
        }

        // Test 2: ne=7 (not divisible by 4)
        {
            std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
            std::vector<float> expected(7);
            cpu_scale(input.data(), expected.data(), 7, 0.5f, 1.0f);

            KargsUnary kargs = {0};
            kargs.ne00 = 7;
            kargs.scale = 0.5f;
            kargs.bias = 1.0f;

            if (!test_unary_op("scale_ne7", pipeline_scale, input, expected, 7, kargs))
                failed++;
        }

        // Test 3: ne=64 (vec4 path)
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)i;
            std::vector<float> expected(64);
            cpu_scale(input.data(), expected.data(), 64, 3.0f, -1.5f);

            KargsUnary kargs = {0};
            kargs.ne00 = 64;
            kargs.scale = 3.0f;
            kargs.bias = -1.5f;

            if (!test_unary_op("scale_ne64_vec4", pipeline_scale, input, expected, 64, kargs))
                failed++;
        }

        // Test 4: Contains INF
        {
            std::vector<float> input = {1.0f, std::numeric_limits<float>::infinity(), 3.0f};
            std::vector<float> expected(3);
            cpu_scale(input.data(), expected.data(), 3, 2.0f, 0.0f);

            KargsUnary kargs = {0};
            kargs.ne00 = 3;
            kargs.scale = 2.0f;
            kargs.bias = 0.0f;

            if (!test_unary_op("scale_inf", pipeline_scale, input, expected, 3, kargs))
                failed++;
        }

        // ========== SQR ==========
        printf("\n========== SQR TESTS ==========\n");

        // Test 5: Single element
        {
            std::vector<float> input = {5.0f};
            std::vector<float> expected(1);
            cpu_sqr(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("sqr_single", pipeline_sqr, input, expected, 1, kargs))
                failed++;
        }

        // Test 6: Negative values
        {
            std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            std::vector<float> expected(5);
            cpu_sqr(input.data(), expected.data(), 5);

            KargsUnary kargs = {0};
            kargs.ne00 = 5;

            if (!test_unary_op("sqr_negatives", pipeline_sqr, input, expected, 5, kargs))
                failed++;
        }

        // Test 7: Very large and very small values
        {
            std::vector<float> input = {1e-20f, 1e20f, 0.0f};
            std::vector<float> expected(3);
            cpu_sqr(input.data(), expected.data(), 3);

            KargsUnary kargs = {0};
            kargs.ne00 = 3;

            if (!test_unary_op("sqr_extremes", pipeline_sqr, input, expected, 3, kargs))
                failed++;
        }

        // Test 8: ne=64
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)i - 32.0f;
            std::vector<float> expected(64);
            cpu_sqr(input.data(), expected.data(), 64);

            KargsUnary kargs = {0};
            kargs.ne00 = 64;

            if (!test_unary_op("sqr_ne64", pipeline_sqr, input, expected, 64, kargs))
                failed++;
        }

        // ========== SQRT ==========
        printf("\n========== SQRT TESTS ==========\n");

        // Test 9: Single element
        {
            std::vector<float> input = {16.0f};
            std::vector<float> expected(1);
            cpu_sqrt(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("sqrt_single", pipeline_sqrt, input, expected, 1, kargs))
                failed++;
        }

        // Test 10: Negative input (should produce NaN)
        {
            std::vector<float> input = {-1.0f, 0.0f, 1.0f};
            std::vector<float> expected(3);
            cpu_sqrt(input.data(), expected.data(), 3);

            KargsUnary kargs = {0};
            kargs.ne00 = 3;

            if (!test_unary_op("sqrt_negative", pipeline_sqrt, input, expected, 3, kargs))
                failed++;
        }

        // Test 11: Zero
        {
            std::vector<float> input = {0.0f};
            std::vector<float> expected(1);
            cpu_sqrt(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("sqrt_zero", pipeline_sqrt, input, expected, 1, kargs))
                failed++;
        }

        // Test 12: ne=64
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)i;
            std::vector<float> expected(64);
            cpu_sqrt(input.data(), expected.data(), 64);

            KargsUnary kargs = {0};
            kargs.ne00 = 64;

            if (!test_unary_op("sqrt_ne64", pipeline_sqrt, input, expected, 64, kargs))
                failed++;
        }

        // ========== RELU ==========
        printf("\n========== RELU TESTS ==========\n");

        // Test 13: Single element (negative)
        {
            std::vector<float> input = {-5.0f};
            std::vector<float> expected(1);
            cpu_relu(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("relu_single_neg", pipeline_relu, input, expected, 1, kargs))
                failed++;
        }

        // Test 14: Single element (positive)
        {
            std::vector<float> input = {5.0f};
            std::vector<float> expected(1);
            cpu_relu(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("relu_single_pos", pipeline_relu, input, expected, 1, kargs))
                failed++;
        }

        // Test 15: Mix of positive and negative
        {
            std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            std::vector<float> expected(5);
            cpu_relu(input.data(), expected.data(), 5);

            KargsUnary kargs = {0};
            kargs.ne00 = 5;

            if (!test_unary_op("relu_mixed", pipeline_relu, input, expected, 5, kargs))
                failed++;
        }

        // Test 16: Zero
        {
            std::vector<float> input = {0.0f};
            std::vector<float> expected(1);
            cpu_relu(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("relu_zero", pipeline_relu, input, expected, 1, kargs))
                failed++;
        }

        // Test 17: ne=64
        {
            std::vector<float> input(64);
            for (int i = 0; i < 64; i++) input[i] = (float)i - 32.0f;
            std::vector<float> expected(64);
            cpu_relu(input.data(), expected.data(), 64);

            KargsUnary kargs = {0};
            kargs.ne00 = 64;

            if (!test_unary_op("relu_ne64", pipeline_relu, input, expected, 64, kargs))
                failed++;
        }

        // Test 18: ne=100 (non-power-of-2)
        {
            std::vector<float> input(100);
            for (int i = 0; i < 100; i++) input[i] = (float)i - 50.0f;
            std::vector<float> expected(100);
            cpu_relu(input.data(), expected.data(), 100);

            KargsUnary kargs = {0};
            kargs.ne00 = 100;

            if (!test_unary_op("relu_ne100", pipeline_relu, input, expected, 100, kargs))
                failed++;
        }

        // Test 19: ne=1024 (large)
        {
            std::vector<float> input(1024);
            for (int i = 0; i < 1024; i++) input[i] = (float)i - 512.0f;
            std::vector<float> expected(1024);
            cpu_relu(input.data(), expected.data(), 1024);

            KargsUnary kargs = {0};
            kargs.ne00 = 1024;

            if (!test_unary_op("relu_ne1024", pipeline_relu, input, expected, 1024, kargs))
                failed++;
        }

        // ========== LOG ==========
        printf("\n========== LOG TESTS ==========\n");

        // Test 20: Negative input (should produce NaN)
        {
            std::vector<float> input = {-1.0f};
            std::vector<float> expected(1);
            cpu_log(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("log_negative", pipeline_log, input, expected, 1, kargs))
                failed++;
        }

        // Test 21: Zero (should produce -INF)
        {
            std::vector<float> input = {0.0f};
            std::vector<float> expected(1);
            cpu_log(input.data(), expected.data(), 1);

            KargsUnary kargs = {0};
            kargs.ne00 = 1;

            if (!test_unary_op("log_zero", pipeline_log, input, expected, 1, kargs))
                failed++;
        }

        // Test 22: Normal values
        {
            std::vector<float> input = {1.0f, M_E, 10.0f};
            std::vector<float> expected(3);
            cpu_log(input.data(), expected.data(), 3);

            KargsUnary kargs = {0};
            kargs.ne00 = 3;

            if (!test_unary_op("log_normal", pipeline_log, input, expected, 3, kargs))
                failed++;
        }

        // ========== SIN/COS ==========
        printf("\n========== SIN/COS TESTS ==========\n");

        // Test 23: SIN basic
        {
            std::vector<float> input = {0.0f, M_PI/2, M_PI};
            std::vector<float> expected(3);
            cpu_sin(input.data(), expected.data(), 3);

            KargsUnary kargs = {0};
            kargs.ne00 = 3;

            if (!test_unary_op("sin_basic", pipeline_sin, input, expected, 3, kargs))
                failed++;
        }

        // Test 24: COS basic
        {
            std::vector<float> input = {0.0f, M_PI/2, M_PI};
            std::vector<float> expected(3);
            cpu_cos(input.data(), expected.data(), 3);

            KargsUnary kargs = {0};
            kargs.ne00 = 3;

            if (!test_unary_op("cos_basic", pipeline_cos, input, expected, 3, kargs))
                failed++;
        }

        // ========== LEAKY_RELU ==========
        printf("\n========== LEAKY_RELU TESTS ==========\n");

        // Test 25: Leaky ReLU
        {
            std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            std::vector<float> expected(5);
            cpu_leaky_relu(input.data(), expected.data(), 5, 0.01f);

            KargsUnary kargs = {0};
            kargs.ne00 = 5;
            kargs.slope = 0.01f;

            if (!test_unary_op("leaky_relu", pipeline_leaky_relu, input, expected, 5, kargs))
                failed++;
        }

        // ========== TANH ==========
        printf("\n========== TANH TESTS ==========\n");

        // Test 26: Tanh
        {
            std::vector<float> input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            std::vector<float> expected(5);
            cpu_tanh(input.data(), expected.data(), 5);

            KargsUnary kargs = {0};
            kargs.ne00 = 5;

            if (!test_unary_op("tanh_basic", pipeline_tanh, input, expected, 5, kargs, 1e-4f))
                failed++;
        }

        // ========== CLAMP ==========
        printf("\n========== CLAMP TESTS ==========\n");

        // Test 27: Clamp basic
        {
            std::vector<float> input = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
            std::vector<float> expected(5);
            cpu_clamp(input.data(), expected.data(), 5, -2.0f, 2.0f);

            KargsUnary kargs = {0};
            kargs.ne00 = 5;
            kargs.min = -2.0f;
            kargs.max = 2.0f;

            if (!test_unary_op("clamp_basic", pipeline_clamp, input, expected, 5, kargs))
                failed++;
        }

        printf("\n=== Summary ===\n");
        printf("Total tests: 27\n");
        printf("Failed: %d\n", failed);
        printf("Passed: %d\n", 27 - failed);

        return failed;
    }
};

int main(int argc, char** argv) {
    TestUnary test;

    if (!test.setup()) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    int failed = test.run_tests();

    return failed > 0 ? 1 : 0;
}
