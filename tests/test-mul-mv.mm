// test-mul-mv.mm - Matrix-vector multiplication kernel unit test (TEST-001)
// Tests kernel_mul_mv_f32_f32_simple from test-mul-mv-simple.metal

#include "test-kernel-base.h"
#include <cstdint>
#include <vector>
#include <cmath>
#include <limits>

// Match ggml_metal_kargs_mul_mv struct from ggml-metal-impl.h
struct KargsMulMv {
    int32_t  ne00;  // matrix width (number of columns)
    int32_t  ne01;  // matrix height (number of rows)
    int32_t  ne02;  // batch dimension
    uint64_t nb00;  // stride of elements in row (bytes)
    uint64_t nb01;  // stride of rows (bytes)
    uint64_t nb02;  // stride of batch (bytes)
    uint64_t nb03;  // stride of higher dimension (bytes)
    int32_t  ne10;  // vector length
    int32_t  ne11;  // number of vectors
    int32_t  ne12;  // batch dimension for vector
    uint64_t nb10;  // stride of elements in vector (bytes)
    uint64_t nb11;  // stride between vectors (bytes)
    uint64_t nb12;  // stride of batch for vector (bytes)
    uint64_t nb13;  // stride of higher dimension for vector (bytes)
    int32_t  ne0;   // output width (same as ne01 - number of rows)
    int32_t  ne1;   // output height (same as ne11 - number of vectors)
    int32_t  nr0;   // rows per threadgroup (only used in non-short variants)
    int16_t  r2;    // broadcast factor for dimension 2
    int16_t  r3;    // broadcast factor for dimension 3
};

// CPU reference implementation (the oracle)
// matrix: ne00 columns × ne01 rows (row-major)
// vector: ne10 elements (must equal ne00)
// output: ne01 elements (one per row)
void cpu_mul_mv(const float* matrix, const float* vector, float* output,
                int ne00, int ne01) {
    for (int row = 0; row < ne01; row++) {
        float sum = 0.0f;
        for (int col = 0; col < ne00; col++) {
            sum += matrix[row * ne00 + col] * vector[col];
        }
        output[row] = sum;
    }
}

class TestMulMv : public TestKernelBase {
private:
    id<MTLComputePipelineState> pipeline;
    const char* kernel_source_path;

public:
    TestMulMv() : pipeline(nil) {
        kernel_source_path = KERNEL_SOURCE_PATH;
    }

    ~TestMulMv() {
        if (pipeline) {
            [pipeline release];
        }
    }

    bool setup() {
        if (!init_device()) {
            return false;
        }

        // Read the standalone simple kernel source
        NSError* error = nil;
        NSString* kernel_path = [NSString stringWithUTF8String:KERNEL_SOURCE_PATH];
        NSString* kernel_src = [NSString stringWithContentsOfFile:kernel_path
                                                          encoding:NSUTF8StringEncoding
                                                             error:&error];
        if (error) {
            fprintf(stderr, "Failed to read kernel source: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Compile the source
        MTLCompileOptions* options = [MTLCompileOptions new];
        library = [device newLibraryWithSource:kernel_src options:options error:&error];
        [options release];

        if (error) {
            fprintf(stderr, "Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }

        // Get kernel function
        id<MTLFunction> function = [library newFunctionWithName:@"kernel_mul_mv_f32_f32_simple"];

        if (!function) {
            fprintf(stderr, "Failed to get kernel_mul_mv_f32_f32_simple function\n");
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
    bool test_case(const char* name,
                   const std::vector<float>& matrix,
                   const std::vector<float>& vector,
                   int ne00, int ne01) {
        printf("\n=== Test case: %s (ne00=%d cols, ne01=%d rows) ===\n", name, ne00, ne01);

        // Allocate output buffers
        std::vector<float> cpu_output(ne01);
        std::vector<float> gpu_output(ne01, 0.0f);

        // Run CPU reference
        cpu_mul_mv(matrix.data(), vector.data(), cpu_output.data(), ne00, ne01);

        // Setup Metal buffers
        KargsMulMv kargs = {0};
        kargs.ne00 = ne00;  // matrix width (columns)
        kargs.ne01 = ne01;  // matrix height (rows)
        kargs.ne02 = 1;     // no batch
        kargs.nb00 = sizeof(float);
        kargs.nb01 = ne00 * sizeof(float);  // row stride
        kargs.nb02 = ne00 * ne01 * sizeof(float);
        kargs.nb03 = kargs.nb02;

        kargs.ne10 = ne00;  // vector length (must match matrix width)
        kargs.ne11 = 1;     // single vector
        kargs.ne12 = 1;     // no batch
        kargs.nb10 = sizeof(float);
        kargs.nb11 = ne00 * sizeof(float);
        kargs.nb12 = kargs.nb11;
        kargs.nb13 = kargs.nb11;

        kargs.ne0 = ne01;   // output size = number of rows
        kargs.ne1 = 1;      // single output vector
        kargs.nr0 = 2;      // rows per threadgroup (not used in short variant)
        kargs.r2 = 1;       // no broadcast
        kargs.r3 = 1;       // no broadcast

        id<MTLBuffer> kargs_buf = create_buffer(sizeof(KargsMulMv), &kargs);
        id<MTLBuffer> matrix_buf = create_buffer(ne00 * ne01 * sizeof(float), matrix.data());
        id<MTLBuffer> vector_buf = create_buffer(ne00 * sizeof(float), vector.data());
        id<MTLBuffer> output_buf = create_buffer(ne01 * sizeof(float));

        if (!kargs_buf || !matrix_buf || !vector_buf || !output_buf) {
            fprintf(stderr, "Failed to create buffers\n");
            return false;
        }

        // Dispatch kernel
        // kernel_mul_mv_f32_f32_simple uses: tgpig.x*32 + tiisg for row index
        // So we need (ne01 + 31) / 32 threadgroups in x dimension
        MTLSize threads_per_tg = MTLSizeMake(32, 1, 1);
        MTLSize threadgroups = MTLSizeMake((ne01 + 31) / 32, 1, 1);

        std::vector<id<MTLBuffer>> buffers = {
            kargs_buf,
            matrix_buf,
            vector_buf,
            output_buf
        };

        bool dispatch_ok = dispatch(pipeline, buffers, threadgroups, threads_per_tg, 0);

        if (!dispatch_ok) {
            fprintf(stderr, "Dispatch failed\n");
            [kargs_buf release];
            [matrix_buf release];
            [vector_buf release];
            [output_buf release];
            return false;
        }

        // Read back GPU result
        memcpy(gpu_output.data(), [output_buf contents], ne01 * sizeof(float));

        // Compare
        bool match = compare(gpu_output.data(), cpu_output.data(), ne01, 1e-5f);

        // Cleanup
        [kargs_buf release];
        [matrix_buf release];
        [vector_buf release];
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

        // Test 1: Single element (1×1 matrix)
        {
            std::vector<float> matrix = {3.0f};
            std::vector<float> vector = {7.0f};
            if (!test_case("single_element_1x1", matrix, vector, 1, 1)) failed++;
        }

        // Test 2: Single row (1×n matrix = vector dot product)
        {
            std::vector<float> matrix = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> vector = {2.0f, 3.0f, 4.0f, 5.0f};
            // Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
            if (!test_case("single_row_1x4", matrix, vector, 4, 1)) failed++;
        }

        // Test 3: Single column (n×1 matrix)
        {
            std::vector<float> matrix = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> vector = {5.0f};
            // Expected: [1*5, 2*5, 3*5, 4*5] = [5, 10, 15, 20]
            if (!test_case("single_column_4x1", matrix, vector, 1, 4)) failed++;
        }

        // Test 4: Non-divisible by 4 width (ne00=7)
        {
            std::vector<float> matrix = {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f
            };
            std::vector<float> vector = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            // Expected: [28, 77] (sum of each row)
            if (!test_case("ne00_7_non_divisible", matrix, vector, 7, 2)) failed++;
        }

        // Test 5: All zeros matrix
        {
            std::vector<float> matrix(64, 0.0f);
            std::vector<float> vector(8, 1.0f);
            // Expected: all zeros
            if (!test_case("all_zeros_matrix", matrix, vector, 8, 8)) failed++;
        }

        // Test 6: All zeros vector
        {
            std::vector<float> matrix(64, 1.0f);
            std::vector<float> vector(8, 0.0f);
            // Expected: all zeros
            if (!test_case("all_zeros_vector", matrix, vector, 8, 8)) failed++;
        }

        // Test 7: Identity-like pattern (diagonal matrix approximation)
        {
            std::vector<float> matrix(16, 0.0f);
            matrix[0] = 1.0f;  // [0,0]
            matrix[5] = 1.0f;  // [1,1]
            matrix[10] = 1.0f; // [2,2]
            matrix[15] = 1.0f; // [3,3]
            std::vector<float> vector = {2.0f, 3.0f, 4.0f, 5.0f};
            // Expected: [2, 3, 4, 5] (diagonal extraction)
            if (!test_case("identity_like_4x4", matrix, vector, 4, 4)) failed++;
        }

        // Test 8: Contains +INF
        {
            std::vector<float> matrix = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> vector = {std::numeric_limits<float>::infinity(), 1.0f, 1.0f, 1.0f};
            // Expected: [INF, INF] (INF + finite = INF)
            if (!test_case("contains_pos_inf", matrix, vector, 4, 1)) failed++;
        }

        // Test 9: Contains -INF
        {
            std::vector<float> matrix = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> vector = {-std::numeric_limits<float>::infinity(), 1.0f, 1.0f, 1.0f};
            // Expected: [-INF, -INF]
            if (!test_case("contains_neg_inf", matrix, vector, 4, 1)) failed++;
        }

        // Test 10: Contains NaN
        {
            std::vector<float> matrix = {1.0f, 2.0f, 3.0f, 4.0f};
            std::vector<float> vector = {std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 1.0f};
            // Expected: [NaN, NaN] (NaN propagates)
            if (!test_case("contains_nan", matrix, vector, 4, 1)) failed++;
        }

        // Test 11: Very large values (test accumulation overflow)
        {
            std::vector<float> matrix = {
                1e20f, 1e20f, 1e20f, 1e20f,
                1e20f, 1e20f, 1e20f, 1e20f
            };
            std::vector<float> vector = {1.0f, 1.0f, 1.0f, 1.0f};
            // Expected: [4e20, 4e20]
            if (!test_case("very_large_values", matrix, vector, 4, 2)) failed++;
        }

        // Test 12: ne00=32 (one simdgroup width)
        {
            std::vector<float> matrix(32 * 4);
            std::vector<float> vector(32);
            for (int i = 0; i < 32; i++) {
                vector[i] = (float)(i + 1);
                for (int row = 0; row < 4; row++) {
                    matrix[row * 32 + i] = (float)(row + 1);
                }
            }
            // Each row has same values [1,1,...,1] or [2,2,...,2] etc
            // Row sum = row_num * sum(vector) = row_num * (1+2+...+32) = row_num * 528
            if (!test_case("ne00_32_one_simdgroup", matrix, vector, 32, 4)) failed++;
        }

        // Test 13: ne00=64 (vec4 path potential)
        {
            std::vector<float> matrix(64 * 8);
            std::vector<float> vector(64);
            for (int i = 0; i < 64; i++) {
                vector[i] = sinf((float)i * 0.1f);
                for (int row = 0; row < 8; row++) {
                    matrix[row * 64 + i] = cosf((float)(row * 64 + i) * 0.1f);
                }
            }
            if (!test_case("ne00_64_vec4", matrix, vector, 64, 8)) failed++;
        }

        // Test 14: ne00=256 (larger)
        {
            std::vector<float> matrix(256 * 16);
            std::vector<float> vector(256);
            for (int i = 0; i < 256; i++) {
                vector[i] = (float)(i % 13) / 13.0f;
                for (int row = 0; row < 16; row++) {
                    matrix[row * 256 + i] = (float)((row * 256 + i) % 17) / 17.0f;
                }
            }
            if (!test_case("ne00_256_large", matrix, vector, 256, 16)) failed++;
        }

        // Test 15: ne01=64 (many rows, tests multiple threadgroups)
        {
            std::vector<float> matrix(32 * 64);
            std::vector<float> vector(32);
            for (int i = 0; i < 32; i++) {
                vector[i] = 1.0f;
            }
            for (int row = 0; row < 64; row++) {
                for (int col = 0; col < 32; col++) {
                    matrix[row * 32 + col] = (float)row;
                }
            }
            // Each row sums to row_num * 32
            if (!test_case("ne01_64_many_rows", matrix, vector, 32, 64)) failed++;
        }

        // Test 16: Mixed positive and negative
        {
            std::vector<float> matrix = {
                1.0f, -2.0f, 3.0f, -4.0f,
                -5.0f, 6.0f, -7.0f, 8.0f,
                9.0f, -10.0f, 11.0f, -12.0f
            };
            std::vector<float> vector = {2.0f, -1.0f, 3.0f, -2.0f};
            // Row 0: 1*2 + (-2)*(-1) + 3*3 + (-4)*(-2) = 2 + 2 + 9 + 8 = 21
            // Row 1: (-5)*2 + 6*(-1) + (-7)*3 + 8*(-2) = -10 - 6 - 21 - 16 = -53
            // Row 2: 9*2 + (-10)*(-1) + 11*3 + (-12)*(-2) = 18 + 10 + 33 + 24 = 85
            if (!test_case("mixed_pos_neg", matrix, vector, 4, 3)) failed++;
        }

        printf("\n=== Summary ===\n");
        printf("Total tests: 16\n");
        printf("Failed: %d\n", failed);
        printf("Passed: %d\n", 16 - failed);

        return failed;
    }
};

int main(int argc, char** argv) {
    TestMulMv test;

    if (!test.setup()) {
        fprintf(stderr, "Test setup failed\n");
        return 1;
    }

    int failed = test.run_tests();

    return failed > 0 ? 1 : 0;
}
