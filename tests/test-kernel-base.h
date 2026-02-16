#ifndef TEST_KERNEL_BASE_H
#define TEST_KERNEL_BASE_H

// Metal kernel unit test base class - Phase 1 of TEST-001
// Raw Metal API, NO ggml dependency. Tests standalone kernel correctness.

#include <Metal/Metal.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

class TestKernelBase {
protected:
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;

public:
    TestKernelBase() : device(nil), command_queue(nil), library(nil) {}

    virtual ~TestKernelBase() {
        if (command_queue) {
            [command_queue release];
            command_queue = nil;
        }
        if (library) {
            [library release];
            library = nil;
        }
        if (device) {
            [device release];
            device = nil;
        }
    }

    // Initialize Metal device and command queue
    bool init_device(const char* device_name = nullptr) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();

        if (device_name) {
            // Find device by name
            for (id<MTLDevice> dev in devices) {
                if (strcmp([dev.name UTF8String], device_name) == 0) {
                    device = [dev retain];
                    break;
                }
            }
        } else {
            // Use default device
            device = MTLCreateSystemDefaultDevice();
        }

        [devices release];

        if (!device) {
            fprintf(stderr, "Failed to create Metal device\n");
            return false;
        }

        printf("Using Metal device: %s\n", [device.name UTF8String]);

        command_queue = [device newCommandQueue];
        if (!command_queue) {
            fprintf(stderr, "Failed to create command queue\n");
            return false;
        }

        return true;
    }

    // Compile Metal shader from source file and get pipeline state
    id<MTLComputePipelineState> compile_kernel(
        const char* metal_source_path,
        const char* function_name,
        MTLFunctionConstantValues* constants = nullptr,
        const char* include_directory = nullptr)
    {
        NSError* error = nil;

        // Read Metal source file
        NSString* path = [NSString stringWithUTF8String:metal_source_path];
        NSString* source = [NSString stringWithContentsOfFile:path
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (error) {
            fprintf(stderr, "Failed to read Metal source from %s: %s\n",
                    metal_source_path, [[error localizedDescription] UTF8String]);
            return nil;
        }

        // Compile library
        MTLCompileOptions* options = [MTLCompileOptions new];

        // Set include path if provided
        if (include_directory) {
            NSString* inc_dir = [NSString stringWithUTF8String:include_directory];
            options.preprocessorMacros = @{};
            // Metal compiler will search in the same directory as the source file
            // No need to set additional include paths for relative includes
        }

        library = [device newLibraryWithSource:source options:options error:&error];
        [options release];

        if (error) {
            fprintf(stderr, "Failed to compile Metal library: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nil;
        }

        // Get function
        NSString* func_name = [NSString stringWithUTF8String:function_name];
        id<MTLFunction> function;

        if (constants) {
            function = [library newFunctionWithName:func_name
                                     constantValues:constants
                                              error:&error];
        } else {
            function = [library newFunctionWithName:func_name];
        }

        if (!function) {
            fprintf(stderr, "Failed to get Metal function '%s': %s\n",
                    function_name,
                    error ? [[error localizedDescription] UTF8String] : "unknown");
            return nil;
        }

        // Create pipeline state
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                      error:&error];
        [function release];

        if (error) {
            fprintf(stderr, "Failed to create pipeline state: %s\n",
                    [[error localizedDescription] UTF8String]);
            return nil;
        }

        return pipeline;
    }

    // Create buffer with StorageModeShared (CPU and GPU accessible)
    id<MTLBuffer> create_buffer(size_t size, const void* data = nullptr) {
        id<MTLBuffer> buffer;

        if (data) {
            buffer = [device newBufferWithBytes:data
                                         length:size
                                        options:MTLResourceStorageModeShared];
        } else {
            buffer = [device newBufferWithLength:size
                                         options:MTLResourceStorageModeShared];
        }

        if (!buffer) {
            fprintf(stderr, "Failed to create buffer of size %zu\n", size);
        }

        return buffer;
    }

    // Dispatch kernel and wait for completion
    bool dispatch(
        id<MTLComputePipelineState> pipeline,
        const std::vector<id<MTLBuffer>>& buffers,
        MTLSize threadgroups,
        MTLSize threads_per_threadgroup,
        size_t threadgroup_memory_length = 0)
    {
        id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
        if (!command_buffer) {
            fprintf(stderr, "Failed to create command buffer\n");
            return false;
        }

        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (!encoder) {
            fprintf(stderr, "Failed to create compute encoder\n");
            return false;
        }

        [encoder setComputePipelineState:pipeline];

        // Bind buffers
        for (size_t i = 0; i < buffers.size(); i++) {
            [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        }

        // Set threadgroup memory if needed
        if (threadgroup_memory_length > 0) {
            [encoder setThreadgroupMemoryLength:threadgroup_memory_length atIndex:0];
        }

        [encoder dispatchThreadgroups:threadgroups
                threadsPerThreadgroup:threads_per_threadgroup];

        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            fprintf(stderr, "Command buffer execution failed: %s\n",
                    [[command_buffer.error localizedDescription] UTF8String]);
            return false;
        }

        return true;
    }

    // Element-wise comparison of float arrays
    // Returns true if all elements match within tolerance
    // Reports first mismatch location
    bool compare(
        const float* gpu_output,
        const float* cpu_expected,
        size_t n,
        float tolerance = 1e-5f)
    {
        bool all_match = true;
        size_t first_mismatch = 0;
        float max_diff = 0.0f;

        for (size_t i = 0; i < n; i++) {
            float gpu_val = gpu_output[i];
            float cpu_val = cpu_expected[i];

            // Handle NaN
            if (std::isnan(gpu_val) && std::isnan(cpu_val)) {
                continue;  // Both NaN is a match
            }
            if (std::isnan(gpu_val) || std::isnan(cpu_val)) {
                if (all_match) {
                    first_mismatch = i;
                    all_match = false;
                }
                fprintf(stderr, "MISMATCH at [%zu]: GPU=%f CPU=%f (NaN mismatch)\n",
                        i, gpu_val, cpu_val);
                continue;
            }

            // Handle infinity
            if (std::isinf(gpu_val) && std::isinf(cpu_val)) {
                if ((gpu_val > 0) == (cpu_val > 0)) {
                    continue;  // Both +INF or both -INF is a match
                }
            }

            float diff = std::fabs(gpu_val - cpu_val);
            max_diff = std::fmax(max_diff, diff);

            if (diff > tolerance) {
                if (all_match) {
                    first_mismatch = i;
                    all_match = false;
                }
                fprintf(stderr, "MISMATCH at [%zu]: GPU=%f CPU=%f (diff=%f, tol=%f)\n",
                        i, gpu_val, cpu_val, diff, tolerance);
            }
        }

        if (all_match) {
            printf("PASS: All %zu elements match (max_diff=%e, tol=%e)\n",
                   n, max_diff, tolerance);
        } else {
            printf("FAIL: First mismatch at element %zu\n", first_mismatch);
        }

        return all_match;
    }

    // Run all tests (implemented by derived class)
    virtual int run_tests() = 0;
};

#endif // TEST_KERNEL_BASE_H
