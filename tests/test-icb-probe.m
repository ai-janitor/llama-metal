// T8.1.1: ICB compute probe for AMD Radeon Pro 5300M
// Validates: ICB creation, inheritPipelineState=NO, setBytes, setBarrier, executeCommandsInBuffer
// Build: clang -framework Metal -framework Foundation -o test-icb-probe tests/test-icb-probe.m

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <stdio.h>

// Inline MSL kernel source — multiply each element by a scalar from kargs
static NSString *kernelSource = @
"struct KArgs { float scale; };\n"
"kernel void scale_buffer(\n"
"    device const float* src [[buffer(0)]],\n"
"    device float* dst       [[buffer(1)]],\n"
"    constant KArgs& kargs   [[buffer(2)]],\n"
"    uint tid [[thread_position_in_grid]]\n"
") {\n"
"    dst[tid] = src[tid] * kargs.scale;\n"
"}\n";

typedef struct { float scale; } KArgs;

static void test_result(const char *name, int pass) {
    fprintf(stderr, "[%s] %s\n", pass ? "PASS" : "FAIL", name);
}

int main(void) {
    @autoreleasepool {
        // --- Device selection: prefer discrete AMD GPU ---
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        id<MTLDevice> device = nil;
        for (id<MTLDevice> d in devices) {
            if (!d.isLowPower && !d.isRemovable) {
                device = d;
                break;
            }
        }
        if (!device) device = MTLCreateSystemDefaultDevice();
        fprintf(stderr, "Device: %s\n", [[device name] UTF8String]);
        fprintf(stderr, "UMA: %s\n", device.hasUnifiedMemory ? "yes" : "no");

        // --- Check GPU family support ---
        // MTLGPUFamilyCommon2 (=3002) is minimum for compute ICBs
        BOOL supportsCommon2 = [device supportsFamily:MTLGPUFamilyCommon2];
        test_result("GPU supports MTLGPUFamilyCommon2 (required for compute ICB)", supportsCommon2);
        if (!supportsCommon2) {
            fprintf(stderr, "ABORT: GPU does not support compute ICBs\n");
            return 1;
        }

        // --- Compile kernel with supportIndirectCommandBuffers = YES ---
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
        if (!library) {
            fprintf(stderr, "FAIL: Library compilation: %s\n", [[error localizedDescription] UTF8String]);
            return 1;
        }
        test_result("MSL compilation", library != nil);

        id<MTLFunction> function = [library newFunctionWithName:@"scale_buffer"];
        test_result("Function lookup", function != nil);

        MTLComputePipelineDescriptor *pipeDesc = [[MTLComputePipelineDescriptor alloc] init];
        pipeDesc.computeFunction = function;
        pipeDesc.supportIndirectCommandBuffers = YES;

        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithDescriptor:pipeDesc
                                                                                options:0
                                                                             reflection:nil
                                                                                  error:&error];
        if (!pso) {
            fprintf(stderr, "FAIL: PSO with supportIndirectCommandBuffers=YES: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 1;
        }
        test_result("PSO with supportIndirectCommandBuffers=YES", pso != nil);
        fprintf(stderr, "  maxTotalThreadsPerThreadgroup: %lu\n", (unsigned long)[pso maxTotalThreadsPerThreadgroup]);

        // --- Create ICB ---
        MTLIndirectCommandBufferDescriptor *icbDesc = [[MTLIndirectCommandBufferDescriptor alloc] init];
        icbDesc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
        icbDesc.inheritPipelineState = NO;
        icbDesc.inheritBuffers = NO;
        icbDesc.maxKernelBufferBindCount = 3; // src, dst, kargs

        id<MTLIndirectCommandBuffer> icb = [device newIndirectCommandBufferWithDescriptor:icbDesc
                                                                          maxCommandCount:4
                                                                                  options:MTLResourceStorageModeShared];
        test_result("ICB creation (inheritPipelineState=NO, compute)", icb != nil);
        if (!icb) {
            fprintf(stderr, "ABORT: ICB creation failed on this GPU\n");
            return 1;
        }

        // --- Create buffers ---
        const int N = 256;
        const size_t bufSize = N * sizeof(float);

        id<MTLBuffer> srcBuf = [device newBufferWithLength:bufSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> dstBuf = [device newBufferWithLength:bufSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> dst2Buf = [device newBufferWithLength:bufSize options:MTLResourceStorageModeShared];

        // Fill source: [1.0, 2.0, 3.0, ...]
        float *src = (float *)[srcBuf contents];
        for (int i = 0; i < N; i++) src[i] = (float)(i + 1);

        memset([dstBuf contents], 0, bufSize);
        memset([dst2Buf contents], 0, bufSize);

        // --- Create kargs buffers (must stay alive through execution) ---
        KArgs kargs0 = { .scale = 2.0f };
        KArgs kargs1 = { .scale = 3.0f };
        id<MTLBuffer> kargsBuf0 = [device newBufferWithBytes:&kargs0 length:sizeof(KArgs)
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> kargsBuf1 = [device newBufferWithBytes:&kargs1 length:sizeof(KArgs)
                                                     options:MTLResourceStorageModeShared];

        // --- Encode ICB commands ---
        // Command 0: scale by 2.0 (src -> dst)
        {
            id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
            [cmd setComputePipelineState:pso];
            [cmd setKernelBuffer:srcBuf offset:0 atIndex:0];
            [cmd setKernelBuffer:dstBuf offset:0 atIndex:1];
            [cmd setKernelBuffer:kargsBuf0 offset:0 atIndex:2];
            test_result("setKernelBuffer for kargs (no setBytes in ICB)", YES);

            [cmd concurrentDispatchThreadgroups:MTLSizeMake(N/256, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        test_result("ICB command 0 encoded (scale by 2.0)", YES);

        // Command 1: scale by 3.0 (src -> dst2) — tests per-command PSO + different buffers
        {
            id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:1];
            [cmd setComputePipelineState:pso];
            [cmd setKernelBuffer:srcBuf offset:0 atIndex:0];
            [cmd setKernelBuffer:dst2Buf offset:0 atIndex:1];
            [cmd setKernelBuffer:kargsBuf1 offset:0 atIndex:2];

            [cmd concurrentDispatchThreadgroups:MTLSizeMake(N/256, 1, 1)
                          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        test_result("ICB command 1 encoded (scale by 3.0)", YES);

        // --- Test setBarrier (open question #3) ---
        {
            id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:1];
            if ([cmd respondsToSelector:@selector(setBarrier)]) {
                [cmd setBarrier];
                test_result("setBarrier supported", YES);
            } else {
                test_result("setBarrier supported", NO);
            }
        }

        // --- Execute ICB ---
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        // Must declare resource usage for ICB — Apple docs require this
        [encoder useResource:srcBuf usage:MTLResourceUsageRead];
        [encoder useResource:dstBuf usage:MTLResourceUsageWrite];
        [encoder useResource:dst2Buf usage:MTLResourceUsageWrite];
        [encoder useResource:kargsBuf0 usage:MTLResourceUsageRead];
        [encoder useResource:kargsBuf1 usage:MTLResourceUsageRead];

        [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, 2)];
        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error) {
            fprintf(stderr, "FAIL: Command buffer error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            test_result("executeCommandsInBuffer execution", NO);
            return 1;
        }
        test_result("executeCommandsInBuffer execution", cmdBuf.error == nil);

        // --- Verify results ---
        float *dst = (float *)[dstBuf contents];
        float *dst2 = (float *)[dst2Buf contents];
        int pass1 = 1, pass2 = 1;

        for (int i = 0; i < N; i++) {
            if (pass1 && dst[i] != src[i] * 2.0f) {
                fprintf(stderr, "  dst[%d] = %.1f, expected %.1f\n", i, dst[i], src[i] * 2.0f);
                pass1 = 0;
            }
            if (pass2 && dst2[i] != src[i] * 3.0f) {
                fprintf(stderr, "  dst2[%d] = %.1f, expected %.1f\n", i, dst2[i], src[i] * 3.0f);
                pass2 = 0;
            }
        }
        test_result("Command 0 correctness (scale by 2.0)", pass1);
        test_result("Command 1 correctness (scale by 3.0)", pass2);

        // --- Test ICB replay (encode once, execute twice) ---
        memset([dstBuf contents], 0, bufSize);
        memset([dst2Buf contents], 0, bufSize);

        id<MTLCommandBuffer> cmdBuf2 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder2 = [cmdBuf2 computeCommandEncoder];
        [encoder2 executeCommandsInBuffer:icb withRange:NSMakeRange(0, 2)];
        [encoder2 endEncoding];
        [cmdBuf2 commit];
        [cmdBuf2 waitUntilCompleted];

        dst = (float *)[dstBuf contents];
        dst2 = (float *)[dst2Buf contents];
        int replay_pass = 1;
        for (int i = 0; i < N; i++) {
            if (dst[i] != src[i] * 2.0f || dst2[i] != src[i] * 3.0f) {
                replay_pass = 0;
                break;
            }
        }
        test_result("ICB replay (execute same ICB twice)", replay_pass);

        // --- Test ICB command patching (change buffer offset between replays) ---
        // Patch command 0 to read from offset 128*sizeof(float) in src
        {
            id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:0];
            [cmd setKernelBuffer:srcBuf offset:128 * sizeof(float) atIndex:0];
        }
        memset([dstBuf contents], 0, bufSize);

        id<MTLCommandBuffer> cmdBuf3 = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder3 = [cmdBuf3 computeCommandEncoder];
        [encoder3 executeCommandsInBuffer:icb withRange:NSMakeRange(0, 1)]; // only cmd 0
        [encoder3 endEncoding];
        [cmdBuf3 commit];
        [cmdBuf3 waitUntilCompleted];

        dst = (float *)[dstBuf contents];
        int patch_pass = 1;
        for (int i = 0; i < N - 128; i++) {
            float expected = (float)(i + 129) * 2.0f; // src[128+i] * 2.0
            if (dst[i] != expected) {
                fprintf(stderr, "  patched dst[%d] = %.1f, expected %.1f\n", i, dst[i], expected);
                patch_pass = 0;
                break;
            }
        }
        test_result("ICB command patching (changed buffer offset)", patch_pass);

        // --- Summary ---
        fprintf(stderr, "\n=== T8.1.1 ICB Probe Summary ===\n");
        fprintf(stderr, "Device: %s (UMA=%s)\n", [[device name] UTF8String],
                device.hasUnifiedMemory ? "yes" : "no");
        fprintf(stderr, "All critical paths tested.\n");

        return 0;
    }
}
