// Production-policy test for ggml-metal-command-buffer-error.m: fake NSObject buffers/queues, no MTLDevice,
// no queue creation, no submission. Links the real error module; asserts stay active under NDEBUG.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <signal.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

#import "ggml-metal.h"
#import "ggml-metal-command-buffer-error.h"

#define TEST_ASSERT(condition) do { if (!(condition)) { fprintf(stderr, "assertion failed: %s:%d: %s\n", __FILE__, __LINE__, #condition); exit(1); } } while (0)

static int g_dealloc;

@interface FakeInfo : NSObject
@property(nonatomic, copy) NSString * fakeLabel;
@property(nonatomic) long fakeState;
@property(nonatomic, retain) NSArray * fakeSignposts;
@end
@implementation FakeInfo
- (NSString *)label { return self.fakeLabel; }
- (MTLCommandEncoderErrorState)errorState { return (MTLCommandEncoderErrorState) self.fakeState; }
- (NSArray *)debugSignposts { return self.fakeSignposts; }
- (void)dealloc { [_fakeLabel release]; [_fakeSignposts release]; [super dealloc]; }
@end

@interface FakeCommandBuffer : NSObject
@property(nonatomic, retain) NSError * fakeError;
@property(nonatomic) MTLCommandBufferStatus fakeStatus;
@property(nonatomic) MTLCommandBufferStatus statusAfterWait;
@property(nonatomic) int waits;
@end
@implementation FakeCommandBuffer
- (NSError *)error { return self.fakeError; }
- (MTLCommandBufferStatus)status { return self.fakeStatus; }
- (void)waitUntilCompleted { self.waits++; self.fakeStatus = self.statusAfterWait; }
- (void)dealloc { g_dealloc++; [_fakeError release]; [super dealloc]; }
@end

@interface FakeQueue : NSObject
@property(nonatomic) int plainCalls;
@property(nonatomic) int unretainedCalls;
@property(nonatomic) int descriptorCalls;
@property(nonatomic) BOOL retainedReferences;
@property(nonatomic) MTLCommandBufferErrorOption errorOptions;
@end
@implementation FakeQueue
- (id)commandBuffer { self.plainCalls++; return [[[FakeCommandBuffer alloc] init] autorelease]; }
- (id)commandBufferWithUnretainedReferences { self.unretainedCalls++; return [[[FakeCommandBuffer alloc] init] autorelease]; }
- (id)commandBufferWithDescriptor:(MTLCommandBufferDescriptor *)descriptor {
    self.descriptorCalls++;
    self.retainedReferences = descriptor.retainedReferences;
    self.errorOptions = descriptor.errorOptions;
    return [[[FakeCommandBuffer alloc] init] autorelease];
}
@end

// Callback record deep-copies the borrowed payload (the consumer contract) so it can be read after the pool drains.
struct capture {
    int calls;
    int64_t code;
    char description[64];
    size_t n_encoders;
    char label[64], state[16], signposts[64];
};
static void capture_error(const struct ggml_backend_metal_gpu_error * error, void * raw) {
    struct capture * capture = raw;
    capture->calls++;
    capture->code = error->code;
    snprintf(capture->description, sizeof(capture->description), "%s", error->description);
    capture->n_encoders = error->n_encoders;
    if (error->n_encoders) {
        snprintf(capture->label, sizeof(capture->label), "%s", error->encoders[0].label);
        snprintf(capture->state, sizeof(capture->state), "%s", error->encoders[0].state);
        snprintf(capture->signposts, sizeof(capture->signposts), "%s", error->encoders[0].signposts);
    }
}


// Returns a +1 (owned) fake buffer.
static FakeCommandBuffer * make_cb(MTLCommandBufferStatus status, MTLCommandBufferStatus after_wait, long code) {
    FakeCommandBuffer * cb = [[FakeCommandBuffer alloc] init];
    cb.fakeStatus = status;
    cb.statusAfterWait = after_wait;
    if (after_wait == MTLCommandBufferStatusError || status == MTLCommandBufferStatusError) {
        cb.fakeError = [NSError errorWithDomain:@"fake" code:code userInfo:@{ NSLocalizedDescriptionKey: @"fake failure" }];
    }
    return cb;
}

// Run block in a forked child; return the raw wait status.
static int run_child(void (^body)(void)) {
    fflush(NULL);
    pid_t pid = fork();
    TEST_ASSERT(pid >= 0);
    if (pid == 0) { body(); _exit(0); }
    int st = 0;
    TEST_ASSERT(waitpid(pid, &st, 0) == pid);
    return st;
}

int main(void) {
    setenv("GGML_NO_BACKTRACE", "1", 1);
    @autoreleasepool {
        FakeQueue * queue = [[[FakeQueue alloc] init] autorelease];
        struct capture capture = {0};

        // factory: unset -> original calls; set -> descriptor with error options and the original reference mode; clear -> original again
        ggml_backend_metal_set_gpu_error_callback(NULL, NULL);
        (void) ggml_metal_new_command_buffer((id<MTLCommandQueue>) queue, false);
        (void) ggml_metal_new_command_buffer((id<MTLCommandQueue>) queue, true);
        TEST_ASSERT(queue.plainCalls == 1 && queue.unretainedCalls == 1 && queue.descriptorCalls == 0);
        ggml_backend_metal_set_gpu_error_callback(capture_error, &capture);
        (void) ggml_metal_new_command_buffer((id<MTLCommandQueue>) queue, true);
        TEST_ASSERT(queue.descriptorCalls == 1 && !queue.retainedReferences);
        TEST_ASSERT(queue.errorOptions == MTLCommandBufferErrorOptionEncoderExecutionStatus);
        (void) ggml_metal_new_command_buffer((id<MTLCommandQueue>) queue, false);
        TEST_ASSERT(queue.descriptorCalls == 2 && queue.retainedReferences);
        ggml_backend_metal_set_gpu_error_callback(NULL, NULL);
        (void) ggml_metal_new_command_buffer((id<MTLCommandQueue>) queue, false);
        TEST_ASSERT(queue.plainCalls == 2 && queue.descriptorCalls == 2);

        // completed buffer: SUCCESS, never a callback
        ggml_backend_metal_set_gpu_error_callback(capture_error, &capture);
        FakeCommandBuffer * ok = make_cb(MTLCommandBufferStatusCompleted, MTLCommandBufferStatusCompleted, 0);
        TEST_ASSERT(ggml_metal_check_command_buffer((id<MTLCommandBuffer>) ok, 0, false, true) == GGML_STATUS_SUCCESS);
        TEST_ASSERT(capture.calls == 0);
        [ok release];

        // failed + hook: production R2 reports copied encoder metadata and returns FAILED (never aborts)
        @autoreleasepool {
            FakeCommandBuffer * failed = make_cb(MTLCommandBufferStatusError, MTLCommandBufferStatusError, 77);
            FakeInfo * info = [[[FakeInfo alloc] init] autorelease];
            info.fakeLabel = @"kernel_mul_mv";
            info.fakeState = 3;
            info.fakeSignposts = @[ @"a", @"b" ];
            failed.fakeError = [NSError errorWithDomain:@"fake" code:77 userInfo:@{
                NSLocalizedDescriptionKey: @"fake failure", MTLCommandBufferEncoderInfoErrorKey: @[ info ] }];
            TEST_ASSERT(ggml_metal_check_command_buffer((id<MTLCommandBuffer>) failed, 0, false, true) == GGML_STATUS_FAILED);
            [failed release];
        }
        TEST_ASSERT(capture.calls == 1 && capture.code == 77 && strcmp(capture.description, "fake failure") == 0);
        TEST_ASSERT(capture.n_encoders == 1 && strcmp(capture.label, "kernel_mul_mv") == 0);
        TEST_ASSERT(strcmp(capture.state, "3") == 0 && strcmp(capture.signposts, "a | b") == 0);

        // missing NSError: explicit code 0 / empty encoders
        FakeCommandBuffer * missing = make_cb(MTLCommandBufferStatusError, MTLCommandBufferStatusError, 0);
        missing.fakeError = nil;
        TEST_ASSERT(ggml_metal_check_command_buffer((id<MTLCommandBuffer>) missing, 1, true, true) == GGML_STATUS_FAILED);
        TEST_ASSERT(capture.calls == 2 && capture.code == 0 && capture.n_encoders == 0);
        TEST_ASSERT(strcmp(capture.description, "missing NSError") == 0);
        [missing release];

        // drain: hook installed, mixed main + extras. Committed/Scheduled are waited, NotEnqueued never is.
        const int dealloc0 = g_dealloc;
        capture.calls = 0;
        struct ggml_metal_command_buffer main[4] = {{0}};
        FakeCommandBuffer * m0 = make_cb(MTLCommandBufferStatusCompleted, MTLCommandBufferStatusCompleted, 0);
        FakeCommandBuffer * m1 = make_cb(MTLCommandBufferStatusCommitted, MTLCommandBufferStatusError, 11);
        FakeCommandBuffer * m2 = make_cb(MTLCommandBufferStatusNotEnqueued, MTLCommandBufferStatusCompleted, 0);
        main[0].obj = (id<MTLCommandBuffer>) m0; main[1].obj = (id<MTLCommandBuffer>) m1; main[2].obj = (id<MTLCommandBuffer>) m2;
        NSMutableArray * extras = [[NSMutableArray alloc] init];
        FakeCommandBuffer * e0 = make_cb(MTLCommandBufferStatusCommitted, MTLCommandBufferStatusCompleted, 0);
        FakeCommandBuffer * e1 = make_cb(MTLCommandBufferStatusScheduled, MTLCommandBufferStatusError, 22);
        // array retain + manual retain, as the context does; then drop the creator's reference
        [extras addObject:e0]; [e0 retain]; [e0 release];
        [extras addObject:e1]; [e1 retain]; [e1 release];
        TEST_ASSERT(ggml_metal_complete_command_buffers(main, 4, extras) == GGML_STATUS_FAILED);
        // m1, m2 (NotEnqueued is not Completed) and e1 reported; every pending one waited exactly as the policy says
        TEST_ASSERT(capture.calls == 3);
        TEST_ASSERT(g_dealloc - dealloc0 == 5); // 3 main + 2 extras released exactly once
        TEST_ASSERT(extras.count == 0);
        for (int i = 0; i < 4; ++i) TEST_ASSERT(main[i].obj == nil);

        // stale-state control: failed -> hook cleared -> second drain finds nothing and must not abort or report
        ggml_backend_metal_set_gpu_error_callback(NULL, NULL);
        TEST_ASSERT(ggml_metal_complete_command_buffers(main, 4, extras) == GGML_STATUS_SUCCESS);
        TEST_ASSERT(capture.calls == 3);

        // wait counters through a second drain with explicit objects we keep alive
        ggml_backend_metal_set_gpu_error_callback(capture_error, &capture);
        FakeCommandBuffer * w_commit = make_cb(MTLCommandBufferStatusCommitted, MTLCommandBufferStatusCompleted, 0);
        FakeCommandBuffer * w_new = make_cb(MTLCommandBufferStatusNotEnqueued, MTLCommandBufferStatusNotEnqueued, 0);
        [w_new setFakeError:nil];
        main[0].obj = (id<MTLCommandBuffer>) [w_commit retain];
        main[1].obj = (id<MTLCommandBuffer>) [w_new retain];
        capture.calls = 0;
        TEST_ASSERT(ggml_metal_complete_command_buffers(main, 2, extras) == GGML_STATUS_FAILED); // NotEnqueued != Completed
        TEST_ASSERT(w_commit.waits == 1 && w_new.waits == 0 && capture.calls == 1);
        [w_commit release]; [w_new release];

        // success with hook: main slots stay (borrowed), extras released/cleared
        FakeCommandBuffer * s_main = make_cb(MTLCommandBufferStatusCompleted, MTLCommandBufferStatusCompleted, 0);
        FakeCommandBuffer * s_ext = make_cb(MTLCommandBufferStatusCompleted, MTLCommandBufferStatusCompleted, 0);
        main[0].obj = (id<MTLCommandBuffer>) s_main; main[1].obj = nil;
        [extras addObject:s_ext]; [s_ext retain]; [s_ext release];
        const int dealloc1 = g_dealloc;
        capture.calls = 0;
        TEST_ASSERT(ggml_metal_complete_command_buffers(main, 2, extras) == GGML_STATUS_SUCCESS);
        TEST_ASSERT(capture.calls == 0 && main[0].obj == (id<MTLCommandBuffer>) s_main && extras.count == 0);
        TEST_ASSERT(g_dealloc - dealloc1 == 1);
        [s_main release]; main[0].obj = nil;
        [extras release];

        // graph-failure cleanup (profile/capture): other pending buffers are waited BEFORE their references drop, no re-report
        ggml_backend_metal_set_gpu_error_callback(capture_error, &capture);
        capture.calls = 0;
        FakeCommandBuffer * p_main = make_cb(MTLCommandBufferStatusCommitted, MTLCommandBufferStatusCompleted, 0);
        FakeCommandBuffer * p_ext = make_cb(MTLCommandBufferStatusScheduled, MTLCommandBufferStatusCompleted, 0);
        NSMutableArray * p_extras = [[NSMutableArray alloc] init];
        [p_extras addObject:p_ext]; [p_ext retain]; // array + manual retain; the creator's reference is kept to read the counters
        main[0].obj = (id<MTLCommandBuffer>) [p_main retain]; // slot reference; creator's kept likewise
        const int dealloc2 = g_dealloc;
        ggml_metal_discard_command_buffers(main, 2, p_extras);
        TEST_ASSERT(p_main.waits == 1 && p_ext.waits == 1 && g_dealloc == dealloc2); // waited, still alive: waits precede dealloc
        TEST_ASSERT(main[0].obj == nil && p_extras.count == 0 && capture.calls == 0);
        [p_main release]; [p_ext release]; [p_extras release];
        TEST_ASSERT(g_dealloc - dealloc2 == 2);

        // no hook: nonfatal failure is FAILED with no callback; fatal failure aborts (child, raw status kept)
        ggml_backend_metal_set_gpu_error_callback(NULL, NULL);
        capture.calls = 0;
        FakeCommandBuffer * bad = make_cb(MTLCommandBufferStatusError, MTLCommandBufferStatusError, 5);
        TEST_ASSERT(ggml_metal_check_command_buffer((id<MTLCommandBuffer>) bad, 0, false, false) == GGML_STATUS_FAILED);
        TEST_ASSERT(capture.calls == 0);
        int st = run_child(^{ (void) ggml_metal_check_command_buffer((id<MTLCommandBuffer>) bad, 0, false, true); });
        TEST_ASSERT(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT);
        st = run_child(^{
            struct ggml_metal_command_buffer one[1] = {{ (id<MTLCommandBuffer>) bad }};
            NSMutableArray * none = [[NSMutableArray alloc] init];
            (void) ggml_metal_complete_command_buffers(one, 1, none);
        });
        TEST_ASSERT(WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT);
        [bad release];
    }
    printf("ok\n");
    return 0;
}
