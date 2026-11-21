// Names and function constants of every function of a compiled Metal library, one sorted line per function:
// <function name> TAB <its constants as name:type:index:required, sorted by name, comma separated, - for none>.
// Run: metal_library_abi <path to .metallib>. Exit 0 when the library loaded, 1 otherwise.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>

int main(int argc, const char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: metal_library_abi <path to .metallib>\n");
        return 1;
    }
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev == nil) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        NSError * err = nil;
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> lib = [dev newLibraryWithURL:url error:&err];
        if (lib == nil) {
            fprintf(stderr, "cannot load %s: %s\n", argv[1], err ? [[err localizedDescription] UTF8String] : "unknown error");
            return 1;
        }
        NSArray<NSString *> * names = [[lib functionNames] sortedArrayUsingSelector:@selector(compare:)];
        for (NSString * name in names) {
            id<MTLFunction> fn = [lib newFunctionWithName:name];
            if (fn == nil) {
                fprintf(stderr, "no function %s\n", [name UTF8String]);
                return 1;
            }
            NSDictionary<NSString *, MTLFunctionConstant *> * fcs = [fn functionConstantsDictionary];
            NSMutableArray<NSString *> * parts = [NSMutableArray array];
            for (NSString * key in [[fcs allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
                MTLFunctionConstant * fc = fcs[key];
                [parts addObject:[NSString stringWithFormat:@"%@:%lu:%lu:%d", fc.name, (unsigned long) fc.type, (unsigned long) fc.index, fc.required ? 1 : 0]];
            }
            printf("%s\t%s\n", [name UTF8String], [parts count] ? [[parts componentsJoinedByString:@","] UTF8String] : "-");
        }
    }
    return 0;
}
