#ifdef __APPLE__
#include "application_catalog.hpp"
#import <AppKit/AppKit.h>

namespace GRIM::Applications {
Backend nativeBackend() {
    return {"macos-bundles", [] {
        @autoreleasepool {
            std::vector<Application> apps;
            NSArray<NSString*>* roots = @[[NSHomeDirectory() stringByAppendingPathComponent:@"Applications"],
                @"/Applications", @"/System/Applications", @"/System/Library/CoreServices/Applications", @"/Network/Applications"];
            for (NSString* root in roots) {
                const auto start = apps.size();
                if (![[NSFileManager defaultManager] fileExistsAtPath:root]) continue;
                __block bool failed = false;
                auto* enumerator = [[NSFileManager defaultManager] enumeratorAtURL:[NSURL fileURLWithPath:root]
                    includingPropertiesForKeys:nil options:NSDirectoryEnumerationSkipsPackageDescendants
                    errorHandler:^BOOL(NSURL*, NSError*) { failed = true; return NO; }];
                if (!enumerator) throw std::runtime_error("Cannot enumerate application bundles");
                for (NSURL* url in enumerator) {
                    if (![[url pathExtension] isEqualToString:@"app"]) continue;
                    NSBundle* bundle = [NSBundle bundleWithURL:url];
                    NSString* identifier = [bundle bundleIdentifier];
                    if (!identifier) continue;
                    NSString* name = [bundle objectForInfoDictionaryKey:@"CFBundleDisplayName"];
                    if (!name) name = [bundle objectForInfoDictionaryKey:@"CFBundleName"];
                    if (!name) name = [[url lastPathComponent] stringByDeletingPathExtension];
                    apps.push_back({"macos.bundle", [identifier UTF8String], [name UTF8String], "bundle", [[url path] UTF8String], ""});
                }
                if (failed) throw std::runtime_error("Application bundle enumeration failed");
                std::sort(apps.begin() + start, apps.end(), [](const Application& a, const Application& b) {
                    return std::tie(a.nativeId, a.launchTarget) < std::tie(b.nativeId, b.launchTarget);
                });
            }
            return apps;
        }
    }};
}
Backend portableBackend() { return {"portable-applications", [] { return std::vector<Application>{}; }}; }
void launchNative(const Application& app) {
    if (app.launchKind != "bundle") throw std::invalid_argument("Unsupported macOS application launch kind");
    @autoreleasepool {
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:app.launchTarget.c_str()]];
        // Launch Services handles application activation; no shell command construction.
        if (![[NSWorkspace sharedWorkspace] openURL:url]) throw std::runtime_error("Launch Services rejected application");
    }
}
}
#endif
