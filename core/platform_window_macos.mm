// =============================================================================
// macOS implementation of platform window/display API
// Objective-C++ (.mm) for Cocoa + Metal + BGFX integration
// =============================================================================
#if defined(__APPLE__)

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>
#include "platform_window.hpp"
#include "logger.hpp"

// =============================================================================
// Metal-backed NSView for BGFX
// =============================================================================
@interface GRIMMetalView : NSView
@end

@implementation GRIMMetalView

- (BOOL)wantsUpdateLayer { return YES; }

+ (Class)layerClass { return [CAMetalLayer class]; }

- (CALayer*)makeBackingLayer {
    CAMetalLayer* layer = [CAMetalLayer layer];
    layer.device = MTLCreateSystemDefaultDevice();
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    layer.framebufferOnly = YES;
    layer.contentsScale = self.window.backingScaleFactor ?: [[NSScreen mainScreen] backingScaleFactor];
    return layer;
}

- (void)viewDidChangeBackingProperties {
    [super viewDidChangeBackingProperties];
    if (self.layer) {
        ((CAMetalLayer*)self.layer).contentsScale = self.window.backingScaleFactor;
    }
}

@end

// =============================================================================
// Overlay window (click-through when no UI, interactive when panels visible)
// =============================================================================
@interface GRIMOverlayWindow : NSWindow
@end

@implementation GRIMOverlayWindow

- (BOOL)canBecomeKeyWindow { return YES; }
- (BOOL)canBecomeMainWindow { return NO; }
- (BOOL)acceptsFirstResponder { return YES; }

@end

// =============================================================================
// App delegate to handle lifecycle
// =============================================================================
@interface GRIMAppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation GRIMAppDelegate

- (BOOL)applicationSupportsSecureRestorableState:(NSApplication*)app {
    (void)app;
    return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)sender {
    (void)sender;
    return NSTerminateNow;
}

@end

// =============================================================================
// Static state
// =============================================================================
static NSWindow*       s_bgfxWindow   = nil;
static GRIMAppDelegate* s_appDelegate = nil;
static bool            s_appInitialized = false;

static void ensureNSApp() {
    if (s_appInitialized) return;

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        s_appDelegate = [[GRIMAppDelegate alloc] init];
        [NSApp setDelegate:s_appDelegate];

        [NSApp finishLaunching];
    }
    s_appInitialized = true;
}

// =============================================================================
// PlatformWindow API
// =============================================================================
namespace PlatformWindow {

void* createBGFXInitWindow() {
    @autoreleasepool {
        ensureNSApp();

        NSRect frame = NSMakeRect(0, 0, 1, 1);

        NSWindowStyleMask style =
            NSWindowStyleMaskTitled |
            NSWindowStyleMaskClosable |
            NSWindowStyleMaskMiniaturizable |
            NSWindowStyleMaskResizable;

        NSWindow* window = [[NSWindow alloc]
            initWithContentRect:frame
                      styleMask:style
                        backing:NSBackingStoreBuffered
                          defer:NO];

        [window setTitle:@"GRIM"];
        [window setReleasedWhenClosed:NO];

        GRIMMetalView* metalView = [[GRIMMetalView alloc] initWithFrame:frame];
        [metalView setWantsLayer:YES];
        [window setContentView:metalView];

        s_bgfxWindow = window;

        LOG_DEBUG("PlatformWindow", "macOS: BGFX init window created (Metal-backed NSView)");
        return (__bridge void*)window;
    }
}

void destroyBGFXInitWindow(void* handle) {
    if (!handle) return;
    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)handle;
        [window orderOut:nil];
        [window close];
        if (window == s_bgfxWindow) s_bgfxWindow = nil;
    }
}

void setWindowVisible(void* handle, bool visible) {
    if (!handle) return;
    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)handle;
        if (visible) {
            [window makeKeyAndOrderFront:nil];
            [NSApp activateIgnoringOtherApps:YES];
        } else {
            [window orderOut:nil];
        }
    }
}

void getVirtualScreenRect(int& x, int& y, int& width, int& height) {
    @autoreleasepool {
        NSRect unionRect = NSZeroRect;
        for (NSScreen* screen in [NSScreen screens]) {
            unionRect = NSUnionRect(unionRect, [screen frame]);
        }
        x = static_cast<int>(NSMinX(unionRect));
        y = static_cast<int>(NSMinY(unionRect));
        width = static_cast<int>(NSWidth(unionRect));
        height = static_cast<int>(NSHeight(unionRect));

        if (width <= 0 || height <= 0) {
            NSRect mainFrame = [[NSScreen mainScreen] frame];
            x = 0;
            y = 0;
            width = static_cast<int>(NSWidth(mainFrame));
            height = static_cast<int>(NSHeight(mainFrame));
        }
    }
}

bool pumpEvents(float& mouseWheelDeltaOut, bool& quitRequested) {
    mouseWheelDeltaOut = 0.0f;
    quitRequested = false;

    @autoreleasepool {
        while (true) {
            NSEvent* event = [NSApp
                nextEventMatchingMask:NSEventMaskAny
                            untilDate:nil
                               inMode:NSDefaultRunLoopMode
                              dequeue:YES];
            if (!event) break;

            if ([event type] == NSEventTypeScrollWheel) {
                mouseWheelDeltaOut += static_cast<float>([event scrollingDeltaY]) * 120.0f;
            }

            if ([event type] == NSEventTypeApplicationDefined &&
                [event subtype] == NSEventSubtypeApplicationActivated) {
            }

            [NSApp sendEvent:event];
        }

        [NSApp updateWindows];
    }

    return true;
}

// =============================================================================
// macOS overlay window creation (called from WindowManager via platform layer)
// =============================================================================
void* createOverlayWindow(int x, int y, int width, int height) {
    @autoreleasepool {
        ensureNSApp();

        // Flip Y: Cocoa origin is bottom-left
        NSRect screenFrame = [[NSScreen mainScreen] frame];
        CGFloat flippedY = NSHeight(screenFrame) - y - height;

        NSRect frame = NSMakeRect(x, flippedY, width, height);

        GRIMOverlayWindow* window = [[GRIMOverlayWindow alloc]
            initWithContentRect:frame
                      styleMask:NSWindowStyleMaskBorderless
                        backing:NSBackingStoreBuffered
                          defer:NO];

        [window setLevel:NSFloatingWindowLevel];
        [window setOpaque:NO];
        [window setBackgroundColor:[NSColor clearColor]];
        [window setHasShadow:NO];
        [window setIgnoresMouseEvents:NO];
        [window setCollectionBehavior:
            NSWindowCollectionBehaviorCanJoinAllSpaces |
            NSWindowCollectionBehaviorFullScreenAuxiliary];
        [window setReleasedWhenClosed:NO];

        GRIMMetalView* metalView = [[GRIMMetalView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
        [metalView setWantsLayer:YES];
        [window setContentView:metalView];

        [window makeKeyAndOrderFront:nil];

        LOG_DEBUG("PlatformWindow", "macOS: Overlay window created (" +
                  std::to_string(width) + "x" + std::to_string(height) + ")");

        return (__bridge void*)window;
    }
}

} // namespace PlatformWindow

#endif // __APPLE__
