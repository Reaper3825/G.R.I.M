// =============================================================================
// macOS implementation of platform window/display API
// Objective-C++ (.mm) for Cocoa + Metal + BGFX integration
// =============================================================================
#if defined(__APPLE__)

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#include "platform_window.hpp"
#include "logger.hpp"
#include <string>
#include <vector>

// =============================================================================
// Metal-backed NSView for BGFX
// =============================================================================
@interface GRIMMetalView : NSView
@end

@implementation GRIMMetalView

- (BOOL)wantsUpdateLayer { return YES; }

- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)canBecomeKeyView { return YES; }

- (BOOL)becomeFirstResponder {
    return [super becomeFirstResponder];
}

- (BOOL)resignFirstResponder {
    return [super resignFirstResponder];
}

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

        // Use a small visible size on macOS so the window can be focused for the backtick hotkey
        NSRect frame = NSMakeRect(0, 0, 320, 200);

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
            NSView* contentView = [window contentView];
            if (contentView && ![window firstResponder]) {
                [window makeFirstResponder:contentView];
            }
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
                            untilDate:[NSDate dateWithTimeIntervalSinceNow:0]
                               inMode:NSDefaultRunLoopMode
                              dequeue:YES];
            if (!event) break;

            if ([event type] == NSEventTypeScrollWheel) {
                mouseWheelDeltaOut += static_cast<float>([event scrollingDeltaY]) * 120.0f;
            }

            if ([event type] == NSEventTypeApplicationDefined &&
                [event subtype] == NSEventSubtypeApplicationActivated) {
            }

            // Debug: log key events to verify keyboard input on macOS
            NSEventType etype = [event type];
            if (etype == NSEventTypeKeyDown || etype == NSEventTypeKeyUp) {
                unsigned short keyCode = [event keyCode];
                NSString* chars = [event characters];
                std::string charsStr = chars && [chars length] ? std::string([chars UTF8String]) : "";
                const char* kind = (etype == NSEventTypeKeyDown) ? "KeyDown" : "KeyUp";
                LOG_DEBUG("PlatformWindow",
                    std::string(kind) + " keyCode=" + std::to_string(keyCode) +
                    (charsStr.empty() ? "" : " char='" + charsStr + "'"));
            }

            [NSApp sendEvent:event];
        }

        [NSApp updateWindows];
    }

    return true;
}

// =============================================================================
// macOS overlay window creation (called from WindowManager via platform layer)
// Uses a regular CALayer (not Metal) for software-rendered 2D overlay
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

        NSView* contentView = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
        [contentView setWantsLayer:YES];
        contentView.layer.contentsGravity = kCAGravityResize;
        contentView.layer.opaque = NO;
        contentView.layer.backgroundColor = [NSColor clearColor].CGColor;  // No tint behind transparent pixels
        [window setContentView:contentView];

        LOG_DEBUG("PlatformWindow", "macOS: Overlay window created (" +
                  std::to_string(width) + "x" + std::to_string(height) + ")");

        return (__bridge void*)window;
    }
}

} // namespace PlatformWindow

// =============================================================================
// Overlay renderer blit: copies BGRA pixel buffer to the overlay window's layer
// Called from OverlayRenderer::endFrame() on macOS
// =============================================================================
void grimOverlayBlit(void* nsWindowHandle, void* pixels, int width, int height) {
    @autoreleasepool {
        if (!nsWindowHandle || !pixels || width <= 0 || height <= 0) return;

        NSWindow* window = (__bridge NSWindow*)nsWindowHandle;
        NSView* contentView = [window contentView];
        if (!contentView) return;

        CALayer* layer = [contentView layer];
        if (!layer) return;

        // Never set contents on CAMetalLayer — it owns its drawables and presentation.
        // Use a dedicated image sublayer so we don't conflict with Metal's ownership.
        CALayer* imageLayer = nil;
        if ([layer isKindOfClass:[CAMetalLayer class]]) {
            for (CALayer* sub in layer.sublayers) {
                if (sub.name && [sub.name isEqualToString:@"GRIMOverlayImage"]) {
                    imageLayer = sub;
                    break;
                }
            }
            if (!imageLayer) {
                imageLayer = [CALayer layer];
                imageLayer.name = @"GRIMOverlayImage";
                imageLayer.frame = layer.bounds;
                imageLayer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
                imageLayer.contentsGravity = kCAGravityResize;
                imageLayer.opaque = NO;
                imageLayer.backgroundColor = [NSColor clearColor].CGColor;
                [layer addSublayer:imageLayer];
            }
            imageLayer.frame = layer.bounds;
            imageLayer.opaque = NO;
            imageLayer.backgroundColor = [NSColor clearColor].CGColor;
            layer = imageLayer;
        } else {
            layer.opaque = NO;
            layer.backgroundColor = [NSColor clearColor].CGColor;
        }

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        // OverlayRenderer uses (a<<24)|(r<<16)|(g<<8)|b → in memory [B,G,R,A].
        // Core Graphics expects RGBA with PremultipliedLast → [R,G,B,A]. Copy and swap R/B.
        size_t numPixels = (size_t)width * (size_t)height;
        std::vector<uint32_t> rgbaBuffer(numPixels);
        const uint32_t* src = static_cast<const uint32_t*>(pixels);
        for (size_t i = 0; i < numPixels; ++i) {
            uint32_t p = src[i];
            uint32_t a = (p >> 24) & 0xFF;
            uint32_t r = (p >> 16) & 0xFF;
            uint32_t g = (p >>  8) & 0xFF;
            uint32_t b = p & 0xFF;
            rgbaBuffer[i] = (a << 24) | (b << 16) | (g << 8) | r;  // store as R,G,B,A for PremultipliedLast
        }
        CGContextRef ctx = CGBitmapContextCreate(
            rgbaBuffer.data(), width, height, 8, width * 4,
            colorSpace,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Little);

        if (ctx) {
            CGImageRef image = CGBitmapContextCreateImage(ctx);
            if (image) {
                [CATransaction begin];
                [CATransaction setDisableActions:YES];
                layer.contents = (__bridge id)image;
                [CATransaction commit];
                CGImageRelease(image);
            }
            CGContextRelease(ctx);
        }
        CGColorSpaceRelease(colorSpace);
    }
}

#endif // __APPLE__
