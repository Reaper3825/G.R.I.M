// =============================================================================
// macOS implementation of platform window/display API
// Objective-C++ (.mm) for Cocoa + Metal + BGFX integration
// =============================================================================
#if defined(__APPLE__)

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreVideo/CoreVideo.h>
#import <os/lock.h>
#include "platform_window.hpp"
#include "platform_input.hpp"
#include "logger.hpp"
#include <string>
#include <functional>
#include <vector>
#include <cstring>
#import <Carbon/Carbon.h>

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
static const NSInteger kOverlayHostTag = 0x4752494D; // 'GRIM' for viewWithTag
static std::function<void(const std::string&)> s_textInputCallback;

static void stopScreenCapture();

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

void setTextInputCallback(std::function<void(const std::string&)> callback) {
    s_textInputCallback = std::move(callback);
}

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

            NSEventType etype = [event type];

            if (etype == NSEventTypeScrollWheel) {
                mouseWheelDeltaOut += static_cast<float>([event scrollingDeltaY]) * 120.0f;
            }

            if (etype == NSEventTypeApplicationDefined &&
                [event subtype] == NSEventSubtypeApplicationActivated) {
            }

            // ---------------------------------------------------------------
            // Event-driven key state tracking (replaces CGEventSourceKeyState polling)
            // ---------------------------------------------------------------
            if (etype == NSEventTypeKeyDown || etype == NSEventTypeKeyUp) {
                bool down = (etype == NSEventTypeKeyDown);
                unsigned short macKeyCode = [event keyCode];
                // Use the reverse mapping in PlatformInput to get VK code
                // We call the macKeyCodeToVK helper indirectly via setKeyDownFromEvent
                int vk = -1;
                switch (macKeyCode) {
                    case kVK_Delete:         vk = 0x08; break;
                    case kVK_Tab:            vk = 0x09; break;
                    case kVK_Return:         vk = 0x0D; break;
                    case kVK_Escape:         vk = 0x1B; break;
                    case kVK_Space:          vk = 0x20; break;
                    case kVK_LeftArrow:      vk = 0x25; break;
                    case kVK_UpArrow:        vk = 0x26; break;
                    case kVK_RightArrow:     vk = 0x27; break;
                    case kVK_DownArrow:      vk = 0x28; break;
                    case kVK_ForwardDelete:  vk = 0x2E; break;
                    case kVK_Home:           vk = 0x24; break;
                    case kVK_End:            vk = 0x23; break;
                    case kVK_PageUp:         vk = 0x21; break;
                    case kVK_PageDown:       vk = 0x22; break;
                    case kVK_Shift:          // fall through
                    case kVK_RightShift:     vk = 0x10; break;
                    case kVK_Control:        // fall through
                    case kVK_RightControl:   vk = 0x11; break;
                    case kVK_Option:         // fall through
                    case kVK_RightOption:    vk = 0x12; break;
                    case kVK_Command:        // fall through
                    case kVK_RightCommand:   vk = 0x11; break; // Cmd → Ctrl VK
                    case kVK_ANSI_A: vk = 'A'; break; case kVK_ANSI_B: vk = 'B'; break;
                    case kVK_ANSI_C: vk = 'C'; break; case kVK_ANSI_D: vk = 'D'; break;
                    case kVK_ANSI_E: vk = 'E'; break; case kVK_ANSI_F: vk = 'F'; break;
                    case kVK_ANSI_G: vk = 'G'; break; case kVK_ANSI_H: vk = 'H'; break;
                    case kVK_ANSI_I: vk = 'I'; break; case kVK_ANSI_J: vk = 'J'; break;
                    case kVK_ANSI_K: vk = 'K'; break; case kVK_ANSI_L: vk = 'L'; break;
                    case kVK_ANSI_M: vk = 'M'; break; case kVK_ANSI_N: vk = 'N'; break;
                    case kVK_ANSI_O: vk = 'O'; break; case kVK_ANSI_P: vk = 'P'; break;
                    case kVK_ANSI_Q: vk = 'Q'; break; case kVK_ANSI_R: vk = 'R'; break;
                    case kVK_ANSI_S: vk = 'S'; break; case kVK_ANSI_T: vk = 'T'; break;
                    case kVK_ANSI_U: vk = 'U'; break; case kVK_ANSI_V: vk = 'V'; break;
                    case kVK_ANSI_W: vk = 'W'; break; case kVK_ANSI_X: vk = 'X'; break;
                    case kVK_ANSI_Y: vk = 'Y'; break; case kVK_ANSI_Z: vk = 'Z'; break;
                    case kVK_ANSI_0: vk = '0'; break; case kVK_ANSI_1: vk = '1'; break;
                    case kVK_ANSI_2: vk = '2'; break; case kVK_ANSI_3: vk = '3'; break;
                    case kVK_ANSI_4: vk = '4'; break; case kVK_ANSI_5: vk = '5'; break;
                    case kVK_ANSI_6: vk = '6'; break; case kVK_ANSI_7: vk = '7'; break;
                    case kVK_ANSI_8: vk = '8'; break; case kVK_ANSI_9: vk = '9'; break;
                    case kVK_ANSI_Grave: vk = 0xC0; break;
                    case kVK_F1:  vk = 0x70; break; case kVK_F2:  vk = 0x71; break;
                    case kVK_F3:  vk = 0x72; break; case kVK_F4:  vk = 0x73; break;
                    case kVK_F5:  vk = 0x74; break; case kVK_F6:  vk = 0x75; break;
                    case kVK_F7:  vk = 0x76; break; case kVK_F8:  vk = 0x77; break;
                    case kVK_F9:  vk = 0x78; break; case kVK_F10: vk = 0x79; break;
                    case kVK_F11: vk = 0x7A; break; case kVK_F12: vk = 0x7B; break;
                    default: break;
                }
                if (vk >= 0) {
                    PlatformInput::setKeyDownFromEvent(vk, down);
                }
                // Update Command key state for isCommandDown()
                if (macKeyCode == kVK_Command || macKeyCode == kVK_RightCommand) {
                    PlatformInput::setCommandDownFromEvent(down);
                }
            }

            // Also track modifier flag changes (covers modifier-only presses without keyDown)  
            if (etype == NSEventTypeFlagsChanged) {
                NSEventModifierFlags mods = [event modifierFlags];
                PlatformInput::setKeyDownFromEvent(0x10, (mods & NSEventModifierFlagShift) != 0);
                PlatformInput::setKeyDownFromEvent(0x11, (mods & NSEventModifierFlagControl) != 0);
                PlatformInput::setKeyDownFromEvent(0x12, (mods & NSEventModifierFlagOption) != 0);
                bool cmdDown = (mods & NSEventModifierFlagCommand) != 0;
                PlatformInput::setCommandDownFromEvent(cmdDown);
                // Map Cmd to Ctrl VK for shortcuts
                if (cmdDown) PlatformInput::setKeyDownFromEvent(0x11, true);
            }

            // macOS equivalent of WM_CHAR: inject printable characters into text input
            // Skip injection when Cmd is held (let clipboard shortcuts pass through to polling)
            if (etype == NSEventTypeKeyDown && s_textInputCallback) {
                NSEventModifierFlags mods = [event modifierFlags];
                bool cmdHeld = (mods & NSEventModifierFlagCommand) != 0;
                if (!cmdHeld) {
                    NSString* chars = [event characters];
                    if (chars && [chars length] > 0) {
                        std::string charsStr = [chars UTF8String];
                        bool injectedAny = false;
                        for (unsigned char c : charsStr) {
                            if (c >= 32 && c < 127) {
                                s_textInputCallback(std::string(1, static_cast<char>(c)));
                                injectedAny = true;
                            }
                        }
                        if (injectedAny) {
                            continue;
                        }
                    }
                }
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
        [window setIgnoresMouseEvents:YES];
        [window setCollectionBehavior:
            NSWindowCollectionBehaviorCanJoinAllSpaces |
            NSWindowCollectionBehaviorFullScreenAuxiliary];
        [window setReleasedWhenClosed:NO];

        // Container hosts (1) N stacked OS blur layers behind and (2) our overlay image on top.
        NSView* containerView = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
        [containerView setWantsLayer:YES];
        containerView.layer.opaque = NO;
        containerView.layer.backgroundColor = [NSColor clearColor].CGColor;

        auto makeBlurView = [&](NSView* parent) {
            NSVisualEffectView* v = [[NSVisualEffectView alloc] initWithFrame:parent.bounds];
            v.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
            v.material = NSVisualEffectMaterialFullScreenUI;
            v.blendingMode = NSVisualEffectBlendingModeBehindWindow;
            v.state = NSVisualEffectStateActive;
            v.alphaValue = 0.99;
            [parent addSubview:v];
        };

        // Two stacked blur passes for a heavier frosted-glass radius.
        makeBlurView(containerView);
        makeBlurView(containerView);

        // Our overlay image (rendered into a layer by grimOverlayBlit).
        NSView* overlayHost = [[NSView alloc] initWithFrame:containerView.bounds];
        overlayHost.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [overlayHost setWantsLayer:YES];
        overlayHost.layer.opaque = NO;
        overlayHost.layer.contentsGravity = kCAGravityResize;
        overlayHost.layer.backgroundColor = [NSColor clearColor].CGColor;
        [containerView addSubview:overlayHost];

        [window setContentView:containerView];

        LOG_DEBUG("PlatformWindow", "macOS: Overlay window created (" +
                  std::to_string(width) + "x" + std::to_string(height) + ")");

        return (__bridge void*)window;
    }
}

void setOverlayBlurMask(void* overlayWindowHandle,
                         const float* panelRects,
                         int panelCount,
                         float cornerRadius)
{
    if (!overlayWindowHandle)
        return;

    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)overlayWindowHandle;
        if (!window) return;

        NSView* containerView = [window contentView];
        if (!containerView) return;

        // No visible panels — hide blur views entirely so no grey tint remains,
        // and stop the screen capture stream to save resources.
        if (!panelRects || panelCount <= 0) {
            for (NSView* sub in containerView.subviews) {
                if (![sub isKindOfClass:[NSVisualEffectView class]]) continue;
                [sub setHidden:YES];
            }
            stopScreenCapture();
            return;
        }

        // Panels visible — ensure blur views are unhidden.
        for (NSView* sub in containerView.subviews) {
            if (![sub isKindOfClass:[NSVisualEffectView class]]) continue;
            [sub setHidden:NO];
        }

        // Build path using CoreGraphics directly (avoids ObjC NSBezierPath overhead).
        CGMutablePathRef path = CGPathCreateMutable();
        const float r = std::max(0.0f, cornerRadius);

        for (int i = 0; i < panelCount; ++i) {
            const float x = panelRects[i * 4 + 0];
            const float y = panelRects[i * 4 + 1];
            const float w = panelRects[i * 4 + 2];
            const float h = panelRects[i * 4 + 3];
            if (w <= 0.0f || h <= 0.0f) continue;
            CGPathAddRoundedRect(path, nullptr, CGRectMake(x, y, w, h), r, r);
        }

        for (NSView* sub in containerView.subviews) {
            if (![sub isKindOfClass:[NSVisualEffectView class]]) continue;

            CALayer* layer = sub.layer; // NSVisualEffectView always has a layer
            if (!layer) continue;

            CAShapeLayer* maskLayer = (CAShapeLayer*)layer.mask;
            if (!maskLayer) {
                maskLayer = [CAShapeLayer layer];
                maskLayer.fillColor = [NSColor whiteColor].CGColor;
                maskLayer.geometryFlipped = YES;
                layer.mask = maskLayer;
            }

            maskLayer.frame = layer.bounds;
            maskLayer.path = path;
        }

        CGPathRelease(path);
    }
}

void setOverlayBlurStyle(void* overlayWindowHandle,
                          bool enabled,
                          float opacity,
                          int intensity)
{
    if (!overlayWindowHandle) return;

    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)overlayWindowHandle;
        if (!window) return;

        NSView* containerView = [window contentView];
        if (!containerView) return;

        intensity = std::max(0, std::min(intensity, 5));

        // Collect existing blur views.
        NSMutableArray<NSVisualEffectView*>* existing = [NSMutableArray array];
        for (NSView* sub in containerView.subviews) {
            if ([sub isKindOfClass:[NSVisualEffectView class]])
                [existing addObject:(NSVisualEffectView*)sub];
        }

        int currentCount = (int)existing.count;
        int desired = enabled ? intensity : 0;

        // Remove excess blur views.
        while (currentCount > desired) {
            [existing.lastObject removeFromSuperview];
            [existing removeLastObject];
            --currentCount;
        }

        // Add missing blur views (insert below the overlay host, i.e. at index 0+).
        while (currentCount < desired) {
            NSVisualEffectView* v = [[NSVisualEffectView alloc] initWithFrame:containerView.bounds];
            v.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
            v.material = NSVisualEffectMaterialFullScreenUI;
            v.blendingMode = NSVisualEffectBlendingModeBehindWindow;
            v.state = NSVisualEffectStateActive;
            [containerView addSubview:v positioned:NSWindowBelow relativeTo:containerView.subviews.lastObject];
            [existing addObject:v];
            ++currentCount;
        }

        // Update opacity on all blur views.
        float clampedOpacity = std::max(0.0f, std::min(1.0f, opacity));
        for (NSVisualEffectView* v in existing) {
            v.alphaValue = clampedOpacity;
        }
    }
}

void setOverlayClickThrough(void* overlayWindowHandle, bool clickThrough) {
    if (!overlayWindowHandle) return;
    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)overlayWindowHandle;
        BOOL shouldIgnore = clickThrough ? YES : NO;
        if ([window ignoresMouseEvents] != shouldIgnore) {
            [window setIgnoresMouseEvents:shouldIgnore];
        }
    }
}

} // namespace PlatformWindow

// =============================================================
// ScreenCaptureKit-based desktop capture (macOS 15+)
// =============================================================
@interface GRIMScreenCapture : NSObject <SCStreamOutput, SCStreamDelegate>
@property (atomic, assign) CVPixelBufferRef latestBuffer;
@property (atomic, assign) CGRect capturedDisplayFrame;
@property (atomic, assign) int capturedPixelWidth;
@property (atomic, assign) int capturedPixelHeight;
@property (atomic, assign) BOOL running;
@end

@implementation GRIMScreenCapture
- (instancetype)init {
    if (self = [super init]) {
        _latestBuffer = nil;
        _capturedDisplayFrame = CGRectZero;
        _capturedPixelWidth = 0;
        _capturedPixelHeight = 0;
        _running = NO;
    }
    return self;
}

- (void)stream:(SCStream*)stream didStopWithError:(NSError*)error {
    (void)stream;
    if (error) {
        LOG_ERROR("PlatformWindow", std::string("ScreenCaptureKit stopped: ") + [[error localizedDescription] UTF8String]);
    }
    self.running = NO;
}

- (void)stream:(SCStream*)stream didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer ofType:(SCStreamOutputType)type {
    (void)stream;
    if (type != SCStreamOutputTypeScreen) return;
    if (!CMSampleBufferIsValid(sampleBuffer)) return;
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    if (!imageBuffer) return;

    CVPixelBufferRef pb = (CVPixelBufferRef)imageBuffer;
    CVPixelBufferRetain(pb);

    extern os_unfair_lock g_grimScreenCapLock;
    os_unfair_lock_lock(&g_grimScreenCapLock);
    CVPixelBufferRef old = self.latestBuffer;
    self.latestBuffer = pb;
    os_unfair_lock_unlock(&g_grimScreenCapLock);
    if (old) CVPixelBufferRelease(old);
}
@end

static GRIMScreenCapture* s_screenCap = nil;
static SCStream* s_stream = nil;
os_unfair_lock g_grimScreenCapLock = OS_UNFAIR_LOCK_INIT;

static void stopScreenCapture() {
    if (s_stream) {
        [s_stream stopCaptureWithCompletionHandler:^(NSError* err) {
            if (err) {
                LOG_ERROR("PlatformWindow", std::string("ScreenCaptureKit stop error: ") + [[err localizedDescription] UTF8String]);
            } else {
                LOG_DEBUG("PlatformWindow", "ScreenCaptureKit capture stopped (no visible panels)");
            }
        }];
        s_stream = nil;
    }
    if (s_screenCap) {
        s_screenCap.running = NO;
        os_unfair_lock_lock(&g_grimScreenCapLock);
        CVPixelBufferRef old = s_screenCap.latestBuffer;
        s_screenCap.latestBuffer = nil;
        os_unfair_lock_unlock(&g_grimScreenCapLock);
        if (old) CVPixelBufferRelease(old);
    }
}

static void ensureScreenCaptureStarted(NSWindow* overlayWindow) {
    if (s_screenCap && s_screenCap.running) return;
    if (!overlayWindow) return;

    if (!s_screenCap) s_screenCap = [[GRIMScreenCapture alloc] init];

    // Start async capture of the main display; exclude our overlay window if possible.
    [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent* content, NSError* error) {
        if (error || !content) {
            if (error) {
                static int sckErrorCount = 0;
                if (sckErrorCount < 3) {
                    LOG_ERROR("PlatformWindow", std::string("ScreenCaptureKit content error: ") + [[error localizedDescription] UTF8String]);
                    if (++sckErrorCount == 3)
                        LOG_ERROR("PlatformWindow", "Suppressing further ScreenCaptureKit errors (grant Screen Recording permission in System Settings)");
                }
            }
            return;
        }

        SCDisplay* display = content.displays.firstObject;
        if (!display) return;

        // Try to find our overlay window in the shareable list so we can exclude it.
        NSMutableArray<SCWindow*>* exclude = [NSMutableArray array];
        for (SCWindow* w in content.windows) {
            if (w.windowID == (CGWindowID)[overlayWindow windowNumber]) {
                [exclude addObject:w];
                break;
            }
        }

        SCContentFilter* filter = nil;
        if (exclude.count > 0) {
            filter = [[SCContentFilter alloc] initWithDisplay:display excludingWindows:exclude];
        } else {
            filter = [[SCContentFilter alloc] initWithDisplay:display excludingWindows:@[]];
        }

        SCStreamConfiguration* cfg = [[SCStreamConfiguration alloc] init];
        cfg.pixelFormat = kCVPixelFormatType_32BGRA;
        cfg.showsCursor = NO;
        cfg.capturesAudio = NO;
        cfg.minimumFrameInterval = CMTimeMake(1, 30); // ~30 FPS
        cfg.queueDepth = 2;

        // Capture full display (pixel dimensions).
        cfg.width = display.width;
        cfg.height = display.height;

        // display.frame is in points (Cocoa space); width/height are pixels.
        s_screenCap.capturedDisplayFrame = CGRectMake(display.frame.origin.x, display.frame.origin.y,
                                                     display.frame.size.width, display.frame.size.height);
        s_screenCap.capturedPixelWidth = (int)display.width;
        s_screenCap.capturedPixelHeight = (int)display.height;

        s_stream = [[SCStream alloc] initWithFilter:filter configuration:cfg delegate:s_screenCap];
        NSError* addErr = nil;
        [s_stream addStreamOutput:s_screenCap type:SCStreamOutputTypeScreen sampleHandlerQueue:dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0) error:&addErr];
        if (addErr) {
            LOG_ERROR("PlatformWindow", std::string("ScreenCaptureKit add output error: ") + [[addErr localizedDescription] UTF8String]);
            return;
        }

        [s_stream startCaptureWithCompletionHandler:^(NSError* startErr) {
            if (startErr) {
                LOG_ERROR("PlatformWindow", std::string("ScreenCaptureKit start error: ") + [[startErr localizedDescription] UTF8String]);
                return;
            }
            s_screenCap.running = YES;
            LOG_DEBUG("PlatformWindow", "ScreenCaptureKit capture started");
        }];
    }];
}

namespace PlatformWindow {

bool captureDesktopBehindOverlay(void* overlayWindowHandle,
                                 int x,
                                 int y,
                                 int width,
                                 int height,
                                 uint32_t* outPixelsARGB)
{
    if (!overlayWindowHandle || !outPixelsARGB || width <= 0 || height <= 0)
        return false;

    @autoreleasepool {
        const int dstW = width;
        const int dstH = height;

        // Clear output (so partially offscreen captures don't leave garbage).
        std::memset(outPixelsARGB, 0, (size_t)dstW * (size_t)dstH * sizeof(uint32_t));

        NSWindow* window = (__bridge NSWindow*)overlayWindowHandle;
        if (!window) return false;

        ensureScreenCaptureStarted(window);
        if (!s_screenCap)
            return false; // not ready yet (permission / first frame)

        CVPixelBufferRef pb = nil;
        os_unfair_lock_lock(&g_grimScreenCapLock);
        pb = s_screenCap.latestBuffer;
        if (pb) CVPixelBufferRetain(pb);
        os_unfair_lock_unlock(&g_grimScreenCapLock);

        if (!pb)
            return false;

        const size_t srcW = CVPixelBufferGetWidth(pb);
        const size_t srcH = CVPixelBufferGetHeight(pb);
        const size_t bpr = CVPixelBufferGetBytesPerRow(pb);
        if (srcW == 0 || srcH == 0 || bpr == 0) {
            CVPixelBufferRelease(pb);
            return false;
        }

        // Map overlay-local coords -> global screen coords (points).
        // overlay-local uses top-left origin; Cocoa screen coords are bottom-left.
        NSRect winFrame = [window frame];
        const CGFloat winTop = winFrame.origin.y + winFrame.size.height;
        const CGFloat screenX = winFrame.origin.x + (CGFloat)x;
        const CGFloat screenYTopDown = winTop - (CGFloat)y; // point at top edge of requested rect

        CGRect disp = s_screenCap.capturedDisplayFrame; // in points, bottom-left origin
        // Convert to display-local (bottom-left origin), in points.
        const CGFloat dx = screenX - disp.origin.x;
        const CGFloat dyTop = screenYTopDown - disp.origin.y;

        // ScreenCaptureKit buffers are in pixels; disp frame is in points.
        // Use the capture's own pixel-per-point ratio to avoid "quadrant" scaling issues.
        const CGFloat dispW = disp.size.width > 0 ? disp.size.width : 1.0;
        const CGFloat dispH = disp.size.height > 0 ? disp.size.height : 1.0;
        const CGFloat sx = (CGFloat)srcW / dispW;
        const CGFloat sy = (CGFloat)srcH / dispH;
        // X: straightforward (left-to-right).
        int srcX0 = (int)std::floor(dx * sx);

        // Y: ScreenCaptureKit pixel buffers are addressed from the top row in memory.
        // Convert bottom-left point Y into top-left pixel row.
        const CGFloat dispHpt = disp.size.height;
        int srcY0 = (int)std::floor((dispHpt - dyTop) * sy);

        CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        const uint8_t* base = (const uint8_t*)CVPixelBufferGetBaseAddress(pb);

        // Output is ARGB; source is BGRA. Nearest-neighbor resample.
        for (int yy = 0; yy < dstH; ++yy) {
            int syi = srcY0 + (int)std::floor((CGFloat)yy * sy);
            if (syi < 0 || syi >= (int)srcH) continue;
            const uint8_t* row = base + (size_t)syi * bpr;
            for (int xx = 0; xx < dstW; ++xx) {
                int sxi = srcX0 + (int)std::floor((CGFloat)xx * sx);
                if (sxi < 0 || sxi >= (int)srcW) continue;
                const uint8_t* p = row + (size_t)sxi * 4;
                const uint8_t b = p[0];
                const uint8_t g = p[1];
                const uint8_t r = p[2];
                const uint8_t a = p[3];
                outPixelsARGB[yy * dstW + xx] =
                    (uint32_t(a) << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
            }
        }

        CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
        CVPixelBufferRelease(pb);
        return true;
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

        // We render the overlay pixels into the dedicated host view above the blur view.
        // Avoid relying on NSView.tag/setTag (varies by SDK); instead, pick the first
        // subview that is NOT the NSVisualEffectView.
        NSView* overlayHost = nil;
        for (NSView* sub in contentView.subviews) {
            if (![sub isKindOfClass:[NSVisualEffectView class]]) {
                overlayHost = sub;
                break;
            }
        }
        if (overlayHost) contentView = overlayHost;

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
