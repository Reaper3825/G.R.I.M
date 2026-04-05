#include "core/grim_platform.h"
#ifndef _WIN32

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>
#include "popup_window.hpp"
#include "logger.hpp"
#include <vector>
#include <algorithm>

// ===========================================================
// macOS Popup Window — NSWindow + CALayer frame presentation
// Equivalent to Win32 layered window with per-pixel alpha.
// ===========================================================

static NSWindow* s_popupWindow = nil;
static NSView*   s_contentView = nil;
static int s_popupWidth  = 0;
static int s_popupHeight = 0;

// ===========================================================
// Custom NSView that hosts a CALayer for bitmap presentation
// ===========================================================
@interface GRIMPopupView : NSView
@end

@implementation GRIMPopupView

- (BOOL)wantsUpdateLayer { return YES; }
- (BOOL)isFlipped { return YES; }

- (void)updateLayer {
    // Layer contents set externally via presentPopup3DFrame
}

@end

// ===========================================================
// Window creation
// ===========================================================
void* createPopupWindow(int width, int height)
{
    if (width <= 0 || height <= 0) {
        LOG_ERROR("PopupWindow", "Invalid popup dimensions: " +
                  std::to_string(width) + "x" + std::to_string(height));
        return nullptr;
    }

    // NSWindow MUST be created on the main thread.
    // Use dispatch_sync to block until creation completes.
    __block NSWindow* window = nil;

    void (^createBlock)(void) = ^{
        @autoreleasepool {
            NSScreen* screen = [NSScreen mainScreen];
            if (!screen) {
                LOG_ERROR("PopupWindow", "No main screen available");
                return;
            }

            NSRect screenFrame = [screen frame];
            CGFloat windowX = NSMaxX(screenFrame) - width - 100;
            CGFloat windowY = 100;  // macOS: bottom-left origin

            NSRect frame = NSMakeRect(windowX, windowY, width, height);

            window = [[NSWindow alloc]
                initWithContentRect:frame
                          styleMask:NSWindowStyleMaskBorderless
                            backing:NSBackingStoreBuffered
                              defer:NO];

            [window setLevel:NSFloatingWindowLevel];
            [window setOpaque:NO];
            [window setBackgroundColor:[NSColor clearColor]];
            [window setHasShadow:YES];
            [window setIgnoresMouseEvents:NO];
            [window setCollectionBehavior:
                NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorFullScreenAuxiliary];
            [window setReleasedWhenClosed:NO];

            GRIMPopupView* contentView = [[GRIMPopupView alloc]
                initWithFrame:NSMakeRect(0, 0, width, height)];
            [contentView setWantsLayer:YES];
            contentView.layer.opaque = NO;
            contentView.layer.backgroundColor = [NSColor clearColor].CGColor;
            contentView.layer.contentsGravity = kCAGravityResize;
            contentView.layer.magnificationFilter = kCAFilterLinear;

            [window setContentView:contentView];

            s_popupWindow = window;
            s_contentView = contentView;
            s_popupWidth  = width;
            s_popupHeight = height;
        }
    };

    if ([NSThread isMainThread]) {
        createBlock();
    } else {
        dispatch_sync(dispatch_get_main_queue(), createBlock);
    }

    if (!window) {
        LOG_ERROR("PopupWindow", "Failed to create macOS popup window");
        return nullptr;
    }

    LOG_DEBUG("PopupWindow", "macOS: Popup window created (" +
              std::to_string(width) + "x" + std::to_string(height) + ")");

    return (__bridge void*)window;
}

// ===========================================================
// Show / Hide / Visibility
// ===========================================================
void showPopupWindow(void* handle)
{
    if (!handle) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
            NSWindow* window = (__bridge NSWindow*)handle;
            [window orderFront:nil];
            [window makeKeyAndOrderFront:nil];
            LOG_DEBUG("PopupWindow", "orderFront dispatched, frame=" +
                      std::to_string((int)[window frame].origin.x) + "," +
                      std::to_string((int)[window frame].origin.y) + " " +
                      std::to_string((int)[window frame].size.width) + "x" +
                      std::to_string((int)[window frame].size.height) +
                      " visible=" + std::to_string([window isVisible]) +
                      " alpha=" + std::to_string([window alphaValue]));
        }
    });
}

void hidePopupWindow(void* handle)
{
    if (!handle) return;
    dispatch_async(dispatch_get_main_queue(), ^{
        @autoreleasepool {
            NSWindow* window = (__bridge NSWindow*)handle;
            [window orderOut:nil];
        }
    });
}

bool isPopupWindowValid(void* handle)
{
    return handle != nullptr;
}

// ===========================================================
// Present a 3D-rendered BGRA frame to the popup window.
// Input: straight-alpha BGRA8 pixels from offscreen readback.
// macOS equivalent of Win32 UpdateLayeredWindow.
// ===========================================================
void presentPopup3DFrame(void* handle, const uint8_t* bgraData, int width, int height)
{
    if (!handle || !bgraData || width <= 0 || height <= 0)
        return;

    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)handle;
        NSView* view = [window contentView];
        if (!view || ![view wantsLayer])
            return;

        // Create a CGImage from the BGRA data.
        // The readback produces straight-alpha BGRA8. CoreGraphics
        // wants premultiplied RGBA, so we convert in-place.
        size_t pixelCount = static_cast<size_t>(width * height);
        size_t byteCount  = pixelCount * 4;

        // Convert BGRA straight → RGBA premultiplied
        std::vector<uint8_t> rgbaPremul(byteCount);
        for (size_t i = 0; i < pixelCount; ++i) {
            uint8_t b = bgraData[i * 4 + 0];
            uint8_t g = bgraData[i * 4 + 1];
            uint8_t r = bgraData[i * 4 + 2];
            uint8_t a = bgraData[i * 4 + 3];
            rgbaPremul[i * 4 + 0] = static_cast<uint8_t>((r * a) / 255);
            rgbaPremul[i * 4 + 1] = static_cast<uint8_t>((g * a) / 255);
            rgbaPremul[i * 4 + 2] = static_cast<uint8_t>((b * a) / 255);
            rgbaPremul[i * 4 + 3] = a;
        }

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(
            rgbaPremul.data(),
            width, height,
            8,                      // bits per component
            width * 4,              // bytes per row
            colorSpace,
            kCGImageAlphaPremultipliedLast | kCGBitmapByteOrderDefault
        );

        if (!ctx) {
            CGColorSpaceRelease(colorSpace);
            return;
        }

        CGImageRef image = CGBitmapContextCreateImage(ctx);
        CGContextRelease(ctx);
        CGColorSpaceRelease(colorSpace);

        if (!image)
            return;

        // Push the image to the CALayer on the main thread
        dispatch_async(dispatch_get_main_queue(), ^{
            CALayer* layer = view.layer;
            if (layer) {
                [CATransaction begin];
                [CATransaction setDisableActions:YES];
                layer.contents = (__bridge id)image;
                [CATransaction commit];
            } else {
                LOG_ERROR("PopupWindow", "CALayer is nil in present dispatch");
            }
            CGImageRelease(image);
        });
    }
}

// ===========================================================
// Handle click on popup window (show console)
// This is checked by the pop_ui loop via mouse position.
// ===========================================================
bool isPointInPopupWindow(void* handle, int screenX, int screenY)
{
    if (!handle) return false;
    @autoreleasepool {
        NSWindow* window = (__bridge NSWindow*)handle;
        NSRect frame = [window frame];
        // macOS screen coords: origin bottom-left
        NSPoint point = NSMakePoint(screenX, screenY);
        return NSPointInRect(point, frame);
    }
}

#endif // !_WIN32
