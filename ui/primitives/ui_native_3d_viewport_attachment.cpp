#include "ui_native_3d_viewport_attachment.hpp"

#include "core/platform_window.hpp"
#include "core/window_manager.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

UINative3DViewportAttachment::UINative3DViewportAttachment(void* overlayWindowHandle, std::string debugName)
    : overlayWindowHandle_(overlayWindowHandle), debugName_(std::move(debugName))
{
    if (!overlayWindowHandle_)
        throw std::runtime_error("UINative3DViewportAttachment requires a valid overlay window handle");
    if (debugName_.empty())
        throw std::runtime_error("UINative3DViewportAttachment requires a non-empty debug name");
}

UINative3DViewportAttachment::~UINative3DViewportAttachment()
{
    onViewportDetached();
}

void UINative3DViewportAttachment::onViewportGeometryChanged(const UI3DViewportGeometry& geometry)
{
    lastGeometry_ = geometry;

    if (!geometry.visible || geometry.pixelWidth <= 0 || geometry.pixelHeight <= 0) {
        hideWindow();
        return;
    }

    ensureWindow();
    PlatformWindow::setViewportWindowBounds(viewportWindowHandle_,
                                            overlayWindowHandle_,
                                            geometry.pixelX,
                                            geometry.pixelY,
                                            geometry.pixelWidth,
                                            geometry.pixelHeight);
    if (geometry.pixelWidth > static_cast<int>(std::numeric_limits<uint16_t>::max())
        || geometry.pixelHeight > static_cast<int>(std::numeric_limits<uint16_t>::max())) {
        throw std::runtime_error("UINative3DViewportAttachment viewport dimensions exceed bgfx uint16 limits for '" + debugName_ + "'");
    }

    ensureFrameBuffer(static_cast<uint16_t>(geometry.pixelWidth),
                      static_cast<uint16_t>(geometry.pixelHeight));
    PlatformWindow::setViewportWindowVisible(viewportWindowHandle_, true);
}

void UINative3DViewportAttachment::onViewportDetached()
{
    destroyFrameBuffer();

    if (!viewportWindowHandle_)
        return;

    PlatformWindow::destroyViewportWindow(viewportWindowHandle_);
    viewportWindowHandle_ = nullptr;
}

void UINative3DViewportAttachment::ensureWindow()
{
    if (viewportWindowHandle_)
        return;

    viewportWindowHandle_ = PlatformWindow::createViewportWindow(overlayWindowHandle_, debugName_.c_str());
    if (!viewportWindowHandle_)
        throw std::runtime_error("UINative3DViewportAttachment failed to create viewport window for '" + debugName_ + "'");
}

void UINative3DViewportAttachment::hideWindow()
{
    if (viewportWindowHandle_)
        PlatformWindow::setViewportWindowVisible(viewportWindowHandle_, false);
}

void UINative3DViewportAttachment::ensureFrameBuffer(uint16_t width, uint16_t height)
{
    if (width == 0 || height == 0)
        throw std::runtime_error("UINative3DViewportAttachment framebuffer dimensions must be > 0 for '" + debugName_ + "'");
    if (!viewportWindowHandle_)
        throw std::runtime_error("UINative3DViewportAttachment cannot create framebuffer before native window for '" + debugName_ + "'");
    if (!WindowManager::isInitialized())
        throw std::runtime_error("UINative3DViewportAttachment cannot create framebuffer before bgfx initialization for '" + debugName_ + "'");

    if (bgfx::isValid(frameBuffer_)
        && frameBufferWidth_ == width
        && frameBufferHeight_ == height) {
        return;
    }

    destroyFrameBuffer();

    frameBuffer_ = bgfx::createFrameBuffer(viewportWindowHandle_,
                                           width,
                                           height,
                                           bgfx::TextureFormat::BGRA8,
                                           bgfx::TextureFormat::D24S8);
    if (!bgfx::isValid(frameBuffer_))
        throw std::runtime_error("UINative3DViewportAttachment failed to create native-window framebuffer for '" + debugName_ + "'");

    bgfx::setName(frameBuffer_, debugName_.c_str());
    frameBufferWidth_ = width;
    frameBufferHeight_ = height;
}

void UINative3DViewportAttachment::destroyFrameBuffer()
{
    if (bgfx::isValid(frameBuffer_))
        bgfx::destroy(frameBuffer_);

    frameBuffer_ = BGFX_INVALID_HANDLE;
    frameBufferWidth_ = 0;
    frameBufferHeight_ = 0;
}