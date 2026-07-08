#pragma once

#include "ui_3d_viewport.hpp"

#include <bgfx/bgfx.h>
#include <string>

class UINative3DViewportAttachment : public UI3DViewportAttachment {
public:
    UINative3DViewportAttachment(void* overlayWindowHandle, std::string debugName);
    ~UINative3DViewportAttachment() override;

    void onViewportGeometryChanged(const UI3DViewportGeometry& geometry) override;
    void onViewportDetached() override;

    void* nativeHandle() const { return viewportWindowHandle_; }
    bgfx::FrameBufferHandle frameBufferHandle() const { return frameBuffer_; }
    bool hasFrameBuffer() const { return bgfx::isValid(frameBuffer_); }
    uint16_t frameBufferWidth() const { return frameBufferWidth_; }
    uint16_t frameBufferHeight() const { return frameBufferHeight_; }
    UI3DViewportGeometry lastGeometry() const { return lastGeometry_; }

private:
    void ensureWindow();
    void hideWindow();
    void ensureFrameBuffer(uint16_t width, uint16_t height);
    void destroyFrameBuffer();

    void* overlayWindowHandle_ = nullptr;
    void* viewportWindowHandle_ = nullptr;
    bgfx::FrameBufferHandle frameBuffer_ = BGFX_INVALID_HANDLE;
    uint16_t frameBufferWidth_ = 0;
    uint16_t frameBufferHeight_ = 0;
    std::string debugName_;
    UI3DViewportGeometry lastGeometry_{};
};