#pragma once

#include "ui_native_3d_viewport_attachment.hpp"

#include <cstdint>
#include <memory>
#include <string>

class UINative3DViewportClearPass {
public:
    static void registerClearPass(const std::string& owner,
                                  std::weak_ptr<UINative3DViewportAttachment> attachment,
                                  uint32_t clearColorRgba);
    static void unregisterClearPass(const std::string& owner);
    static void render(uint32_t bgfxFrame);
};