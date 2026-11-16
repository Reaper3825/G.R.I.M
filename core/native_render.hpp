#pragma once

#include "core/host_exports.hpp"
#include <bgfx/embedded_shader.h>
#include <span>

namespace GRIM::Render {

struct VertexAttribute {
    bgfx::Attrib::Enum semantic;
    uint8_t count;
    bgfx::AttribType::Enum type;
    bool normalized = false;
};

GRIM_HOST_API bgfx::VertexLayout makeVertexLayout(std::span<const VertexAttribute> attributes);

// Forward native plugin submissions to the bgfx context owned by the host.
#define GRIM_RENDER_CALL(result, name, params, defaults, args) \
    GRIM_HOST_API result name defaults;
#include "core/native_render_calls.inc"
#undef GRIM_RENDER_CALL

} // namespace GRIM::Render
