#include "core/native_render.hpp"

namespace GRIM::Render {

bgfx::VertexLayout makeVertexLayout(std::span<const VertexAttribute> attributes) {
    bgfx::VertexLayout layout;
    layout.begin();
    for (const auto& attribute : attributes)
        layout.add(attribute.semantic, attribute.count, attribute.type, attribute.normalized);
    layout.end();
    return layout;
}

#define GRIM_RENDER_CALL(result, name, params, defaults, args) \
    result name params { return bgfx::name args; }
#include "core/native_render_calls.inc"
#undef GRIM_RENDER_CALL

} // namespace GRIM::Render
