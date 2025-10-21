#include <bgfx/bgfx.h>
#include <bgfx/embedded_shader.h>

// Define the embedded shader array
static const bgfx::EmbeddedShader s_embeddedShaders_data[] = {
    BGFX_EMBEDDED_SHADER_END()
};

// Global pointer that matches ui_renderer.cpp extern
const bgfx::EmbeddedShader* s_embeddedShaders = s_embeddedShaders_data;
