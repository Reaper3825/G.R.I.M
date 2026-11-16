GRIM_RENDER_CALL(const bgfx::Caps*, getCaps, (), (), ())
GRIM_RENDER_CALL(const bgfx::Stats*, getStats, (), (), ())
GRIM_RENDER_CALL(bgfx::RendererType::Enum, getRendererType, (), (), ())
GRIM_RENDER_CALL(const bgfx::Memory*, copy, (const void* data, uint32_t size),
                 (const void* data, uint32_t size), (data, size))
GRIM_RENDER_CALL(bgfx::ShaderHandle, createEmbeddedShader,
                 (const bgfx::EmbeddedShader* shaders, bgfx::RendererType::Enum renderer, const char* name),
                 (const bgfx::EmbeddedShader* shaders, bgfx::RendererType::Enum renderer, const char* name),
                 (shaders, renderer, name))
GRIM_RENDER_CALL(bgfx::ProgramHandle, createProgram, (bgfx::ShaderHandle vs, bgfx::ShaderHandle fs, bool destroyShaders),
                 (bgfx::ShaderHandle vs, bgfx::ShaderHandle fs, bool destroyShaders), (vs, fs, destroyShaders))
GRIM_RENDER_CALL(bgfx::UniformHandle, createUniform, (const char* name, bgfx::UniformType::Enum type, uint16_t count),
                 (const char* name, bgfx::UniformType::Enum type, uint16_t count = 1), (name, type, count))
GRIM_RENDER_CALL(bgfx::TextureHandle, createTexture2D,
                 (uint16_t width, uint16_t height, bool mips, uint16_t layers, bgfx::TextureFormat::Enum format,
                  uint64_t flags, const bgfx::Memory* data),
                 (uint16_t width, uint16_t height, bool mips, uint16_t layers, bgfx::TextureFormat::Enum format,
                  uint64_t flags, const bgfx::Memory* data), (width, height, mips, layers, format, flags, data))
GRIM_RENDER_CALL(bgfx::VertexBufferHandle, createVertexBuffer,
                 (const bgfx::Memory* data, const bgfx::VertexLayout& layout, uint16_t flags),
                 (const bgfx::Memory* data, const bgfx::VertexLayout& layout, uint16_t flags = BGFX_BUFFER_NONE),
                 (data, layout, flags))
GRIM_RENDER_CALL(bgfx::IndexBufferHandle, createIndexBuffer, (const bgfx::Memory* data, uint16_t flags),
                 (const bgfx::Memory* data, uint16_t flags), (data, flags))

#define GRIM_RENDER_DESTROY(type) \
    GRIM_RENDER_CALL(void, destroy, (bgfx::type handle), (bgfx::type handle), (handle))
GRIM_RENDER_DESTROY(ShaderHandle)
GRIM_RENDER_DESTROY(ProgramHandle)
GRIM_RENDER_DESTROY(UniformHandle)
GRIM_RENDER_DESTROY(TextureHandle)
GRIM_RENDER_DESTROY(VertexBufferHandle)
GRIM_RENDER_DESTROY(IndexBufferHandle)
#undef GRIM_RENDER_DESTROY

GRIM_RENDER_CALL(uint32_t, getAvailTransientVertexBuffer, (uint32_t count, const bgfx::VertexLayout& layout),
                 (uint32_t count, const bgfx::VertexLayout& layout), (count, layout))
GRIM_RENDER_CALL(uint32_t, getAvailTransientIndexBuffer, (uint32_t count, bool index32),
                 (uint32_t count, bool index32), (count, index32))
GRIM_RENDER_CALL(void, allocTransientVertexBuffer,
                 (bgfx::TransientVertexBuffer* buffer, uint32_t count, const bgfx::VertexLayout& layout),
                 (bgfx::TransientVertexBuffer* buffer, uint32_t count, const bgfx::VertexLayout& layout),
                 (buffer, count, layout))
GRIM_RENDER_CALL(void, allocTransientIndexBuffer, (bgfx::TransientIndexBuffer* buffer, uint32_t count, bool index32),
                 (bgfx::TransientIndexBuffer* buffer, uint32_t count, bool index32), (buffer, count, index32))
GRIM_RENDER_CALL(void, setViewFrameBuffer, (bgfx::ViewId view, bgfx::FrameBufferHandle buffer),
                 (bgfx::ViewId view, bgfx::FrameBufferHandle buffer), (view, buffer))
GRIM_RENDER_CALL(void, setViewRect, (bgfx::ViewId view, uint16_t x, uint16_t y, uint16_t width, uint16_t height),
                 (bgfx::ViewId view, uint16_t x, uint16_t y, uint16_t width, uint16_t height), (view, x, y, width, height))
GRIM_RENDER_CALL(void, setViewTransform, (bgfx::ViewId view, const void* viewMatrix, const void* projection),
                 (bgfx::ViewId view, const void* viewMatrix, const void* projection), (view, viewMatrix, projection))
GRIM_RENDER_CALL(void, setViewMode, (bgfx::ViewId view, bgfx::ViewMode::Enum mode),
                 (bgfx::ViewId view, bgfx::ViewMode::Enum mode), (view, mode))
GRIM_RENDER_CALL(uint32_t, setTransform, (const void* matrix), (const void* matrix), (matrix))

#define GRIM_RENDER_BUFFER(type) \
    GRIM_RENDER_CALL(void, setVertexBuffer, (uint8_t stream, type buffer, uint32_t start, uint32_t count), \
                     (uint8_t stream, type buffer, uint32_t start, uint32_t count), (stream, buffer, start, count))
GRIM_RENDER_BUFFER(bgfx::VertexBufferHandle)
GRIM_RENDER_BUFFER(const bgfx::TransientVertexBuffer*)
#undef GRIM_RENDER_BUFFER
#define GRIM_RENDER_BUFFER(type) \
    GRIM_RENDER_CALL(void, setIndexBuffer, (type buffer, uint32_t start, uint32_t count), \
                     (type buffer, uint32_t start, uint32_t count), (buffer, start, count))
GRIM_RENDER_BUFFER(bgfx::IndexBufferHandle)
GRIM_RENDER_BUFFER(const bgfx::TransientIndexBuffer*)
#undef GRIM_RENDER_BUFFER

GRIM_RENDER_CALL(void, setUniform, (bgfx::UniformHandle uniform, const void* value),
                 (bgfx::UniformHandle uniform, const void* value), (uniform, value))
GRIM_RENDER_CALL(void, setTexture, (uint8_t stage, bgfx::UniformHandle sampler, bgfx::TextureHandle texture),
                 (uint8_t stage, bgfx::UniformHandle sampler, bgfx::TextureHandle texture), (stage, sampler, texture))
GRIM_RENDER_CALL(void, setState, (uint64_t state), (uint64_t state), (state))
GRIM_RENDER_CALL(void, submit, (bgfx::ViewId view, bgfx::ProgramHandle program),
                 (bgfx::ViewId view, bgfx::ProgramHandle program), (view, program))
GRIM_RENDER_CALL(void, touch, (bgfx::ViewId view), (bgfx::ViewId view), (view))
GRIM_RENDER_CALL(void, setViewClear, (bgfx::ViewId view, uint16_t flags, uint32_t rgba, float depth, uint8_t stencil),
                 (bgfx::ViewId view, uint16_t flags, uint32_t rgba, float depth, uint8_t stencil),
                 (view, flags, rgba, depth, stencil))
GRIM_RENDER_CALL(void, resetView, (bgfx::ViewId view), (bgfx::ViewId view), (view))
