#include "cesium_bgfx_render_adapter.hpp"

#ifdef OPAQUE
#undef OPAQUE
#endif

#include "core/window_manager.hpp"
#include "popup_ui/popup_3d_shaders.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"

#include <Cesium3DTilesSelection/TileContent.h>
#include <Cesium3DTilesSelection/TileLoadResult.h>
#include <Cesium3DTilesSelection/ViewUpdateResult.h>
#include <CesiumAsync/AsyncSystem.h>
#include <CesiumGltf/Accessor.h>
#include <CesiumGltf/Buffer.h>
#include <CesiumGltf/BufferView.h>
#include <CesiumGltf/Image.h>
#include <CesiumGltf/Material.h>
#include <CesiumGltf/MaterialPBRMetallicRoughness.h>
#include <CesiumGltf/MeshPrimitive.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltf/Texture.h>
#include <CesiumGltf/TextureInfo.h>
#include <CesiumImage/ImageAsset.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace GRIM::GeoSpatial {

namespace {
    constexpr const char* kRenderPassName = "geospatial_cesium_bgfx_render_adapter";
    constexpr uint64_t kOpaqueState = BGFX_STATE_WRITE_RGB |
                                      BGFX_STATE_WRITE_A |
                                      BGFX_STATE_WRITE_Z |
                                      BGFX_STATE_DEPTH_TEST_LESS |
                                      BGFX_STATE_CULL_CW;
    constexpr uint64_t kAlphaState = kOpaqueState |
                                     BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA,
                                                           BGFX_STATE_BLEND_INV_SRC_ALPHA);

    struct CesiumBgfxVertex {
        float px = 0.0f;
        float py = 0.0f;
        float pz = 0.0f;
        float nx = 0.0f;
        float ny = 0.0f;
        float nz = 1.0f;
        float tx = 1.0f;
        float ty = 0.0f;
        float tz = 0.0f;
        float tw = 1.0f;
        float u = 0.0f;
        float v = 0.0f;
        uint32_t color = 0xFFFFFFFFu;
    };

    struct CpuImage {
        uint16_t width = 0;
        uint16_t height = 0;
        std::vector<std::byte> rgba8;
    };

    struct CpuPrimitive {
        std::vector<CesiumBgfxVertex> vertices;
        std::vector<uint32_t> indices;
        glm::dmat4 transform = glm::dmat4(1.0);
        int32_t baseColorTexture = -1;
        std::array<float, 4> baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
        bool doubleSided = false;
        bool translucent = false;
    };

    struct LoadTileResources {
        std::vector<CpuPrimitive> primitives;
        std::vector<CpuImage> images;
    };

    struct GpuTexture {
        bgfx::TextureHandle handle = BGFX_INVALID_HANDLE;
    };

    struct RasterTextureResources {
        GpuTexture texture;
    };

    struct GpuPrimitive {
        bgfx::VertexBufferHandle vertexBuffer = BGFX_INVALID_HANDLE;
        bgfx::IndexBufferHandle indexBuffer = BGFX_INVALID_HANDLE;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
        glm::dmat4 transform = glm::dmat4(1.0);
        int32_t baseColorTexture = -1;
        std::array<float, 4> baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
        bool doubleSided = false;
        bool translucent = false;
    };

    struct RasterAttachment {
        int32_t overlayTextureCoordinateID = -1;
        RasterTextureResources* resources = nullptr;
        glm::dvec2 translation{0.0, 0.0};
        glm::dvec2 scale{1.0, 1.0};
    };

    struct TileGpuResources {
        std::vector<GpuPrimitive> primitives;
        std::vector<GpuTexture> textures;
        std::vector<RasterAttachment> rasterAttachments;
    };

    std::mutex g_registryMutex;
    std::vector<CesiumBgfxRenderAdapter*> g_adapters;
    bool g_renderPassRegistered = false;
    bgfx::VertexLayout g_vertexLayout;
    bool g_vertexLayoutInitialized = false;
    PopupShaderState* g_shaderState = nullptr;
    bgfx::TextureHandle g_whiteTexture = BGFX_INVALID_HANDLE;

    bgfx::VertexLayout& vertexLayout()
    {
        if (!g_vertexLayoutInitialized) {
            g_vertexLayout.begin()
                .add(bgfx::Attrib::Position, 3, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Normal, 3, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Tangent, 4, bgfx::AttribType::Float)
                .add(bgfx::Attrib::TexCoord0, 2, bgfx::AttribType::Float)
                .add(bgfx::Attrib::Color0, 4, bgfx::AttribType::Uint8, true)
            .end();
            g_vertexLayoutInitialized = true;
        }
        return g_vertexLayout;
    }

    void destroyTexture(GpuTexture& texture) noexcept
    {
        if (bgfx::isValid(texture.handle)) {
            bgfx::destroy(texture.handle);
            texture.handle = BGFX_INVALID_HANDLE;
        }
    }

    void destroyGpuPrimitive(GpuPrimitive& primitive) noexcept
    {
        if (bgfx::isValid(primitive.vertexBuffer)) {
            bgfx::destroy(primitive.vertexBuffer);
            primitive.vertexBuffer = BGFX_INVALID_HANDLE;
        }
        if (bgfx::isValid(primitive.indexBuffer)) {
            bgfx::destroy(primitive.indexBuffer);
            primitive.indexBuffer = BGFX_INVALID_HANDLE;
        }
    }

    void destroyTileResources(TileGpuResources* resources) noexcept
    {
        if (!resources)
            return;

        for (GpuPrimitive& primitive : resources->primitives)
            destroyGpuPrimitive(primitive);
        for (GpuTexture& texture : resources->textures)
            destroyTexture(texture);
        delete resources;
    }

    void destroyRasterResources(RasterTextureResources* resources) noexcept
    {
        if (!resources)
            return;

        destroyTexture(resources->texture);
        delete resources;
    }

    void ensureGpuGlobals()
    {
        vertexLayout();

        if (!g_shaderState)
            g_shaderState = popupShadersCreate();

        if (!bgfx::isValid(g_whiteTexture)) {
            const uint32_t white = 0xFFFFFFFFu;
            const bgfx::Memory* memory = bgfx::copy(&white, sizeof(white));
            g_whiteTexture = bgfx::createTexture2D(1, 1, false, 1,
                                                   bgfx::TextureFormat::RGBA8,
                                                   BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP,
                                                   memory);
            if (!bgfx::isValid(g_whiteTexture))
                throw std::runtime_error("CesiumBgfxRenderAdapter failed to create white fallback texture");
        }
    }

    void shutdownGpuGlobalsIfUnused()
    {
        if (!g_adapters.empty())
            return;

        if (bgfx::isValid(g_whiteTexture)) {
            bgfx::destroy(g_whiteTexture);
            g_whiteTexture = BGFX_INVALID_HANDLE;
        }
        if (g_shaderState) {
            popupShadersDestroy(g_shaderState);
            g_shaderState = nullptr;
        }
    }

    void registerAdapter(CesiumBgfxRenderAdapter* adapter)
    {
        if (!adapter)
            throw std::runtime_error("CesiumBgfxRenderAdapter registry received NULL adapter");

        std::lock_guard lock(g_registryMutex);
        g_adapters.push_back(adapter);
        if (!g_renderPassRegistered) {
            WindowManager::registerRenderPass(kRenderPassName, &CesiumBgfxRenderAdapter::renderAll, true);
            g_renderPassRegistered = true;
        }
    }

    void unregisterAdapter(CesiumBgfxRenderAdapter* adapter)
    {
        std::lock_guard lock(g_registryMutex);
        auto it = std::remove(g_adapters.begin(), g_adapters.end(), adapter);
        g_adapters.erase(it, g_adapters.end());
        if (g_adapters.empty() && g_renderPassRegistered) {
            WindowManager::unregisterRenderPass(kRenderPassName);
            g_renderPassRegistered = false;
            shutdownGpuGlobalsIfUnused();
        }
    }

    const CesiumGltf::Accessor& checkedAccessor(const CesiumGltf::Model& model, int32_t accessorIndex, const char* semantic)
    {
        if (accessorIndex < 0 || static_cast<size_t>(accessorIndex) >= model.accessors.size())
            throw std::runtime_error(std::string("Cesium glTF primitive has invalid accessor for ") + semantic);
        return model.accessors[static_cast<size_t>(accessorIndex)];
    }

    const std::byte* accessorBase(const CesiumGltf::Model& model, const CesiumGltf::Accessor& accessor, const char* semantic)
    {
        if (accessor.bufferView < 0 || static_cast<size_t>(accessor.bufferView) >= model.bufferViews.size())
            throw std::runtime_error(std::string("Cesium glTF accessor has invalid bufferView for ") + semantic);

        const CesiumGltf::BufferView& bufferView = model.bufferViews[static_cast<size_t>(accessor.bufferView)];
        if (bufferView.buffer < 0 || static_cast<size_t>(bufferView.buffer) >= model.buffers.size())
            throw std::runtime_error(std::string("Cesium glTF bufferView has invalid buffer for ") + semantic);

        const std::vector<std::byte>& data = model.buffers[static_cast<size_t>(bufferView.buffer)].cesium.data;
        const int64_t start = bufferView.byteOffset + accessor.byteOffset;
        const int64_t stride = accessor.computeByteStride(model);
        if (start < 0 || stride <= 0 || accessor.count < 0)
            throw std::runtime_error(std::string("Cesium glTF accessor has invalid byte layout for ") + semantic);

        const int64_t bytesNeeded = start + (accessor.count == 0 ? 0 : (accessor.count - 1) * stride + accessor.computeBytesPerVertex());
        if (bytesNeeded < 0 || static_cast<size_t>(bytesNeeded) > data.size())
            throw std::runtime_error(std::string("Cesium glTF accessor exceeds buffer bounds for ") + semantic);

        return data.data() + start;
    }

    std::optional<int32_t> findAttribute(const CesiumGltf::MeshPrimitive& primitive, const char* name)
    {
        auto it = primitive.attributes.find(name);
        if (it == primitive.attributes.end())
            return std::nullopt;
        return it->second;
    }

    std::array<float, 3> readVec3Float(const std::byte* base, int64_t stride, int64_t index)
    {
        const float* values = reinterpret_cast<const float*>(base + index * stride);
        return {values[0], values[1], values[2]};
    }

    std::array<float, 2> readVec2Float(const std::byte* base, int64_t stride, int64_t index)
    {
        const float* values = reinterpret_cast<const float*>(base + index * stride);
        return {values[0], values[1]};
    }

    uint32_t readIndexValue(const std::byte* base, int64_t stride, int64_t index, int32_t componentType)
    {
        const std::byte* address = base + index * stride;
        switch (componentType) {
            case CesiumGltf::Accessor::ComponentType::UNSIGNED_BYTE:
                return static_cast<uint32_t>(*reinterpret_cast<const uint8_t*>(address));
            case CesiumGltf::Accessor::ComponentType::UNSIGNED_SHORT:
                return static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(address));
            case CesiumGltf::Accessor::ComponentType::UNSIGNED_INT:
                return *reinterpret_cast<const uint32_t*>(address);
            default:
                throw std::runtime_error("Cesium glTF primitive uses unsupported index component type");
        }
    }

    void requireFloatAccessor(const CesiumGltf::Accessor& accessor, const std::string& type, const char* semantic)
    {
        if (accessor.componentType != CesiumGltf::Accessor::ComponentType::FLOAT || accessor.type != type)
            throw std::runtime_error(std::string("Cesium glTF ") + semantic + " accessor must be float " + type);
        if (accessor.sparse)
            throw std::runtime_error(std::string("Cesium glTF ") + semantic + " sparse accessors are not supported yet");
    }

    CpuImage makeCpuImage(const CesiumImage::ImageAsset& image)
    {
        if (image.width <= 0 || image.height <= 0)
            throw std::runtime_error("Cesium image has invalid dimensions");
        if (image.width > 65535 || image.height > 65535)
            throw std::runtime_error("Cesium image dimensions exceed bgfx texture2D uint16 limits");
        if (image.compressedPixelFormat != CesiumImage::GpuCompressedPixelFormat::NONE)
            throw std::runtime_error("Cesium compressed image formats are not supported by the bgfx adapter yet");
        if (image.bytesPerChannel != 1)
            throw std::runtime_error("Cesium image bytesPerChannel must be 1 for RGBA8 upload");
        if (image.channels < 1 || image.channels > 4)
            throw std::runtime_error("Cesium image channel count must be 1..4");

        const size_t pixelCount = static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
        const size_t expectedBytes = pixelCount * static_cast<size_t>(image.channels);
        if (image.pixelData.size() < expectedBytes)
            throw std::runtime_error("Cesium image pixel data is shorter than dimensions require");

        CpuImage result;
        result.width = static_cast<uint16_t>(image.width);
        result.height = static_cast<uint16_t>(image.height);
        result.rgba8.resize(pixelCount * 4);

        for (size_t i = 0; i < pixelCount; ++i) {
            const std::byte* source = image.pixelData.data() + i * static_cast<size_t>(image.channels);
            std::byte* target = result.rgba8.data() + i * 4;
            target[0] = source[0];
            target[1] = image.channels >= 3 ? source[1] : source[0];
            target[2] = image.channels >= 3 ? source[2] : source[0];
            target[3] = image.channels == 2 ? source[1] : (image.channels == 4 ? source[3] : std::byte{255});
        }

        return result;
    }

    std::vector<CpuImage> buildImages(const CesiumGltf::Model& model)
    {
        std::vector<CpuImage> images;
        images.reserve(model.images.size());
        for (const CesiumGltf::Image& image : model.images) {
            if (!image.pAsset) {
                images.push_back(CpuImage{});
                continue;
            }
            images.push_back(makeCpuImage(*image.pAsset));
        }
        return images;
    }

    int32_t materialBaseTexture(const CesiumGltf::Model& model, const CesiumGltf::Material& material)
    {
        if (!material.pbrMetallicRoughness || !material.pbrMetallicRoughness->baseColorTexture)
            return -1;

        const int32_t textureIndex = material.pbrMetallicRoughness->baseColorTexture->index;
        if (textureIndex < 0 || static_cast<size_t>(textureIndex) >= model.textures.size())
            throw std::runtime_error("Cesium material references invalid base color texture");

        const int32_t imageIndex = model.textures[static_cast<size_t>(textureIndex)].source;
        if (imageIndex < 0 || static_cast<size_t>(imageIndex) >= model.images.size())
            throw std::runtime_error("Cesium texture references invalid image source");

        return imageIndex;
    }

    std::array<float, 4> materialBaseColor(const CesiumGltf::Material& material)
    {
        std::array<float, 4> color{1.0f, 1.0f, 1.0f, 1.0f};
        if (!material.pbrMetallicRoughness)
            return color;

        const std::vector<double>& source = material.pbrMetallicRoughness->baseColorFactor;
        for (size_t i = 0; i < std::min(source.size(), color.size()); ++i)
            color[i] = static_cast<float>(source[i]);
        return color;
    }

    CpuPrimitive buildPrimitive(const CesiumGltf::Model& model,
                                const CesiumGltf::MeshPrimitive& primitive,
                                const glm::dmat4& transform)
    {
        if (primitive.mode != CesiumGltf::MeshPrimitive::Mode::TRIANGLES)
            throw std::runtime_error("Cesium bgfx adapter only supports triangle primitives");

        const std::optional<int32_t> positionIndex = findAttribute(primitive, "POSITION");
        if (!positionIndex)
            throw std::runtime_error("Cesium glTF primitive is missing POSITION");

        const CesiumGltf::Accessor& positionAccessor = checkedAccessor(model, *positionIndex, "POSITION");
        requireFloatAccessor(positionAccessor, CesiumGltf::Accessor::Type::VEC3, "POSITION");

        const std::optional<int32_t> normalIndex = findAttribute(primitive, "NORMAL");
        const std::optional<int32_t> texcoordIndex = findAttribute(primitive, "TEXCOORD_0");
        const CesiumGltf::Accessor* normalAccessor = nullptr;
        const CesiumGltf::Accessor* texcoordAccessor = nullptr;
        if (normalIndex) {
            normalAccessor = &checkedAccessor(model, *normalIndex, "NORMAL");
            requireFloatAccessor(*normalAccessor, CesiumGltf::Accessor::Type::VEC3, "NORMAL");
        }
        if (texcoordIndex) {
            texcoordAccessor = &checkedAccessor(model, *texcoordIndex, "TEXCOORD_0");
            requireFloatAccessor(*texcoordAccessor, CesiumGltf::Accessor::Type::VEC2, "TEXCOORD_0");
        }

        const std::byte* positionBase = accessorBase(model, positionAccessor, "POSITION");
        const int64_t positionStride = positionAccessor.computeByteStride(model);
        const std::byte* normalBase = normalAccessor ? accessorBase(model, *normalAccessor, "NORMAL") : nullptr;
        const int64_t normalStride = normalAccessor ? normalAccessor->computeByteStride(model) : 0;
        const std::byte* texcoordBase = texcoordAccessor ? accessorBase(model, *texcoordAccessor, "TEXCOORD_0") : nullptr;
        const int64_t texcoordStride = texcoordAccessor ? texcoordAccessor->computeByteStride(model) : 0;

        CpuPrimitive result;
        result.transform = transform;
        result.vertices.resize(static_cast<size_t>(positionAccessor.count));
        for (int64_t i = 0; i < positionAccessor.count; ++i) {
            const std::array<float, 3> position = readVec3Float(positionBase, positionStride, i);
            const std::array<float, 3> normal = normalBase ? readVec3Float(normalBase, normalStride, i)
                                                          : std::array<float, 3>{0.0f, 0.0f, 1.0f};
            const std::array<float, 2> texcoord = texcoordBase ? readVec2Float(texcoordBase, texcoordStride, i)
                                                               : std::array<float, 2>{0.0f, 0.0f};

            CesiumBgfxVertex& vertex = result.vertices[static_cast<size_t>(i)];
            vertex.px = position[0];
            vertex.py = position[1];
            vertex.pz = position[2];
            vertex.nx = normal[0];
            vertex.ny = normal[1];
            vertex.nz = normal[2];
            vertex.u = texcoord[0];
            vertex.v = texcoord[1];
        }

        if (primitive.indices >= 0) {
            const CesiumGltf::Accessor& indexAccessor = checkedAccessor(model, primitive.indices, "INDICES");
            if (indexAccessor.type != CesiumGltf::Accessor::Type::SCALAR || indexAccessor.sparse)
                throw std::runtime_error("Cesium glTF index accessor must be non-sparse SCALAR");

            const std::byte* indexBase = accessorBase(model, indexAccessor, "INDICES");
            const int64_t indexStride = indexAccessor.computeByteStride(model);
            result.indices.resize(static_cast<size_t>(indexAccessor.count));
            for (int64_t i = 0; i < indexAccessor.count; ++i)
                result.indices[static_cast<size_t>(i)] = readIndexValue(indexBase, indexStride, i, indexAccessor.componentType);
        } else {
            result.indices.resize(result.vertices.size());
            for (uint32_t i = 0; i < result.indices.size(); ++i)
                result.indices[i] = i;
        }

        if (primitive.material >= 0) {
            if (static_cast<size_t>(primitive.material) >= model.materials.size())
                throw std::runtime_error("Cesium glTF primitive references invalid material");

            const CesiumGltf::Material& material = model.materials[static_cast<size_t>(primitive.material)];
            result.baseColorTexture = materialBaseTexture(model, material);
            result.baseColorFactor = materialBaseColor(material);
            result.doubleSided = material.doubleSided;
            result.translucent = material.alphaMode == CesiumGltf::Material::AlphaMode::BLEND || result.baseColorFactor[3] < 1.0f;
        }

        return result;
    }

    LoadTileResources* buildTileResources(Cesium3DTilesSelection::TileLoadResult& tileLoadResult,
                                          const glm::dmat4& tileTransform)
    {
        auto* resources = new LoadTileResources();
        CesiumGltf::Model* model = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
        if (!model)
            return resources;

        resources->images = buildImages(*model);
        model->forEachPrimitiveInScene(model->scene,
            [&](const CesiumGltf::Model& gltf,
                const CesiumGltf::Node&,
                const CesiumGltf::Mesh&,
                const CesiumGltf::MeshPrimitive& primitive,
                const glm::dmat4& nodeTransform) {
                    resources->primitives.push_back(buildPrimitive(gltf, primitive, tileTransform * nodeTransform));
                });
        return resources;
    }

    bgfx::TextureHandle createTexture(const CpuImage& image)
    {
        if (image.width == 0 || image.height == 0 || image.rgba8.empty())
            return BGFX_INVALID_HANDLE;

        const bgfx::Memory* memory = bgfx::copy(image.rgba8.data(), static_cast<uint32_t>(image.rgba8.size()));
        bgfx::TextureHandle handle = bgfx::createTexture2D(image.width,
                                                           image.height,
                                                           false,
                                                           1,
                                                           bgfx::TextureFormat::RGBA8,
                                                           BGFX_SAMPLER_U_CLAMP | BGFX_SAMPLER_V_CLAMP,
                                                           memory);
        if (!bgfx::isValid(handle))
            throw std::runtime_error("CesiumBgfxRenderAdapter failed to create bgfx texture");
        return handle;
    }

    TileGpuResources* createGpuResources(LoadTileResources* loadResources)
    {
        if (!loadResources)
            throw std::runtime_error("CesiumBgfxRenderAdapter main-thread preparation received NULL load resources");

        ensureGpuGlobals();
        auto* resources = new TileGpuResources();
        try {
            resources->textures.reserve(loadResources->images.size());
            for (const CpuImage& image : loadResources->images) {
                GpuTexture texture;
                texture.handle = createTexture(image);
                resources->textures.push_back(texture);
            }

            resources->primitives.reserve(loadResources->primitives.size());
            for (const CpuPrimitive& cpuPrimitive : loadResources->primitives) {
                if (cpuPrimitive.vertices.empty() || cpuPrimitive.indices.empty())
                    continue;

                GpuPrimitive primitive;
                primitive.vertexCount = static_cast<uint32_t>(cpuPrimitive.vertices.size());
                primitive.indexCount = static_cast<uint32_t>(cpuPrimitive.indices.size());
                primitive.transform = cpuPrimitive.transform;
                primitive.baseColorTexture = cpuPrimitive.baseColorTexture;
                primitive.baseColorFactor = cpuPrimitive.baseColorFactor;
                primitive.doubleSided = cpuPrimitive.doubleSided;
                primitive.translucent = cpuPrimitive.translucent;

                const bgfx::Memory* vertexMemory = bgfx::copy(cpuPrimitive.vertices.data(),
                                                              static_cast<uint32_t>(cpuPrimitive.vertices.size() * sizeof(CesiumBgfxVertex)));
                primitive.vertexBuffer = bgfx::createVertexBuffer(vertexMemory, vertexLayout());
                if (!bgfx::isValid(primitive.vertexBuffer))
                    throw std::runtime_error("CesiumBgfxRenderAdapter failed to create vertex buffer");

                const bgfx::Memory* indexMemory = bgfx::copy(cpuPrimitive.indices.data(),
                                                             static_cast<uint32_t>(cpuPrimitive.indices.size() * sizeof(uint32_t)));
                primitive.indexBuffer = bgfx::createIndexBuffer(indexMemory, BGFX_BUFFER_INDEX32);
                if (!bgfx::isValid(primitive.indexBuffer))
                    throw std::runtime_error("CesiumBgfxRenderAdapter failed to create index buffer");

                resources->primitives.push_back(primitive);
            }
        } catch (...) {
            destroyTileResources(resources);
            throw;
        }

        return resources;
    }

    void matrixToFloatColumnMajor(const glm::dmat4& source, float target[16])
    {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row)
                target[column * 4 + row] = static_cast<float>(source[column][row]);
        }
    }

    bgfx::TextureHandle textureForPrimitive(const TileGpuResources& resources, const GpuPrimitive& primitive)
    {
        if (!resources.rasterAttachments.empty() && resources.rasterAttachments.front().resources &&
            bgfx::isValid(resources.rasterAttachments.front().resources->texture.handle)) {
            return resources.rasterAttachments.front().resources->texture.handle;
        }

        if (primitive.baseColorTexture >= 0 && static_cast<size_t>(primitive.baseColorTexture) < resources.textures.size() &&
            bgfx::isValid(resources.textures[static_cast<size_t>(primitive.baseColorTexture)].handle)) {
            return resources.textures[static_cast<size_t>(primitive.baseColorTexture)].handle;
        }

        return g_whiteTexture;
    }

    TileGpuResources* tileResources(const Cesium3DTilesSelection::Tile& tile)
    {
        const Cesium3DTilesSelection::TileRenderContent* renderContent = tile.getContent().getRenderContent();
        if (!renderContent)
            return nullptr;
        return static_cast<TileGpuResources*>(renderContent->getRenderResources());
    }
}

CesiumBgfxRenderAdapter::CesiumBgfxRenderAdapter(std::string owner)
    : owner_(std::move(owner))
{
    if (owner_.empty())
        throw std::runtime_error("CesiumBgfxRenderAdapter requires a non-empty owner");

    viewId_ = WindowManager::reserveViewIds(owner_, WindowManager::ViewIdRange::PanelViewport, 1).at(0);
    registerAdapter(this);
}

CesiumBgfxRenderAdapter::~CesiumBgfxRenderAdapter()
{
    unregisterAdapter(this);
    WindowManager::releaseViewIds(owner_);
}

void CesiumBgfxRenderAdapter::setViewportAttachment(std::weak_ptr<UINative3DViewportAttachment> attachment)
{
    std::lock_guard lock(mutex_);
    viewportAttachment_ = std::move(attachment);
}

void CesiumBgfxRenderAdapter::setFrameSelection(std::vector<Cesium3DTilesSelection::Tile::ConstPointer> tiles,
                                                glm::dmat4 view,
                                                glm::dmat4 projection)
{
    std::lock_guard lock(mutex_);
    frameTiles_ = std::move(tiles);
    view_ = view;
    projection_ = projection;
}

CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
CesiumBgfxRenderAdapter::prepareInLoadThread(const CesiumAsync::AsyncSystem& asyncSystem,
                                             Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
                                             const glm::dmat4& transform,
                                             const std::any&)
{
    LoadTileResources* resources = buildTileResources(tileLoadResult, transform);
    Cesium3DTilesSelection::TileLoadResultAndRenderResources result{
        std::move(tileLoadResult),
        resources
    };
    return asyncSystem.createResolvedFuture(std::move(result));
}

void* CesiumBgfxRenderAdapter::prepareInMainThread(Cesium3DTilesSelection::Tile&, void* pLoadThreadResult)
{
    std::unique_ptr<LoadTileResources> loadResources(static_cast<LoadTileResources*>(pLoadThreadResult));
    return createGpuResources(loadResources.get());
}

void CesiumBgfxRenderAdapter::free(Cesium3DTilesSelection::Tile&, void* pLoadThreadResult, void* pMainThreadResult) noexcept
{
    delete static_cast<LoadTileResources*>(pLoadThreadResult);
    destroyTileResources(static_cast<TileGpuResources*>(pMainThreadResult));
}

void* CesiumBgfxRenderAdapter::prepareRasterInLoadThread(CesiumImage::ImageAsset& image, const std::any&)
{
    return new CpuImage(makeCpuImage(image));
}

void* CesiumBgfxRenderAdapter::prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile&, void* pLoadThreadResult)
{
    std::unique_ptr<CpuImage> image(static_cast<CpuImage*>(pLoadThreadResult));
    auto* resources = new RasterTextureResources();
    try {
        ensureGpuGlobals();
        resources->texture.handle = createTexture(*image);
    } catch (...) {
        destroyRasterResources(resources);
        throw;
    }
    return resources;
}

void CesiumBgfxRenderAdapter::freeRaster(const CesiumRasterOverlays::RasterOverlayTile&, void* pLoadThreadResult, void* pMainThreadResult) noexcept
{
    delete static_cast<CpuImage*>(pLoadThreadResult);
    destroyRasterResources(static_cast<RasterTextureResources*>(pMainThreadResult));
}

void CesiumBgfxRenderAdapter::attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
                                                       int32_t overlayTextureCoordinateID,
                                                       const CesiumRasterOverlays::RasterOverlayTile&,
                                                       void* pMainThreadRendererResources,
                                                       const glm::dvec2& translation,
                                                       const glm::dvec2& scale)
{
    TileGpuResources* resources = tileResources(tile);
    if (!resources)
        return;

    auto* rasterResources = static_cast<RasterTextureResources*>(pMainThreadRendererResources);
    if (!rasterResources)
        throw std::runtime_error("CesiumBgfxRenderAdapter attachRaster received NULL raster resources");

    resources->rasterAttachments.push_back(RasterAttachment{
        overlayTextureCoordinateID,
        rasterResources,
        translation,
        scale
    });
}

void CesiumBgfxRenderAdapter::detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
                                                       int32_t overlayTextureCoordinateID,
                                                       const CesiumRasterOverlays::RasterOverlayTile&,
                                                       void* pMainThreadRendererResources) noexcept
{
    TileGpuResources* resources = tileResources(tile);
    if (!resources)
        return;

    auto* rasterResources = static_cast<RasterTextureResources*>(pMainThreadRendererResources);
    auto it = std::remove_if(resources->rasterAttachments.begin(), resources->rasterAttachments.end(),
        [&](const RasterAttachment& attachment) {
            return attachment.overlayTextureCoordinateID == overlayTextureCoordinateID &&
                   attachment.resources == rasterResources;
        });
    resources->rasterAttachments.erase(it, resources->rasterAttachments.end());
}

void CesiumBgfxRenderAdapter::renderAll(uint32_t bgfxFrame)
{
    std::vector<CesiumBgfxRenderAdapter*> adapters;
    {
        std::lock_guard lock(g_registryMutex);
        adapters = g_adapters;
    }

    for (CesiumBgfxRenderAdapter* adapter : adapters) {
        if (adapter)
            adapter->render(bgfxFrame);
    }
}

void CesiumBgfxRenderAdapter::render(uint32_t)
{
    std::vector<Cesium3DTilesSelection::Tile::ConstPointer> tiles;
    glm::dmat4 view;
    glm::dmat4 projection;
    std::shared_ptr<UINative3DViewportAttachment> attachment;
    {
        std::lock_guard lock(mutex_);
        tiles = frameTiles_;
        view = view_;
        projection = projection_;
        attachment = viewportAttachment_.lock();
    }

    if (!attachment || !attachment->hasFrameBuffer())
        return;

    const UI3DViewportGeometry geometry = attachment->lastGeometry();
    if (!geometry.visible || attachment->frameBufferWidth() == 0 || attachment->frameBufferHeight() == 0)
        return;

    ensureGpuGlobals();

    float viewMatrix[16];
    float projectionMatrix[16];
    matrixToFloatColumnMajor(view, viewMatrix);
    matrixToFloatColumnMajor(projection, projectionMatrix);

    bgfx::setViewFrameBuffer(viewId_, attachment->frameBufferHandle());
    bgfx::setViewRect(viewId_, 0, 0, attachment->frameBufferWidth(), attachment->frameBufferHeight());
    bgfx::setViewTransform(viewId_, viewMatrix, projectionMatrix);

    const float lightDir[4] = {0.35f, 0.55f, 0.75f, 0.0f};
    const float lightParams[4] = {1.2f, 0.35f, 0.0f, 0.0f};
    const float alpha[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float emissive[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    bool submitted = false;
    for (const Cesium3DTilesSelection::Tile::ConstPointer& tilePointer : tiles) {
        if (!tilePointer)
            continue;

        TileGpuResources* resources = tileResources(*tilePointer);
        if (!resources)
            continue;

        for (const GpuPrimitive& primitive : resources->primitives) {
            if (!bgfx::isValid(primitive.vertexBuffer) || !bgfx::isValid(primitive.indexBuffer))
                continue;

            float modelMatrix[16];
            matrixToFloatColumnMajor(primitive.transform, modelMatrix);
            bgfx::setTransform(modelMatrix);
            bgfx::setVertexBuffer(0, primitive.vertexBuffer, 0, primitive.vertexCount);
            bgfx::setIndexBuffer(primitive.indexBuffer, 0, primitive.indexCount);
            bgfx::setUniform(popupShadersGetUniform(g_shaderState, PopupShaderUniform::LightDir), lightDir);
            bgfx::setUniform(popupShadersGetUniform(g_shaderState, PopupShaderUniform::LightParams), lightParams);
            bgfx::setUniform(popupShadersGetUniform(g_shaderState, PopupShaderUniform::Alpha), alpha);
            bgfx::setUniform(popupShadersGetUniform(g_shaderState, PopupShaderUniform::Emissive), emissive);

            const bgfx::TextureHandle texture = textureForPrimitive(*resources, primitive);
            bgfx::setTexture(0, popupShadersGetUniform(g_shaderState, PopupShaderUniform::AlbedoSampler), texture);
            bgfx::setTexture(1, popupShadersGetUniform(g_shaderState, PopupShaderUniform::NormalSampler), g_whiteTexture);
            bgfx::setTexture(2, popupShadersGetUniform(g_shaderState, PopupShaderUniform::PackedSampler), g_whiteTexture);

            uint64_t state = primitive.translucent ? kAlphaState : kOpaqueState;
            if (primitive.doubleSided)
                state &= ~(BGFX_STATE_CULL_CW | BGFX_STATE_CULL_CCW);
            bgfx::setState(state);
            bgfx::submit(viewId_, popupShadersGetProgram(g_shaderState));
            submitted = true;
        }
    }

    if (!submitted)
        bgfx::touch(viewId_);
}

} // namespace GRIM::GeoSpatial