#include "cesium_bgfx_render_adapter.hpp"
#include "cesium_bgfx_shaders.hpp"

#ifdef OPAQUE
#undef OPAQUE
#endif

#include "core/window_manager.hpp"
#include "logger.hpp"
#include "ui/primitives/ui_native_3d_viewport_attachment.hpp"

#include <stb/stb_image.h>

#include <Cesium3DTilesSelection/BoundingVolume.h>
#include <Cesium3DTilesSelection/TileContent.h>
#include <Cesium3DTilesSelection/TileID.h>
#include <CesiumGeometry/QuadtreeTileID.h>
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
#include <CesiumGeospatial/BoundingRegion.h>
#include <CesiumGeospatial/Ellipsoid.h>
#include <CesiumGeospatial/GlobeRectangle.h>
#include <CesiumImage/ImageAsset.h>
#include <CesiumRasterOverlays/RasterOverlayTile.h>
#include <CesiumRasterOverlays/RasterOverlayTileProvider.h>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat3x3.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
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
                                      BGFX_STATE_DEPTH_TEST_LEQUAL;
    constexpr uint64_t kAlphaState = kOpaqueState |
                                     BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA,
                                                           BGFX_STATE_BLEND_INV_SRC_ALPHA);
    constexpr uint64_t kRasterCompositeState = BGFX_STATE_WRITE_RGB |
                                               BGFX_STATE_WRITE_A |
                                               BGFX_STATE_DEPTH_TEST_LEQUAL |
                                               BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA,
                                                                     BGFX_STATE_BLEND_INV_SRC_ALPHA);
    constexpr uint64_t kAreaState = BGFX_STATE_WRITE_RGB |
                                    BGFX_STATE_WRITE_A |
                                    BGFX_STATE_DEPTH_TEST_LEQUAL |
                                    BGFX_STATE_BLEND_FUNC(BGFX_STATE_BLEND_SRC_ALPHA,
                                                          BGFX_STATE_BLEND_INV_SRC_ALPHA);
    constexpr bool kEnableCesiumRasterTextureRendering = true;
    constexpr bool kDebugRasterOverlayUsesGlobeTexture = false;
    constexpr bool kDebugRasterOverlayUsesSphericalTextureCoordinates = false;
    constexpr uint64_t kTerrainSamplerFlags = BGFX_SAMPLER_U_CLAMP |
                                              BGFX_SAMPLER_V_CLAMP |
                                              BGFX_SAMPLER_MIN_ANISOTROPIC |
                                              BGFX_SAMPLER_MAG_ANISOTROPIC;

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
        uint8_t mipCount = 1;
        std::vector<std::byte> rgba8; // full mip chain, tightly packed, largest level first
    };

    struct CpuPrimitive {
        std::vector<CesiumBgfxVertex> vertices;
        std::vector<uint32_t> indices;
        glm::dmat4 transform = glm::dmat4(1.0);
        uint32_t surfaceID = 0;
        int32_t baseColorTexture = -1;
        int32_t rasterOverlayTextureCoordinateID = -1;
        bool requiresRasterOverlay = false;
        bool generatedTextureCoordinates = false;
        bool hasNonFiniteOverlayTextureCoordinates = false;
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
        uint32_t surfaceID = 0;
        int32_t baseColorTexture = -1;
        int32_t rasterOverlayTextureCoordinateID = -1;
        bool requiresRasterOverlay = false;
        std::array<float, 4> baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
        bool doubleSided = false;
        bool translucent = false;
    };

    struct RasterAttachment {
        int32_t overlayTextureCoordinateID = -1;
        uintptr_t resourceID = 0;
        bgfx::TextureHandle textureHandle = BGFX_INVALID_HANDLE;
        glm::dvec2 translation{0.0, 0.0};
        glm::dvec2 scale{1.0, 1.0};
    };

    struct RenderMaterial {
        bgfx::TextureHandle texture = BGFX_INVALID_HANDLE;
        std::array<float, 4> baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
        glm::dvec2 translation{0.0, 0.0};
        glm::dvec2 scale{1.0, 1.0};
        bool clipToUnitUv = false;
        float fadeAlpha = 1.0f;
        float opacity = 1.0f;
    };

    struct TileGpuResources {
        std::vector<GpuPrimitive> primitives;
        std::vector<GpuTexture> textures;
        mutable std::mutex rasterAttachmentsMutex;
        std::vector<RasterAttachment> rasterAttachments;
    };

    std::mutex g_registryMutex;
    std::vector<CesiumBgfxRenderAdapter*> g_adapters;
    bool g_renderPassRegistered = false;
    bgfx::VertexLayout g_vertexLayout;
    bool g_vertexLayoutInitialized = false;
    CesiumShaderState* g_shaderState = nullptr;
    bgfx::TextureHandle g_whiteTexture = BGFX_INVALID_HANDLE;
    bgfx::TextureHandle g_globeTexture = BGFX_INVALID_HANDLE;
    std::string g_globeTextureSource;
    bool g_rasterTextureRenderingEnabled = kEnableCesiumRasterTextureRendering;
    bool g_loggedCesiumOverlayUv = false;
    bool g_loggedCesiumOverlayUvRange = false;
    bool g_loggedGeneratedSphericalUv = false;
    bool g_loggedNonFiniteOverlayUvRepair = false;
    std::mutex g_tileResourcesRegistryMutex;
    std::vector<TileGpuResources*> g_tileResourcesRegistry;

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

        std::lock_guard registryLock(g_tileResourcesRegistryMutex);
        auto registryIt = std::remove(g_tileResourcesRegistry.begin(), g_tileResourcesRegistry.end(), resources);
        g_tileResourcesRegistry.erase(registryIt, g_tileResourcesRegistry.end());

        for (GpuPrimitive& primitive : resources->primitives)
            destroyGpuPrimitive(primitive);
        for (GpuTexture& texture : resources->textures)
            destroyTexture(texture);
        delete resources;
    }

    bool isRegisteredTileResources(const TileGpuResources* resources)
    {
        return std::find(g_tileResourcesRegistry.begin(), g_tileResourcesRegistry.end(), resources) != g_tileResourcesRegistry.end();
    }

    void registerTileResources(TileGpuResources* resources)
    {
        if (!resources)
            throw std::runtime_error("CesiumBgfxRenderAdapter registry received NULL tile resources");

        std::lock_guard registryLock(g_tileResourcesRegistryMutex);
        g_tileResourcesRegistry.push_back(resources);
    }

    void purgeRasterResourceAttachments(uintptr_t resourceID) noexcept
    {
        if (resourceID == 0)
            return;

        std::lock_guard registryLock(g_tileResourcesRegistryMutex);
        for (TileGpuResources* tileResources : g_tileResourcesRegistry) {
            if (!tileResources)
                continue;

            std::lock_guard attachmentsLock(tileResources->rasterAttachmentsMutex);
            auto attachmentIt = std::remove_if(tileResources->rasterAttachments.begin(), tileResources->rasterAttachments.end(),
                [&](const RasterAttachment& attachment) {
                    return attachment.resourceID == resourceID;
                });
            tileResources->rasterAttachments.erase(attachmentIt, tileResources->rasterAttachments.end());
        }
    }

    void destroyRasterResources(RasterTextureResources* resources) noexcept
    {
        if (!resources)
            return;

        destroyTexture(resources->texture);
        delete resources;
    }

    bgfx::TextureHandle createProceduralGlobeTexture()
    {
        constexpr uint16_t width = 1024;
        constexpr uint16_t height = 512;
        std::vector<std::byte> rgba(static_cast<size_t>(width) * height * 4);

        for (uint16_t y = 0; y < height; ++y) {
            const double v = static_cast<double>(y) / static_cast<double>(height - 1);
            const double latitude = (0.5 - v) * 3.1415926535897932384626433832795;
            for (uint16_t x = 0; x < width; ++x) {
                const double u = static_cast<double>(x) / static_cast<double>(width - 1);
                const double longitude = (u * 2.0 - 1.0) * 3.1415926535897932384626433832795;
                const double landNoise = std::sin(longitude * 2.1 + std::sin(latitude * 3.0)) +
                                         0.55 * std::sin(longitude * 5.4 - latitude * 1.7) +
                                         0.35 * std::cos(longitude * 8.0 + latitude * 4.0);
                const double polar = std::abs(latitude) / 1.5707963267948966192313216916398;
                const bool ice = polar > 0.82;
                const bool land = !ice && landNoise > 0.52 - 0.28 * std::cos(latitude);
                const double shade = 0.86 + 0.14 * std::cos(latitude);

                uint8_t r = 26;
                uint8_t g = 73;
                uint8_t b = 116;
                if (land) {
                    r = static_cast<uint8_t>(std::clamp(72.0 * shade + landNoise * 18.0, 0.0, 255.0));
                    g = static_cast<uint8_t>(std::clamp(118.0 * shade + landNoise * 22.0, 0.0, 255.0));
                    b = static_cast<uint8_t>(std::clamp(70.0 * shade, 0.0, 255.0));
                } else if (ice) {
                    r = 218;
                    g = 231;
                    b = 232;
                } else {
                    r = static_cast<uint8_t>(std::clamp(20.0 * shade + polar * 16.0, 0.0, 255.0));
                    g = static_cast<uint8_t>(std::clamp(78.0 * shade + polar * 18.0, 0.0, 255.0));
                    b = static_cast<uint8_t>(std::clamp(128.0 * shade + polar * 22.0, 0.0, 255.0));
                }

                const size_t offset = (static_cast<size_t>(y) * width + x) * 4;
                rgba[offset + 0] = static_cast<std::byte>(r);
                rgba[offset + 1] = static_cast<std::byte>(g);
                rgba[offset + 2] = static_cast<std::byte>(b);
                rgba[offset + 3] = static_cast<std::byte>(255);
            }
        }

        const bgfx::Memory* memory = bgfx::copy(rgba.data(), static_cast<uint32_t>(rgba.size()));
        bgfx::TextureHandle texture = bgfx::createTexture2D(width,
                                                            height,
                                                            false,
                                                            1,
                                                            bgfx::TextureFormat::RGBA8,
                                                            BGFX_SAMPLER_V_CLAMP |
                                                                BGFX_SAMPLER_MIN_ANISOTROPIC |
                                                                BGFX_SAMPLER_MAG_ANISOTROPIC,
                                                            memory);
        if (!bgfx::isValid(texture))
            throw std::runtime_error("CesiumBgfxRenderAdapter failed to create procedural globe texture");
        return texture;
    }

    bgfx::TextureHandle createGlobeTextureFromFile(const std::string& texturePath)
    {
        int width = 0;
        int height = 0;
        int channels = 0;
        stbi_uc* pixels = stbi_load(texturePath.c_str(), &width, &height, &channels, 4);
        if (!pixels)
            throw std::runtime_error("CesiumBgfxRenderAdapter failed to load globe texture: " + texturePath);
        if (width <= 0 || height <= 0 || width > 65535 || height > 65535) {
            stbi_image_free(pixels);
            throw std::runtime_error("CesiumBgfxRenderAdapter globe texture dimensions are invalid: " + texturePath);
        }

        const size_t byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        if (byteCount > std::numeric_limits<uint32_t>::max()) {
            stbi_image_free(pixels);
            throw std::runtime_error("CesiumBgfxRenderAdapter globe texture is too large for bgfx copy: " + texturePath);
        }

        const bgfx::Memory* memory = bgfx::copy(pixels, static_cast<uint32_t>(byteCount));
        stbi_image_free(pixels);
        bgfx::TextureHandle texture = bgfx::createTexture2D(static_cast<uint16_t>(width),
                                                            static_cast<uint16_t>(height),
                                                            false,
                                                            1,
                                                            bgfx::TextureFormat::RGBA8,
                                                            BGFX_SAMPLER_V_CLAMP |
                                                                BGFX_SAMPLER_MIN_ANISOTROPIC |
                                                                BGFX_SAMPLER_MAG_ANISOTROPIC,
                                                            memory);
        if (!bgfx::isValid(texture))
            throw std::runtime_error("CesiumBgfxRenderAdapter failed to create globe texture: " + texturePath);
        return texture;
    }

    void ensureGlobeTexture(const std::string& texturePath)
    {
        // Empty path means "no opinion": tile/raster preparation calls this
        // without knowing the configured texture, and must not tear down the
        // file texture the render pass installed (destroying a texture that
        // in-flight frames still sample is undefined). Only an explicit path
        // change swaps the texture.
        if (texturePath.empty()) {
            if (bgfx::isValid(g_globeTexture))
                return;
            g_globeTexture = createProceduralGlobeTexture();
            g_globeTextureSource = "<procedural>";
            LOG_DEBUG("CesiumBgfxRenderAdapter", "Globe texture active: <procedural>");
            return;
        }

        if (bgfx::isValid(g_globeTexture) && g_globeTextureSource == texturePath)
            return;

        if (bgfx::isValid(g_globeTexture)) {
            bgfx::destroy(g_globeTexture);
            g_globeTexture = BGFX_INVALID_HANDLE;
        }

        g_globeTexture = createGlobeTextureFromFile(texturePath);
        g_globeTextureSource = texturePath;
        LOG_DEBUG("CesiumBgfxRenderAdapter", "Globe texture active: " + texturePath);
    }

    void ensureGpuGlobals(const std::string& globeTexturePath = {})
    {
        vertexLayout();

        if (!g_shaderState)
            g_shaderState = cesiumShadersCreate();

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
        ensureGlobeTexture(globeTexturePath);
    }

    void shutdownGpuGlobalsIfUnused()
    {
        if (!g_adapters.empty())
            return;

        if (bgfx::isValid(g_whiteTexture)) {
            bgfx::destroy(g_whiteTexture);
            g_whiteTexture = BGFX_INVALID_HANDLE;
        }
        if (bgfx::isValid(g_globeTexture)) {
            bgfx::destroy(g_globeTexture);
            g_globeTexture = BGFX_INVALID_HANDLE;
            g_globeTextureSource.clear();
        }
        if (g_shaderState) {
            cesiumShadersDestroy(g_shaderState);
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

    struct CesiumOverlayAttribute {
        int32_t textureCoordinateID = -1;
        int32_t accessorIndex = -1;
        std::string name;
    };

    int32_t parseCesiumOverlayTextureCoordinateID(const std::string& attributeName)
    {
        constexpr const char* prefix = "_CESIUMOVERLAY_";
        constexpr size_t prefixLength = 15;
        if (attributeName.rfind(prefix, 0) != 0 || attributeName.size() == prefixLength)
            return -1;

        int64_t value = 0;
        for (size_t index = prefixLength; index < attributeName.size(); ++index) {
            const unsigned char digit = static_cast<unsigned char>(attributeName[index]);
            if (!std::isdigit(digit))
                return -1;
            value = value * 10 + static_cast<int64_t>(digit - '0');
            if (value > std::numeric_limits<int32_t>::max())
                throw std::runtime_error("Cesium raster overlay texture coordinate ID exceeds int32 range");
        }
        return static_cast<int32_t>(value);
    }

    std::vector<CesiumOverlayAttribute> findCesiumOverlayAttributes(const CesiumGltf::MeshPrimitive& primitive)
    {
        std::vector<CesiumOverlayAttribute> result;
        for (const auto& [name, accessorIndex] : primitive.attributes) {
            const int32_t textureCoordinateID = parseCesiumOverlayTextureCoordinateID(name);
            if (textureCoordinateID < 0)
                continue;
            result.push_back(CesiumOverlayAttribute{
                textureCoordinateID,
                accessorIndex,
                name
            });
        }
        std::sort(result.begin(), result.end(), [](const CesiumOverlayAttribute& lhs, const CesiumOverlayAttribute& rhs) {
            return lhs.textureCoordinateID < rhs.textureCoordinateID;
        });
        return result;
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

    // Ellipsoid-surface normal expressed in the primitive's local space, used
    // when the glTF has no NORMAL attribute. The previous constant (0,0,1)
    // fallback lit whole tiles near-black depending on their orientation.
    std::array<float, 3> geodeticNormalFromWorldPosition(const glm::dmat4& transform,
                                                         const glm::dmat3& worldToLocalRotation,
                                                         const std::array<float, 3>& position)
    {
        const glm::dvec4 worldPosition = transform * glm::dvec4(position[0], position[1], position[2], 1.0);
        const glm::dvec3 world(worldPosition.x, worldPosition.y, worldPosition.z);
        if (glm::length(world) <= 0.0)
            throw std::runtime_error("Cesium geodetic normal generation received a zero-length vertex");

        const glm::dvec3 worldNormal = CesiumGeospatial::Ellipsoid::WGS84.geodeticSurfaceNormal(world);
        const glm::dvec3 localNormal = glm::normalize(worldToLocalRotation * worldNormal);
        return {
            static_cast<float>(localNormal.x),
            static_cast<float>(localNormal.y),
            static_cast<float>(localNormal.z)
        };
    }

    std::array<float, 2> sphericalUvFromWorldPosition(const glm::dmat4& transform,
                                                      const std::array<float, 3>& position)
    {
        constexpr double pi = 3.1415926535897932384626433832795;
        const glm::dvec4 worldPosition = transform * glm::dvec4(position[0], position[1], position[2], 1.0);
        const double radius = std::sqrt(worldPosition.x * worldPosition.x +
                                        worldPosition.y * worldPosition.y +
                                        worldPosition.z * worldPosition.z);
        if (radius <= 0.0)
            throw std::runtime_error("Cesium spherical UV generation received a zero-radius vertex");

        const double longitude = std::atan2(worldPosition.y, worldPosition.x);
        const double latitude = std::asin(std::clamp(worldPosition.z / radius, -1.0, 1.0));
        return {
            static_cast<float>((longitude + pi) / (2.0 * pi)),
            static_cast<float>(0.5 - latitude / pi)
        };
    }

    std::optional<CesiumGeometry::Rectangle> tileRectangle(const Cesium3DTilesSelection::Tile& tile)
    {
        if (const auto* region = std::get_if<CesiumGeospatial::BoundingRegion>(&tile.getBoundingVolume())) {
            const CesiumGeospatial::GlobeRectangle& rectangle = region->getRectangle();
            return CesiumGeometry::Rectangle(
                rectangle.getWest(),
                rectangle.getSouth(),
                rectangle.getEast(),
                rectangle.getNorth());
        }
        return std::nullopt;
    }

    struct GeneratedUvStats {
        size_t primitives = 0;
        size_t vertices = 0;
        size_t polarVertices = 0;
        size_t polarRepaired = 0;
        size_t nonFiniteOverlayRepaired = 0;
        float minU = std::numeric_limits<float>::max();
        float maxU = std::numeric_limits<float>::lowest();
        float minV = std::numeric_limits<float>::max();
        float maxV = std::numeric_limits<float>::lowest();
    };

    // Rewrites generated (fallback) texture coordinates as global
    // equirectangular UVs using the tile's globe rectangle. The load-thread
    // fallback (sphericalUvFromWorldPosition) breaks down at the poles: atan2
    // on near-zero x/y is numerically arbitrary, so every triangle touching a
    // pole smears a random horizontal strip of the texture across itself (the
    // radial "vortex" artifact). Here longitude is unwrapped against the tile
    // rectangle so the antimeridian edge stays continuous, and pole vertices
    // inherit u from triangle neighbours whose longitude is well defined.
    GeneratedUvStats regenerateGeneratedTextureCoordinates(LoadTileResources& resources,
                                                           const CesiumGeometry::Rectangle& rectangle)
    {
        GeneratedUvStats stats;
        constexpr double pi = 3.1415926535897932384626433832795;
        constexpr double twoPi = 2.0 * pi;
        // Vertices this close to the rotation axis have no reliable longitude:
        // tile positions are float32 relative to the tile corner, so a pole
        // vertex reconstructs with metres of x/y noise and atan2 returns an
        // arbitrary angle (the radial texture smear). 1 km is far above that
        // noise floor but far below the first off-pole grid row of any tile
        // coarse enough to actually contain a pole.
        constexpr double polarAxisDistanceMeters = 1000.0;
        constexpr double unwrapEpsilon = 1.0e-9;

        for (CpuPrimitive& primitive : resources.primitives) {
            const bool regenerateAllUvs = primitive.generatedTextureCoordinates;
            // cesium-native derives _CESIUMOVERLAY_N coordinates from the same
            // float32 vertex positions, so its pole vertices carry the same
            // atan2 longitude noise and smear the imagery layer at the poles.
            // For those primitives only the polar u is repaired; everything
            // else keeps cesium's values.
            const bool repairOverlayPolesOnly = !regenerateAllUvs && primitive.rasterOverlayTextureCoordinateID >= 0;
            if (!regenerateAllUvs && !repairOverlayPolesOnly)
                continue;

            ++stats.primitives;
            const size_t vertexCount = primitive.vertices.size();
            stats.vertices += vertexCount;
            std::vector<bool> isPolar(vertexCount, false);
            for (size_t i = 0; i < vertexCount; ++i) {
                CesiumBgfxVertex& vertex = primitive.vertices[i];
                const glm::dvec4 world = primitive.transform * glm::dvec4(vertex.px, vertex.py, vertex.pz, 1.0);
                isPolar[i] = std::sqrt(world.x * world.x + world.y * world.y) < polarAxisDistanceMeters;
                const bool repairNonFiniteOverlayUv = !regenerateAllUvs &&
                                                      (!std::isfinite(vertex.u) || !std::isfinite(vertex.v));
                if (!regenerateAllUvs && !repairNonFiniteOverlayUv)
                    continue;

                const auto cartographic = CesiumGeospatial::Ellipsoid::WGS84.cartesianToCartographic(
                    glm::dvec3(world.x, world.y, world.z));
                if (!cartographic)
                    throw std::runtime_error("Cesium generated UV regeneration received an invalid ellipsoid position");

                double longitude = cartographic->longitude;
                if (longitude < rectangle.minimumX && longitude + twoPi <= rectangle.maximumX + unwrapEpsilon)
                    longitude += twoPi;
                else if (longitude > rectangle.maximumX && longitude - twoPi >= rectangle.minimumX - unwrapEpsilon)
                    longitude -= twoPi;

                if (regenerateAllUvs) {
                    vertex.u = static_cast<float>((longitude + pi) / twoPi);
                    vertex.v = static_cast<float>(0.5 - cartographic->latitude / pi);
                } else {
                    const double rectangleWidth = rectangle.maximumX - rectangle.minimumX;
                    const double rectangleHeight = rectangle.maximumY - rectangle.minimumY;
                    if (rectangleWidth <= 0.0 || rectangleHeight <= 0.0)
                        throw std::runtime_error("Cesium overlay UV regeneration received an invalid tile rectangle");
                    vertex.u = static_cast<float>(std::clamp(
                        (longitude - rectangle.minimumX) / rectangleWidth, 0.0, 1.0));
                    vertex.v = static_cast<float>(std::clamp(
                        (cartographic->latitude - rectangle.minimumY) / rectangleHeight, 0.0, 1.0));
                    ++stats.nonFiniteOverlayRepaired;
                }
            }

            std::vector<float> polarUSum(vertexCount, 0.0f);
            std::vector<uint32_t> polarUCount(vertexCount, 0);
            for (size_t i = 0; i + 2 < primitive.indices.size(); i += 3) {
                const std::array<uint32_t, 3> triangle{
                    primitive.indices[i],
                    primitive.indices[i + 1],
                    primitive.indices[i + 2]};
                for (uint32_t polarIndex : triangle) {
                    if (polarIndex >= vertexCount || !isPolar[polarIndex])
                        continue;
                    for (uint32_t neighbourIndex : triangle) {
                        if (neighbourIndex >= vertexCount || isPolar[neighbourIndex])
                            continue;
                        polarUSum[polarIndex] += primitive.vertices[neighbourIndex].u;
                        ++polarUCount[polarIndex];
                    }
                }
            }
            for (size_t i = 0; i < vertexCount; ++i) {
                if (isPolar[i]) {
                    ++stats.polarVertices;
                    if (polarUCount[i] > 0) {
                        primitive.vertices[i].u = polarUSum[i] / static_cast<float>(polarUCount[i]);
                        ++stats.polarRepaired;
                    }
                }
                stats.minU = std::min(stats.minU, primitive.vertices[i].u);
                stats.maxU = std::max(stats.maxU, primitive.vertices[i].u);
                stats.minV = std::min(stats.minV, primitive.vertices[i].v);
                stats.maxV = std::max(stats.maxV, primitive.vertices[i].v);
                if (!std::isfinite(primitive.vertices[i].u) || !std::isfinite(primitive.vertices[i].v))
                    throw std::runtime_error("Cesium tile-aware overlay UV regeneration produced non-finite values");
            }
            primitive.hasNonFiniteOverlayTextureCoordinates = false;
        }
        return stats;
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

    bool isNoDataBlackPixel(const std::byte* pixel)
    {
        // GIBS pads no-data regions (dateline seam, polar night, uncaptured
        // swaths) with black, but JPEG compression leaves noise up to a few
        // counts, so a strict 0/1 threshold lets most of the pad through as
        // opaque near-black. Real imagery this dark is rare (open ocean at
        // dusk), and keyed pixels now fall back to the base layer anyway.
        constexpr uint8_t kNoDataBlackThreshold = 5;
        return static_cast<uint8_t>(pixel[0]) <= kNoDataBlackThreshold &&
               static_cast<uint8_t>(pixel[1]) <= kNoDataBlackThreshold &&
               static_cast<uint8_t>(pixel[2]) <= kNoDataBlackThreshold;
    }

    // Box-filters one RGBA8 mip level into the next smaller one, clamping the
    // 2x2 footprint on odd dimensions. RGB is alpha-weighted so transparent
    // no-data pixels do not darken lower mip levels.
    void appendNextMipLevel(const std::byte* source,
                            uint32_t sourceWidth,
                            uint32_t sourceHeight,
                            uint32_t targetWidth,
                            uint32_t targetHeight,
                            std::byte* target)
    {
        for (uint32_t y = 0; y < targetHeight; ++y) {
            const uint32_t sy0 = std::min(y * 2, sourceHeight - 1);
            const uint32_t sy1 = std::min(y * 2 + 1, sourceHeight - 1);
            for (uint32_t x = 0; x < targetWidth; ++x) {
                const uint32_t sx0 = std::min(x * 2, sourceWidth - 1);
                const uint32_t sx1 = std::min(x * 2 + 1, sourceWidth - 1);
                const std::byte* p00 = source + (static_cast<size_t>(sy0) * sourceWidth + sx0) * 4;
                const std::byte* p01 = source + (static_cast<size_t>(sy0) * sourceWidth + sx1) * 4;
                const std::byte* p10 = source + (static_cast<size_t>(sy1) * sourceWidth + sx0) * 4;
                const std::byte* p11 = source + (static_cast<size_t>(sy1) * sourceWidth + sx1) * 4;
                std::byte* out = target + (static_cast<size_t>(y) * targetWidth + x) * 4;
                const std::array<const std::byte*, 4> pixels{p00, p01, p10, p11};
                uint32_t alphaSum = 0;
                uint32_t weightedColor[3]{};
                for (const std::byte* pixel : pixels) {
                    const uint32_t alpha = static_cast<uint32_t>(pixel[3]);
                    alphaSum += alpha;
                    weightedColor[0] += static_cast<uint32_t>(pixel[0]) * alpha;
                    weightedColor[1] += static_cast<uint32_t>(pixel[1]) * alpha;
                    weightedColor[2] += static_cast<uint32_t>(pixel[2]) * alpha;
                }

                if (alphaSum == 0) {
                    out[0] = std::byte{0};
                    out[1] = std::byte{0};
                    out[2] = std::byte{0};
                } else {
                    out[0] = static_cast<std::byte>((weightedColor[0] + alphaSum / 2) / alphaSum);
                    out[1] = static_cast<std::byte>((weightedColor[1] + alphaSum / 2) / alphaSum);
                    out[2] = static_cast<std::byte>((weightedColor[2] + alphaSum / 2) / alphaSum);
                }
                out[3] = static_cast<std::byte>((alphaSum + 2) / 4);
            }
        }
    }

    // Appends the full mip chain after the base level so bgfx can sample
    // trilinearly; without mips distant/oblique tiles shimmer badly while
    // zooming.
    void buildMipChain(CpuImage& image)
    {
        uint32_t width = image.width;
        uint32_t height = image.height;
        const size_t baseBytes = static_cast<size_t>(width) * height * 4;

        size_t chainBytes = baseBytes;
        uint8_t mipCount = 1;
        for (uint32_t w = width, h = height; w > 1 || h > 1;) {
            w = std::max(w / 2, 1u);
            h = std::max(h / 2, 1u);
            chainBytes += static_cast<size_t>(w) * h * 4;
            ++mipCount;
        }

        image.rgba8.resize(chainBytes);
        image.mipCount = mipCount;

        size_t sourceOffset = 0;
        size_t targetOffset = baseBytes;
        while (width > 1 || height > 1) {
            const uint32_t nextWidth = std::max(width / 2, 1u);
            const uint32_t nextHeight = std::max(height / 2, 1u);
            appendNextMipLevel(image.rgba8.data() + sourceOffset,
                               width, height,
                               nextWidth, nextHeight,
                               image.rgba8.data() + targetOffset);
            sourceOffset = targetOffset;
            targetOffset += static_cast<size_t>(nextWidth) * nextHeight * 4;
            width = nextWidth;
            height = nextHeight;
        }
    }

    CpuImage makeCpuImage(const CesiumImage::ImageAsset& image, bool keyNoDataBlack, bool buildMips)
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
            if (keyNoDataBlack && isNoDataBlackPixel(target))
                target[3] = std::byte{0};
        }

        if (buildMips)
            buildMipChain(result);
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
            images.push_back(makeCpuImage(*image.pAsset, false, true));
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
                                const glm::dmat4& transform,
                                uint32_t surfaceID,
                                const std::optional<CesiumOverlayAttribute>& cesiumOverlayAttribute,
                                bool requiresRasterOverlay)
    {
        if (primitive.mode != CesiumGltf::MeshPrimitive::Mode::TRIANGLES)
            throw std::runtime_error("Cesium bgfx adapter only supports triangle primitives");

        const std::optional<int32_t> positionIndex = findAttribute(primitive, "POSITION");
        if (!positionIndex)
            throw std::runtime_error("Cesium glTF primitive is missing POSITION");

        const CesiumGltf::Accessor& positionAccessor = checkedAccessor(model, *positionIndex, "POSITION");
        requireFloatAccessor(positionAccessor, CesiumGltf::Accessor::Type::VEC3, "POSITION");

        const std::optional<int32_t> normalIndex = findAttribute(primitive, "NORMAL");
        const std::optional<int32_t> texcoordIndex = cesiumOverlayAttribute ? std::optional<int32_t>(cesiumOverlayAttribute->accessorIndex)
                                                                            : findAttribute(primitive, "TEXCOORD_0");
        const CesiumGltf::Accessor* normalAccessor = nullptr;
        const CesiumGltf::Accessor* texcoordAccessor = nullptr;
        if (normalIndex) {
            normalAccessor = &checkedAccessor(model, *normalIndex, "NORMAL");
            requireFloatAccessor(*normalAccessor, CesiumGltf::Accessor::Type::VEC3, "NORMAL");
            if (normalAccessor->count != positionAccessor.count)
                throw std::runtime_error("Cesium glTF NORMAL accessor count does not match POSITION count");
        }
        if (texcoordIndex) {
            const char* texcoordSemantic = cesiumOverlayAttribute ? cesiumOverlayAttribute->name.c_str() : "TEXCOORD_0";
            texcoordAccessor = &checkedAccessor(model, *texcoordIndex, texcoordSemantic);
            requireFloatAccessor(*texcoordAccessor, CesiumGltf::Accessor::Type::VEC2, texcoordSemantic);
            if (texcoordAccessor->count != positionAccessor.count)
                throw std::runtime_error(std::string("Cesium glTF ") + texcoordSemantic + " accessor count does not match POSITION count");
            if (cesiumOverlayAttribute && !g_loggedCesiumOverlayUv) {
                LOG_DEBUG("CesiumBgfxRenderAdapter", "Using Cesium raster overlay UV attribute " + cesiumOverlayAttribute->name);
                g_loggedCesiumOverlayUv = true;
            }
        } else if (!g_loggedGeneratedSphericalUv) {
            LOG_DEBUG("CesiumBgfxRenderAdapter", "Generating spherical UVs for Cesium primitive without _CESIUMOVERLAY_N or TEXCOORD_0");
            g_loggedGeneratedSphericalUv = true;
        }

        const std::byte* positionBase = accessorBase(model, positionAccessor, "POSITION");
        const int64_t positionStride = positionAccessor.computeByteStride(model);
        const std::byte* normalBase = normalAccessor ? accessorBase(model, *normalAccessor, "NORMAL") : nullptr;
        const int64_t normalStride = normalAccessor ? normalAccessor->computeByteStride(model) : 0;
        const char* texcoordSemantic = cesiumOverlayAttribute ? cesiumOverlayAttribute->name.c_str() : "TEXCOORD_0";
        const std::byte* texcoordBase = texcoordAccessor ? accessorBase(model, *texcoordAccessor, texcoordSemantic) : nullptr;
        const int64_t texcoordStride = texcoordAccessor ? texcoordAccessor->computeByteStride(model) : 0;

        CpuPrimitive result;
        result.transform = transform;
        result.surfaceID = surfaceID;
        result.requiresRasterOverlay = requiresRasterOverlay;
        result.generatedTextureCoordinates = !texcoordAccessor;
        if (cesiumOverlayAttribute)
            result.rasterOverlayTextureCoordinateID = cesiumOverlayAttribute->textureCoordinateID;
        result.vertices.resize(static_cast<size_t>(positionAccessor.count));
        const glm::dmat3 worldToLocalRotation = glm::inverse(glm::dmat3(transform));
        for (int64_t i = 0; i < positionAccessor.count; ++i) {
            const std::array<float, 3> position = readVec3Float(positionBase, positionStride, i);
            const std::array<float, 3> normal = normalBase ? readVec3Float(normalBase, normalStride, i)
                                                          : geodeticNormalFromWorldPosition(transform, worldToLocalRotation, position);
            const std::array<float, 2> texcoord = texcoordBase && !(cesiumOverlayAttribute && kDebugRasterOverlayUsesSphericalTextureCoordinates)
                ? readVec2Float(texcoordBase, texcoordStride, i)
                : sphericalUvFromWorldPosition(transform, position);
            const bool finiteTexcoord = std::isfinite(texcoord[0]) && std::isfinite(texcoord[1]);
            if (!finiteTexcoord && !cesiumOverlayAttribute)
                throw std::runtime_error(std::string("Cesium glTF ") + texcoordSemantic + " contains non-finite values");
            result.hasNonFiniteOverlayTextureCoordinates |= !finiteTexcoord;

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

        float minU = std::numeric_limits<float>::max();
        float minV = std::numeric_limits<float>::max();
        float maxU = std::numeric_limits<float>::lowest();
        float maxV = std::numeric_limits<float>::lowest();
        for (const CesiumBgfxVertex& vertex : result.vertices) {
            if (!std::isfinite(vertex.u) || !std::isfinite(vertex.v))
                continue;
            minU = std::min(minU, vertex.u);
            minV = std::min(minV, vertex.v);
            maxU = std::max(maxU, vertex.u);
            maxV = std::max(maxV, vertex.v);
        }
        if (cesiumOverlayAttribute && !g_loggedCesiumOverlayUvRange) {
            LOG_DEBUG("CesiumBgfxRenderAdapter", "Cesium raster overlay UV range before clamp u=[" +
                                                  std::to_string(minU) + "," + std::to_string(maxU) +
                                                  "] v=[" + std::to_string(minV) + "," + std::to_string(maxV) + "]");
            g_loggedCesiumOverlayUvRange = true;
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

    std::vector<CpuPrimitive> buildPrimitiveVariants(const CesiumGltf::Model& model,
                                                     const CesiumGltf::MeshPrimitive& primitive,
                                                     const glm::dmat4& transform,
                                                     uint32_t surfaceID)
    {
        std::vector<CpuPrimitive> variants;
        variants.push_back(buildPrimitive(model, primitive, transform, surfaceID, std::nullopt, false));

        for (const CesiumOverlayAttribute& overlayAttribute : findCesiumOverlayAttributes(primitive))
            variants.push_back(buildPrimitive(model, primitive, transform, surfaceID, overlayAttribute, true));

        return variants;
    }

    LoadTileResources* buildTileResources(Cesium3DTilesSelection::TileLoadResult& tileLoadResult,
                                          const glm::dmat4& tileTransform)
    {
        auto* resources = new LoadTileResources();
        CesiumGltf::Model* model = std::get_if<CesiumGltf::Model>(&tileLoadResult.contentKind);
        if (!model)
            return resources;

        resources->images = buildImages(*model);
        uint32_t surfaceID = 0;
        model->forEachPrimitiveInScene(model->scene,
            [&](const CesiumGltf::Model& gltf,
                const CesiumGltf::Node&,
                const CesiumGltf::Mesh&,
                const CesiumGltf::MeshPrimitive& primitive,
                const glm::dmat4& nodeTransform) {
                    std::vector<CpuPrimitive> primitiveVariants = buildPrimitiveVariants(gltf, primitive, tileTransform * nodeTransform, surfaceID++);
                    for (CpuPrimitive& primitiveVariant : primitiveVariants)
                        resources->primitives.push_back(std::move(primitiveVariant));
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
                                                           image.mipCount > 1,
                                                           1,
                                                           bgfx::TextureFormat::RGBA8,
                                                           kTerrainSamplerFlags,
                                                           memory);
        if (!bgfx::isValid(handle)) {
            const bgfx::Stats* stats = bgfx::getStats();
            const bgfx::Caps* caps = bgfx::getCaps();
            if (!stats || !caps)
                throw std::runtime_error("CesiumBgfxRenderAdapter failed to create bgfx texture and resource statistics are unavailable");
            const std::string message = "CesiumBgfxRenderAdapter failed to create bgfx texture: textures=" +
                                        std::to_string(stats->numTextures) + "/" + std::to_string(caps->limits.maxTextures);
            LOG_ERROR("CesiumBgfxRenderAdapter", message);
            throw std::runtime_error(message);
        }
        return handle;
    }

    bgfx::VertexBufferHandle createVertexBuffer(const std::vector<CesiumBgfxVertex>& vertices)
    {
        if (vertices.empty())
            return BGFX_INVALID_HANDLE;

        const bgfx::Memory* vertexMemory = bgfx::copy(vertices.data(),
                                                      static_cast<uint32_t>(vertices.size() * sizeof(CesiumBgfxVertex)));
        bgfx::VertexBufferHandle vertexBuffer = bgfx::createVertexBuffer(vertexMemory, vertexLayout());
        if (!bgfx::isValid(vertexBuffer)) {
            const bgfx::Stats* stats = bgfx::getStats();
            const bgfx::Caps* caps = bgfx::getCaps();
            if (!stats || !caps)
                throw std::runtime_error("CesiumBgfxRenderAdapter failed to create vertex buffer and resource statistics are unavailable");
            const std::string message = "CesiumBgfxRenderAdapter failed to create vertex buffer: vertexBuffers=" +
                                        std::to_string(stats->numVertexBuffers) + "/" + std::to_string(caps->limits.maxVertexBuffers);
            LOG_ERROR("CesiumBgfxRenderAdapter", message);
            throw std::runtime_error(message);
        }
        return vertexBuffer;
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
                primitive.surfaceID = cpuPrimitive.surfaceID;
                primitive.baseColorTexture = cpuPrimitive.baseColorTexture;
                primitive.rasterOverlayTextureCoordinateID = cpuPrimitive.rasterOverlayTextureCoordinateID;
                primitive.requiresRasterOverlay = cpuPrimitive.requiresRasterOverlay;
                primitive.baseColorFactor = cpuPrimitive.baseColorFactor;
                primitive.doubleSided = cpuPrimitive.doubleSided;
                primitive.translucent = cpuPrimitive.translucent;

                primitive.vertexBuffer = createVertexBuffer(cpuPrimitive.vertices);

                const bgfx::Memory* indexMemory = bgfx::copy(cpuPrimitive.indices.data(),
                                                             static_cast<uint32_t>(cpuPrimitive.indices.size() * sizeof(uint32_t)));
                primitive.indexBuffer = bgfx::createIndexBuffer(indexMemory, BGFX_BUFFER_INDEX32);
                if (!bgfx::isValid(primitive.indexBuffer)) {
                    const bgfx::Stats* stats = bgfx::getStats();
                    const bgfx::Caps* caps = bgfx::getCaps();
                    if (!stats || !caps)
                        throw std::runtime_error("CesiumBgfxRenderAdapter failed to create index buffer and resource statistics are unavailable");
                    const std::string message = "CesiumBgfxRenderAdapter failed to create index buffer: indexBuffers=" +
                                                std::to_string(stats->numIndexBuffers) + "/" + std::to_string(caps->limits.maxIndexBuffers);
                    LOG_ERROR("CesiumBgfxRenderAdapter", message);
                    throw std::runtime_error(message);
                }

                resources->primitives.push_back(primitive);
            }
        } catch (...) {
            destroyTileResources(resources);
            throw;
        }

        registerTileResources(resources);
        return resources;
    }

    void matrixToFloatColumnMajor(const glm::dmat4& source, float target[16])
    {
        for (int column = 0; column < 4; ++column) {
            for (int row = 0; row < 4; ++row)
                target[column * 4 + row] = static_cast<float>(source[column][row]);
        }
    }

    glm::dmat4 projectionForBgfx(const glm::dmat4& projection)
    {
        if (bgfx::getCaps()->homogeneousDepth)
            return projection;

        glm::dmat4 converted = projection;
        for (int column = 0; column < 4; ++column)
            converted[column][2] = projection[column][2] * 0.5 + projection[column][3] * 0.5;
        return converted;
    }

    // Base-layer material: the tile's own glTF texture when it has one,
    // otherwise the shared globe texture. This is rendered only when the
    // surface has no attached Cesium raster imagery.
    RenderMaterial materialForPrimitive(const TileGpuResources& resources, const GpuPrimitive& primitive)
    {
        RenderMaterial material;
        material.baseColorFactor = primitive.baseColorFactor;

        if (primitive.baseColorTexture >= 0 && static_cast<size_t>(primitive.baseColorTexture) < resources.textures.size() &&
            bgfx::isValid(resources.textures[static_cast<size_t>(primitive.baseColorTexture)].handle)) {
            material.texture = resources.textures[static_cast<size_t>(primitive.baseColorTexture)].handle;
            return material;
        }

        material.texture = bgfx::isValid(g_globeTexture) ? g_globeTexture : g_whiteTexture;
        return material;
    }

    std::vector<RenderMaterial> rasterMaterialsForPrimitive(const TileGpuResources& resources,
                                                            const GpuPrimitive& primitive,
                                                            const std::vector<float>& overlayOpacities)
    {
        std::vector<RenderMaterial> materials;
        if (!g_rasterTextureRenderingEnabled || !primitive.requiresRasterOverlay)
            return materials;

        std::lock_guard lock(resources.rasterAttachmentsMutex);
        for (const RasterAttachment& attachment : resources.rasterAttachments) {
            if (attachment.overlayTextureCoordinateID != primitive.rasterOverlayTextureCoordinateID ||
                !bgfx::isValid(attachment.textureHandle)) {
                continue;
            }

            RenderMaterial material;
            material.texture = kDebugRasterOverlayUsesGlobeTexture && bgfx::isValid(g_globeTexture)
                ? g_globeTexture
                : attachment.textureHandle;
            material.baseColorFactor = primitive.baseColorFactor;
            material.translation = kDebugRasterOverlayUsesGlobeTexture ? glm::dvec2(0.0, 0.0) : attachment.translation;
            material.scale = kDebugRasterOverlayUsesGlobeTexture ? glm::dvec2(1.0, 1.0) : attachment.scale;
            material.clipToUnitUv = !kDebugRasterOverlayUsesGlobeTexture;
            if (attachment.overlayTextureCoordinateID < 0 ||
                static_cast<size_t>(attachment.overlayTextureCoordinateID) >= overlayOpacities.size()) {
                throw std::runtime_error("Cesium raster attachment has no configured opacity for its texture coordinate ID");
            }
            material.opacity = overlayOpacities[static_cast<size_t>(attachment.overlayTextureCoordinateID)];
            materials.push_back(material);
        }
        return materials;
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

    const bgfx::Caps* caps = bgfx::getCaps();
    if (!caps)
        throw std::runtime_error("CesiumBgfxRenderAdapter requires initialized bgfx capabilities");
    viewId_ = WindowManager::reserveViewIds(owner_, WindowManager::ViewIdRange::PanelViewport, 1).at(0);
    registerAdapter(this);
    // Bumped with every rendering change so a run's log proves which adapter
    // revision the binary actually contains.
    LOG_DEBUG("CesiumBgfxRenderAdapter",
              "Adapter revision r5: authoritative single-pass raster imagery, no terrain face culling, "
              "rect-based generated UVs with axis-distance pole repair, sticky globe texture; "
              "expects projected imagery coverage; resource limits vertex=" +
              std::to_string(caps->limits.maxVertexBuffers) +
              " index=" + std::to_string(caps->limits.maxIndexBuffers) +
              " texture=" + std::to_string(caps->limits.maxTextures));
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

void CesiumBgfxRenderAdapter::setGlobeTexturePath(std::string texturePath)
{
    std::lock_guard lock(mutex_);
    globeTexturePath_ = std::move(texturePath);
}

void CesiumBgfxRenderAdapter::setRasterOverlayOpacities(std::vector<float> opacities)
{
    for (float opacity : opacities) {
        if (opacity < 0.0f || opacity > 1.0f)
            throw std::runtime_error("CesiumBgfxRenderAdapter raster overlay opacity must be within [0, 1]");
    }
    std::lock_guard lock(mutex_);
    rasterOverlayOpacities_ = std::move(opacities);
}

void CesiumBgfxRenderAdapter::setFrameSelection(std::vector<Cesium3DTilesSelection::Tile::ConstPointer> tiles,
                                                std::vector<Cesium3DTilesSelection::Tile::ConstPointer> fadingOutTiles,
                                                glm::dmat4 view,
                                                glm::dmat4 projection)
{
    std::lock_guard lock(mutex_);
    frameTiles_ = std::move(tiles);
    fadingOutTiles_ = std::move(fadingOutTiles);
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

void* CesiumBgfxRenderAdapter::prepareInMainThread(Cesium3DTilesSelection::Tile& tile, void* pLoadThreadResult)
{
    auto* loadResources = static_cast<LoadTileResources*>(pLoadThreadResult);
    if (!loadResources)
        throw std::runtime_error("CesiumBgfxRenderAdapter main-thread preparation received NULL load resources");
    const std::optional<CesiumGeometry::Rectangle> rectangle = tileRectangle(tile);
    if (rectangle) {
        const GeneratedUvStats stats = regenerateGeneratedTextureCoordinates(*loadResources, *rectangle);
        if (stats.nonFiniteOverlayRepaired > 0 && !g_loggedNonFiniteOverlayUvRepair) {
            LOG_DEBUG("CesiumBgfxRenderAdapter",
                      "Reconstructed " + std::to_string(stats.nonFiniteOverlayRepaired) +
                          " non-finite overlay UV vertices from tile cartographic bounds");
            g_loggedNonFiniteOverlayUvRepair = true;
        }
        // Verification for the pole-smear fix: polarRepaired must equal
        // polar on any tile touching a pole, and the u range must be a clean
        // sub-interval of [0,1] matching the tile's longitude span. Logged
        // for the first tiles only; runs on the main thread.
        static int s_uvRegenLogBudget = 12;
        if (stats.primitives > 0 && s_uvRegenLogBudget > 0) {
            --s_uvRegenLogBudget;
            LOG_DEBUG("CesiumBgfxRenderAdapter",
                      "UV regen " + Cesium3DTilesSelection::TileIdUtilities::createTileIdString(tile.getTileID()) +
                      " prims=" + std::to_string(stats.primitives) +
                      " verts=" + std::to_string(stats.vertices) +
                      " polar=" + std::to_string(stats.polarVertices) +
                      " repaired=" + std::to_string(stats.polarRepaired) +
                      " overlayRepaired=" + std::to_string(stats.nonFiniteOverlayRepaired) +
                      " u=[" + std::to_string(stats.minU) + "," + std::to_string(stats.maxU) + "]" +
                      " v=[" + std::to_string(stats.minV) + "," + std::to_string(stats.maxV) + "]");
        }
    }
    for (const CpuPrimitive& primitive : loadResources->primitives) {
        if (primitive.hasNonFiniteOverlayTextureCoordinates)
            throw std::runtime_error("Cesium tile has non-finite overlay UVs but no usable cartographic bounds");
    }
    TileGpuResources* gpuResources = createGpuResources(loadResources);
    delete loadResources;
    return gpuResources;
}

void CesiumBgfxRenderAdapter::free(Cesium3DTilesSelection::Tile&, void* pLoadThreadResult, void* pMainThreadResult) noexcept
{
    delete static_cast<LoadTileResources*>(pLoadThreadResult);
    destroyTileResources(static_cast<TileGpuResources*>(pMainThreadResult));
}

void* CesiumBgfxRenderAdapter::prepareRasterInLoadThread(CesiumImage::ImageAsset& image, const std::any&)
{
    // No-data black keying is OFF: dark real imagery (e.g. BlueMarble's deep
    // ocean, single-digit RGB) is indistinguishable from provider padding, and
    // keying it punched the imagery layer full of holes (84% of the Pacific
    // hemisphere tile was being keyed out). Imagery renders opaque as-is; the
    return new CpuImage(makeCpuImage(image, false, false));
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
    auto* resources = static_cast<RasterTextureResources*>(pMainThreadResult);
    if (resources)
        purgeRasterResourceAttachments(reinterpret_cast<uintptr_t>(resources));
    destroyRasterResources(resources);
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

    {
        std::lock_guard lock(resources->rasterAttachmentsMutex);
        resources->rasterAttachments.push_back(RasterAttachment{
            overlayTextureCoordinateID,
            reinterpret_cast<uintptr_t>(rasterResources),
            rasterResources->texture.handle,
            translation,
            scale
        });
    }
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
    const uintptr_t resourceID = reinterpret_cast<uintptr_t>(rasterResources);
    std::lock_guard lock(resources->rasterAttachmentsMutex);
    auto it = std::remove_if(resources->rasterAttachments.begin(), resources->rasterAttachments.end(),
        [&](const RasterAttachment& attachment) {
            return attachment.overlayTextureCoordinateID == overlayTextureCoordinateID &&
                   attachment.resourceID == resourceID;
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

void CesiumBgfxRenderAdapter::setPointMarkers(std::vector<CesiumBgfxPointMarker> markers)
{
    std::lock_guard lock(mutex_);
    pointMarkers_ = std::move(markers);
}

void CesiumBgfxRenderAdapter::setAreaShapes(std::vector<CesiumBgfxAreaShape> shapes)
{
    std::lock_guard lock(mutex_);
    areaShapes_ = std::move(shapes);
}

void CesiumBgfxRenderAdapter::render(uint32_t)
{
    std::vector<Cesium3DTilesSelection::Tile::ConstPointer> tiles;
    std::vector<Cesium3DTilesSelection::Tile::ConstPointer> fadingOutTiles;
    glm::dmat4 view;
    glm::dmat4 projection;
    std::vector<CesiumBgfxPointMarker> pointMarkers;
    std::vector<CesiumBgfxAreaShape> areaShapes;
    std::vector<float> rasterOverlayOpacities;
    std::shared_ptr<UINative3DViewportAttachment> attachment;
    std::string globeTexturePath;
    {
        std::lock_guard lock(mutex_);
        tiles = frameTiles_;
        fadingOutTiles = fadingOutTiles_;
        view = view_;
        projection = projection_;
        pointMarkers = pointMarkers_;
        areaShapes = areaShapes_;
        rasterOverlayOpacities = rasterOverlayOpacities_;
        attachment = viewportAttachment_.lock();
        globeTexturePath = globeTexturePath_;
    }

    if (!attachment || !attachment->hasFrameBuffer())
        return;

    const UI3DViewportGeometry geometry = attachment->lastGeometry();
    if (!geometry.visible || attachment->frameBufferWidth() == 0 || attachment->frameBufferHeight() == 0)
        return;

    ensureGpuGlobals(globeTexturePath);

    float viewMatrix[16];
    float projectionMatrix[16];
    matrixToFloatColumnMajor(glm::dmat4(1.0), viewMatrix);
    matrixToFloatColumnMajor(projectionForBgfx(projection), projectionMatrix);
    const float cameraPos[4] = {0.0f, 0.0f, 0.0f, 1.0f};

    bgfx::setViewFrameBuffer(viewId_, attachment->frameBufferHandle());
    bgfx::setViewRect(viewId_, 0, 0, attachment->frameBufferWidth(), attachment->frameBufferHeight());
    bgfx::setViewTransform(viewId_, viewMatrix, projectionMatrix);
    bgfx::setViewMode(viewId_, bgfx::ViewMode::Sequential);

    const float lightDir[4] = {0.35f, 0.55f, 0.75f, 0.0f};
    const float lightParams[4] = {0.75f, 0.38f, 0.0f, 0.0f};
    const float colorParams[4] = {1.0f, 1.0f, 0.0f, 0.0f};
    const float surfaceParams[4] = {0.12f, 0.72f, 0.035f, 0.035f};

    auto submitPrimitive = [&](const GpuPrimitive& primitive, const RenderMaterial& material, uint64_t state) {
        if (!bgfx::isValid(material.texture))
            return false;
        if (!bgfx::isValid(primitive.vertexBuffer))
            throw std::runtime_error("CesiumBgfxRenderAdapter submitPrimitive received invalid vertex buffer");
        if (!bgfx::isValid(primitive.indexBuffer))
            throw std::runtime_error("CesiumBgfxRenderAdapter submitPrimitive received invalid index buffer");
        if (primitive.vertexCount == 0 || primitive.indexCount == 0)
            throw std::runtime_error("CesiumBgfxRenderAdapter submitPrimitive received empty primitive");

        float modelMatrix[16];
        matrixToFloatColumnMajor(view * primitive.transform, modelMatrix);
        bgfx::setTransform(modelMatrix);
        bgfx::setVertexBuffer(0, primitive.vertexBuffer, 0, primitive.vertexCount);
        bgfx::setIndexBuffer(primitive.indexBuffer, 0, primitive.indexCount);
        const float baseColorFactor[4] = {
            material.baseColorFactor[0],
            material.baseColorFactor[1],
            material.baseColorFactor[2],
            material.baseColorFactor[3]
        };
        const float overlayTransform[4] = {
            static_cast<float>(material.translation.x),
            static_cast<float>(material.translation.y),
            static_cast<float>(material.scale.x),
            static_cast<float>(material.scale.y)
        };
        const float textureParams[4] = {
            material.clipToUnitUv ? 1.0f : 0.0f,
            material.fadeAlpha,
            material.opacity,
            0.0f
        };

        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::LightDir), lightDir);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::LightParams), lightParams);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::BaseColorFactor), baseColorFactor);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::OverlayTransform), overlayTransform);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::TextureParams), textureParams);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::CameraPos), cameraPos);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::ColorParams), colorParams);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::SurfaceParams), surfaceParams);
        bgfx::setTexture(0, cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::ImagerySampler), material.texture);

        if (primitive.doubleSided)
            state &= ~(BGFX_STATE_CULL_CW | BGFX_STATE_CULL_CCW);
        bgfx::setState(state);
        bgfx::submit(viewId_, cesiumShadersGetProgram(g_shaderState));
        return true;
    };

    bool submitted = false;
    size_t baseDrawCount = 0;
    size_t overlayDrawCount = 0;
    auto submitTile = [&](const Cesium3DTilesSelection::Tile& tile, float fadeAlpha) {
        std::lock_guard registryLock(g_tileResourcesRegistryMutex);
        TileGpuResources* resources = tileResources(tile);
        if (!resources || !isRegisteredTileResources(resources))
            return;

        for (const GpuPrimitive& primitive : resources->primitives) {
            if (!bgfx::isValid(primitive.vertexBuffer) || !bgfx::isValid(primitive.indexBuffer))
                continue;
            if (primitive.requiresRasterOverlay) {
                for (RenderMaterial rasterMaterial : rasterMaterialsForPrimitive(*resources, primitive, rasterOverlayOpacities)) {
                    rasterMaterial.fadeAlpha = fadeAlpha;
                    if (submitPrimitive(primitive, rasterMaterial, kRasterCompositeState)) {
                        submitted = true;
                        ++overlayDrawCount;
                    }
                }
            } else {
                RenderMaterial material = materialForPrimitive(*resources, primitive);
                material.fadeAlpha = fadeAlpha;
                if (submitPrimitive(primitive, material,
                                    primitive.translucent ? kAlphaState : kOpaqueState)) {
                    submitted = true;
                    ++baseDrawCount;
                }
            }
        }
    };

    auto tileFadePercentage = [](const Cesium3DTilesSelection::Tile& tile) {
        const Cesium3DTilesSelection::TileRenderContent* renderContent = tile.getContent().getRenderContent();
        if (!renderContent)
            return 0.0f;
        return std::clamp(renderContent->getLodTransitionFadePercentage(), 0.0f, 1.0f);
    };

    static uint32_t s_diagFrame = 0;
    const bool diagFrameDue = (s_diagFrame++ % 300u) == 0;

    // Selected tiles first (fade percentage 0->1 is their opacity), then
    // fading-out tiles on top. For fading-out tiles Cesium's percentage is
    // transition *progress* (0 = just started fading, 1 = gone), so their
    // opacity is the inverse. Drawing old tiles last means they keep covering
    // the new ones at equal depth until the dither reveals the replacement.
    for (const Cesium3DTilesSelection::Tile::ConstPointer& tilePointer : tiles) {
        if (tilePointer)
            submitTile(*tilePointer, tileFadePercentage(*tilePointer));
    }
    for (const Cesium3DTilesSelection::Tile::ConstPointer& tilePointer : fadingOutTiles) {
        if (!tilePointer)
            continue;
        const float fadeAlpha = 1.0f - tileFadePercentage(*tilePointer);
        if (fadeAlpha > 0.004f)
            submitTile(*tilePointer, fadeAlpha);
    }

    size_t areaDrawCount = 0;
    size_t areaMinimumPixelScaleCount = 0;
    const double areaProjectionY = std::abs(projection[1][1]);
    if (areaProjectionY <= std::numeric_limits<double>::epsilon())
        throw std::runtime_error("CesiumBgfxRenderAdapter area projection has zero Y scale");
    for (const CesiumBgfxAreaShape& area : areaShapes) {
        if (area.geometryKind != GeoSpatialGeometryKind::CubeArea &&
            area.geometryKind != GeoSpatialGeometryKind::SphereArea) {
            throw std::runtime_error("CesiumBgfxRenderAdapter received non-area geometry in area payload");
        }
        if (!std::isfinite(area.sizeMeters) || area.sizeMeters <= 0.0)
            throw std::runtime_error("CesiumBgfxRenderAdapter received invalid area size");

        std::vector<CesiumBgfxVertex> areaVertices;
        std::vector<uint16_t> areaIndices;
        const glm::dmat3 viewRotation(view);

        auto appendVertex = [&](const glm::dvec3& ecefPosition, const glm::dvec3& ecefNormal) {
            if (areaVertices.size() >= static_cast<size_t>(std::numeric_limits<uint16_t>::max()))
                throw std::runtime_error("CesiumBgfxRenderAdapter area mesh exceeds uint16 vertex indexing");
            const glm::dvec3 viewPosition = glm::dvec3(view * glm::dvec4(ecefPosition, 1.0));
            const glm::dvec3 viewNormal = glm::normalize(viewRotation * ecefNormal);
            CesiumBgfxVertex vertex;
            vertex.px = static_cast<float>(viewPosition.x);
            vertex.py = static_cast<float>(viewPosition.y);
            vertex.pz = static_cast<float>(viewPosition.z);
            vertex.nx = static_cast<float>(viewNormal.x);
            vertex.ny = static_cast<float>(viewNormal.y);
            vertex.nz = static_cast<float>(viewNormal.z);
            areaVertices.push_back(vertex);
        };

        const glm::dvec4 anchorViewPosition = view * glm::dvec4(area.anchorEcef, 1.0);
        if (anchorViewPosition.z >= -0.01)
            continue;
        constexpr double minimumAreaPixels = 6.0;
        const double minimumDisplaySizeMeters =
            (minimumAreaPixels * -anchorViewPosition.z) /
            (areaProjectionY * static_cast<double>(attachment->frameBufferHeight()));
        const double displaySizeMeters = std::max(area.sizeMeters, minimumDisplaySizeMeters);
        if (displaySizeMeters > area.sizeMeters)
            ++areaMinimumPixelScaleCount;

        const double halfSize = displaySizeMeters * 0.5;
        const glm::dvec3 center = area.anchorEcef + area.upEcef * halfSize;
        auto ecefFromLocal = [&](const glm::dvec3& local) {
            return center + area.eastEcef * local.x + area.northEcef * local.y + area.upEcef * local.z;
        };
        auto ecefNormalFromLocal = [&](const glm::dvec3& localNormal) {
            return glm::normalize(area.eastEcef * localNormal.x +
                                  area.northEcef * localNormal.y +
                                  area.upEcef * localNormal.z);
        };

        if (area.geometryKind == GeoSpatialGeometryKind::CubeArea) {
            const std::array<glm::dvec3, 8> corners = {
                glm::dvec3{-halfSize, -halfSize, -halfSize},
                glm::dvec3{ halfSize, -halfSize, -halfSize},
                glm::dvec3{ halfSize,  halfSize, -halfSize},
                glm::dvec3{-halfSize,  halfSize, -halfSize},
                glm::dvec3{-halfSize, -halfSize,  halfSize},
                glm::dvec3{ halfSize, -halfSize,  halfSize},
                glm::dvec3{ halfSize,  halfSize,  halfSize},
                glm::dvec3{-halfSize,  halfSize,  halfSize}
            };
            const std::array<std::array<uint8_t, 4>, 6> faces = {{
                {{0, 3, 2, 1}}, {{4, 5, 6, 7}}, {{0, 1, 5, 4}},
                {{3, 7, 6, 2}}, {{0, 4, 7, 3}}, {{1, 2, 6, 5}}
            }};
            const std::array<glm::dvec3, 6> normals = {
                glm::dvec3{0.0, 0.0, -1.0}, glm::dvec3{0.0, 0.0, 1.0},
                glm::dvec3{0.0, -1.0, 0.0}, glm::dvec3{0.0, 1.0, 0.0},
                glm::dvec3{-1.0, 0.0, 0.0}, glm::dvec3{1.0, 0.0, 0.0}
            };
            for (size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
                const uint16_t baseVertex = static_cast<uint16_t>(areaVertices.size());
                const glm::dvec3 normal = ecefNormalFromLocal(normals[faceIndex]);
                for (uint8_t cornerIndex : faces[faceIndex])
                    appendVertex(ecefFromLocal(corners[cornerIndex]), normal);
                areaIndices.insert(areaIndices.end(), {
                    baseVertex, static_cast<uint16_t>(baseVertex + 1), static_cast<uint16_t>(baseVertex + 2),
                    baseVertex, static_cast<uint16_t>(baseVertex + 2), static_cast<uint16_t>(baseVertex + 3)
                });
            }
        } else {
            constexpr uint16_t stackCount = 12;
            constexpr uint16_t sliceCount = 24;
            constexpr double pi = 3.1415926535897932384626433832795;
            for (uint16_t stack = 0; stack <= stackCount; ++stack) {
                const double latitude = -0.5 * pi + pi * static_cast<double>(stack) / stackCount;
                const double ringRadius = std::cos(latitude);
                const double localUp = std::sin(latitude);
                for (uint16_t slice = 0; slice <= sliceCount; ++slice) {
                    const double longitude = 2.0 * pi * static_cast<double>(slice) / sliceCount;
                    const glm::dvec3 localNormal{
                        ringRadius * std::cos(longitude),
                        ringRadius * std::sin(longitude),
                        localUp
                    };
                    const glm::dvec3 ecefNormal = ecefNormalFromLocal(localNormal);
                    appendVertex(center + ecefNormal * halfSize, ecefNormal);
                }
            }
            for (uint16_t stack = 0; stack < stackCount; ++stack) {
                for (uint16_t slice = 0; slice < sliceCount; ++slice) {
                    const uint16_t first = static_cast<uint16_t>(stack * (sliceCount + 1) + slice);
                    const uint16_t second = static_cast<uint16_t>(first + sliceCount + 1);
                    areaIndices.insert(areaIndices.end(), {
                        first, second, static_cast<uint16_t>(first + 1),
                        second, static_cast<uint16_t>(second + 1), static_cast<uint16_t>(first + 1)
                    });
                }
            }
        }

        const uint32_t vertexCount = static_cast<uint32_t>(areaVertices.size());
        const uint32_t indexCount = static_cast<uint32_t>(areaIndices.size());
        if (vertexCount == 0 || indexCount == 0)
            throw std::runtime_error("CesiumBgfxRenderAdapter generated empty area geometry");
        if (bgfx::getAvailTransientVertexBuffer(vertexCount, vertexLayout()) < vertexCount ||
            bgfx::getAvailTransientIndexBuffer(indexCount, false) < indexCount) {
            throw std::runtime_error("CesiumBgfxRenderAdapter lacks transient buffer space for area geometry");
        }

        bgfx::TransientVertexBuffer vertexBuffer;
        bgfx::TransientIndexBuffer indexBuffer;
        bgfx::allocTransientVertexBuffer(&vertexBuffer, vertexCount, vertexLayout());
        bgfx::allocTransientIndexBuffer(&indexBuffer, indexCount, false);
        std::memcpy(vertexBuffer.data, areaVertices.data(), areaVertices.size() * sizeof(CesiumBgfxVertex));
        std::memcpy(indexBuffer.data, areaIndices.data(), areaIndices.size() * sizeof(uint16_t));

        const float areaOverlayTransform[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        const float areaTextureParams[4] = {0.0f, 1.0f, 0.0f, 0.0f};
        float identity[16];
        matrixToFloatColumnMajor(glm::dmat4(1.0), identity);
        bgfx::setTransform(identity);
        bgfx::setVertexBuffer(0, &vertexBuffer, 0, vertexCount);
        bgfx::setIndexBuffer(&indexBuffer, 0, indexCount);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::LightDir), lightDir);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::LightParams), lightParams);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::BaseColorFactor), area.color.data());
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::OverlayTransform), areaOverlayTransform);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::TextureParams), areaTextureParams);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::CameraPos), cameraPos);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::ColorParams), colorParams);
        bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::SurfaceParams), surfaceParams);
        bgfx::setTexture(0, cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::ImagerySampler), g_whiteTexture);
        bgfx::setState(kAreaState);
        bgfx::submit(viewId_, cesiumShadersGetProgram(g_shaderState));
        submitted = true;
        ++areaDrawCount;
    }

    struct PreparedMarker {
        glm::dvec3 center;
        double radius = 0.0;
        std::array<float, 4> color{};
    };
    std::vector<PreparedMarker> preparedMarkers;
    preparedMarkers.reserve(pointMarkers.size());
    const double projectionY = std::abs(projection[1][1]);
    if (projectionY <= std::numeric_limits<double>::epsilon())
        throw std::runtime_error("CesiumBgfxRenderAdapter point marker projection has zero Y scale");

    for (const CesiumBgfxPointMarker& marker : pointMarkers) {
        const glm::dvec4 viewPosition = view * glm::dvec4(marker.positionEcef, 1.0);
        if (viewPosition.z >= -0.01)
            continue;

        const double radius = std::max(
            1.0,
            (14.0 * -viewPosition.z) /
                (projectionY * static_cast<double>(attachment->frameBufferHeight())));
        preparedMarkers.push_back({glm::dvec3(viewPosition), radius, marker.color});
    }

    if (!preparedMarkers.empty()) {
        const uint32_t vertexCount = static_cast<uint32_t>(preparedMarkers.size() * 4);
        const uint32_t indexCount = static_cast<uint32_t>(preparedMarkers.size() * 6);
        if (bgfx::getAvailTransientVertexBuffer(vertexCount, vertexLayout()) < vertexCount ||
            bgfx::getAvailTransientIndexBuffer(indexCount, false) < indexCount) {
            throw std::runtime_error("CesiumBgfxRenderAdapter lacks transient buffer space for point markers");
        }

        bgfx::TransientVertexBuffer vertexBuffer;
        bgfx::TransientIndexBuffer indexBuffer;
        bgfx::allocTransientVertexBuffer(&vertexBuffer, vertexCount, vertexLayout());
        bgfx::allocTransientIndexBuffer(&indexBuffer, indexCount, false);
        auto* vertices = reinterpret_cast<CesiumBgfxVertex*>(vertexBuffer.data);
        auto* indices = reinterpret_cast<uint16_t*>(indexBuffer.data);

        for (size_t markerIndex = 0; markerIndex < preparedMarkers.size(); ++markerIndex) {
            const PreparedMarker& marker = preparedMarkers[markerIndex];
            const double z = marker.center.z + marker.radius * 0.25;
            const size_t vertexOffset = markerIndex * 4;
            const size_t indexOffset = markerIndex * 6;
            vertices[vertexOffset + 0].px = static_cast<float>(marker.center.x - marker.radius);
            vertices[vertexOffset + 0].py = static_cast<float>(marker.center.y - marker.radius);
            vertices[vertexOffset + 0].pz = static_cast<float>(z);
            vertices[vertexOffset + 1].px = static_cast<float>(marker.center.x + marker.radius);
            vertices[vertexOffset + 1].py = static_cast<float>(marker.center.y - marker.radius);
            vertices[vertexOffset + 1].pz = static_cast<float>(z);
            vertices[vertexOffset + 2].px = static_cast<float>(marker.center.x + marker.radius);
            vertices[vertexOffset + 2].py = static_cast<float>(marker.center.y + marker.radius);
            vertices[vertexOffset + 2].pz = static_cast<float>(z);
            vertices[vertexOffset + 3].px = static_cast<float>(marker.center.x - marker.radius);
            vertices[vertexOffset + 3].py = static_cast<float>(marker.center.y + marker.radius);
            vertices[vertexOffset + 3].pz = static_cast<float>(z);
            indices[indexOffset + 0] = 0;
            indices[indexOffset + 1] = 1;
            indices[indexOffset + 2] = 2;
            indices[indexOffset + 3] = 2;
            indices[indexOffset + 4] = 3;
            indices[indexOffset + 5] = 0;
        }

        const float markerLightDir[4] = {0.0f, 0.0f, 1.0f, 0.0f};
        const float markerLightParams[4] = {0.0f, 1.0f, 0.0f, 0.0f};
        const float markerOverlayTransform[4] = {0.0f, 0.0f, 1.0f, 1.0f};
        const float markerTextureParams[4] = {0.0f, 1.0f, 0.0f, 0.0f};
        const float markerSurfaceParams[4] = {0.0f, 1.0f, 0.0f, 1.0f};
        float identity[16];
        matrixToFloatColumnMajor(glm::dmat4(1.0), identity);

        for (size_t markerIndex = 0; markerIndex < preparedMarkers.size(); ++markerIndex) {
            const PreparedMarker& marker = preparedMarkers[markerIndex];
            bgfx::setTransform(identity);
            bgfx::setVertexBuffer(0, &vertexBuffer, static_cast<uint32_t>(markerIndex * 4), 4);
            bgfx::setIndexBuffer(&indexBuffer, static_cast<uint32_t>(markerIndex * 6), 6);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::LightDir), markerLightDir);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::LightParams), markerLightParams);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::BaseColorFactor), marker.color.data());
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::OverlayTransform), markerOverlayTransform);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::TextureParams), markerTextureParams);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::CameraPos), cameraPos);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::ColorParams), colorParams);
            bgfx::setUniform(cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::SurfaceParams), markerSurfaceParams);
            bgfx::setTexture(0, cesiumShadersGetUniform(g_shaderState, CesiumShaderUniform::ImagerySampler), g_whiteTexture);
            bgfx::setState(kAlphaState);
            bgfx::submit(viewId_, cesiumShadersGetProgram(g_shaderState));
            submitted = true;
        }
    }

    // Every surface contributes exactly one authoritative draw: raster
    // imagery when attached, otherwise its glTF/globe base material.
    if (diagFrameDue) {
        LOG_DEBUG("CesiumBgfxRenderAdapter",
                  "render passes base=" + std::to_string(baseDrawCount) +
                  " overlay=" + std::to_string(overlayDrawCount) +
                  " tiles=" + std::to_string(tiles.size()) +
                  " fadingOut=" + std::to_string(fadingOutTiles.size()) +
                  " markersVisible=" + std::to_string(pointMarkers.size()) +
                  " markersPrepared=" + std::to_string(preparedMarkers.size()) +
                  " areasVisible=" + std::to_string(areaShapes.size()) +
                  " areasDrawn=" + std::to_string(areaDrawCount) +
                  " areasMinPixelScaled=" + std::to_string(areaMinimumPixelScaleCount));
    }

    if (!submitted)
        bgfx::touch(viewId_);
}

} // namespace GRIM::GeoSpatial