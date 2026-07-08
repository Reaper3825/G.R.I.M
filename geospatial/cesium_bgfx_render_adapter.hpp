#pragma once

#ifdef OPAQUE
#undef OPAQUE
#endif

#include <Cesium3DTilesSelection/IPrepareRendererResources.h>
#include <Cesium3DTilesSelection/Tile.h>

#include <bgfx/bgfx.h>
#include <glm/mat4x4.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

class UINative3DViewportAttachment;

namespace GRIM::GeoSpatial {

class CesiumBgfxRenderAdapter final : public Cesium3DTilesSelection::IPrepareRendererResources {
public:
    explicit CesiumBgfxRenderAdapter(std::string owner);
    ~CesiumBgfxRenderAdapter() override;

    CesiumBgfxRenderAdapter(const CesiumBgfxRenderAdapter&) = delete;
    CesiumBgfxRenderAdapter& operator=(const CesiumBgfxRenderAdapter&) = delete;

    void setViewportAttachment(std::weak_ptr<UINative3DViewportAttachment> attachment);
    void setFrameSelection(std::vector<Cesium3DTilesSelection::Tile::ConstPointer> tiles,
                           glm::dmat4 view,
                           glm::dmat4 projection);

    CesiumAsync::Future<Cesium3DTilesSelection::TileLoadResultAndRenderResources>
    prepareInLoadThread(const CesiumAsync::AsyncSystem& asyncSystem,
                        Cesium3DTilesSelection::TileLoadResult&& tileLoadResult,
                        const glm::dmat4& transform,
                        const std::any& rendererOptions) override;

    void* prepareInMainThread(Cesium3DTilesSelection::Tile& tile,
                              void* pLoadThreadResult) override;

    void free(Cesium3DTilesSelection::Tile& tile,
              void* pLoadThreadResult,
              void* pMainThreadResult) noexcept override;

    void* prepareRasterInLoadThread(CesiumImage::ImageAsset& image,
                                    const std::any& rendererOptions) override;

    void* prepareRasterInMainThread(CesiumRasterOverlays::RasterOverlayTile& rasterTile,
                                    void* pLoadThreadResult) override;

    void freeRaster(const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
                    void* pLoadThreadResult,
                    void* pMainThreadResult) noexcept override;

    void attachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
                                  int32_t overlayTextureCoordinateID,
                                  const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
                                  void* pMainThreadRendererResources,
                                  const glm::dvec2& translation,
                                  const glm::dvec2& scale) override;

    void detachRasterInMainThread(const Cesium3DTilesSelection::Tile& tile,
                                  int32_t overlayTextureCoordinateID,
                                  const CesiumRasterOverlays::RasterOverlayTile& rasterTile,
                                  void* pMainThreadRendererResources) noexcept override;

    static void renderAll(uint32_t bgfxFrame);

private:
    void render(uint32_t bgfxFrame);

    std::string owner_;
    bgfx::ViewId viewId_ = 0;
    std::weak_ptr<UINative3DViewportAttachment> viewportAttachment_;
    std::vector<Cesium3DTilesSelection::Tile::ConstPointer> frameTiles_;
    glm::dmat4 view_ = glm::dmat4(1.0);
    glm::dmat4 projection_ = glm::dmat4(1.0);
    mutable std::mutex mutex_;
};

} // namespace GRIM::GeoSpatial