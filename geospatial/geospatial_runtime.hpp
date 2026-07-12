#pragma once

#include "../geospatial_panel_controller.hpp"
#include "core/platform_window.hpp"
#include "ui/ui_camera.hpp"

#include <exception>
#include <memory>
#include <mutex>

class UINative3DViewportAttachment;

namespace GRIM::GeoSpatial {

class GeoSpatialRuntime final : public GeoSpatialPanelController {
public:
    GeoSpatialRuntime();
    ~GeoSpatialRuntime() override;

    GeoSpatialPanelSnapshot snapshot() const override;
    void requestInitializeCesium() override;
    void requestReloadTileset() override;
    void requestRecenterHome() override;
    void requestResetCamera() override;
    void requestSetLayerVisibility(const std::string& id, bool visible) override;
    void requestSetLayerOpacity(const std::string& id, float opacity) override;
    void requestSetLodOverrideEnabled(bool enabled) override;
    void requestSetLodOverrideLevel(int level) override;
    void requestUpsertGroup(const std::string& originalId,
                            const std::string& name,
                            const std::string& color) override;
    void requestRemoveGroup(const std::string& id) override;
    void requestToggleGroupVisibility(const std::string& id) override;
    void requestUpsertPoint(const std::string& originalId,
                            const GeoSpatialPointDefinition& point) override;
    void requestRemovePoint(const std::string& id) override;
    void requestTogglePointVisibility(const std::string& id) override;
    void requestUpsertArea(const std::string& originalId,
                           const GeoSpatialAreaDefinition& area) override;
    void requestRemoveArea(const std::string& id) override;
    void requestToggleAreaVisibility(const std::string& id) override;
    void requestSavePointCatalog() override;

    void setViewportReady(bool ready);
    void setViewportAttachment(std::weak_ptr<UINative3DViewportAttachment> attachment);
    void tick(float dtSeconds);

private:
    struct CesiumRuntimeContext;

    void refreshCameraStatusLocked();
    void setNotInitializedStatusLocked(const char* action);
    void setActionFailedStatusLocked(const char* action, const std::exception& error);
    void updateProviderStatusLocked();
    void handleViewportInput(const PlatformWindow::ViewportInputEvent& event);
    void applyViewportKeyboardLocked(float dtSeconds);
    void resetCameraLocked();
    void loadPointCatalogLocked();
    void savePointCatalogLocked();
    void syncPointMarkersLocked();
    void markPointCatalogChangedLocked(const std::string& status);
    void setPointCatalogFailedStatusLocked(const char* action, const std::exception& error);
    std::string generatePointCatalogIdLocked(const char* prefix) const;

    mutable std::mutex mutex_;
    GeoSpatialPanelSnapshot snapshot_{};
    std::unique_ptr<CesiumRuntimeContext> cesium_;
    std::weak_ptr<UINative3DViewportAttachment> viewportAttachment_;
    UICamera camera_;
    double elapsedSeconds_ = 0.0;
    int viewportLastX_ = 0;
    int viewportLastY_ = 0;
    double viewportDragPixels_ = 0.0;
    bool viewportInputActive_ = false;
    bool viewportDraggingLeft_ = false;
    bool viewportDraggingRight_ = false;
    bool viewportDraggingMiddle_ = false;
    bool firstFrameLogged_ = false;
    bool initialized_ = false;
};

} // namespace GRIM::GeoSpatial