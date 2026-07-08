#pragma once

#include "../geospatial_panel_controller.hpp"
#include "core/platform_window.hpp"
#include "ui/ui_camera.hpp"

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
    void requestToggleTerrain() override;
    void requestToggleImagery() override;

    void setViewportReady(bool ready);
    void setViewportAttachment(std::weak_ptr<UINative3DViewportAttachment> attachment);
    void tick(float dtSeconds);

private:
    struct CesiumRuntimeContext;

    void refreshCameraStatusLocked();
    void setNotInitializedStatusLocked(const char* action);
    void updateProviderStatusLocked();
    void handleViewportInput(const PlatformWindow::ViewportInputEvent& event);
    void applyViewportKeyboardLocked(float dtSeconds);
    void resetCameraLocked();

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
    bool initialized_ = false;
};

} // namespace GRIM::GeoSpatial