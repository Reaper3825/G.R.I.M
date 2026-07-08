#pragma once

#include "../geospatial_panel_controller.hpp"

#include <glm/vec3.hpp>

#include <mutex>

namespace GRIM::GeoSpatial {

class GeoSpatialRuntime final : public GeoSpatialPanelController {
public:
    GeoSpatialRuntime();

    GeoSpatialPanelSnapshot snapshot() const override;
    void requestInitializeCesium() override;
    void requestReloadTileset() override;
    void requestRecenterHome() override;
    void requestResetCamera() override;
    void requestToggleTerrain() override;
    void requestToggleImagery() override;

    void setViewportReady(bool ready);
    void tick(float dtSeconds);

private:
    void refreshCameraStatusLocked();
    void setNotInitializedStatusLocked(const char* action);

    mutable std::mutex mutex_;
    GeoSpatialPanelSnapshot snapshot_{};
    glm::dvec3 homeEcef_{0.0, 0.0, 0.0};
    double elapsedSeconds_ = 0.0;
    bool initialized_ = false;
};

} // namespace GRIM::GeoSpatial