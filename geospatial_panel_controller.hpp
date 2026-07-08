#pragma once

#include <string>

namespace GRIM::GeoSpatial {

struct GeoSpatialPanelSnapshot {
    std::string cesium_status = "Controller not attached";
    std::string active_tileset = "No tileset loaded";
    std::string camera_mode = "Detached";
    bool cesium_ready = false;
    bool viewport_ready = false;
    bool terrain_enabled = false;
    bool imagery_enabled = false;
};

class GeoSpatialPanelController {
public:
    virtual ~GeoSpatialPanelController() = default;

    virtual GeoSpatialPanelSnapshot snapshot() const = 0;
    virtual void requestInitializeCesium() = 0;
    virtual void requestReloadTileset() = 0;
    virtual void requestRecenterHome() = 0;
    virtual void requestResetCamera() = 0;
    virtual void requestToggleTerrain() = 0;
    virtual void requestToggleImagery() = 0;
};

} // namespace GRIM::GeoSpatial