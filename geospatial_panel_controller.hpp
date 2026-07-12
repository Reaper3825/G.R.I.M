#pragma once

#include "geospatial/geospatial_geometry.hpp"

#include <string>
#include <vector>

namespace GRIM::GeoSpatial {

struct GeoSpatialLayerSnapshot {
    std::string id;
    std::string name;
    bool visible = false;
    float opacity = 1.0f;
};

struct GeoSpatialPanelSnapshot {
    std::string cesium_status = "Controller not attached";
    std::string active_tileset = "No tileset loaded";
    std::string camera_mode = "Detached";
    bool cesium_ready = false;
    bool viewport_ready = false;
    bool terrain_enabled = false;
    bool imagery_enabled = false;
    std::vector<GeoSpatialLayerSnapshot> layers;
    bool point_catalog_dirty = false;
    std::string point_catalog_status = "Point catalog not loaded";
    std::vector<GeoSpatialGroupDefinition> groups;
    bool picked_location_valid = false;
    double picked_longitude_degrees = 0.0;
    double picked_latitude_degrees = 0.0;
    double picked_height_meters = 0.0;
};

class GeoSpatialPanelController {
public:
    virtual ~GeoSpatialPanelController() = default;

    virtual GeoSpatialPanelSnapshot snapshot() const = 0;
    virtual void requestInitializeCesium() = 0;
    virtual void requestReloadTileset() = 0;
    virtual void requestRecenterHome() = 0;
    virtual void requestResetCamera() = 0;
    virtual void requestSetLayerVisibility(const std::string& id, bool visible) = 0;
    virtual void requestSetLayerOpacity(const std::string& id, float opacity) = 0;
    virtual void requestUpsertGroup(const std::string& originalId,
                                    const std::string& name,
                                    const std::string& color) = 0;
    virtual void requestRemoveGroup(const std::string& id) = 0;
    virtual void requestToggleGroupVisibility(const std::string& id) = 0;
    virtual void requestUpsertPoint(const std::string& originalId,
                                    const GeoSpatialPointDefinition& point) = 0;
    virtual void requestRemovePoint(const std::string& id) = 0;
    virtual void requestTogglePointVisibility(const std::string& id) = 0;
    virtual void requestUpsertArea(const std::string& originalId,
                                   const GeoSpatialAreaDefinition& area) = 0;
    virtual void requestRemoveArea(const std::string& id) = 0;
    virtual void requestToggleAreaVisibility(const std::string& id) = 0;
    virtual void requestSavePointCatalog() = 0;
};

} // namespace GRIM::GeoSpatial