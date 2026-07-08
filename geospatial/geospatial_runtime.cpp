#include "geospatial_runtime.hpp"

#include <CesiumGeospatial/Cartographic.h>
#include <CesiumGeospatial/Ellipsoid.h>

#include <iomanip>
#include <stdexcept>
#include <sstream>

namespace GRIM::GeoSpatial {

namespace {
    std::string formatEcef(const glm::dvec3& ecef)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(1)
               << "ECEF(" << ecef.x << ", " << ecef.y << ", " << ecef.z << ")";
        return stream.str();
    }
}

GeoSpatialRuntime::GeoSpatialRuntime()
{
    snapshot_.cesium_status = "Cesium Native linked; waiting for initialization";
    snapshot_.active_tileset = "No tileset loaded";
    snapshot_.camera_mode = "Home camera not initialized";
}

GeoSpatialPanelSnapshot GeoSpatialRuntime::snapshot() const
{
    std::lock_guard lock(mutex_);
    return snapshot_;
}

void GeoSpatialRuntime::requestInitializeCesium()
{
    std::lock_guard lock(mutex_);

    const CesiumGeospatial::Cartographic home = CesiumGeospatial::Cartographic::fromDegrees(
        -75.59777,
        40.03883,
        24000000.0);
    homeEcef_ = CesiumGeospatial::Ellipsoid::WGS84.cartographicToCartesian(home);

    initialized_ = true;
    snapshot_.cesium_ready = true;
    snapshot_.cesium_status = "Cesium Native initialized; bgfx tile renderer pending";
    snapshot_.active_tileset = "Ready for 3D Tiles / terrain provider binding";
    refreshCameraStatusLocked();
}

void GeoSpatialRuntime::requestReloadTileset()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("reload tileset");
        return;
    }

    snapshot_.active_tileset = "Tileset reload requested; loader adapter pending";
    snapshot_.cesium_status = "Cesium Native ready for tileset reload";
}

void GeoSpatialRuntime::requestRecenterHome()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("recenter home");
        return;
    }

    refreshCameraStatusLocked();
    snapshot_.cesium_status = "Camera recentered to WGS84 home";
}

void GeoSpatialRuntime::requestResetCamera()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("reset camera");
        return;
    }

    refreshCameraStatusLocked();
    snapshot_.cesium_status = "Camera reset to default globe orbit";
}

void GeoSpatialRuntime::requestToggleTerrain()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("toggle terrain");
        return;
    }

    snapshot_.terrain_enabled = !snapshot_.terrain_enabled;
    snapshot_.cesium_status = snapshot_.terrain_enabled ? "Terrain enabled" : "Terrain disabled";
}

void GeoSpatialRuntime::requestToggleImagery()
{
    std::lock_guard lock(mutex_);
    if (!initialized_) {
        setNotInitializedStatusLocked("toggle imagery");
        return;
    }

    snapshot_.imagery_enabled = !snapshot_.imagery_enabled;
    snapshot_.cesium_status = snapshot_.imagery_enabled ? "Imagery enabled" : "Imagery disabled";
}

void GeoSpatialRuntime::setViewportReady(bool ready)
{
    std::lock_guard lock(mutex_);
    snapshot_.viewport_ready = ready;
    if (!ready) {
        snapshot_.cesium_status = "Native viewport attachment unavailable";
    }
}

void GeoSpatialRuntime::tick(float dtSeconds)
{
    std::lock_guard lock(mutex_);
    if (dtSeconds < 0.0f)
        throw std::runtime_error("GeoSpatialRuntime::tick requires non-negative dtSeconds");

    elapsedSeconds_ += static_cast<double>(dtSeconds);
}

void GeoSpatialRuntime::refreshCameraStatusLocked()
{
    snapshot_.camera_mode = "Home orbit " + formatEcef(homeEcef_);
}

void GeoSpatialRuntime::setNotInitializedStatusLocked(const char* action)
{
    snapshot_.cesium_status = std::string("Initialize Cesium before ") + action;
}

} // namespace GRIM::GeoSpatial