#pragma once

struct GRIMWindow;

namespace GRIM::GeoSpatial {
class GeoSpatialRuntime;
}

namespace GRIM {

void tickApplicationFrame(
    GeoSpatial::GeoSpatialRuntime& geoSpatialRuntime,
    const GRIMWindow& overlayWindow);

} // namespace GRIM