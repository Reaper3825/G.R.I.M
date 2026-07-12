#pragma once

#include "cesium_bgfx_render_adapter.hpp"
#include "geospatial_geometry.hpp"

#include <vector>

namespace GRIM::GeoSpatial {

struct GeoSpatialRenderPayloads {
    std::vector<CesiumBgfxPointMarker> point_markers;
    std::vector<CesiumBgfxAreaShape> area_shapes;
};

GeoSpatialRenderPayloads aggregateRenderPayloads(
    const std::vector<GeoSpatialGroupDefinition>& groups);

} // namespace GRIM::GeoSpatial