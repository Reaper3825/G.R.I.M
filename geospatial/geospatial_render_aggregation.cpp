#include "geospatial_render_aggregation.hpp"

#include "ui/ui_camera.hpp"

#include <CesiumGeospatial/Cartographic.h>

#include <cmath>

namespace GRIM::GeoSpatial {

GeoSpatialRenderPayloads aggregateRenderPayloads(
    const std::vector<GeoSpatialGroupDefinition>& groups)
{
    GeoSpatialRenderPayloads payloads;
    for (const GeoSpatialGroupDefinition& group : groups) {
        if (!group.visible)
            continue;
        for (const GeoSpatialPointDefinition& point : group.points) {
            if (!point.visible)
                continue;
            const CesiumGeospatial::Cartographic cartographic = CesiumGeospatial::Cartographic::fromDegrees(
                point.longitude_degrees, point.latitude_degrees, point.height_meters);
            payloads.point_markers.push_back({
                UICamera::cartographicToEcef(cartographic),
                colorToFloat(point.color)
            });
        }
        for (const GeoSpatialAreaDefinition& area : group.areas) {
            if (!area.visible)
                continue;
            const CesiumGeospatial::Cartographic cartographic = CesiumGeospatial::Cartographic::fromDegrees(
                area.longitude_degrees, area.latitude_degrees, area.height_meters);
            const double longitude = cartographic.longitude;
            const double latitude = cartographic.latitude;
            const glm::dvec3 east{-std::sin(longitude), std::cos(longitude), 0.0};
            const glm::dvec3 north{
                -std::sin(latitude) * std::cos(longitude),
                -std::sin(latitude) * std::sin(longitude),
                std::cos(latitude)
            };
            const glm::dvec3 up{
                std::cos(latitude) * std::cos(longitude),
                std::cos(latitude) * std::sin(longitude),
                std::sin(latitude)
            };
            payloads.area_shapes.push_back({
                area.geometry_kind,
                UICamera::cartographicToEcef(cartographic),
                east,
                north,
                up,
                area.size_meters,
                colorToFloat(area.color, area.opacity)
            });
        }
    }
    return payloads;
}

} // namespace GRIM::GeoSpatial