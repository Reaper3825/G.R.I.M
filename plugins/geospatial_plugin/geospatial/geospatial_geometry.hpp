#pragma once

#include <array>
#include <string>
#include <vector>

namespace GRIM::GeoSpatial {

enum class GeoSpatialGeometryKind {
    Point,
    CubeArea,
    SphereArea
};

struct GeoSpatialEntityDefinition {
    std::string id;
    std::string name;
    std::string group_id;
    std::string color = "#FFFFFF";
    double longitude_degrees = 0.0;
    double latitude_degrees = 0.0;
    double height_meters = 0.0;
    bool visible = true;
    GeoSpatialGeometryKind geometry_kind = GeoSpatialGeometryKind::Point;
};

struct GeoSpatialPointDefinition final : GeoSpatialEntityDefinition {
    GeoSpatialPointDefinition()
    {
        geometry_kind = GeoSpatialGeometryKind::Point;
    }
};

struct GeoSpatialAreaDefinition final : GeoSpatialEntityDefinition {
    GeoSpatialAreaDefinition()
    {
        geometry_kind = GeoSpatialGeometryKind::CubeArea;
    }

    double size_meters = 1000.0;
    float opacity = 0.35f;
};

struct GeoSpatialGroupDefinition {
    std::string id;
    std::string name;
    std::string color = "#FFFFFF";
    bool visible = true;
    std::vector<GeoSpatialPointDefinition> points;
    std::vector<GeoSpatialAreaDefinition> areas;
};

const char* geometryKindName(GeoSpatialGeometryKind kind);
GeoSpatialGeometryKind parseGeometryKind(const std::string& value);
void validateIdentifier(const std::string& id, const char* kind);
void validateName(const std::string& name, const char* kind);
void validateCoordinates(const GeoSpatialEntityDefinition& entity);
void validateArea(const GeoSpatialAreaDefinition& area);
std::string normalizeColor(const std::string& color);
std::array<float, 4> colorToFloat(const std::string& color, float alpha = 1.0f);

} // namespace GRIM::GeoSpatial