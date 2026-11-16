#include "geospatial_geometry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace GRIM::GeoSpatial {

const char* geometryKindName(GeoSpatialGeometryKind kind)
{
    switch (kind) {
        case GeoSpatialGeometryKind::Point:
            return "point";
        case GeoSpatialGeometryKind::CubeArea:
            return "cube_area";
        case GeoSpatialGeometryKind::SphereArea:
            return "sphere_area";
    }
    throw std::runtime_error("GeoSpatial geometry kind is invalid");
}

GeoSpatialGeometryKind parseGeometryKind(const std::string& value)
{
    if (value == "point")
        return GeoSpatialGeometryKind::Point;
    if (value == "cube_area")
        return GeoSpatialGeometryKind::CubeArea;
    if (value == "sphere_area")
        return GeoSpatialGeometryKind::SphereArea;
    throw std::runtime_error("GeoSpatial geometry_kind must be 'point', 'cube_area', or 'sphere_area'");
}

void validateIdentifier(const std::string& id, const char* kind)
{
    if (!kind || kind[0] == '\0')
        throw std::runtime_error("GeoSpatial identifier validation requires a kind");
    if (id.empty() || id.size() > 64)
        throw std::runtime_error(std::string("GeoSpatial ") + kind + " id must contain 1-64 characters");
    if (!std::all_of(id.begin(), id.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '-' || character == '_' || character == '.';
        })) {
        throw std::runtime_error(std::string("GeoSpatial ") + kind + " id may only use letters, digits, '-', '_', and '.'");
    }
}

void validateName(const std::string& name, const char* kind)
{
    if (!kind || kind[0] == '\0')
        throw std::runtime_error("GeoSpatial name validation requires a kind");
    if (name.empty() || name.size() > 128)
        throw std::runtime_error(std::string("GeoSpatial ") + kind + " name must contain 1-128 characters");
}

void validateCoordinates(const GeoSpatialEntityDefinition& entity)
{
    if (!std::isfinite(entity.longitude_degrees) || entity.longitude_degrees < -180.0 || entity.longitude_degrees > 180.0)
        throw std::runtime_error("GeoSpatial entity longitude must be finite and within [-180, 180]");
    if (!std::isfinite(entity.latitude_degrees) || entity.latitude_degrees < -90.0 || entity.latitude_degrees > 90.0)
        throw std::runtime_error("GeoSpatial entity latitude must be finite and within [-90, 90]");
    if (!std::isfinite(entity.height_meters) || entity.height_meters < -10000.0 || entity.height_meters > 100000000.0)
        throw std::runtime_error("GeoSpatial entity height must be finite and within [-10000, 100000000] meters");
}

void validateArea(const GeoSpatialAreaDefinition& area)
{
    if (area.geometry_kind != GeoSpatialGeometryKind::CubeArea &&
        area.geometry_kind != GeoSpatialGeometryKind::SphereArea) {
        throw std::runtime_error("GeoSpatial area geometry must be cube_area or sphere_area");
    }
    if (!std::isfinite(area.size_meters) || area.size_meters <= 0.0 || area.size_meters > 10000000.0)
        throw std::runtime_error("GeoSpatial area size must be finite and within (0, 10000000] meters");
    if (!std::isfinite(area.opacity) || area.opacity < 0.05f || area.opacity > 0.95f)
        throw std::runtime_error("GeoSpatial area opacity must be finite and within [0.05, 0.95]");
}

std::string normalizeColor(const std::string& color)
{
    if (color.size() != 7 || color.front() != '#' ||
        !std::all_of(color.begin() + 1, color.end(), [](unsigned char character) { return std::isxdigit(character); })) {
        throw std::runtime_error("GeoSpatial color must use #RRGGBB format");
    }

    std::string normalized = color;
    std::transform(normalized.begin() + 1, normalized.end(), normalized.begin() + 1,
                   [](unsigned char character) { return static_cast<char>(std::toupper(character)); });
    return normalized;
}

std::array<float, 4> colorToFloat(const std::string& color, float alpha)
{
    const std::string normalized = normalizeColor(color);
    const auto channel = [&](size_t offset) {
        return static_cast<float>(std::stoul(normalized.substr(offset, 2), nullptr, 16)) / 255.0f;
    };
    return {channel(1), channel(3), channel(5), alpha};
}

} // namespace GRIM::GeoSpatial