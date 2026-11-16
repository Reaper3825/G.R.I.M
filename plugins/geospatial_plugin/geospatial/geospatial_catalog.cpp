#include "geospatial_catalog.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifndef GRIM_ROOT_DIR
#error GRIM_ROOT_DIR must be defined by the build system
#endif

namespace GRIM::GeoSpatial {

namespace {
    constexpr int kPointCatalogVersion = 2;

    std::string requireCatalogString(const nlohmann::json& object, const char* key, const std::string& path)
    {
        if (!object.contains(key) || !object.at(key).is_string())
            throw std::runtime_error("GeoSpatial point catalog requires string " + path + "." + key);
        const std::string value = object.at(key).get<std::string>();
        if (value.empty())
            throw std::runtime_error("GeoSpatial point catalog requires non-empty string " + path + "." + key);
        return value;
    }

    bool requireCatalogBool(const nlohmann::json& object, const char* key, const std::string& path)
    {
        if (!object.contains(key) || !object.at(key).is_boolean())
            throw std::runtime_error("GeoSpatial point catalog requires boolean " + path + "." + key);
        return object.at(key).get<bool>();
    }

    double requireCatalogNumber(const nlohmann::json& object, const char* key, const std::string& path)
    {
        if (!object.contains(key) || !object.at(key).is_number())
            throw std::runtime_error("GeoSpatial point catalog requires number " + path + "." + key);
        const double value = object.at(key).get<double>();
        if (!std::isfinite(value))
            throw std::runtime_error("GeoSpatial point catalog requires finite number " + path + "." + key);
        return value;
    }

    bool entityIdExists(const std::vector<GeoSpatialGroupDefinition>& groups, const std::string& id)
    {
        for (const GeoSpatialGroupDefinition& group : groups) {
            if (group.id == id)
                return true;
            if (std::any_of(group.points.begin(), group.points.end(),
                            [&](const GeoSpatialPointDefinition& point) { return point.id == id; })) {
                return true;
            }
            if (std::any_of(group.areas.begin(), group.areas.end(),
                            [&](const GeoSpatialAreaDefinition& area) { return area.id == id; })) {
                return true;
            }
        }
        return false;
    }
}

std::filesystem::path pointCatalogPath()
{
    return std::filesystem::path(GRIM_ROOT_DIR) / "data" / "geospatial" / "point_groups.json";
}

std::vector<GeoSpatialGroupDefinition> loadPointCatalog(const std::filesystem::path& path)
{
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("GeoSpatial point catalog path is not a regular file: " + path.string());

    std::ifstream input(path);
    if (!input.is_open())
        throw std::runtime_error("GeoSpatial point catalog cannot be opened: " + path.string());
    nlohmann::json catalog;
    input >> catalog;
    if (!catalog.is_object() || !catalog.contains("version") || !catalog.at("version").is_number_integer())
        throw std::runtime_error("GeoSpatial point catalog requires integer version");
    if (catalog.at("version").get<int>() != kPointCatalogVersion)
        throw std::runtime_error("GeoSpatial point catalog version must be " + std::to_string(kPointCatalogVersion));
    if (!catalog.contains("groups") || !catalog.at("groups").is_array())
        throw std::runtime_error("GeoSpatial point catalog requires groups array");

    std::vector<GeoSpatialGroupDefinition> loadedGroups;
    for (size_t groupIndex = 0; groupIndex < catalog.at("groups").size(); ++groupIndex) {
        const nlohmann::json& sourceGroup = catalog.at("groups").at(groupIndex);
        const std::string groupPath = "groups[" + std::to_string(groupIndex) + "]";
        if (!sourceGroup.is_object())
            throw std::runtime_error("GeoSpatial point catalog " + groupPath + " must be an object");

        GeoSpatialGroupDefinition group;
        group.id = requireCatalogString(sourceGroup, "id", groupPath);
        group.name = requireCatalogString(sourceGroup, "name", groupPath);
        group.color = normalizeColor(requireCatalogString(sourceGroup, "color", groupPath));
        group.visible = requireCatalogBool(sourceGroup, "visible", groupPath);
        validateIdentifier(group.id, "group");
        validateName(group.name, "group");
        if (!sourceGroup.contains("points") || !sourceGroup.at("points").is_array())
            throw std::runtime_error("GeoSpatial point catalog requires array " + groupPath + ".points");
        if (!sourceGroup.contains("areas") || !sourceGroup.at("areas").is_array())
            throw std::runtime_error("GeoSpatial point catalog requires array " + groupPath + ".areas");
        if (entityIdExists(loadedGroups, group.id))
            throw std::runtime_error("GeoSpatial point catalog duplicate group id: " + group.id);

        for (size_t pointIndex = 0; pointIndex < sourceGroup.at("points").size(); ++pointIndex) {
            const nlohmann::json& sourcePoint = sourceGroup.at("points").at(pointIndex);
            const std::string pointPath = groupPath + ".points[" + std::to_string(pointIndex) + "]";
            if (!sourcePoint.is_object())
                throw std::runtime_error("GeoSpatial point catalog " + pointPath + " must be an object");
            GeoSpatialPointDefinition point;
            point.id = requireCatalogString(sourcePoint, "id", pointPath);
            point.name = requireCatalogString(sourcePoint, "name", pointPath);
            point.group_id = group.id;
            point.geometry_kind = parseGeometryKind(requireCatalogString(sourcePoint, "geometry_kind", pointPath));
            if (point.geometry_kind != GeoSpatialGeometryKind::Point)
                throw std::runtime_error("GeoSpatial point catalog " + pointPath + " geometry_kind must be 'point'");
            point.color = normalizeColor(requireCatalogString(sourcePoint, "color", pointPath));
            point.longitude_degrees = requireCatalogNumber(sourcePoint, "longitude_degrees", pointPath);
            point.latitude_degrees = requireCatalogNumber(sourcePoint, "latitude_degrees", pointPath);
            point.height_meters = requireCatalogNumber(sourcePoint, "height_meters", pointPath);
            point.visible = requireCatalogBool(sourcePoint, "visible", pointPath);
            validateIdentifier(point.id, "point");
            validateName(point.name, "point");
            validateCoordinates(point);
            if (entityIdExists(loadedGroups, point.id) ||
                std::any_of(group.points.begin(), group.points.end(),
                            [&](const GeoSpatialPointDefinition& candidate) { return candidate.id == point.id; })) {
                throw std::runtime_error("GeoSpatial point catalog duplicate point id: " + point.id);
            }
            group.points.push_back(std::move(point));
        }

        for (size_t areaIndex = 0; areaIndex < sourceGroup.at("areas").size(); ++areaIndex) {
            const nlohmann::json& sourceArea = sourceGroup.at("areas").at(areaIndex);
            const std::string areaPath = groupPath + ".areas[" + std::to_string(areaIndex) + "]";
            if (!sourceArea.is_object())
                throw std::runtime_error("GeoSpatial point catalog " + areaPath + " must be an object");
            GeoSpatialAreaDefinition area;
            area.id = requireCatalogString(sourceArea, "id", areaPath);
            area.name = requireCatalogString(sourceArea, "name", areaPath);
            area.group_id = group.id;
            area.geometry_kind = parseGeometryKind(requireCatalogString(sourceArea, "geometry_kind", areaPath));
            area.color = normalizeColor(requireCatalogString(sourceArea, "color", areaPath));
            area.longitude_degrees = requireCatalogNumber(sourceArea, "longitude_degrees", areaPath);
            area.latitude_degrees = requireCatalogNumber(sourceArea, "latitude_degrees", areaPath);
            area.height_meters = requireCatalogNumber(sourceArea, "height_meters", areaPath);
            area.size_meters = requireCatalogNumber(sourceArea, "size_meters", areaPath);
            area.opacity = static_cast<float>(requireCatalogNumber(sourceArea, "opacity", areaPath));
            area.visible = requireCatalogBool(sourceArea, "visible", areaPath);
            validateIdentifier(area.id, "area");
            validateName(area.name, "area");
            validateCoordinates(area);
            validateArea(area);
            const bool duplicateInCurrentGroup =
                std::any_of(group.points.begin(), group.points.end(),
                            [&](const GeoSpatialPointDefinition& candidate) { return candidate.id == area.id; }) ||
                std::any_of(group.areas.begin(), group.areas.end(),
                            [&](const GeoSpatialAreaDefinition& candidate) { return candidate.id == area.id; });
            if (entityIdExists(loadedGroups, area.id) || duplicateInCurrentGroup)
                throw std::runtime_error("GeoSpatial point catalog duplicate entity id: " + area.id);
            group.areas.push_back(std::move(area));
        }
        loadedGroups.push_back(std::move(group));
    }
    return loadedGroups;
}

void savePointCatalog(const std::filesystem::path& path,
                      const std::vector<GeoSpatialGroupDefinition>& groups)
{
    nlohmann::json catalog;
    catalog["version"] = kPointCatalogVersion;
    catalog["groups"] = nlohmann::json::array();
    for (const GeoSpatialGroupDefinition& group : groups) {
        nlohmann::json savedGroup = {
            {"id", group.id}, {"name", group.name}, {"color", group.color},
            {"visible", group.visible}, {"points", nlohmann::json::array()},
            {"areas", nlohmann::json::array()}
        };
        for (const GeoSpatialPointDefinition& point : group.points) {
            savedGroup["points"].push_back({
                {"id", point.id}, {"name", point.name},
                {"geometry_kind", geometryKindName(point.geometry_kind)}, {"color", point.color},
                {"longitude_degrees", point.longitude_degrees}, {"latitude_degrees", point.latitude_degrees},
                {"height_meters", point.height_meters}, {"visible", point.visible}
            });
        }
        for (const GeoSpatialAreaDefinition& area : group.areas) {
            savedGroup["areas"].push_back({
                {"id", area.id}, {"name", area.name},
                {"geometry_kind", geometryKindName(area.geometry_kind)}, {"color", area.color},
                {"longitude_degrees", area.longitude_degrees}, {"latitude_degrees", area.latitude_degrees},
                {"height_meters", area.height_meters}, {"size_meters", area.size_meters},
                {"opacity", area.opacity}, {"visible", area.visible}
            });
        }
        catalog["groups"].push_back(std::move(savedGroup));
    }

    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output.is_open())
        throw std::runtime_error("GeoSpatial point catalog cannot be opened for writing: " + path.string());
    output << catalog.dump(2) << '\n';
    output.flush();
    if (!output.good())
        throw std::runtime_error("GeoSpatial point catalog write failed: " + path.string());
}

std::string generatePointCatalogId(const std::vector<GeoSpatialGroupDefinition>& groups,
                                   std::string_view prefix)
{
    if (prefix.empty())
        throw std::runtime_error("GeoSpatial point catalog ID generator requires a prefix");
    for (uint64_t sequence = 1; sequence < std::numeric_limits<uint64_t>::max(); ++sequence) {
        std::ostringstream stream;
        stream << prefix << '-' << std::setw(6) << std::setfill('0') << sequence;
        const std::string candidate = stream.str();
        if (!entityIdExists(groups, candidate))
            return candidate;
    }
    throw std::runtime_error("GeoSpatial exhausted generated IDs for prefix '" + std::string(prefix) + "'");
}

} // namespace GRIM::GeoSpatial