#include "geospatial_runtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace GRIM::GeoSpatial {

void GeoSpatialRuntime::requestUpsertGroup(const std::string& originalId,
                                           const std::string& name,
                                           const std::string& color)
{
    std::lock_guard lock(mutex_);
    try {
        validateName(name, "group");
        const std::string normalizedColor = normalizeColor(color);

        if (originalId.empty()) {
            const std::string id = generatePointCatalogIdLocked("group");
            snapshot_.groups.push_back({id, name, normalizedColor, true, {}});
            markPointCatalogChangedLocked("Added group '" + name + "'");
            return;
        }

        auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
            [&](const GeoSpatialGroupDefinition& candidate) { return candidate.id == originalId; });
        if (group == snapshot_.groups.end())
            throw std::runtime_error("GeoSpatial group does not exist: " + originalId);

        group->name = name;
        group->color = normalizedColor;
        markPointCatalogChangedLocked("Updated group '" + name + "'");
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Save group", error);
    }
}

void GeoSpatialRuntime::requestRemoveGroup(const std::string& id)
{
    std::lock_guard lock(mutex_);
    try {
        auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
            [&](const GeoSpatialGroupDefinition& candidate) { return candidate.id == id; });
        if (group == snapshot_.groups.end())
            throw std::runtime_error("GeoSpatial group does not exist: " + id);
        const size_t removedEntityCount = group->points.size() + group->areas.size();
        snapshot_.groups.erase(group);
        markPointCatalogChangedLocked("Removed group '" + id + "' and " + std::to_string(removedEntityCount) + " object(s)");
        syncPointMarkersLocked();
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Remove group", error);
    }
}

void GeoSpatialRuntime::requestToggleGroupVisibility(const std::string& id)
{
    std::lock_guard lock(mutex_);
    try {
        auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
            [&](const GeoSpatialGroupDefinition& candidate) { return candidate.id == id; });
        if (group == snapshot_.groups.end())
            throw std::runtime_error("GeoSpatial group does not exist: " + id);
        group->visible = !group->visible;
        if (group->visible)
            markPointCatalogChangedLocked("Showing group '" + group->name + "'");
        else
            markPointCatalogChangedLocked("Hiding group '" + group->name + "'");
        syncPointMarkersLocked();
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Toggle group visibility", error);
    }
}

} // namespace GRIM::GeoSpatial