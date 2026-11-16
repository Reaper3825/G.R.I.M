#include "geospatial_runtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace GRIM::GeoSpatial {

void GeoSpatialRuntime::requestUpsertArea(const std::string& originalId,
                                          const GeoSpatialAreaDefinition& requestedArea)
{
    std::lock_guard lock(mutex_);
    try {
        GeoSpatialAreaDefinition area = requestedArea;
        validateIdentifier(area.group_id, "area group");
        validateName(area.name, "area");
        area.color = normalizeColor(area.color);
        validateCoordinates(area);
        validateArea(area);

        if (originalId.empty())
            area.id = generatePointCatalogIdLocked("area");
        else
            area.id = originalId;
        validateIdentifier(area.id, "area");

        auto targetGroup = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
            [&](const GeoSpatialGroupDefinition& group) { return group.id == area.group_id; });
        if (targetGroup == snapshot_.groups.end())
            throw std::runtime_error("GeoSpatial area group does not exist: " + area.group_id);

        GeoSpatialAreaDefinition* existingArea = nullptr;
        GeoSpatialGroupDefinition* existingGroup = nullptr;
        for (GeoSpatialGroupDefinition& group : snapshot_.groups) {
            for (GeoSpatialAreaDefinition& candidate : group.areas) {
                if (!originalId.empty() && candidate.id == originalId) {
                    existingArea = &candidate;
                    existingGroup = &group;
                }
            }
        }

        if (originalId.empty()) {
            area.visible = true;
            targetGroup->areas.push_back(area);
            markPointCatalogChangedLocked("Added area '" + area.name + "'");
            syncPointMarkersLocked();
            return;
        }
        if (!existingArea || !existingGroup)
            throw std::runtime_error("GeoSpatial area does not exist: " + originalId);

        area.visible = existingArea->visible;
        if (existingGroup == &*targetGroup) {
            *existingArea = area;
        } else {
            auto areaIt = std::find_if(existingGroup->areas.begin(), existingGroup->areas.end(),
                [&](const GeoSpatialAreaDefinition& candidate) { return candidate.id == originalId; });
            existingGroup->areas.erase(areaIt);
            targetGroup->areas.push_back(area);
        }
        markPointCatalogChangedLocked("Updated area '" + area.name + "'");
        syncPointMarkersLocked();
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Save area", error);
    }
}

void GeoSpatialRuntime::requestRemoveArea(const std::string& id)
{
    std::lock_guard lock(mutex_);
    try {
        for (GeoSpatialGroupDefinition& group : snapshot_.groups) {
            auto area = std::find_if(group.areas.begin(), group.areas.end(),
                [&](const GeoSpatialAreaDefinition& candidate) { return candidate.id == id; });
            if (area == group.areas.end())
                continue;
            group.areas.erase(area);
            markPointCatalogChangedLocked("Removed area '" + id + "'");
            syncPointMarkersLocked();
            return;
        }
        throw std::runtime_error("GeoSpatial area does not exist: " + id);
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Remove area", error);
    }
}

void GeoSpatialRuntime::requestToggleAreaVisibility(const std::string& id)
{
    std::lock_guard lock(mutex_);
    try {
        for (GeoSpatialGroupDefinition& group : snapshot_.groups) {
            auto area = std::find_if(group.areas.begin(), group.areas.end(),
                [&](const GeoSpatialAreaDefinition& candidate) { return candidate.id == id; });
            if (area == group.areas.end())
                continue;
            area->visible = !area->visible;
            if (area->visible)
                markPointCatalogChangedLocked("Showing area '" + area->name + "'");
            else
                markPointCatalogChangedLocked("Hiding area '" + area->name + "'");
            syncPointMarkersLocked();
            return;
        }
        throw std::runtime_error("GeoSpatial area does not exist: " + id);
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Toggle area visibility", error);
    }
}

} // namespace GRIM::GeoSpatial