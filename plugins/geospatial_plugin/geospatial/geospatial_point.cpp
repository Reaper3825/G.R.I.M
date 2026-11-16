#include "geospatial_runtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace GRIM::GeoSpatial {

void GeoSpatialRuntime::requestUpsertPoint(const std::string& originalId,
                                           const GeoSpatialPointDefinition& requestedPoint)
{
    std::lock_guard lock(mutex_);
    try {
        GeoSpatialPointDefinition point = requestedPoint;
        validateIdentifier(point.group_id, "point group");
        validateName(point.name, "point");
        point.color = normalizeColor(point.color);
        validateCoordinates(point);

        if (originalId.empty())
            point.id = generatePointCatalogIdLocked("point");
        else
            point.id = originalId;
        validateIdentifier(point.id, "point");

        auto targetGroup = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
            [&](const GeoSpatialGroupDefinition& group) { return group.id == point.group_id; });
        if (targetGroup == snapshot_.groups.end())
            throw std::runtime_error("GeoSpatial point group does not exist: " + point.group_id);

        GeoSpatialPointDefinition* existingPoint = nullptr;
        GeoSpatialGroupDefinition* existingGroup = nullptr;
        for (GeoSpatialGroupDefinition& group : snapshot_.groups) {
            for (GeoSpatialPointDefinition& candidate : group.points) {
                if (!originalId.empty() && candidate.id == originalId) {
                    existingPoint = &candidate;
                    existingGroup = &group;
                }
            }
        }

        if (originalId.empty()) {
            point.visible = true;
            targetGroup->points.push_back(point);
            markPointCatalogChangedLocked("Added point '" + point.name + "'");
            syncPointMarkersLocked();
            return;
        }
        if (!existingPoint || !existingGroup)
            throw std::runtime_error("GeoSpatial point does not exist: " + originalId);

        point.visible = existingPoint->visible;
        if (existingGroup == &*targetGroup) {
            *existingPoint = point;
        } else {
            auto pointIt = std::find_if(existingGroup->points.begin(), existingGroup->points.end(),
                [&](const GeoSpatialPointDefinition& candidate) { return candidate.id == originalId; });
            existingGroup->points.erase(pointIt);
            targetGroup->points.push_back(point);
        }
        markPointCatalogChangedLocked("Updated point '" + point.name + "'");
        syncPointMarkersLocked();
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Save point", error);
    }
}

void GeoSpatialRuntime::requestRemovePoint(const std::string& id)
{
    std::lock_guard lock(mutex_);
    try {
        for (GeoSpatialGroupDefinition& group : snapshot_.groups) {
            auto point = std::find_if(group.points.begin(), group.points.end(),
                [&](const GeoSpatialPointDefinition& candidate) { return candidate.id == id; });
            if (point == group.points.end())
                continue;
            group.points.erase(point);
            markPointCatalogChangedLocked("Removed point '" + id + "'");
            syncPointMarkersLocked();
            return;
        }
        throw std::runtime_error("GeoSpatial point does not exist: " + id);
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Remove point", error);
    }
}

void GeoSpatialRuntime::requestTogglePointVisibility(const std::string& id)
{
    std::lock_guard lock(mutex_);
    try {
        for (GeoSpatialGroupDefinition& group : snapshot_.groups) {
            auto point = std::find_if(group.points.begin(), group.points.end(),
                [&](const GeoSpatialPointDefinition& candidate) { return candidate.id == id; });
            if (point == group.points.end())
                continue;
            point->visible = !point->visible;
            if (point->visible)
                markPointCatalogChangedLocked("Showing point '" + point->name + "'");
            else
                markPointCatalogChangedLocked("Hiding point '" + point->name + "'");
            syncPointMarkersLocked();
            return;
        }
        throw std::runtime_error("GeoSpatial point does not exist: " + id);
    } catch (const std::exception& error) {
        setPointCatalogFailedStatusLocked("Toggle point visibility", error);
    }
}

} // namespace GRIM::GeoSpatial