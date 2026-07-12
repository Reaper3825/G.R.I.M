#pragma once

#include "geospatial_geometry.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM::GeoSpatial {

std::filesystem::path pointCatalogPath();
std::vector<GeoSpatialGroupDefinition> loadPointCatalog(const std::filesystem::path& path);
void savePointCatalog(const std::filesystem::path& path,
                      const std::vector<GeoSpatialGroupDefinition>& groups);
std::string generatePointCatalogId(const std::vector<GeoSpatialGroupDefinition>& groups,
                                   std::string_view prefix);

} // namespace GRIM::GeoSpatial