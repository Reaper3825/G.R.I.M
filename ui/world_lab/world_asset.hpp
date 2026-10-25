#pragma once
#include "world_document.hpp"

namespace GRIM::WorldLab {
// A self-contained entity tree. External passages, goals and player state are
// deliberately not part of a reusable room or object.
struct Asset {
    std::string name, root;
    std::vector<Entity> entities;
    Document preview() const;
    nlohmann::json json() const;
    static Asset parse(const nlohmann::json&);
    static Asset capture(const Document&, const std::string& root, const std::string& name);
    static Asset create(const std::string& kind);
    std::string insert(Document&, const std::string& parent, float x, float y) const;
};
struct AssetEntry {
    std::filesystem::path path;
    std::string name, kind, searchText;
    size_t entityCount = 0;
    bool valid = false;
};
std::vector<AssetEntry> listAssets(const std::filesystem::path& directory);
// Indices ordered by textual relevance, with deterministic library-order ties.
std::vector<size_t> searchAssets(const std::vector<AssetEntry>&, const std::string& query, const std::string& kind);
std::filesystem::path newAssetPath(const std::filesystem::path& directory, const Asset&);
}
