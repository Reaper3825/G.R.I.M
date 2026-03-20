#include "dataset_io_json.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace GRIM {
namespace Pipeline {

using json = nlohmann::json;

static TaggedEntry parseTaggedEntry(const json& j) {
    TaggedEntry e;
    e.id               = j.value("id", std::string());
    e.content          = j.value("content", std::string());
    e.sourceUrl        = j.value("source_url", std::string());
    e.sourceType       = j.value("source_type", std::string());
    e.qualityTier      = j.value("quality_tier", std::string());
    e.subject          = j.value("subject", std::string());
    e.reliabilityScore = j.value("reliability_score", 0.0f);
    e.timestamp        = j.value("timestamp", int64_t(0));
    e.verified         = j.value("verified", false);
    if (j.contains("tags") && j["tags"].is_array()) {
        for (const auto& t : j["tags"]) {
            if (t.is_string()) e.tags.push_back(t.get<std::string>());
        }
    }
    return e;
}

static json taggedEntryToJson(const TaggedEntry& e) {
    json j;
    j["id"]                = e.id;
    j["content"]           = e.content;
    j["source_url"]        = e.sourceUrl;
    j["source_type"]       = e.sourceType;
    j["quality_tier"]      = e.qualityTier;
    j["subject"]           = e.subject;
    j["reliability_score"] = e.reliabilityScore;
    j["timestamp"]         = e.timestamp;
    j["verified"]          = e.verified;
    j["tags"]              = e.tags;
    return j;
}

// ─── DatasetIOJson ──────────────────────────────────────

bool DatasetIOJson::loadAllEntries(const fs::path& datasetPath,
                                   std::vector<TaggedEntry>& entries) {
    entries.clear();
    std::ifstream file(datasetPath);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            entries.push_back(parseTaggedEntry(json::parse(line)));
        } catch (...) { continue; }
    }
    return true;
}

bool DatasetIOJson::saveAllEntries(const fs::path& datasetPath,
                                   const std::vector<TaggedEntry>& entries) {
    std::error_code ec;
    fs::create_directories(datasetPath.parent_path(), ec);

    // Atomic write via temp file
    fs::path tmpPath = datasetPath;
    tmpPath += ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        if (!out.is_open()) return false;
        for (const auto& e : entries) {
            out << taggedEntryToJson(e).dump() << "\n";
        }
        if (!out.good()) return false;
    }
    fs::rename(tmpPath, datasetPath, ec);
    return !ec;
}

bool DatasetIOJson::appendEntries(const fs::path& datasetPath,
                                  const std::vector<TaggedEntry>& entries) {
    if (entries.empty()) return true;

    std::error_code ec;
    fs::create_directories(datasetPath.parent_path(), ec);

    std::ofstream out(datasetPath, std::ios::app);
    if (!out.is_open()) return false;
    for (const auto& e : entries) {
        out << taggedEntryToJson(e).dump() << "\n";
    }
    return out.good();
}

size_t DatasetIOJson::countEntries(const fs::path& datasetPath) {
    std::ifstream file(datasetPath);
    if (!file.is_open()) return 0;
    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) count++;
    }
    return count;
}

bool DatasetIOJson::loadAssignments(const fs::path& path, std::vector<std::string>& ids) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    try {
        json j = json::parse(file);
        ids.clear();
        if (j.contains("assigned") && j["assigned"].is_array()) {
            for (const auto& id : j["assigned"]) {
                if (id.is_string()) ids.push_back(id.get<std::string>());
            }
        }
        return true;
    } catch (...) { return false; }
}

bool DatasetIOJson::saveAssignments(const fs::path& path, const std::vector<std::string>& ids) {
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    json j;
    j["assigned"] = ids;
    std::ofstream file(path, std::ios::trunc);
    if (!file.is_open()) return false;
    file << j.dump(2) << "\n";
    return file.good();
}

} // namespace Pipeline
} // namespace GRIM
