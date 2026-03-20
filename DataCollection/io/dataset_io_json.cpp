#include "dataset_io_json.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace GRIM {
namespace Pipeline {

using json = nlohmann::json;

static TaggedEntry parseTaggedEntry(const json& j) {
    TaggedEntry e;
    e.id              = j.value("id", std::string());
    e.content         = j.value("content", std::string());
    e.sourceUrl       = j.value("source_url", std::string());
    e.sourceType      = j.value("source_type", std::string());
    e.qualityTier     = j.value("quality_tier", std::string());
    e.subject         = j.value("subject", std::string());
    e.reliabilityScore = j.value("reliability_score", 0.0f);
    e.timestamp       = j.value("timestamp", int64_t(0));
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
    j["tags"]              = e.tags;
    return j;
}

// ─── DatasetIOJson ──────────────────────────────────────

bool DatasetIOJson::iterateShard(const fs::path& shardPath,
                                 std::function<bool(const TaggedEntry&)> visitor) {
    std::ifstream file(shardPath);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            json j = json::parse(line);
            TaggedEntry entry = parseTaggedEntry(j);
            if (!visitor(entry)) break;
        } catch (...) {
            continue;
        }
    }
    return true;
}

size_t DatasetIOJson::countEntries(const fs::path& manifestPath) {
    std::ifstream mf(manifestPath);
    if (!mf.is_open()) return 0;

    try {
        json manifest = json::parse(mf);
        return manifest.value("total_entries", size_t(0));
    } catch (...) {
        return 0;
    }
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
    } catch (...) {
        return false;
    }
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

// ─── AppendOnlyDatasetWriterJson ────────────────────────

bool AppendOnlyDatasetWriterJson::beginRun(const PipelineRunLayout& run) {
    shardCounter_ = 0;
    shardsDir_ = run.outputRoot / "shards";
    std::error_code ec;
    fs::create_directories(shardsDir_, ec);
    return !ec;
}

bool AppendOnlyDatasetWriterJson::appendTaggedChunk(const EntryChunk<TaggedEntry>& chunk,
                                                     fs::path& writtenShardPath) {
    if (chunk.items.empty()) return true;

    std::ostringstream name;
    name << "shard_" << std::setw(6) << std::setfill('0') << shardCounter_++ << ".jsonl";
    writtenShardPath = shardsDir_ / name.str();

    std::ofstream file(writtenShardPath, std::ios::trunc);
    if (!file.is_open()) return false;

    for (const auto& entry : chunk.items) {
        file << taggedEntryToJson(entry).dump() << "\n";
    }
    return file.good();
}

bool AppendOnlyDatasetWriterJson::commitRunManifest(const PipelineRunLayout& run,
                                                     const std::vector<fs::path>& newShards) {
    json manifest;
    size_t totalEntries = 0;

    // Load existing manifest if present
    {
        std::ifstream existing(run.manifestPath);
        if (existing.is_open()) {
            try {
                manifest = json::parse(existing);
                totalEntries = manifest.value("total_entries", size_t(0));
            } catch (...) {
                manifest = json::object();
            }
        }
    }

    if (!manifest.contains("schema_version")) {
        manifest["schema_version"] = 1;
    }
    if (!manifest.contains("shards") || !manifest["shards"].is_array()) {
        manifest["shards"] = json::array();
    }

    for (const auto& shardPath : newShards) {
        size_t count = 0;
        std::ifstream sf(shardPath);
        if (sf.is_open()) {
            std::string line;
            while (std::getline(sf, line)) {
                if (!line.empty()) count++;
            }
        }
        totalEntries += count;

        json shardRecord;
        shardRecord["file"]    = shardPath.filename().string();
        shardRecord["entries"] = count;
        shardRecord["run_id"]  = run.runId;
        manifest["shards"].push_back(shardRecord);
    }

    manifest["total_entries"] = totalEntries;

    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    manifest["last_updated"] = epoch;

    // Atomic write via temp file
    fs::path tmpPath = run.manifestPath;
    tmpPath += ".tmp";
    {
        std::error_code ec;
        fs::create_directories(run.manifestPath.parent_path(), ec);
        std::ofstream out(tmpPath, std::ios::trunc);
        if (!out.is_open()) return false;
        out << manifest.dump(2) << "\n";
        if (!out.good()) return false;
    }
    std::error_code ec;
    fs::rename(tmpPath, run.manifestPath, ec);
    return !ec;
}

bool AppendOnlyDatasetWriterJson::abortRun(const PipelineRunLayout& run) {
    std::error_code ec;
    if (fs::exists(shardsDir_, ec)) {
        // Remove only shards written in this run
        for (const auto& entry : fs::directory_iterator(shardsDir_, ec)) {
            if (entry.is_regular_file()) {
                fs::remove(entry.path(), ec);
            }
        }
    }
    return true;
}

} // namespace Pipeline
} // namespace GRIM
