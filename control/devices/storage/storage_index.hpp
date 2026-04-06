#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  StorageEntry — metadata for one file in shared storage
// ─────────────────────────────────────────────────────────

struct StorageEntry {
    std::string file_id;            // UUID
    std::string relative_path;      // e.g. "documents/readme.txt"
    uint64_t    size_bytes = 0;
    std::string sha256_hash;
    std::string content_type;       // inferred from extension
    std::string source_device_id;   // which device uploaded it
    std::string created_at;         // ISO 8601
    std::string modified_at;        // ISO 8601
};

inline void to_json(nlohmann::json& j, const StorageEntry& e) {
    j = {
        { "file_id",          e.file_id          },
        { "relative_path",    e.relative_path    },
        { "size_bytes",       e.size_bytes       },
        { "sha256_hash",      e.sha256_hash      },
        { "content_type",     e.content_type     },
        { "source_device_id", e.source_device_id },
        { "created_at",       e.created_at       },
        { "modified_at",      e.modified_at      }
    };
}

inline void from_json(const nlohmann::json& j, StorageEntry& e) {
    auto require = [&](const char* field) {
        if (!j.contains(field))
            throw std::runtime_error(
                std::string("StorageEntry: missing required field '") + field + "'");
    };
    require("file_id");
    require("relative_path");
    require("size_bytes");
    require("sha256_hash");
    require("source_device_id");
    require("created_at");
    require("modified_at");

    j.at("file_id").get_to(e.file_id);
    j.at("relative_path").get_to(e.relative_path);
    j.at("size_bytes").get_to(e.size_bytes);
    j.at("sha256_hash").get_to(e.sha256_hash);
    j.at("content_type").get_to(e.content_type);
    j.at("source_device_id").get_to(e.source_device_id);
    j.at("created_at").get_to(e.created_at);
    j.at("modified_at").get_to(e.modified_at);
}

// ─────────────────────────────────────────────────────────
//  DirectoryEntry — virtual folder derived from paths
// ─────────────────────────────────────────────────────────

struct DirectoryEntry {
    std::string name;
    bool        is_directory;
    uint64_t    size_bytes = 0;
    std::string modified_at;
    std::string file_id;            // empty for directories
    std::string source_device_id;   // empty for directories
};

// ─────────────────────────────────────────────────────────
//  StorageIndex — file metadata catalog
//
//  Thread-safe.  Persisted to .storage_index.json via AtomicWriter.
// ─────────────────────────────────────────────────────────

class StorageIndex {
public:
    explicit StorageIndex(const std::string& storage_root);

    // CRUD
    void addEntry(StorageEntry entry);
    void removeEntry(const std::string& file_id);
    void updateEntry(const StorageEntry& entry);

    // Lookups
    const StorageEntry* findById(const std::string& file_id) const;
    const StorageEntry* findByPath(const std::string& relative_path) const;

    // Directory listing — returns files and virtual subdirectories at given path
    std::vector<DirectoryEntry> listDirectory(const std::string& dir_path) const;

    // Search by substring match on relative_path
    std::vector<StorageEntry> search(const std::string& query) const;

    // All entries
    std::vector<StorageEntry> listAll() const;

    size_t entryCount() const;

private:
    void load();
    void persist() const;
    static void validatePath(const std::string& path);

    std::string index_path_;
    mutable std::mutex mutex_;
    std::vector<StorageEntry> entries_;
};

} // namespace GRIM
