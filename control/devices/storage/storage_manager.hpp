#pragma once

#include "storage_index.hpp"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace GRIM {

// ─────────────────────────────────────────────────────────
//  StorageManager — file operations on shared storage
//
//  Root: ~/.grim/shared_storage/
//  All paths are relative to root.  Path traversal is
//  rejected at the boundary (OWASP).
// ─────────────────────────────────────────────────────────

class StorageManager {
public:
    explicit StorageManager(const std::string& storage_root);

    // File operations
    std::string storeFile(const std::string& relative_path,
                          const std::vector<uint8_t>& data,
                          const std::string& source_device_id);

    std::filesystem::path retrieveFile(const std::string& file_id) const;
    void deleteFile(const std::string& file_id);
    void createDirectory(const std::string& relative_path);

    // Hash
    static std::string calculateSHA256(const std::filesystem::path& path);

    // Stats
    uint64_t getDiskUsage() const;

    // Index access
    StorageIndex& index() { return index_; }
    const StorageIndex& index() const { return index_; }

    const std::string& storageRoot() const { return storage_root_; }

private:
    static void sanitizePath(const std::string& path);
    static std::string inferContentType(const std::string& filename);
    static std::string generateUUID();
    static std::string nowISO8601();

    std::string storage_root_;
    StorageIndex index_;
};

} // namespace GRIM
