#include "storage_index.hpp"
#include "memory/atomic_writer.hpp"
#include "logger.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <nlohmann/json.hpp>

namespace GRIM {

static constexpr const char* TAG = "StorageIndex";
static constexpr const char* INDEX_FILENAME = ".storage_index.json";

// ─── Path security ───────────────────────────────────────

void StorageIndex::validatePath(const std::string& path) {
    if (path.empty()) return; // root listing

    // Reject absolute paths
    if (path.front() == '/' || path.front() == '\\')
        throw std::runtime_error("StorageIndex: absolute paths rejected: " + path);
#ifdef _WIN32
    if (path.size() >= 2 && path[1] == ':')
        throw std::runtime_error("StorageIndex: absolute paths rejected: " + path);
#endif

    // Reject null bytes
    if (path.find('\0') != std::string::npos)
        throw std::runtime_error("StorageIndex: null bytes in path rejected");

    // Reject path traversal
    if (path.find("..") != std::string::npos)
        throw std::runtime_error("StorageIndex: path traversal '..' rejected: " + path);
}

// ─── Construction ────────────────────────────────────────

StorageIndex::StorageIndex(const std::string& storage_root) {
    std::filesystem::path root(storage_root);
    std::filesystem::create_directories(root);
    index_path_ = (root / INDEX_FILENAME).string();
    load();
    LOG_DEBUG(TAG, "Initialized with " + std::to_string(entries_.size()) + " entries");
}

// ─── CRUD ────────────────────────────────────────────────

void StorageIndex::addEntry(StorageEntry entry) {
    validatePath(entry.relative_path);

    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto& e : entries_) {
        if (e.file_id == entry.file_id)
            throw std::runtime_error("StorageIndex: duplicate file_id '" + entry.file_id + "'");
        if (e.relative_path == entry.relative_path)
            throw std::runtime_error("StorageIndex: duplicate path '" + entry.relative_path + "'");
    }

    entries_.push_back(std::move(entry));
    persist();
}

void StorageIndex::removeEntry(const std::string& file_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = std::find_if(entries_.begin(), entries_.end(),
                           [&](const StorageEntry& e) { return e.file_id == file_id; });
    if (it == entries_.end())
        throw std::runtime_error("StorageIndex: file_id '" + file_id + "' not found");

    entries_.erase(it);
    persist();
}

void StorageIndex::updateEntry(const StorageEntry& entry) {
    validatePath(entry.relative_path);

    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& e : entries_) {
        if (e.file_id == entry.file_id) {
            e = entry;
            persist();
            return;
        }
    }
    throw std::runtime_error("StorageIndex::updateEntry: file_id '" + entry.file_id + "' not found");
}

// ─── Lookups ─────────────────────────────────────────────

const StorageEntry* StorageIndex::findById(const std::string& file_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : entries_) {
        if (e.file_id == file_id) return &e;
    }
    return nullptr;
}

const StorageEntry* StorageIndex::findByPath(const std::string& relative_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& e : entries_) {
        if (e.relative_path == relative_path) return &e;
    }
    return nullptr;
}

// ─── Directory listing ───────────────────────────────────

std::vector<DirectoryEntry> StorageIndex::listDirectory(const std::string& dir_path) const {
    validatePath(dir_path);

    std::lock_guard<std::mutex> lock(mutex_);

    std::string prefix = dir_path;
    if (!prefix.empty() && prefix.back() != '/')
        prefix += '/';

    std::set<std::string> seen_dirs;
    std::vector<DirectoryEntry> result;

    for (const auto& e : entries_) {
        // Check if this entry is under dir_path
        const std::string& path = e.relative_path;

        if (prefix.empty()) {
            // Root listing
            auto slash = path.find('/');
            if (slash == std::string::npos) {
                // Direct file in root
                DirectoryEntry de;
                de.name             = path;
                de.is_directory     = false;
                de.size_bytes       = e.size_bytes;
                de.modified_at      = e.modified_at;
                de.file_id          = e.file_id;
                de.source_device_id = e.source_device_id;
                result.push_back(de);
            } else {
                // Subdirectory
                std::string dirname = path.substr(0, slash);
                if (seen_dirs.insert(dirname).second) {
                    DirectoryEntry de;
                    de.name         = dirname;
                    de.is_directory = true;
                    result.push_back(de);
                }
            }
        } else if (path.size() > prefix.size() &&
                   path.compare(0, prefix.size(), prefix) == 0) {
            // Under this directory
            std::string remainder = path.substr(prefix.size());
            auto slash = remainder.find('/');
            if (slash == std::string::npos) {
                // Direct child file
                DirectoryEntry de;
                de.name             = remainder;
                de.is_directory     = false;
                de.size_bytes       = e.size_bytes;
                de.modified_at      = e.modified_at;
                de.file_id          = e.file_id;
                de.source_device_id = e.source_device_id;
                result.push_back(de);
            } else {
                // Subdirectory
                std::string dirname = remainder.substr(0, slash);
                if (seen_dirs.insert(dirname).second) {
                    DirectoryEntry de;
                    de.name         = dirname;
                    de.is_directory = true;
                    result.push_back(de);
                }
            }
        }
    }

    // Sort: directories first, then by name
    std::sort(result.begin(), result.end(),
              [](const DirectoryEntry& a, const DirectoryEntry& b) {
                  if (a.is_directory != b.is_directory) return a.is_directory > b.is_directory;
                  return a.name < b.name;
              });

    return result;
}

// ─── Search ──────────────────────────────────────────────

std::vector<StorageEntry> StorageIndex::search(const std::string& query) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<StorageEntry> result;
    for (const auto& e : entries_) {
        if (e.relative_path.find(query) != std::string::npos)
            result.push_back(e);
    }
    return result;
}

std::vector<StorageEntry> StorageIndex::listAll() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_;
}

size_t StorageIndex::entryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

// ─── Persistence ─────────────────────────────────────────

void StorageIndex::load() {
    if (!std::filesystem::exists(index_path_)) return;

    std::ifstream in(index_path_);
    if (!in)
        throw std::runtime_error("StorageIndex: cannot open " + index_path_);

    nlohmann::json j;
    in >> j;

    if (!j.is_array())
        throw std::runtime_error("StorageIndex: expected JSON array in " + index_path_);

    entries_.clear();
    for (const auto& entry : j) {
        entries_.push_back(entry.get<StorageEntry>());
    }
}

void StorageIndex::persist() const {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& e : entries_) {
        j.push_back(e);
    }
    AtomicWriter::writeString(index_path_, j.dump(2));
}

} // namespace GRIM
