#include "memory_storage.hpp"
#include "memory_router.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace GRIM {
bool MemoryStorage::recentlyModified(const std::string& key, int seconds) {
    return false; // or TODO placeholder
}

// =====================================
// Initialization / Shutdown
// =====================================


void MemoryStorage::initialize(const std::string& longTermPath) {
    std::scoped_lock lock(mtx);
    longTermFile = longTermPath;
    loadFromDisk();
}

void MemoryStorage::shutdown() {
    std::scoped_lock lock(mtx);
    saveToDisk();
    shortTerm.clear();
    longTerm.clear();
}

// =====================================
// Storage Functions
// =====================================

void MemoryStorage::storeShortTerm(const MemoryObject& obj) {
    std::scoped_lock lock(mtx);

    if (shortTerm.size() >= SHORT_TERM_MAX)
        shortTerm.pop_front();

    shortTerm.push_back(obj);
}

void MemoryStorage::storeLongTerm(const MemoryObject& obj) {
    std::scoped_lock lock(mtx);
    longTerm[obj.id] = obj;
    saveToDisk();
}

// =====================================
// Retrieval
// =====================================

std::optional<MemoryObject> MemoryStorage::getById(const std::string& id) {
    std::scoped_lock lock(mtx);

    if (auto it = longTerm.find(id); it != longTerm.end())
        return it->second;

    for (auto& m : shortTerm)
        if (m.id == id)
            return m;

    return std::nullopt;
}

std::vector<MemoryObject> MemoryStorage::search(const std::string& query, int maxResults) {
    std::scoped_lock lock(mtx);
    std::vector<MemoryObject> results;

    auto match = [&](const MemoryObject& obj) {
        return obj.raw.find(query) != std::string::npos ||
               obj.normalized.find(query) != std::string::npos;
    };

    // Search short-term first
    for (auto it = shortTerm.rbegin(); it != shortTerm.rend() && results.size() < maxResults; ++it)
        if (match(*it)) results.push_back(*it);

    // Then long-term
    for (auto& [id, obj] : longTerm)
        if (match(obj)) {
            results.push_back(obj);
            if (results.size() >= maxResults) break;
        }

    return results;
}

// =====================================
// Maintenance
// =====================================

void MemoryStorage::decay(float rate) {
    std::scoped_lock lock(mtx);

    for (auto& [id, obj] : longTerm) {
        float ageDays = static_cast<float>((std::time(nullptr) - obj.timestamp) / 86400.0);
        obj.confidence -= rate * ageDays;
        if (obj.confidence < 0.2f)
            obj.confidence = 0.2f; // floor for now
    }
}

void MemoryStorage::flush() {
    std::scoped_lock lock(mtx);
    saveToDisk();
}

// =====================================
// Internal Save/Load
// =====================================

void MemoryStorage::saveToDisk() {
    if (longTermFile.empty()) return;

    std::filesystem::create_directories(std::filesystem::path(longTermFile).parent_path());

    json j;
    for (auto& [id, obj] : longTerm)
        j["memories"].push_back(json::parse(obj.toJSON()));

    std::ofstream ofs(longTermFile, std::ios::trunc);
    if (!ofs.is_open()) return;
    ofs << j.dump(2);
}


void MemoryStorage::loadFromDisk() {
    if (longTermFile.empty()) return;

    std::ifstream ifs(longTermFile);
    if (!ifs.is_open()) return;

    json j;
    ifs >> j;

    if (!j.contains("memories")) return;

    for (auto& m : j["memories"]) {
        MemoryObject obj = MemoryObject::fromJSON(m.dump());
        longTerm[obj.id] = obj;
    }
}


std::time_t GRIM::MemoryStorage::getLastModified(const std::string& key) {
    extern nlohmann::json longTermMemory;
    if (!longTermMemory.contains(key)) return 0;
    if (longTermMemory[key].contains("timestamp"))
        return longTermMemory[key]["timestamp"].get<std::time_t>();
    return 0;
}

} // namespace GRIM


