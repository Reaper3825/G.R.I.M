#pragma once
#include <string>
#include <deque>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <vector>
#include "memory_manager.hpp"

namespace GRIM {

class MemoryStorage {
public:
    void initialize(const std::string& longTermPath);
    void shutdown();
    static bool recentlyModified(const std::string& key, int seconds);
    static std::time_t getLastModified(const std::string& key);
    void storeShortTerm(const MemoryObject& obj);
    void storeLongTerm(const MemoryObject& obj);

    std::optional<MemoryObject> getById(const std::string& id);
    std::vector<MemoryObject> search(const std::string& query, int maxResults = 10);
    std::optional<MemoryObject> findLearnedCommand(const std::string& phrase);
    void storeLearnedCommand(const std::string& phrase,
                             const std::string& action,
                             float confidence = 1.0f);
                             
    void decay(float rate);
    void flush();

private:
    void saveToDisk();
    void loadFromDisk();

    static constexpr size_t SHORT_TERM_MAX = 50;

    std::deque<MemoryObject> shortTerm;
    std::unordered_map<std::string, MemoryObject> longTerm;

    std::string longTermFile;
    std::mutex mtx;
};

} // namespace GRIM
