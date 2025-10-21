#pragma once
#include <string>
#include <unordered_map>
#include <filesystem>
#include <mutex>
#include <nlohmann/json.hpp>

namespace Voice {
namespace TTSCache {

    // =========================================================
    // TTS Cache System
    // Reuses audio files for repeated phrases
    // Auto-cleanup for old temp files
    // =========================================================

    struct CacheEntry {
        std::string filePath;
        std::time_t lastUsed;
        bool isPermanent;  // true = system message, false = temp
        size_t useCount;
    };

    // Initialize cache system
    void init();
    
    // Shutdown and cleanup
    void shutdown();

    // Get cached audio file for text (or nullptr if not cached)
    std::string getCached(const std::string& text, const std::string& speaker, double speed);

    // Store audio file in cache
    void store(const std::string& text, const std::string& speaker, double speed, 
               const std::string& filePath, bool isPermanent = false);

    // Mark system messages for permanent caching
    void markPermanent(const std::string& text);

    // Cleanup old temp files (older than N hours)
    void cleanupOldFiles(int maxAgeHours = 24);

    // Get cache statistics
    nlohmann::json getStats();

} // namespace TTSCache
} // namespace Voice
