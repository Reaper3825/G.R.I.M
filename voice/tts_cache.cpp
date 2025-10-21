#include "tts_cache.hpp"
#include "logger.hpp"
#include <fstream>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace Voice {
namespace TTSCache {

    // =========================================================
    // State
    // =========================================================
    static std::unordered_map<std::string, CacheEntry> g_cache;
    static std::mutex g_cacheMutex;
    static fs::path g_cacheDir;
    static fs::path g_tempDir;
    static fs::path g_indexFile;

    // System messages that should be permanently cached
    static const std::vector<std::string> PERMANENT_PHRASES = {
        "Was that what you wanted?",
        "Got it — I'll keep doing that.",
        "Understood — I'll adjust next time.",
        "Opening notepad",
        "Opening",
        "Launching",
        "I processed",
        "commands. Was that correct?",
        "Got it — I'll remember that next time.",
        "I couldn't save that one — try again later.",
        "Sorry, I couldn't interpret that."
    };

    // =========================================================
    // Hash function for cache key
    // =========================================================
    static std::string getCacheKey(const std::string& text, const std::string& speaker, double speed) {
        // Simple hash: normalize text and combine with params
        std::string normalized = text;
        
        // ? FIX: Cast to unsigned char before calling tolower to handle non-ASCII
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), 
            [](char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); });
        
        // ? FIX: Cast to unsigned char before calling isspace to handle non-ASCII
        normalized.erase(std::unique(normalized.begin(), normalized.end(),
            [](char a, char b) { 
                return std::isspace(static_cast<unsigned char>(a)) && 
                       std::isspace(static_cast<unsigned char>(b)); 
            }), normalized.end());
        
        return normalized + "_" + speaker + "_" + std::to_string(static_cast<int>(speed * 100));
    }

    // =========================================================
    // Load cache index from disk
    // =========================================================
    static void loadIndex() {
        if (!fs::exists(g_indexFile)) {
            LOG_DEBUG("TTSCache", "No existing cache index found");
            return;
        }

        try {
            std::ifstream in(g_indexFile);
            json j;
            in >> j;

            for (auto& [key, entry] : j.items()) {
                CacheEntry ce;
                ce.filePath = entry["file"].get<std::string>();
                ce.lastUsed = entry.value("lastUsed", static_cast<std::time_t>(0));
                ce.isPermanent = entry.value("permanent", false);
                ce.useCount = entry.value("useCount", 0);

                // Verify file still exists
                if (fs::exists(ce.filePath)) {
                    g_cache[key] = ce;
                } else {
                    LOG_DEBUG("TTSCache", "Cached file missing, skipping: " + ce.filePath);
                }
            }

            LOG_DEBUG("TTSCache", "Loaded " + std::to_string(g_cache.size()) + " cached entries");
        } catch (const std::exception& e) {
            LOG_ERROR("TTSCache", std::string("Failed to load cache index: ") + e.what());
        }
    }

    // =========================================================
    // Save cache index to disk
    // =========================================================
    static void saveIndex() {
        try {
            json j;
            for (const auto& [key, entry] : g_cache) {
                j[key] = {
                    {"file", entry.filePath},
                    {"lastUsed", entry.lastUsed},
                    {"permanent", entry.isPermanent},
                    {"useCount", entry.useCount}
                };
            }

            std::ofstream out(g_indexFile);
            out << j.dump(2);
            LOG_DEBUG("TTSCache", "Saved cache index with " + std::to_string(g_cache.size()) + " entries");
        } catch (const std::exception& e) {
            LOG_ERROR("TTSCache", std::string("Failed to save cache index: ") + e.what());
        }
    }

    // =========================================================
    // Public API
    // =========================================================
    void init() {
        g_cacheDir = fs::path("D:/G.R.I.M/resources/tts_out/cache");
        g_tempDir = fs::path("D:/G.R.I.M/resources/tts_out/temp");
        g_indexFile = fs::path("D:/G.R.I.M/resources/tts_out/cache_index.json");

        fs::create_directories(g_cacheDir);
        fs::create_directories(g_tempDir);

        loadIndex();

        // Auto-cleanup on init
        cleanupOldFiles(24);

        LOG_PHASE("TTS Cache initialized", true);
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        saveIndex();
        LOG_PHASE("TTS Cache shutdown", true);
    }

    std::string getCached(const std::string& text, const std::string& speaker, double speed) {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        std::string key = getCacheKey(text, speaker, speed);
        auto it = g_cache.find(key);
        
        if (it != g_cache.end()) {
            // Update usage stats
            it->second.lastUsed = std::time(nullptr);
            it->second.useCount++;
            
            LOG_DEBUG("TTSCache", "Cache HIT for: " + text.substr(0, 30) + "...");
            return it->second.filePath;
        }
        
        LOG_DEBUG("TTSCache", "Cache MISS for: " + text.substr(0, 30) + "...");
        return "";
    }

    void store(const std::string& text, const std::string& speaker, double speed,
               const std::string& filePath, bool isPermanent) {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        std::string key = getCacheKey(text, speaker, speed);
        
        // Check if this phrase should be permanent
        bool shouldBePermanent = isPermanent;
        for (const auto& phrase : PERMANENT_PHRASES) {
            if (text.find(phrase) != std::string::npos) {
                shouldBePermanent = true;
                break;
            }
        }

        // Determine target directory and path
        fs::path sourceFile = fs::path(filePath);
        fs::path targetDir = shouldBePermanent ? g_cacheDir : g_tempDir;
        fs::path targetPath = targetDir / sourceFile.filename();

        try {
            // Make sure source file exists
            if (!fs::exists(sourceFile)) {
                LOG_ERROR("TTSCache", "Source file doesn't exist: " + sourceFile.string());
                return;
            }

            // If file needs to be moved to a different directory
            if (sourceFile.parent_path() != targetDir) {
                // Use copy + remove instead of rename for cross-directory moves
                fs::copy_file(sourceFile, targetPath, fs::copy_options::overwrite_existing);
                fs::remove(sourceFile);
                LOG_DEBUG("TTSCache", "Moved file: " + sourceFile.string() + " ? " + targetPath.string());
            } else {
                targetPath = sourceFile; // Already in the right place
            }

            CacheEntry entry;
            entry.filePath = targetPath.string();
            entry.lastUsed = std::time(nullptr);
            entry.isPermanent = shouldBePermanent;
            entry.useCount = 1;

            g_cache[key] = entry;

            LOG_DEBUG("TTSCache", std::string("Stored ") + (shouldBePermanent ? "PERMANENT" : "TEMP") + 
                      " cache: " + text.substr(0, 30) + "...");
            
            // Save index after adding entry
            saveIndex();
        } catch (const std::exception& e) {
            LOG_ERROR("TTSCache", std::string("Failed to store cache entry: ") + e.what());
        }
    }

    void markPermanent(const std::string& text) {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        // Find all cache entries containing this text
        for (auto& [key, entry] : g_cache) {
            if (key.find(text) != std::string::npos && !entry.isPermanent) {
                entry.isPermanent = true;
                
                // Move file from temp to cache
                fs::path oldPath = fs::path(entry.filePath);
                fs::path newPath = g_cacheDir / oldPath.filename();
                
                try {
                    if (fs::exists(oldPath)) {
                        // Use copy + remove for reliable cross-directory move
                        fs::copy_file(oldPath, newPath, fs::copy_options::overwrite_existing);
                        fs::remove(oldPath);
                        entry.filePath = newPath.string();
                        LOG_DEBUG("TTSCache", "Marked as permanent: " + text);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("TTSCache", std::string("Failed to mark permanent: ") + e.what());
                }
            }
        }
        
        saveIndex();
    }

    void cleanupOldFiles(int maxAgeHours) {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        auto now = std::time(nullptr);
        int maxAgeSeconds = maxAgeHours * 3600;
        int deletedCount = 0;

        // Clean up temp directory
        std::vector<std::string> keysToRemove;
        
        for (auto& [key, entry] : g_cache) {
            if (!entry.isPermanent) {
                auto age = now - entry.lastUsed;
                if (age > maxAgeSeconds) {
                    try {
                        if (fs::exists(entry.filePath)) {
                            fs::remove(entry.filePath);
                            deletedCount++;
                        }
                        keysToRemove.push_back(key);
                    } catch (const std::exception& e) {
                        LOG_ERROR("TTSCache", std::string("Failed to delete old file: ") + e.what());
                    }
                }
            }
        }

        // Remove deleted entries from cache
        for (const auto& key : keysToRemove) {
            g_cache.erase(key);
        }

        if (deletedCount > 0) {
            LOG_DEBUG("TTSCache", "Cleaned up " + std::to_string(deletedCount) + " old temp files");
            saveIndex();
        }
    }

    json getStats() {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        size_t permanentCount = 0;
        size_t tempCount = 0;
        size_t totalSize = 0;

        for (const auto& [key, entry] : g_cache) {
            if (entry.isPermanent) {
                permanentCount++;
            } else {
                tempCount++;
            }

            try {
                if (fs::exists(entry.filePath)) {
                    totalSize += fs::file_size(entry.filePath);
                }
            } catch (...) {}
        }

        return {
            {"total_entries", g_cache.size()},
            {"permanent", permanentCount},
            {"temp", tempCount},
            {"total_size_mb", totalSize / (1024.0 * 1024.0)}
        };
    }

} // namespace TTSCache
} // namespace Voice
