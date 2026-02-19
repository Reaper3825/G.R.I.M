#include "tts_cache.hpp"
#include "logger.hpp"
#include "resources.hpp"
#include <fstream>
#include <chrono>
#include <algorithm>
#include <unordered_map>

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
        "Got it I'll keep doing that.",
        "Understood I'll adjust next time.",
        "Opening notepad",
        "Opening",
        "Launching",
        "I processed",
        "commands. Was that correct?",
        "Got it I'll remember that next time.",
        "I couldn't save that one try again later.",
        "Sorry, I couldn't interpret that.",

        "Welcome back",
        "Grim is online",
        "Initializing",
        "System ready",
        "Voice recognition active",
        "Processing request"
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
        
        // ? NEW: Speaker is now part of directory path, not the key
        // This prevents cache collisions between speakers
        return speaker + "/" + normalized + "_" + std::to_string(static_cast<int>(speed * 100));
    }
    
    // =========================================================
    // Get speaker-specific directory
    // =========================================================
    static fs::path getSpeakerDir(const std::string& speaker, bool isPermanent) {
        fs::path baseDir = isPermanent ? g_cacheDir : g_tempDir;
        return baseDir / speaker;
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
    std::string grimRoot = getGrimRootDir();
    g_cacheDir = fs::path(grimRoot) / "resources/tts_out/cache";
    g_tempDir  = fs::path(grimRoot) / "resources/tts_out/temp";
    g_indexFile = fs::path(grimRoot) / "resources/tts_out/cache_index.json";

    // Create base directories (speaker subdirs created on-demand)
    fs::create_directories(g_cacheDir);
    fs::create_directories(g_tempDir);

    loadIndex();
    
    // Migrate old cache files to speaker-specific directories
    // This runs once and moves any loose .wav files into default/ subdirectory
    migrateToSpeakerDirs("default");

    // 🔹 Disable cleanup on startup — this was deleting new files before reuse
    // cleanupOldFiles(24);

    LOG_PHASE("TTS Cache initialized (per-speaker organization)", true);
}


    void shutdown() {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        saveIndex();
        LOG_PHASE("TTS Cache shutdown", true);
    }

    // =========================================================
    // Get cached audio file
    // =========================================================
    std::string getCached(const std::string& text, const std::string& speaker, double speed)
    {
        std::lock_guard<std::mutex> lock(g_cacheMutex);

        std::string key = getCacheKey(text, speaker, speed);

        auto it = g_cache.find(key);
        if (it != g_cache.end())
        {
            // ✅ FIX: Verify file actually exists before returning path
            if (fs::exists(it->second.filePath))
            {
                // Update last used time
                it->second.lastUsed = std::time(nullptr);
                it->second.useCount++;

                std::string logText = text.length() > 30 ? text.substr(0, 30) + "..." : text;
                LOG_DEBUG("TTSCache", "Cache HIT for: " + logText);
                return it->second.filePath;
            }
            else
            {
                // ✅ File doesn't exist - remove from cache
                std::string logText = text.length() > 30 ? text.substr(0, 30) + "..." : text;
                LOG_DEBUG("TTSCache", "Cache entry invalid (file missing), removing: " + logText);
                g_cache.erase(it);
                saveIndex();
                return "";
            }
        }

        std::string logText = text.length() > 30 ? text.substr(0, 30) + "..." : text;
        LOG_DEBUG("TTSCache", "Cache MISS for: " + logText);
        return "";
    }

    // =========================================================
    // Store audio in cache
    // =========================================================
    std::string store(const std::string& text, const std::string& speaker, double speed,
                      const std::string& filePath, bool isPermanent)
    {
        std::lock_guard<std::mutex> lock(g_cacheMutex);

        std::string key = getCacheKey(text, speaker, speed);

        // Decide if this should be permanent
        bool shouldBePermanent = isPermanent;

        // Get speaker-specific directory
        fs::path destDir = getSpeakerDir(speaker, shouldBePermanent);
        fs::create_directories(destDir);

        // Generate destination filename (keep the same random name)
        fs::path srcPath(filePath);
        fs::path destPath = destDir / srcPath.filename();

        // Move file from temp to speaker-specific cache/temp
        if (fs::exists(srcPath) && srcPath != destPath)
        {
            try
            {
                fs::rename(srcPath, destPath);
                LOG_DEBUG("TTSCache", "Moved file: " + srcPath.string() + " → " + destPath.string());
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("TTSCache", std::string("Failed to move file: ") + e.what());
                // If move fails, just use the original path
                destPath = srcPath;
            }
        }
        else if (fs::exists(destPath))
        {
            // File is already in the destination
            LOG_DEBUG("TTSCache", "File already in destination: " + destPath.string());
        }
        else
        {
            LOG_ERROR("TTSCache", "Source file not found: " + srcPath.string());
            return "";  // ✅ Return empty string on error
        }

        // Store in cache index
        CacheEntry entry;
        entry.filePath = destPath.string();
        entry.lastUsed = std::time(nullptr);
        entry.isPermanent = shouldBePermanent;
        entry.useCount = 1;

        g_cache[key] = entry;

        std::string cacheType = shouldBePermanent ? "PERMANENT" : "TEMP";
        std::string logText = text.length() > 30 ? text.substr(0, 30) + "..." : text;
        LOG_DEBUG("TTSCache", "Stored " + cacheType + " cache [" + speaker + "]: " + logText);

        // Save index to disk
        saveIndex();

        return destPath.string();  // ✅ Return the final path
    }

    void markPermanent(const std::string& text) {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        // Find all cache entries containing this text
        for (auto& [key, entry] : g_cache) {
            if (key.find(text) != std::string::npos && !entry.isPermanent) {
                entry.isPermanent = true;
                
                // Extract speaker from key (format: "speaker/normalized_text_speed")
                std::string speaker = "default";
                size_t slashPos = key.find('/');
                if (slashPos != std::string::npos) {
                    speaker = key.substr(0, slashPos);
                }
                
                // Move file from temp/{speaker}/ to cache/{speaker}/
                fs::path oldPath = fs::path(entry.filePath);
                fs::path speakerCacheDir = g_cacheDir / speaker;
                fs::create_directories(speakerCacheDir);
                fs::path newPath = speakerCacheDir / oldPath.filename();
                
                try {
                    if (fs::exists(oldPath)) {
                        // Use copy + remove for reliable cross-directory move
                        fs::copy_file(oldPath, newPath, fs::copy_options::overwrite_existing);
                        fs::remove(oldPath);
                        entry.filePath = newPath.string();
                        LOG_DEBUG("TTSCache", "Marked as permanent [" + speaker + "]: " + text);
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

    // =========================================================
    // Migrate old cache files to speaker-specific directories
    // =========================================================
    void migrateToSpeakerDirs(const std::string& defaultSpeaker) {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        LOG_DEBUG("TTSCache", "Migrating old cache files to speaker-specific directories...");
        
        int migratedCount = 0;
        
        // Scan old cache and temp directories for orphaned files
        std::vector<fs::path> oldDirs = {g_cacheDir, g_tempDir};
        
        for (const auto& dir : oldDirs) {
            if (!fs::exists(dir)) continue;
            
            bool isPermanent = (dir == g_cacheDir);
            fs::path targetDir = getSpeakerDir(defaultSpeaker, isPermanent);
            fs::create_directories(targetDir);
            
            try {
                for (const auto& entry : fs::directory_iterator(dir)) {
                    // Skip subdirectories (those are speaker dirs)
                    if (entry.is_directory()) continue;
                    
                    // Skip if not a .wav file
                    if (entry.path().extension() != ".wav") continue;
                    
                    // Move to speaker-specific directory
                    fs::path oldPath = entry.path();
                    fs::path newPath = targetDir / oldPath.filename();
                    
                    if (!fs::exists(newPath)) {
                        fs::rename(oldPath, newPath);
                        migratedCount++;
                        LOG_DEBUG("TTSCache", "Migrated: " + oldPath.filename().string() + " → " + defaultSpeaker);
                    }
                }
            } catch (const std::exception& e) {
                LOG_ERROR("TTSCache", std::string("Migration error: ") + e.what());
            }
        }
        
        if (migratedCount > 0) {
            LOG_DEBUG("TTSCache", "Migrated " + std::to_string(migratedCount) + " files to [" + defaultSpeaker + "] directory");
        } else {
            LOG_DEBUG("TTSCache", "No files to migrate");
        }
    }

    json getStats() {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        
        size_t permanentCount = 0;
        size_t tempCount = 0;
        size_t totalSize = 0;
        std::unordered_map<std::string, size_t> speakerCounts;

        for (const auto& [key, entry] : g_cache) {
            if (entry.isPermanent) {
                permanentCount++;
            } else {
                tempCount++;
            }

            // Extract speaker from key (format: "speaker/...")
            std::string speaker = "unknown";
            size_t slashPos = key.find('/');
            if (slashPos != std::string::npos) {
                speaker = key.substr(0, slashPos);
            }
            speakerCounts[speaker]++;

            try {
                if (fs::exists(entry.filePath)) {
                    totalSize += fs::file_size(entry.filePath);
                }
            } catch (...) {}
        }

        json speakerJson = json::object();
        for (const auto& [speaker, count] : speakerCounts) {
            speakerJson[speaker] = count;
        }

        return {
            {"total_entries", g_cache.size()},
            {"permanent", permanentCount},
            {"temp", tempCount},
            {"total_size_mb", totalSize / (1024.0 * 1024.0)},
            {"per_speaker", speakerJson}
        };
    }

} // namespace TTSCache
} // namespace Voice
