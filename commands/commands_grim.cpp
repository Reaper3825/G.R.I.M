#include "commands_grim.hpp"
#include "response_manager.hpp"
#include "logger.hpp"
#include "voice/tts_cache.hpp"
#include "voice/voice_speak.hpp"  // ? ADD: For preCacheCommonPhrases()
#include "helpers/color.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// ====================================================
// Clear TTS Cache (Complete Wipe)
// ====================================================
CommandResult cmdClearCache(const std::string& arg) {
    (void)arg;
    
    LOG_DEBUG("Cache", "Clearing TTS cache (full wipe)...");
    
    try {
        int totalRemoved = 0;
        
        fs::path tempDir = "D:/G.R.I.M/resources/tts_out/temp";
        fs::path cacheDir = "D:/G.R.I.M/resources/tts_out/cache";
        
        // Shutdown cache system
        Voice::TTSCache::shutdown();
        
        // Delete cache index
        fs::path cacheIndex = "D:/G.R.I.M/resources/tts_out/cache_index.json";
        if (fs::exists(cacheIndex)) {
            fs::remove(cacheIndex);
            LOG_DEBUG("Cache", "Deleted cache index");
        }
        
        // Delete ALL temp files
        if (fs::exists(tempDir)) {
            for (const auto& entry : fs::directory_iterator(tempDir)) {
                if (entry.path().extension() == ".wav") {
                    fs::remove(entry.path());
                    totalRemoved++;
                }
            }
            LOG_DEBUG("Cache", "Cleared temp directory");
        }
        
        // Delete ALL permanent cache files
        if (fs::exists(cacheDir)) {
            for (const auto& entry : fs::directory_iterator(cacheDir)) {
                if (entry.path().extension() == ".wav") {
                    fs::remove(entry.path());
                    totalRemoved++;
                }
            }
            LOG_DEBUG("Cache", "Cleared permanent cache directory");
        }
        
        // Reinitialize empty cache
        Voice::TTSCache::init();
        
        std::string resp = "TTS cache cleared. Removed " + std::to_string(totalRemoved) + 
                          " cached audio files. Cache will rebuild as needed.";
        
        LOG_DEBUG("Cache", resp);
        
        return {
            true,
            resp,
            "ERR_NONE",
            "routine",
            "Cache cleared successfully",
            Colors::Green
        };
        
    } catch (const std::exception& e) {
        std::string error = "Failed to clear cache: " + std::string(e.what());
        LOG_ERROR("Cache", error);
        
        return {
            false,
            error,
            "ERR_CACHE_CLEAR_FAILED",
            "error",
            "Cache clear failed",
            Colors::Red
        };
    }
}

// ====================================================
// Reset TTS Cache (Wipe + Regenerate Common Phrases)
// ====================================================
CommandResult cmdResetCache(const std::string& arg) {
    (void)arg;
    
    LOG_DEBUG("Cache", "Resetting TTS cache (wipe + regenerate)...");
    
    try {
        int totalRemoved = 0;
        
        fs::path tempDir = "D:/G.R.I.M/resources/tts_out/temp";
        fs::path cacheDir = "D:/G.R.I.M/resources/tts_out/cache";
        
        // Shutdown cache system
        Voice::TTSCache::shutdown();
        
        // Delete cache index
        fs::path cacheIndex = "D:/G.R.I.M/resources/tts_out/cache_index.json";
        if (fs::exists(cacheIndex)) {
            fs::remove(cacheIndex);
            LOG_DEBUG("Cache", "Deleted cache index");
        }
        
        // Delete ALL temp files
        if (fs::exists(tempDir)) {
            for (const auto& entry : fs::directory_iterator(tempDir)) {
                if (entry.path().extension() == ".wav") {
                    fs::remove(entry.path());
                    totalRemoved++;
                }
            }
            LOG_DEBUG("Cache", "Cleared temp directory");
        }
        
        // Delete ALL permanent cache files
        if (fs::exists(cacheDir)) {
            for (const auto& entry : fs::directory_iterator(cacheDir)) {
                if (entry.path().extension() == ".wav") {
                    fs::remove(entry.path());
                    totalRemoved++;
                }
            }
            LOG_DEBUG("Cache", "Cleared permanent cache directory");
        }
        
        // Reinitialize clean cache
        Voice::TTSCache::init();
        
        LOG_DEBUG("Cache", "Cache wiped. Regenerating common phrases...");
        
        // ? NEW: Regenerate common phrases immediately
        Voice::preCacheCommonPhrases();
        
        std::string resp = "TTS cache reset complete. Removed " + std::to_string(totalRemoved) + 
                          " old files and regenerated common phrases.";
        
        LOG_DEBUG("Cache", resp);
        
        return {
            true,
            resp,
            "ERR_NONE",
            "routine",
            "Cache reset with fresh audio",
            Colors::Green
        };
        
    } catch (const std::exception& e) {
        std::string error = "Failed to reset cache: " + std::string(e.what());
        LOG_ERROR("Cache", error);
        
        return {
            false,
            error,
            "ERR_CACHE_RESET_FAILED",
            "error",
            "Cache reset failed",
            Colors::Red
        };
    }
}
