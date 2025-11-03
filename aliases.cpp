#include "aliases.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "ui/ui_helpers.hpp"
#include "commands/commands_core.hpp"
#include "logger.hpp"
#include "pch.hpp"
#include "aliases_generated.h"
#include <flatbuffers/flatbuffers.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <shlobj.h>  // For SHGetFolderPath (Start Menu)
#include <ctime>

namespace fs = std::filesystem;

// ====================================================
// Globals
// ====================================================
static std::unordered_map<std::string, std::string> g_userAliases;
static std::unordered_map<std::string, std::string> g_autoAliases;
static std::mutex g_aliasMutex;
static const std::string ALIAS_FILE = "app_aliases.fb";

// 🔹 Reentrancy guard for async refresh
static std::atomic<bool> isRefreshing{false};

// ====================================================
// Internal helpers
// ====================================================
static fs::path getAliasFilePath() {
    return fs::path(getResourcePath()) / ALIAS_FILE;
}

static void saveLocked() {
    try {
        flatbuffers::FlatBufferBuilder builder(1024);
        
        // Build user aliases vector
        std::vector<flatbuffers::Offset<AliasesSchema::AliasEntry>> userEntries;
        for (const auto& [key, path] : g_userAliases) {
            auto keyStr = builder.CreateString(key);
            auto pathStr = builder.CreateString(path);
            auto entry = AliasesSchema::CreateAliasEntry(builder, keyStr, pathStr);
            userEntries.push_back(entry);
        }
        
        // Build auto aliases vector
        std::vector<flatbuffers::Offset<AliasesSchema::AliasEntry>> autoEntries;
        for (const auto& [key, path] : g_autoAliases) {
            auto keyStr = builder.CreateString(key);
            auto pathStr = builder.CreateString(path);
            auto entry = AliasesSchema::CreateAliasEntry(builder, keyStr, pathStr);
            autoEntries.push_back(entry);
        }
        
        auto userVec = builder.CreateVector(userEntries);
        auto autoVec = builder.CreateVector(autoEntries);
        
        auto aliases = AliasesSchema::CreateAliases(builder, userVec, autoVec, std::time(nullptr));
        builder.Finish(aliases);
        
        // Write to file
        fs::path filePath = getAliasFilePath();
        std::ofstream out(filePath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        out.close();

        LOG_PHASE("Aliases saved", true);
        LOG_DEBUG("Aliases", "Saved aliases → " + filePath.string());
    } catch (const std::exception& e) {
        LOG_ERROR("Aliases", std::string("Could not save aliases: ") + e.what());
        LOG_PHASE("Aliases save", false);
    }
}

// Normalize a filename to create an alias key
static std::string normalizeKey(const std::string& filename) {
    std::string key = filename;
    
    // Remove extension
    size_t dotPos = key.find_last_of('.');
    if (dotPos != std::string::npos) {
        key = key.substr(0, dotPos);
    }
    
    // Convert to lowercase
    std::transform(key.begin(), key.end(), key.begin(), ::tolower);
    
    return key;
}

// Scan a directory for executables (non-recursive for performance)
static void scanDirectory(const fs::path& dir, std::unordered_map<std::string, std::string>& aliases, int& count) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return;
    }
    
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                // Only index executables
                if (ext == ".exe" || ext == ".bat" || ext == ".cmd") {
                    std::string key = normalizeKey(entry.path().filename().string());
                    std::string path = entry.path().string();
                    
                    // Only add if not already present (first occurrence wins)
                    if (aliases.find(key) == aliases.end()) {
                        aliases[key] = path;
                        count++;
                        
                        // Report progress every 50 items
                        if (count % 50 == 0) {
                            LOG_DEBUG("Aliases", "Scan progress: " + std::to_string(count) + " aliases found...");
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        // Silently skip directories we can't access
        LOG_DEBUG("Aliases", "Skipped directory " + dir.string() + ": " + e.what());
    }
}

// Scan Windows Start Menu for shortcuts
static void scanStartMenu(std::unordered_map<std::string, std::string>& aliases, int& count) {
    char path[MAX_PATH];
    
    // Common Start Menu locations
    std::vector<int> folders = {
        CSIDL_COMMON_PROGRAMS,  // All Users Start Menu
        CSIDL_PROGRAMS          // Current User Start Menu
    };
    
    LOG_DEBUG("Aliases", "Scanning Start Menu...");
    
    for (int folder : folders) {
        if (SUCCEEDED(SHGetFolderPathA(NULL, folder, NULL, 0, path))) {
            try {
                fs::path startMenuPath(path);
                
                // Scan recursively but limit depth
                for (const auto& entry : fs::recursive_directory_iterator(
                    startMenuPath,
                    fs::directory_options::skip_permission_denied)) {
                    
                    if (entry.is_regular_file() && entry.path().extension() == ".lnk") {
                        std::string key = normalizeKey(entry.path().stem().string());
                        std::string path = entry.path().string();
                        
                        if (aliases.find(key) == aliases.end()) {
                            aliases[key] = path;
                            count++;
                            
                            // Report progress every 50 items
                            if (count % 50 == 0) {
                                LOG_DEBUG("Aliases", "Scan progress: " + std::to_string(count) + " aliases found...");
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("Aliases", "Error scanning Start Menu: " + std::string(e.what()));
            }
        }
    }
}

// Perform the actual scan
static void performScan() {
    std::unordered_map<std::string, std::string> newAutoAliases;
    int count = 0;
    
    LOG_DEBUG("Aliases", "=== Starting comprehensive alias scan ===");
    LOG_DEBUG("Aliases", "Scanning common program directories...");
    
    // 1. Scan common program directories on D: drive
    std::vector<std::string> commonDirs = {
        "D:\\Program Files",
        "D:\\Program Files (x86)",
        "D:\\Apps",
        "D:\\Games",
        "C:\\Program Files",
        "C:\\Program Files (x86)"
    };
    
    int dirIndex = 0;
    for (const auto& dirStr : commonDirs) {
        dirIndex++;
        fs::path dir(dirStr);
        
        if (fs::exists(dir) && fs::is_directory(dir)) {
            LOG_DEBUG("Aliases", "[" + std::to_string(dirIndex) + "/" + std::to_string(commonDirs.size()) + "] Scanning: " + dirStr);
            
            try {
                // Scan top-level directories only (non-recursive)
                int subdirCount = 0;
                for (const auto& subEntry : fs::directory_iterator(dir)) {
                    if (subEntry.is_directory()) {
                        subdirCount++;
                        scanDirectory(subEntry.path(), newAutoAliases, count);
                    }
                }
                LOG_DEBUG("Aliases", "  → Scanned " + std::to_string(subdirCount) + " subdirectories");
            } catch (const std::exception& e) {
                LOG_DEBUG("Aliases", "  → Skipped: " + std::string(e.what()));
            }
        } else {
            LOG_DEBUG("Aliases", "[" + std::to_string(dirIndex) + "/" + std::to_string(commonDirs.size()) + "] Not found: " + dirStr);
        }
    }
    
    // 2. Scan Start Menu shortcuts
    LOG_DEBUG("Aliases", "Scanning Windows Start Menu shortcuts...");
    int startMenuCount = count;
    scanStartMenu(newAutoAliases, count);
    LOG_DEBUG("Aliases", "  → Found " + std::to_string(count - startMenuCount) + " Start Menu shortcuts");
    
    // 3. Update global auto aliases
    {
        std::scoped_lock lock(g_aliasMutex);
        g_autoAliases = std::move(newAutoAliases);
    }
    
    LOG_DEBUG("Aliases", "=== Scan complete: " + std::to_string(count) + " total aliases discovered ===");
}

// ====================================================
// Public API
// ====================================================
namespace aliases {

void init() {
    LOG_PHASE("Aliases init", true);
    LOG_DEBUG("Aliases", "Bootstrap: initializing (cache only, no scan)");

    try {
        // Clear in-memory structures
        g_userAliases.clear();
        g_autoAliases.clear();
        load(); // load() locks internally
    } catch (const std::exception& e) {
        LOG_ERROR("Aliases", std::string("Exception during init: ") + e.what());
        LOG_PHASE("Aliases init", false);

        g_userAliases.clear();
        g_autoAliases.clear();
        saveLocked();
    }
}

void load() {
    std::scoped_lock lock(g_aliasMutex);

    fs::path filePath = getAliasFilePath();
    if (!fs::exists(filePath)) {
        LOG_ERROR("Aliases", ALIAS_FILE + " not found — creating defaults");
        LOG_PHASE("Aliases load", false);
        saveLocked();
        return;
    }

    try {
        // Read binary file
        std::ifstream in(filePath, std::ios::binary);
        std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();

        // Verify and parse FlatBuffer
        flatbuffers::Verifier verifier(buffer.data(), buffer.size());
        if (!AliasesSchema::VerifyAliasesBuffer(verifier)) {
            LOG_ERROR("Aliases", "FlatBuffer verification failed");
            LOG_PHASE("Aliases load", false);
            saveLocked();
            return;
        }

        const AliasesSchema::Aliases* aliases = AliasesSchema::GetAliases(buffer.data());
        
        // Load user aliases
        g_userAliases.clear();
        if (aliases->user()) {
            for (const auto* entry : *aliases->user()) {
                if (entry->key() && entry->path()) {
                    g_userAliases[entry->key()->str()] = entry->path()->str();
                }
            }
        }
        
        // Load auto aliases
        g_autoAliases.clear();
        if (aliases->auto_()) {
            for (const auto* entry : *aliases->auto_()) {
                if (entry->key() && entry->path()) {
                    g_autoAliases[entry->key()->str()] = entry->path()->str();
                }
            }
        }

        LOG_PHASE("Aliases load", true);
        LOG_DEBUG("Aliases", "Loaded " + ALIAS_FILE + " successfully");
    } catch (const std::exception& e) {
        LOG_ERROR("Aliases", std::string("Failed to parse ") + ALIAS_FILE + ": " + e.what());
        LOG_PHASE("Aliases load", false);

        g_userAliases.clear();
        g_autoAliases.clear();
        saveLocked();
    }
}

void refreshAsync() {
    if (isRefreshing.exchange(true)) {
        LOG_DEBUG("Aliases", "refreshAsync skipped (already running)");
        return;
    }

    std::thread([] {
        LOG_DEBUG("Aliases", "refreshAsync launched");

        performScan();
        saveLocked();
        
        LOG_DEBUG("Aliases", "Background refresh complete");
        isRefreshing.store(false);
    }).detach();
}

CommandResult refreshNow() {
    CommandResult result;
    result.success = true;
    result.color   = Colors::Green;

    if (isRefreshing.load()) {
        result.message = "Alias refresh already in progress. Please wait...";
        result.color = Colors::Yellow;
        return result;
    }

    LOG_DEBUG("Aliases", "Manual refresh started");
    
    auto startTime = std::chrono::steady_clock::now();
    performScan();
    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    saveLocked();

    int userCount = 0;
    int autoCount = 0;
    {
        std::scoped_lock lock(g_aliasMutex);
        userCount = g_userAliases.size();
        autoCount = g_autoAliases.size();
    }
    
    std::ostringstream oss;
    oss << "Alias scan complete!\n";
    oss << "  User aliases: " << userCount << "\n";
    oss << "  Auto-discovered: " << autoCount << "\n";
    oss << "  Total: " << (userCount + autoCount) << "\n";
    oss << "  Scan time: " << (duration / 1000.0) << "s";
    
    result.message = oss.str();
    return result;
}

std::string resolve(const std::string& key) {
    std::scoped_lock lock(g_aliasMutex);

    // Normalize the lookup key
    std::string normalizedKey = key;
    std::transform(normalizedKey.begin(), normalizedKey.end(), normalizedKey.begin(), ::tolower);

    // Check user aliases first
    auto userIt = g_userAliases.find(normalizedKey);
    if (userIt != g_userAliases.end()) {
        return userIt->second;
    }
    
    // Check auto aliases
    auto autoIt = g_autoAliases.find(normalizedKey);
    if (autoIt != g_autoAliases.end()) {
        return autoIt->second;
    }

    return {};
}

std::unordered_map<std::string, std::string> getAll() {
    std::scoped_lock lock(g_aliasMutex);
    std::unordered_map<std::string, std::string> all;

    for (const auto& [k, v] : g_userAliases) {
        all[k] = v;
    }
    for (const auto& [k, v] : g_autoAliases) {
        all[k] = v;
    }
    return all;
}

std::string info(const std::string& key) {
    std::scoped_lock lock(g_aliasMutex);

    std::string normalizedKey = key;
    std::transform(normalizedKey.begin(), normalizedKey.end(), normalizedKey.begin(), ::tolower);

    std::ostringstream oss;
    auto userIt = g_userAliases.find(normalizedKey);
    if (userIt != g_userAliases.end()) {
        oss << key << " → " << userIt->second << " (user)";
    } else {
        auto autoIt = g_autoAliases.find(normalizedKey);
        if (autoIt != g_autoAliases.end()) {
            oss << key << " → " << autoIt->second << " (auto)";
        } else {
            oss << key << " not found in aliases.";
        }
    }
    return oss.str();
}

} // namespace aliases
