#include "aliases.hpp"
#include "console_history.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "ui_helpers.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <thread>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
static nlohmann::json g_aliases;          // { "user": {}, "auto": {} }
static std::mutex g_aliasMutex;
static const std::string ALIAS_FILE = "aliases.json";
static const std::string JUNK_FILE  = "auto_junk.json";

// ------------------------------------------------------------
// Simple Levenshtein distance (for fuzzy fallback)
// ------------------------------------------------------------
static int levenshtein(const std::string& s1, const std::string& s2) {
    const size_t m = s1.size();
    const size_t n = s2.size();
    std::vector<int> prev(n + 1), curr(n + 1);

    for (size_t j = 0; j <= n; j++) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= m; i++) {
        curr[0] = static_cast<int>(i);
        for (size_t j = 1; j <= n; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            curr[j] = std::min({
                prev[j] + 1,        // deletion
                curr[j - 1] + 1,    // insertion
                prev[j - 1] + cost  // substitution
            });
        }
        prev.swap(curr);
    }
    return prev[n];
}

// ------------------------------------------------------------
// Helpers: Save/Load JSON
// ------------------------------------------------------------
static void save() {
    std::lock_guard<std::mutex> lock(g_aliasMutex);
    std::ofstream out(ALIAS_FILE);
    if (out) out << g_aliases.dump(2);
}

void aliases::load() {
    std::lock_guard<std::mutex> lock(g_aliasMutex);

    std::ifstream in(ALIAS_FILE);
    if (!in.is_open()) {
        g_aliases = { {"user", nlohmann::json::object()},
                      {"auto", nlohmann::json::object()} };
        save();
        grimLog("[aliases] Created new aliases.json");
        return;
    }

    try {
        in >> g_aliases;
        if (!g_aliases.contains("user")) g_aliases["user"] = nlohmann::json::object();
        if (!g_aliases.contains("auto")) g_aliases["auto"] = nlohmann::json::object();
        grimLog("[aliases] Loaded aliases.json");
    } catch (const std::exception& e) {
        grimLog(std::string("[aliases] Failed to parse aliases.json: ") + e.what());
        g_aliases = { {"user", nlohmann::json::object()},
                      {"auto", nlohmann::json::object()} };
    }
}

// ------------------------------------------------------------
// Scan Helpers
// ------------------------------------------------------------
static void scanPath(std::unordered_map<std::string, std::string>& results) {
    const char* pathEnv = std::getenv("PATH");
    if (!pathEnv) return;

    std::stringstream ss(pathEnv);
    std::string dir;
    while (std::getline(ss, dir, ';')) {
        if (!fs::exists(dir)) continue;
        for (auto& p : fs::directory_iterator(dir)) {
            if (!p.is_regular_file()) continue;
            if (p.path().extension() == ".exe") {
                std::string name = p.path().stem().string();
                std::string full = p.path().string();
                // Basic filters
                std::string lower = name;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                if (lower.find("uninstall") != std::string::npos ||
                    lower.find("setup") != std::string::npos ||
                    lower.find("helper") != std::string::npos ||
                    lower.find("update") != std::string::npos) {
                    continue; // skip junk
                }
                results[name] = full;
            }
        }
    }
}

static void scanStartMenu(std::unordered_map<std::string, std::string>& results) {
#ifdef _WIN32
    char* appData = std::getenv("APPDATA");
    if (!appData) return;
    fs::path startMenu = fs::path(appData) / "Microsoft/Windows/Start Menu/Programs";
    if (!fs::exists(startMenu)) return;

    for (auto& p : fs::recursive_directory_iterator(startMenu)) {
        if (p.path().extension() == ".lnk") {
            std::string name = p.path().stem().string();
            results[name] = p.path().string();
        }
    }
#endif
}

// ------------------------------------------------------------
// Refresh Core
// ------------------------------------------------------------
static void refreshCore(bool userTriggered) {
    std::unordered_map<std::string, std::string> autoResults;

    scanPath(autoResults);
    scanStartMenu(autoResults);

    // Merge into g_aliases["auto"]
    {
        std::lock_guard<std::mutex> lock(g_aliasMutex);
        g_aliases["auto"] = nlohmann::json::object();
        for (auto& [k, v] : autoResults) {
            g_aliases["auto"][k] = {
                {"path", v},
                {"source", "auto"},
                {"last_seen", (int)time(nullptr)}
            };
        }
    }
    save();

    if (userTriggered) {
        history.push("[aliases] Manual refresh complete (" +
                     std::to_string(autoResults.size()) + " apps found)",
                     sf::Color::Green);
    } else {
        grimLog("[aliases] Background auto refresh complete (" +
                std::to_string(autoResults.size()) + " apps)");
    }
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
void aliases::init() {
    load();
    // Spawn background thread for auto refresh
    std::thread([]() {
        refreshCore(false);
    }).detach();
}

void aliases::refreshAsync() {
    std::thread([]() {
        refreshCore(false);
    }).detach();
}

void aliases::refreshNow() {
    refreshCore(true);
}

std::string aliases::resolve(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_aliasMutex);

    // 1. User alias
    if (g_aliases["user"].contains(key)) {
        return g_aliases["user"][key].value("path", "");
    }

    // 2. Auto alias
    if (g_aliases["auto"].contains(key)) {
        return g_aliases["auto"][key].value("path", "");
    }

    // 3. Fuzzy match
    int bestDistance = 999;
    std::string bestMatch;
    for (auto& [k, v] : g_aliases["auto"].items()) {
        int dist = levenshtein(key, k);
        if (dist < bestDistance) {
            bestDistance = dist;
            bestMatch = v.value("path", "");
        }
    }
    if (bestDistance <= 2 && !bestMatch.empty()) {
        return bestMatch;
    }

    // Not found
    return {};
}

const nlohmann::json& aliases::getAll() {
    return g_aliases;
}

std::string aliases::info(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_aliasMutex);
    if (g_aliases["user"].contains(key)) {
        return "[USER] " + key + " → " + g_aliases["user"][key].dump();
    }
    if (g_aliases["auto"].contains(key)) {
        return "[AUTO] " + key + " → " + g_aliases["auto"][key].dump();
    }
    return "[Alias] Not found: " + key;
}
