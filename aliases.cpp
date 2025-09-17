#include "aliases.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "ui_helpers.hpp"
#include "commands/commands_core.hpp"   // 🔹 CommandResult

#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <vector>
#include <thread>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <atomic>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
static nlohmann::json g_aliases;          // { "user": {}, "auto": {} }
static std::mutex g_aliasMutex;
static const std::string ALIAS_FILE = "app_aliases.json";

// 🔹 Reentrancy guard
static std::atomic<bool> isRefreshing{false};

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
    if (out) {
        out << g_aliases.dump(2);
        grimLog("[aliases] aliases.json saved");
    } else {
        grimLog("[aliases] ERROR: could not save aliases.json");
    }
}

void aliases::load() {
    std::lock_guard<std::mutex> lock(g_aliasMutex);
    grimLog("[aliases] load() starting");

    // Sanity check file before parsing
    std::error_code ec;
    auto fileSize = fs::file_size(ALIAS_FILE, ec);
    if (ec) {
        grimLog("[aliases] aliases.json not found or stat failed → resetting to defaults");
        g_aliases = { {"user", nlohmann::json::object()},
                      {"auto", nlohmann::json::object()} };
        save();
        return;
    }

    if (fileSize > 5 * 1024 * 1024) { // 5 MB limit
        grimLog("[aliases] aliases.json too large (" + std::to_string(fileSize) + " bytes) → resetting");
        g_aliases = { {"user", nlohmann::json::object()},
                      {"auto", nlohmann::json::object()} };
        save();
        return;
    }

    grimLog("[aliases] Opening aliases.json (" + std::to_string(fileSize) + " bytes)");
    std::ifstream in(ALIAS_FILE);
    if (!in.is_open()) {
        grimLog("[aliases] Failed to open aliases.json → resetting");
        g_aliases = { {"user", nlohmann::json::object()},
                      {"auto", nlohmann::json::object()} };
        save();
        return;
    }

    try {
        grimLog("[aliases] Parsing aliases.json...");
        in >> g_aliases;
        grimLog("[aliases] Parsing finished OK");
        if (!g_aliases.contains("user")) g_aliases["user"] = nlohmann::json::object();
        if (!g_aliases.contains("auto")) g_aliases["auto"] = nlohmann::json::object();
        grimLog("[aliases] Loaded aliases.json successfully");
    } catch (const std::exception& e) {
        grimLog(std::string("[aliases] Failed to parse aliases.json: ") + e.what());
        g_aliases = { {"user", nlohmann::json::object()},
                      {"auto", nlohmann::json::object()} };
        save();
    }

    grimLog("[aliases] load() finished");
}

// ------------------------------------------------------------
// Refresh Core
// ------------------------------------------------------------
static CommandResult refreshCore(bool userTriggered) {
    grimLog(userTriggered ? "[aliases] Manual refresh started"
                          : "[aliases] Background refresh started");

    // 🚨 Temporarily stubbed scanning out for debugging
    std::unordered_map<std::string, std::string> autoResults;

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

    CommandResult result;
    result.success = true;
    result.color   = sf::Color::Green;
    result.errorCode = "ERR_NONE";

    if (userTriggered) {
        result.message  = "[aliases] Manual refresh complete (0 apps found)";
        result.voice    = "";          // silent
        result.category = "routine";
    } else {
        grimLog("[aliases] Background refresh complete (0 apps)");
        result.message  = "[aliases] Background refresh complete (0 apps)";
        result.voice    = "";
        result.category = "routine";
    }
    return result;
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
void aliases::init() {
    grimLog("[aliases] init() called");
    load();   // only load JSON here
    grimLog("[aliases] init() finished (no scan triggered)");
}

void aliases::refreshAsync() {
    if (isRefreshing.exchange(true)) {
        grimLog("[aliases] refreshAsync skipped (already running)");
        return;
    }
    grimLog("[aliases] refreshAsync launched");
    std::thread([]() {
        CommandResult res = refreshCore(false);
        (void)res; // background mode → ignored
        isRefreshing = false;
        grimLog("[aliases] refreshAsync thread finished, flag released");
    }).detach();
}

CommandResult aliases::refreshNow() {
    if (isRefreshing.exchange(true)) {
        grimLog("[aliases] refreshNow skipped (already running)");
        CommandResult result;
        result.success   = false;
        result.message   = "[aliases] Refresh already running, skipping manual";
        result.color     = sf::Color::Yellow;
        result.errorCode = "ERR_ALIAS_BUSY";
        result.voice     = "";
        result.category  = "routine";
        return result;
    }
    grimLog("[aliases] refreshNow started");
    CommandResult res = refreshCore(true);
    isRefreshing = false;
    grimLog("[aliases] refreshNow finished, flag released");
    return res;
}

std::string aliases::resolve(const std::string& key) {
    std::lock_guard<std::mutex> lock(g_aliasMutex);

    if (g_aliases["user"].contains(key)) {
        return g_aliases["user"][key].value("path", "");
    }

    if (g_aliases["auto"].contains(key)) {
        return g_aliases["auto"][key].value("path", "");
    }

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

    return {};
}

// 🔹 New: returns a flattened map of all aliases
std::unordered_map<std::string, std::string> aliases::getAll() {
    std::lock_guard<std::mutex> lock(g_aliasMutex);
    std::unordered_map<std::string, std::string> result;

    for (auto& [k, v] : g_aliases["user"].items()) {
        result[k] = v.value("path", "");
    }
    for (auto& [k, v] : g_aliases["auto"].items()) {
        result[k] = v.value("path", "");
    }

    return result;
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
