#include "aliases.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "ui_helpers.hpp"
#include "commands/commands_core.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <atomic>
#include <filesystem>
#include <ctime>

namespace fs = std::filesystem;

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
static nlohmann::json g_aliases;          // { "user": {}, "auto": {} }
static std::mutex g_aliasMutex;
static const std::string ALIAS_FILE = "app_aliases.json";

// 🔹 Reentrancy guard for async refresh
static std::atomic<bool> isRefreshing{false};

// ------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------
static fs::path getAliasFilePath() {
    return fs::path(getResourcePath()) / ALIAS_FILE;
}

static void ensureStructure() {
    if (!g_aliases.is_object()) g_aliases = nlohmann::json::object();
    if (!g_aliases.contains("user") || !g_aliases["user"].is_object())
        g_aliases["user"] = nlohmann::json::object();
    if (!g_aliases.contains("auto") || !g_aliases["auto"].is_object())
        g_aliases["auto"] = nlohmann::json::object();
}

static void saveLocked() {
    try {
        fs::path filePath = getAliasFilePath();
        std::ofstream out(filePath, std::ios::trunc);
        out << g_aliases.dump(4);
        out.close();
        std::cout << "[aliases] Saved aliases → " << filePath.string() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[aliases] ERROR: Could not save aliases: " << e.what() << "\n";
    }
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------
namespace aliases {

void init() {
    std::cout << "[aliases] Bootstrap: initializing (cache only, no scan)\n";

    try {
        // Always start with clean structure
        g_aliases = { {"user", nlohmann::json::object()}, {"auto", nlohmann::json::object()} };
        load(); // load() locks internally
    } catch (const std::exception& e) {
        std::cerr << "[aliases] ERROR during init: " << e.what() << "\n";
        g_aliases = { {"user", nlohmann::json::object()}, {"auto", nlohmann::json::object()} };
        saveLocked();
    }

    ensureStructure();
}

void load() {
    std::scoped_lock lock(g_aliasMutex);

    fs::path filePath = getAliasFilePath();
    if (!fs::exists(filePath)) {
        std::cerr << "[aliases] " << ALIAS_FILE << " not found ─ creating defaults\n";
        saveLocked();
        return;
    }

    try {
        std::ifstream in(filePath);
        nlohmann::json loaded = nlohmann::json::parse(in);

        g_aliases = loaded;
        ensureStructure();

        std::cout << "[aliases] Loaded " << ALIAS_FILE << " successfully\n";
    } catch (const std::exception& e) {
        std::cerr << "[aliases] Failed to parse " << ALIAS_FILE
                  << " ─ resetting to defaults (" << e.what() << ")\n";
        g_aliases = { {"user", nlohmann::json::object()}, {"auto", nlohmann::json::object()} };
        saveLocked();
    }
}

void refreshAsync() {
    if (isRefreshing.exchange(true)) {
        std::cout << "[aliases] refreshAsync skipped (already running)\n";
        return;
    }

    std::thread([] {
        std::cout << "[aliases] refreshAsync launched\n";

        {
            std::scoped_lock lock(g_aliasMutex);
            g_aliases["auto"]["timestamp"] = std::time(nullptr);
        }

        saveLocked();
        std::cout << "[aliases] Background refresh complete\n";
        isRefreshing.store(false);
    }).detach();
}

CommandResult refreshNow() {
    CommandResult result;
    result.success = true;
    result.color = sf::Color::Green;

    {
        std::scoped_lock lock(g_aliasMutex);
        g_aliases["auto"]["timestamp"] = std::time(nullptr);
    }

    saveLocked();
    result.message = "[aliases] Refresh complete (manual trigger).";
    return result;
}

std::string resolve(const std::string& key) {
    std::scoped_lock lock(g_aliasMutex);

    if (g_aliases["user"].contains(key)) {
        return g_aliases["user"][key].get<std::string>();
    }
    if (g_aliases["auto"].contains(key)) {
        return g_aliases["auto"][key].get<std::string>();
    }

    return {};
}

std::unordered_map<std::string, std::string> getAll() {
    std::scoped_lock lock(g_aliasMutex);
    std::unordered_map<std::string, std::string> all;

    for (auto& [k, v] : g_aliases["user"].items()) {
        all[k] = v.get<std::string>();
    }
    for (auto& [k, v] : g_aliases["auto"].items()) {
        all[k] = v.get<std::string>();
    }
    return all;
}

std::string info(const std::string& key) {
    std::scoped_lock lock(g_aliasMutex);

    std::ostringstream oss;
    if (g_aliases["user"].contains(key)) {
        oss << key << " → " << g_aliases["user"][key].get<std::string>() << " (user)";
    } else if (g_aliases["auto"].contains(key)) {
        oss << key << " → " << g_aliases["auto"][key].get<std::string>() << " (auto)";
    } else {
        oss << key << " not found in aliases.";
    }
    return oss.str();
}

} // namespace aliases
