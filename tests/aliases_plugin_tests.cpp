#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "aliases.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

// Replace only the host alias store/scanner with controlled state. The real
// DLL, API, registry, ownership, and result writer remain under test.
namespace {
std::unordered_map<std::string, std::string> fixtureAliases;
int refreshCalls = 0, infoCalls = 0, refreshMode = 0;
constexpr const char* arrow = "\xE2\x86\x92";
}
namespace aliases {
std::unordered_map<std::string, std::string> getAll() { return fixtureAliases; }
std::string info(const std::string& key) {
    ++infoCalls;
    if (key == "__empty__") return {}; // Also cover the command's empty-info branch.
    auto normalized = key;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const auto found = fixtureAliases.find(normalized);
    if (found == fixtureAliases.end()) return key + " not found in aliases.";
    return key + " " + arrow + " " + found->second + " (user)";
}
CommandResult refreshNow() {
    ++refreshCalls;
    if (refreshMode == 3) throw std::runtime_error("fixture scan failure");
    if (refreshMode == 1) {
        CommandResult result;
        result.message = "Alias refresh already in progress. Please wait...";
        result.color = Colors::Yellow;
        return result;
    }
    if (refreshMode == 2) { CommandResult result; result.success = false; return result; }
    fixtureAliases["refreshed"] = "C:/fresh.exe";
    return CommandResult{};
}
}

void runAliasesPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    fixtureAliases.clear(); refreshCalls = infoCalls = refreshMode = 0;
    auto* reader = PluginAPI::createPluginAPI("alias-reader", "", 0);
    const auto take = [&](const char* text) {
        check(text != nullptr, "Host alias API returned null");
        const std::string copy(text);
        reader->string_free(reader, text);
        return nlohmann::json::parse(copy);
    };
    const auto denied = take(reader->refresh_aliases(reader));
    check(!denied.at("completed").get<bool>() && denied.at("error") == "Filesystem permission denied" && refreshCalls == 0,
          "Alias refresh bypassed filesystem permission");
    check(reader->get_alias_snapshot(nullptr, nullptr) == nullptr && reader->refresh_aliases(nullptr) == nullptr,
          "Invalid alias API identity accepted");
    const auto empty = take(reader->get_alias_snapshot(reader, nullptr));
    check(empty.at("format") == "grim-alias-snapshot" && empty.at("version") == 1 && empty.at("entries").empty(),
          "Empty alias snapshot incorrect");
    PluginAPI::cleanupPluginContext(reader);

    const auto folder = fs::temp_directory_path() / ("grim-aliases-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create aliases test folder");
    struct Cleanup {
        fs::path folder;
        ~Cleanup() {
            PluginManager::unloadAll();
            fixtureAliases.clear();
            std::error_code error;
            fs::remove(folder / "aliases_plugin.dll", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    fs::copy_file(pluginDirectory / "aliases_plugin.dll", folder / "aliases_plugin.dll");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 3 && refreshCalls == 0 && infoCalls == 0,
          "Alias plugin load changed host alias state");
    for (const auto* id : {"alias list", "alias info", "alias refresh"}) {
        const auto d = registry.getTool(id);
        const bool refresh = std::string(id) == "alias refresh";
        check(d && d->provider_name == "Aliases Plugin" && d->provider_type == ToolProviderType::Plugin &&
              d->is_informational != refresh && d->needs_confirmation == refresh &&
              d->permission_bits == (refresh ? GRIM_PERM_FILESYSTEM : 0) && d->examples.size() == 1 &&
              d->parameters.size() == (std::string(id) == "alias info" ? 1 : 0), "Alias metadata/policy changed");
    }
    const auto invoke = [&](const char* id, const std::string& argument = "") { return registry.invoke(id, argument); };
    const auto matches = [&](const CommandResult& r, bool ok, const std::string& message, const char* code,
                             const char* category, const std::string& voice, Color color) {
        check(r.success == ok && r.message == message && r.errorCode == code && r.category == category &&
              r.voice == voice && r.color.toUInt() == color.toUInt() && r.structuredOutputJson.empty(),
              "Alias command result differs from original behavior");
    };
    matches(invoke("alias list", "ignored"), true, "[Alias] No aliases loaded.", "ERR_NONE", "summary", "No aliases loaded", Colors::Yellow);
    matches(invoke("alias info"), false, "[Alias] Usage: alias info <name>", "ERR_ALIAS_NOT_FOUND", "error", "Alias name required", Colors::Red);
    check(infoCalls == 0, "Empty info argument called host lookup");
    // The current real host info() returns a nonempty missing-name message.
    matches(invoke("alias info", "Missing"), true, "[Alias] Missing not found in aliases.", "ERR_NONE", "summary", "Alias info for Missing", Colors::Green);
    matches(invoke("alias info", "__empty__"), false, "configured:ERR_ALIAS_NOT_FOUND: __empty__", "ERR_ALIAS_NOT_FOUND", "error", "Alias not found", Colors::Red);
    fixtureAliases = {{"editor", "C:/Program Files/Editor.exe"}, {"terminal", "C:/Terminal.exe"}};
    std::ostringstream expectedList;
    expectedList << "[Alias] Listing loaded aliases:\n [ALL ALIASES]\n";
    for (const auto& [name, target] : aliases::getAll()) expectedList << "   " << name << " " << arrow << " " << target << "\n";
    matches(invoke("alias list"), true, expectedList.str(), "ERR_NONE", "summary", "Aliases listed", Colors::Cyan);
    matches(invoke("alias info", "EDITOR"), true, std::string("[Alias] EDITOR ") + arrow + " C:/Program Files/Editor.exe (user)", "ERR_NONE", "summary", "Alias info for EDITOR", Colors::Green);
    matches(invoke("alias info", " editor "), true, "[Alias]  editor  not found in aliases.", "ERR_NONE", "summary", "Alias info for  editor ", Colors::Green);
    fixtureAliases["editor"] = "C:/Updated.exe";
    matches(invoke("alias info", "editor"), true, std::string("[Alias] editor ") + arrow + " C:/Updated.exe (user)", "ERR_NONE", "summary", "Alias info for editor", Colors::Green);
    for (const int mode : {0, 1, 2}) {
        refreshMode = mode;
        matches(invoke("alias refresh", "ignored"), true, "[Alias] Manual refresh complete.", "ERR_NONE", "routine", "Alias refresh complete", Colors::Green);
    }
    check(refreshCalls == 3 && fixtureAliases.count("refreshed") == 1, "Refresh did not use shared host state");
    refreshMode = 3;
    matches(invoke("alias refresh"), false, "[Alias] Refresh failed: fixture scan failure", "ERR_ALIAS_NOT_FOUND", "error", "Alias refresh failed", Colors::Red);
    fixtureAliases["large"] = std::string(70000, 'x');
    const auto large = invoke("alias list");
    check(large.success && large.message.find(fixtureAliases["large"]) != std::string::npos, "Large alias listing truncated");
    const auto beforeRemoval = refreshCalls;
    fs::remove(folder / "aliases_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("alias list") && !registry.isRegistered("alias info") &&
          !registry.isRegistered("alias refresh") && fixtureAliases.count("refreshed") == 1 && refreshCalls == beforeRemoval,
          "Removing alias plugin changed host state or retained commands");
    std::cout << "Aliases plugin exact results, shared state, refresh permissions/status, and removal tests passed\n";
}
