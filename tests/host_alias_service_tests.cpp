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

// Controlled host alias cache/scanner; no filesystem scan or database writes.
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

void runHostAliasServiceTests() {
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
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
    check(!registry.isRegistered("alias list") && !registry.isRegistered("alias info") &&
          !registry.isRegistered("alias refresh"), "Removed alias tools remain registered");
    auto* writer = PluginAPI::createPluginAPI("alias-service-test", "", GRIM_PERM_FILESYSTEM);
    const auto read = [&](const char* text) {
        check(text != nullptr, "Host alias service returned null");
        const std::string copy(text); writer->string_free(writer, text);
        return nlohmann::json::parse(copy);
    };
    fixtureAliases = {{"editor", "C:/Editor.exe"}};
    check(read(writer->get_alias_snapshot(writer, nullptr)).at("entries").size() == 1,
          "Retained host alias cache unavailable");
    check(read(writer->get_alias_snapshot(writer, "EDITOR")).at("info").get<std::string>().find("C:/Editor.exe") != std::string::npos,
          "Retained alias lookup failed");
    for (const int mode : {0, 1, 2}) {
        refreshMode = mode;
        check(read(writer->refresh_aliases(writer)).at("completed").get<bool>(), "Host refresh completion contract changed");
    }
    refreshMode = 3;
    check(!read(writer->refresh_aliases(writer)).at("completed").get<bool>(), "Refresh exception was hidden");
    PluginAPI::cleanupPluginContext(writer);
    fixtureAliases.clear(); refreshMode = 0;
    std::cout << "Retained host alias service tests passed\n";
}
