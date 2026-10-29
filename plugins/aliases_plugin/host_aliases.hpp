#pragma once
#include "core/plugin_api.hpp"
#include "commands/command_result.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hostAliases {
inline const GrimPluginAPI* host = nullptr;
inline std::string take(const char* text) {
    if (!text) throw std::runtime_error("Host alias service unavailable");
    struct Release {
        const char* text;
        ~Release() { host->string_free(host, text); }
    } release{text};
    return text;
}
inline nlohmann::json parse(const char* text, const char* format) {
    const auto value = nlohmann::json::parse(take(text));
    if (value.at("format") != format || value.at("version") != 1)
        throw std::runtime_error("Unsupported host alias response");
    return value;
}
inline auto getAll() {
    const auto value = parse(host->get_alias_snapshot(host, nullptr), "grim-alias-snapshot");
    // Retain host iteration order rather than rebuilding an unordered_map.
    std::vector<std::pair<std::string, std::string>> entries;
    for (const auto& entry : value.at("entries"))
        entries.emplace_back(entry.at("name").get<std::string>(), entry.at("target").get<std::string>());
    return entries;
}
inline std::string info(const std::string& key) {
    return parse(host->get_alias_snapshot(host, key.c_str()), "grim-alias-snapshot").at("info");
}
inline void refreshNow() {
    const auto value = parse(host->refresh_aliases(host), "grim-alias-refresh");
    if (!value.at("completed").get<bool>()) throw std::runtime_error(value.at("error").get<std::string>());
}
inline std::string errorMessage(const std::string& code) {
    return take(host->get_user_error_message(host, code.c_str()));
}
}
CommandResult cmdAliasList(const std::string&);
CommandResult cmdAliasInfo(const std::string&);
CommandResult cmdAliasRefresh(const std::string&);
