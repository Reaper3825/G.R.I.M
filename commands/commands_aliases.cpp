#include "commands_aliases.hpp"
#include "aliases.hpp"
#include "console_history.hpp"
#include "response_manager.hpp"

#include <sstream>

// ------------------------------------------------------------
// alias list → display user + auto keys
// ------------------------------------------------------------
CommandResult cmdAliasList(const std::string& args) {
    std::ostringstream out;
    const auto& all = aliases::getAll();

    out << "[Alias] Listing aliases\n";

    if (all.contains("user") && !all["user"].empty()) {
        out << " [USER]\n";
        for (auto& [k, v] : all["user"].items()) {
            out << "   " << k << " → " << v.value("path", "") << "\n";
        }
    } else {
        out << " [USER] (none)\n";
    }

    if (all.contains("auto") && !all["auto"].empty()) {
        out << " [AUTO]\n";
        for (auto& [k, v] : all["auto"].items()) {
            out << "   " << k << " → " << v.value("path", "") << "\n";
        }
    } else {
        out << " [AUTO] (none)\n";
    }

    out << " [FALLBACK]\n";
    out << "   Uses fuzzy matching when no direct alias is found.\n";

    return {true, out.str()};
}

// ------------------------------------------------------------
// alias info <name>
// ------------------------------------------------------------
CommandResult cmdAliasInfo(const std::string& args) {
    if (args.empty()) {
        return {false, "[Alias] Usage: alias info <name>"};
    }
    return {true, aliases::info(args)};
}

// ------------------------------------------------------------
// alias refresh → blocking refreshNow()
// ------------------------------------------------------------
CommandResult cmdAliasRefresh(const std::string& args) {
    aliases::refreshNow();
    return {true, "[Alias] Manual refresh triggered."};
}

// ------------------------------------------------------------
// Register alias commands
// ------------------------------------------------------------
void registerAliasCommands() {
    commandMap["alias list"]    = cmdAliasList;
    commandMap["alias info"]    = cmdAliasInfo;
    commandMap["alias refresh"] = cmdAliasRefresh;
}
