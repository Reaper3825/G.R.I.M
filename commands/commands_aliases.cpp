#include "commands_aliases.hpp"
#include "aliases.hpp"
#include "console_history.hpp"
#include "error_manager.hpp"

// ====================================================
// alias list → dump aliases by section
// ====================================================
CommandResult cmdAliasList(const std::string& /*arg*/) {
    const nlohmann::json& all = aliases::getAll();

    if (all.empty()) {
        return {
            true,                               // success
            "[Alias] No aliases loaded.",       // message
            "ERR_NONE",                         // errorCode
            "summary",                          // category
            "No aliases loaded",                // voice
            Colors::Yellow                      // color
        };
    }

    std::ostringstream oss;
    oss << "[Alias] Listing loaded aliases:\n";

    if (all.contains("user")) {
        oss << " [USER]\n";
        for (auto& [k, v] : all["user"].items()) {
            oss << "   " << k << " → " << v << "\n";
        }
    }
    if (all.contains("auto")) {
        oss << " [AUTO]\n";
        for (auto& [k, v] : all["auto"].items()) {
            oss << "   " << k << " → " << v << "\n";
        }
    }
    if (all.contains("fallback")) {
        oss << " [FALLBACK]\n";
        for (auto& [k, v] : all["fallback"].items()) {
            oss << "   " << k << " → " << v << "\n";
        }
    }

    return {
        true,                   // success
        oss.str(),              // message
        "ERR_NONE",             // errorCode
        "summary",              // category
        "Aliases listed",       // voice
        Colors::Cyan            // color
    };
}

// ====================================================
// alias info <name> → metadata about a specific alias
// ====================================================
CommandResult cmdAliasInfo(const std::string& arg) {
    if (arg.empty()) {
        return {
            false,                                      // success
            "[Alias] Usage: alias info <name>",         // message
            "ERR_ALIAS_NOT_FOUND",                      // errorCode
            "error",                                    // category
            "Alias name required",                      // voice
            Colors::Red                                 // color
        };
    }

    std::string meta = aliases::info(arg);
    if (meta.empty()) {
        return {
            false,                                                                  // success
            ErrorManager::getUserMessage("ERR_ALIAS_NOT_FOUND") + ": " + arg,      // message
            "ERR_ALIAS_NOT_FOUND",                                                  // errorCode
            "error",                                                                // category
            "Alias not found",                                                      // voice
            Colors::Red                                                             // color
        };
    }

    return {
        true,                               // success
        "[Alias] " + meta,                  // message
        "ERR_NONE",                         // errorCode
        "summary",                          // category
        "Alias info for " + arg,            // voice
        Colors::Green                       // color
    };
}

// ====================================================
// alias refresh → run blocking refresh, push into history
// ====================================================
CommandResult cmdAliasRefresh(const std::string& /*arg*/) {
    try {
        aliases::refreshNow();
        return {
            true,                                       // success
            "[Alias] Manual refresh complete.",         // message
            "ERR_NONE",                                 // errorCode
            "routine",                                  // category
            "Alias refresh complete",                   // voice
            Colors::Green                               // color
        };
    } catch (const std::exception& e) {
        return {
            false,                                                  // success
            std::string("[Alias] Refresh failed: ") + e.what(),    // message
            "ERR_ALIAS_NOT_FOUND",                                  // errorCode
            "error",                                                // category
            "Alias refresh failed",                                 // voice
            Colors::Red                                             // color
        };
    }
}
