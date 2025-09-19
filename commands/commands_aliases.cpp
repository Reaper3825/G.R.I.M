#include "commands_aliases.hpp"
#include "aliases.hpp"
#include "console_history.hpp"
#include "error_manager.hpp"

// ------------------------------------------------------------
// alias list → dump aliases by section
// ------------------------------------------------------------
CommandResult cmdAliasList(const std::string& /*arg*/) {
    const nlohmann::json& all = aliases::getAll();

    if (all.empty()) {
        return {
            "[Alias] No aliases loaded.",
            true,
            sf::Color::Yellow,
            "ERR_NONE",
            "No aliases loaded",   // voice
            "summary"
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
        oss.str(),
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "Aliases listed",   // short voice-friendly message
        "summary"
    };
}

// ------------------------------------------------------------
// alias info <name> → metadata about a specific alias
// ------------------------------------------------------------
CommandResult cmdAliasInfo(const std::string& arg) {
    if (arg.empty()) {
        return {
            "[Alias] Usage: alias info <name>",
            false,
            sf::Color::Red,
            "ERR_ALIAS_NOT_FOUND",
            "Alias name required",
            "error"
        };
    }

    std::string meta = aliases::info(arg);
    if (meta.empty()) {
        return {
            ErrorManager::getUserMessage("ERR_ALIAS_NOT_FOUND") + ": " + arg,
            false,
            sf::Color::Red,
            "ERR_ALIAS_NOT_FOUND",
            "Alias not found",
            "error"
        };
    }

    return {
        "[Alias] " + meta,
        true,
        sf::Color::Green,
        "ERR_NONE",
        "Alias info for " + arg,
        "summary"
    };
}

// ------------------------------------------------------------
// alias refresh → run blocking refresh, push into history
// ------------------------------------------------------------
CommandResult cmdAliasRefresh(const std::string& /*arg*/) {
    try {
        aliases::refreshNow();
        return {
            "[Alias] Manual refresh complete.",
            true,
            sf::Color::Green,
            "ERR_NONE",
            "Alias refresh complete",
            "routine"
        };
    } catch (const std::exception& e) {
        return {
            std::string("[Alias] Refresh failed: ") + e.what(),
            false,
            sf::Color::Red,
            "ERR_ALIAS_NOT_FOUND",
            "Alias refresh failed",
            "error"
        };
    }
}
