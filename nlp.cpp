#include "nlp.hpp"
#include "nlp_rules.hpp"
#include "commands.hpp"

#include <algorithm>
#include <filesystem>
#include <mutex>
#include <string>
#include <cctype>

namespace fs = std::filesystem;

static std::once_flag g_rulesOnce;

// small helpers
static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// main entry
std::string parseNaturalLanguage(const std::string& line, fs::path& currentDir) {
    // lazy-load rules once
    std::call_once(g_rulesOnce, [](){
        // ignore failure silently; you can push a “rules not found” message into chat if desired
        loadNlpRules("nlp_rules.json");
    });

    std::string lowered = toLower(trim(line));

    // 1) dynamic rules from JSON
    if (auto mapped = mapTextToCommand(lowered); !mapped.empty()) {
        return handleCommand(mapped, currentDir);
    }

    // 2) lightweight hardcoded fallbacks
    // where am i / pwd
    if (lowered.find("where am i") != std::string::npos ||
        lowered.find("current directory") != std::string::npos) {
        return handleCommand("pwd", currentDir);
    }

    // list files
    if (lowered.find("list files") != std::string::npos ||
        lowered.find("show files") != std::string::npos ||
        lowered == "list") {
        return handleCommand("ls", currentDir); // use your ls if you have it
    }

    // go to root
    if (lowered == "go to root" || lowered == "root" || lowered == "open root") {
        return handleCommand("root", currentDir);
    }

    // up one level
    if (lowered == "up one level" || lowered == "go up" || lowered == "go back" || lowered == "back") {
        return handleCommand("up", currentDir);
    }

    // simple "go to/open <path or alias>"
    // e.g., "go to desktop", "open downloads", "open c:\projects"
    {
        // find a space after "go to " or "open "
        auto startsWith = [&](const char* pfx){
            return lowered.rfind(pfx, 0) == 0;
        };
        if (startsWith("go to ") || startsWith("open ")) {
            auto pos = lowered.find(' ');
            if (pos != std::string::npos) {
                std::string tail = trim(lowered.substr(pos + 1));
                // let commands.cpp resolve aliases/paths
                return handleCommand(std::string("cd ") + tail, currentDir);
            }
        }
    }

    // let commands.cpp show its unknown-command message
    return "";
}
