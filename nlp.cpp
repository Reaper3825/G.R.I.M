#include "nlp.hpp"
#include "nlp_rules.hpp"
#include "commands.hpp"
#include <algorithm>

static bool g_rulesLoaded = false;

std::string parseNaturalLanguage(const std::string& line, fs::path& currentDir) {
    if (!g_rulesLoaded) {
        // Try to load once; ignore failure silently (you can add a chat hint if you want)
        loadNlpRules("nlp_rules.json");
        g_rulesLoaded = true;
    }

    std::string lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);

    // 1) Dynamic rules from JSON
    if (auto mapped = mapTextToCommand(lowered); !mapped.empty()) {
        return handleCommand(mapped, currentDir);
    }

    // 2) Your existing hardcoded fallbacks (keep as you had)
    if (lowered.find("where am i") != std::string::npos ||
        lowered.find("current directory") != std::string::npos) {
        return handleCommand("pwd", currentDir);
    }

    if (lowered.find("list files") != std::string::npos ||
        lowered.find("show files") != std::string::npos) {
        return handleCommand("list", currentDir);
    }

    // ... keep any other special-cases you’ve already written ...

    return ""; // let commands.cpp show the unknown‑command message
}
