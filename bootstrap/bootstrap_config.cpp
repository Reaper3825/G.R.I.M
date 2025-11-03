#include "bootstrap_config.hpp"
#include "error_manager.hpp"
#include "resources.hpp"
#include "aliases.hpp"
#include "nlp/nlp.hpp"
#include "console_history.hpp"
#include "ai/ai.hpp"
#include "logger.hpp"
#include "nlp/grammar_parser.hpp"  // ✅ NEW: Grammar parser

namespace fs = std::filesystem;

extern NLP g_nlp;
extern nlohmann::json aiConfig;

// ✅ NEW: Global grammar parser instance DEFINITION
namespace GRIM {
    GrammarParser g_grammarParser;
}

// ----------------- helpers -----------------
static bool mergeDefaults(nlohmann::json& cfg,
                          const nlohmann::json& defs,
                          const std::string& prefix = "",
                          int* patchedCount = nullptr) {
    bool patched = false;
    for (auto& [key, defVal] : defs.items()) {
        if (!cfg.contains(key) || cfg[key].is_null()) {
            cfg[key] = defVal;
            patched = true;
            if (patchedCount) (*patchedCount)++;
        } else if (defVal.is_object() && cfg[key].is_object()) {
            if (mergeDefaults(cfg[key], defVal,
                              prefix.empty() ? key : prefix + "." + key,
                              patchedCount))
                patched = true;
        } else if (cfg[key].type() != defVal.type()) {
            cfg[key] = defVal;
            patched = true;
            if (patchedCount) (*patchedCount)++;
        }
    }
    return patched;
}

// ----------------- defaults -----------------
namespace bootstrap_config {

nlohmann::json defaultAliases() {
    return nlohmann::json::object();
}

// ----------------- loader -----------------
bool loadConfig(const fs::path& path,
                const nlohmann::json& defaults,
                nlohmann::json& outConfig,
                const std::string& name,
                const std::string& errorCode) {
    if (!fs::exists(path)) {
        outConfig = defaults;
        std::ofstream(path) << outConfig.dump(2);

        LOG_PHASE(name + " created", true);
        return true;
    }

    try {
        std::ifstream f(path);
        f >> outConfig;

        int patchedCount = 0;
        if (mergeDefaults(outConfig, defaults, "", &patchedCount)) {
            std::ofstream(path) << outConfig.dump(2);
            LOG_PHASE(name + " patched", true);
            LOG_DEBUG("Config", name + " patched (" + std::to_string(patchedCount) + " keys)");
        } else {
            LOG_PHASE(name + " load", true);
        }
        return true;
    } catch (...) {
        LOG_ERROR("Config", name + " invalid → reset to defaults");
        LOG_PHASE(name + " load", false);

        if (!errorCode.empty())
            ErrorManager::report(errorCode);

        outConfig = defaults;
        std::ofstream(path) << outConfig.dump(2);
        return false;
    }
}

// ----------------- entry -----------------
void initAll() {
    // ai_config.json - Simple load from root
    try {
        std::ifstream f(AI_CONFIG_FILE);
        if (f.is_open()) {
            f >> aiConfig;
            LOG_PHASE("AI config load", true);
        } else {
            LOG_ERROR("Config", "ai_config.json not found");
            LOG_PHASE("AI config load", false);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Config", std::string("ai_config.json parse error: ") + e.what());
        LOG_PHASE("AI config load", false);
    }

    // ✅ Grammar-based NLP rules (try binary first, then JSON fallback)
    bool grammarLoaded = false;
    
    // Try binary FlatBuffer format first (faster loading)
    fs::path grammarBinary = fs::path(getResourcePath()) / "grammar_rules.fb";
    if (fs::exists(grammarBinary)) {
        LOG_DEBUG("Config", "Loading binary grammar from: " + grammarBinary.string());
        if (GRIM::g_grammarParser.loadBinary(grammarBinary.string())) {
            LOG_PHASE("Grammar parser initialized (binary)", true);
            grammarLoaded = true;
            
            // Log statistics
            auto stats = GRIM::g_grammarParser.getStats();
            LOG_DEBUG("Grammar", "Loaded " + std::to_string(stats["components_loaded"].get<int>()) + 
                " components, " + std::to_string(stats["verbs_loaded"].get<int>()) + 
                " verbs, " + std::to_string(stats["templates_loaded"].get<int>()) + " templates");
        } else {
            LOG_ERROR("Config", "Failed to load binary grammar from: " + grammarBinary.string());
        }
    }
    
    // Fallback to JSON if binary not available or failed
    if (!grammarLoaded) {
        fs::path grammarPath = fs::path(getResourcePath()) / "nlp_grammar.json";
        if (fs::exists(grammarPath)) {
            LOG_DEBUG("Config", "Loading JSON grammar from: " + grammarPath.string());
            if (GRIM::g_grammarParser.load(grammarPath.string())) {
                LOG_PHASE("Grammar parser initialized (JSON)", true);
                grammarLoaded = true;
                
                // Log statistics
                auto stats = GRIM::g_grammarParser.getStats();
                LOG_DEBUG("Grammar", "Loaded " + std::to_string(stats["components_loaded"].get<int>()) + 
                    " components, " + std::to_string(stats["verbs_loaded"].get<int>()) + 
                    " verbs, " + std::to_string(stats["templates_loaded"].get<int>()) + " templates");
            } else {
                LOG_ERROR("Config", "Failed to load grammar rules from: " + grammarPath.string());
                LOG_PHASE("Grammar parser initialization", false);
            }
        } else {
            LOG_DEBUG("Config", "Grammar file not found: " + grammarPath.string() + " - falling back to regex-only NLP");
        }
    }

    // NLP rules (regex-based - now serves as fallback)
fs::path nlpPath = fs::path(getResourcePath()) / "nlp_rules.json";
    if (!fs::exists(nlpPath)) {
        std::ofstream(nlpPath) << "[]\n";
        LOG_PHASE("NLP rules created", true);
    }
    std::string err;
    if (!g_nlp.load_rules(nlpPath.string(), &err)) {
        LOG_ERROR("Config", "Failed to load NLP rules: " + err);
   LOG_PHASE("NLP rules load", false);
    } else {
      LOG_PHASE("NLP rules load", true);
    }

    // synonyms.json
    fs::path synPath = fs::path(getResourcePath()) / "synonyms.json";
    if (!fs::exists(synPath)) {
        std::ofstream(synPath) << "{}\n";
        LOG_PHASE("Synonyms config created", true);
    } else {
        LOG_PHASE("Synonyms config load", true);
    }
}

} // namespace bootstrap_config
