#include "nlp.hpp"
#include "pch.hpp"

// ====================================================
// Load NLP rules from a JSON file into the global g_nlp object
// ====================================================
bool loadNlpRules(const std::string& path)
{
    std::ifstream in(path);
    if (!in.is_open())
    {
        LOG_ERROR("NLP", "Could not open NLP rules file: " + path);
        return false;
    }

    try
    {
        nlohmann::json j;
        in >> j;

        if (!j.is_array())
        {
            LOG_ERROR("NLP", "Invalid NLP rules JSON (expected array)");
            return false;
        }

        // Convert the parsed JSON back to string
        std::string rulesText = j.dump();

        std::string err;
        if (!g_nlp.load_rules_from_string(rulesText, &err))
        {
            LOG_ERROR("NLP", "Failed to load NLP rules: " + err);
            return false;
        }

        LOG_DEBUG("NLP", "Loaded " + std::to_string(g_nlp.rule_count()) +
                          " rules from " + path);
        return true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("NLP", std::string("Failed to parse NLP rules: ") + e.what());
        return false;
    }
}
