#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>

struct AIStyle {
    std::string tone;
    double optimism_level;
    std::vector<std::string> encouragement_phrases;
    std::string formality;
};

AIStyle loadAIStyle(const std::string& filename) {
    AIStyle style;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open AI style config file\n";
        return style;
    }
    nlohmann::json j;
    file >> j;

    style.tone = j.value("tone", "neutral");
    style.optimism_level = j.value("optimism_level", 0.5);
    style.encouragement_phrases = j.value("encouragement_phrases", std::vector<std::string>{});
    style.formality = j.value("formality", "neutral");
    return style;
}
