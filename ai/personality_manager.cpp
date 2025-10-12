#include "personality_manager.hpp"
#include "logger.hpp"
#include <algorithm>

using namespace GRIM;

bool PersonalityManager::isStable() {
    return state.energy > 0.4f && state.confidence > 0.4f;
}

PersonalityState PersonalityManager::state;

void PersonalityManager::init(nlohmann::json& memory) {
    if (memory.contains("personality")) {
        auto j = memory["personality"];
        std::string m = j.value("mood", "calm");
        if (m == "focused") state.mood = Mood::Focused;
        else if (m == "curious") state.mood = Mood::Curious;
        else if (m == "irritated") state.mood = Mood::Irritated;
        else if (m == "playful") state.mood = Mood::Playful;
        else if (m == "tired") state.mood = Mood::Tired;
        else state.mood = Mood::Calm;

        state.energy = j.value("energy", 0.75f);
        state.confidence = j.value("confidence", 0.75f);
    }
}

void PersonalityManager::save(nlohmann::json& memory) {
    memory["personality"] = {
        {"mood", moodToString(state.mood)},
        {"energy", state.energy},
        {"confidence", state.confidence}
    };
}

void PersonalityManager::updateAfterCommand(bool success) {
    extern nlohmann::json longTermMemory;  // ✅ access global memory safely

    if (success) {
        state.confidence = std::min(1.0f, state.confidence + 0.05f);
        state.energy     = std::min(1.0f, state.energy + 0.02f);
    } else {
        state.confidence = std::max(0.0f, state.confidence - 0.05f);
        state.energy     = std::max(0.0f, state.energy - 0.05f);
    }

    // Mood logic
    if (state.confidence < 0.3f)
        state.mood = Mood::Tired;
    else if (state.energy < 0.3f)
        state.mood = Mood::Irritated;
    else if (state.confidence > 0.85f && state.energy > 0.7f)
        state.mood = Mood::Playful;
    else
        state.mood = Mood::Calm;

    save(longTermMemory); // ✅ use the global json
}


void PersonalityManager::decayOverTime() {
    std::time_t now = std::time(nullptr);
    if (difftime(now, state.lastUpdate) > 60) {
        state.energy = std::max(0.0f, state.energy - 0.01f);
        state.confidence = std::max(0.0f, state.confidence - 0.005f);
        state.lastUpdate = now;
    }
}

std::string PersonalityManager::moodToString(Mood mood) {
    switch (mood) {
        case Mood::Focused: return "focused";
        case Mood::Curious: return "curious";
        case Mood::Irritated: return "irritated";
        case Mood::Playful: return "playful";
        case Mood::Tired: return "tired";
        default: return "calm";
    }
}

std::string PersonalityManager::generatePrefix() {
    switch (state.mood) {
        case Mood::Playful: return "😏 (Playful tone)";
        case Mood::Curious: return "🤔 (Curious tone)";
        case Mood::Irritated: return "😒 (Short tone)";
        case Mood::Tired:    return "😴 (Low energy)";
        default: return "";
    }
}

PersonalityState& PersonalityManager::get() {
    return state;
}
