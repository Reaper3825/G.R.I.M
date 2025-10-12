#pragma once
#include <string>
#include <ctime>
#include <nlohmann/json.hpp>

namespace GRIM {

enum class Mood {
    Calm,
    Focused,
    Curious,
    Irritated,
    Playful,
    Tired
};

struct PersonalityState {
    Mood mood = Mood::Calm;
    float energy = 0.75f;
    float confidence = 0.75f;
    std::time_t lastUpdate = std::time(nullptr);
};

class PersonalityManager {
public:
    static void init(nlohmann::json& memory);
    static void updateAfterCommand(bool success);
    static void decayOverTime();
    static std::string moodToString(Mood mood);
    static std::string generatePrefix();  // influences AI tone
    static bool isStable();
    static PersonalityState& get();
    static void save(nlohmann::json& memory);

private:
    static PersonalityState state;
};

} // namespace GRIM
