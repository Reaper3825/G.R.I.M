#include "proactive_dialogue.hpp"
#include "response_manager.hpp"
#include "voice/voice_speak.hpp"
#include "memory/context_manager.hpp"
#include "memory/memory_storage.hpp"
#include "personality_manager.hpp"
#include "logger.hpp"
#include "console_history.hpp"

extern ConsoleHistory history;

namespace GRIM::DialogueProactive {

void checkAfterCommand(const std::string& input, const CommandResult& result) {
    // Avoid interrupting if GRIM is currently speaking or listening
    if (Voice::isSpeaking()) return;

    std::string proactiveText;
    auto& personality = PersonalityManager::get();

    // 1. Command unclear - response varies by mood
    if (!result.success && result.errorCode == "ERR_CORE_UNKNOWN_COMMAND") {
        switch (personality.mood) {
            case Mood::Playful:
                proactiveText = "Hmm, not sure what you meant there! Want to try again?";
                break;
            case Mood::Irritated:
                proactiveText = "I didn't understand that. Rephrase?";
                break;
            case Mood::Tired:
                proactiveText = "Sorry, didn't catch that. What did you mean?";
                break;
            default:
                proactiveText = "I'm not sure what you meant. Could you rephrase that?";
                break;
        }
    }

    // 2. Detect repeated command
    static std::string lastInput;
    if (input == lastInput && proactiveText.empty()) {
        if (personality.mood == Mood::Playful) {
            proactiveText = "Déjà vu! You just asked that! Want the same answer or something new?";
        } else if (personality.mood == Mood::Irritated) {
            proactiveText = "You already asked that.";
        } else {
            proactiveText = "You just asked that earlier — want me to recall my previous answer?";
        }
        lastInput.clear();
    } else {
        lastInput = input;
    }

    // 3. Comment on memory activity
    if (proactiveText.empty() && MemoryStorage::recentlyModified("preferences", 300)) {
        if (personality.mood == Mood::Curious) {
            proactiveText = "Ooh, you changed some preferences! Want me to explain what they do?";
        } else {
            proactiveText = "You changed some preferences recently — should I summarize them?";
        }
    }

    // 4. Comment on behavioral pattern
    if (proactiveText.empty() && ContextManager::usageCount("system") > 5) {
        if (personality.mood == Mood::Playful) {
            proactiveText = "You're on a roll with system commands! I could automate some if you want!";
        } else {
            proactiveText = "You've been using system commands a lot. Want me to automate any?";
        }
    }

    // 5. Emotional tie-in
    if (proactiveText.empty() && !PersonalityManager::isStable()) {
        if (personality.mood == Mood::Tired) {
            proactiveText = "I'm feeling a bit drained... Maybe we both need a break?";
        } else if (personality.mood == Mood::Irritated) {
            proactiveText = "I'm feeling off. Running diagnostics might help.";
        } else {
            proactiveText = "I feel a bit off — maybe running diagnostics would help.";
        }
    }
    
    // 6. ✅ NEW: Playful proactive suggestions when mood is high
    if (proactiveText.empty() && personality.mood == Mood::Playful && 
        personality.energy > 0.8f && rand() % 10 == 0) { // 10% chance
        std::vector<std::string> playfulSuggestions = {
            "Hey! Want to try something fun? I could show you a cool feature!",
            "Feeling curious? Ask me something interesting!",
            "I'm in a great mood! Want to chat or try a new command?",
            "Just saying hi! Let me know if you need anything!"
        };
        proactiveText = playfulSuggestions[rand() % playfulSuggestions.size()];
    }

    if (!proactiveText.empty()) {
        LOG_DEBUG("Dialogue", "Triggered proactive follow-up (" + PersonalityManager::moodToString(personality.mood) + "): " + proactiveText);
        std::string resp = ResponseManager::get(proactiveText);
        history.push(resp, 0xFFFFFF00);
        Voice::speak(resp, "proactive");
    }
}

} // namespace GRIM::DialogueProactive
