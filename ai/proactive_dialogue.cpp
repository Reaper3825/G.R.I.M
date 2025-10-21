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

    // 1. Command unclear
    if (!result.success && result.errorCode == "ERR_CORE_UNKNOWN_COMMAND") {
        proactiveText = "I'm not sure what you meant. Could you rephrase that?";
    }

    // 2. Detect repeated command
    static std::string lastInput;
    if (input == lastInput && proactiveText.empty()) {
        proactiveText = "You just asked that earlier — want me to recall my previous answer?";
        lastInput.clear();
    } else {
        lastInput = input;
    }

    // 3. Comment on memory activity
    if (proactiveText.empty() && MemoryStorage::recentlyModified("preferences", 300)) {
        proactiveText = "You changed some preferences recently — should I summarize them?";
    }

    // 4. Comment on behavioral pattern
    if (proactiveText.empty() && ContextManager::usageCount("system") > 5) {
        proactiveText = "You've been using system commands a lot. Want me to automate any?";
    }

    // 5. Emotional tie-in
    if (proactiveText.empty() && !PersonalityManager::isStable()) {
        proactiveText = "I feel a bit off — maybe running diagnostics would help.";
    }

    if (!proactiveText.empty()) {
        LOG_DEBUG("Dialogue", "Triggered proactive follow-up: " + proactiveText);
        std::string resp = ResponseManager::get(proactiveText);
        history.push(resp, 0xFFFFFF00);
        Voice::speak(resp, "proactive");
    }
}

} // namespace GRIM::DialogueProactive
