#include <unordered_map>
#include <random>
#include <ctime>
#include <deque>
#include <mutex>
#include "response_manager.hpp"
#include "error_manager.hpp"
#include "voice/voice_speak.hpp"
#include "resources.hpp"

CommandResult ResponseManager::systemMessage(const std::string& msg,
                                             const Color& color) {
    history.push(msg, (color.a << 24) | (color.b << 16) | (color.g << 8) | color.r);
    Voice::speak(msg, "system");

    return {
        true,             // success
        msg,              // message
        "ERR_NONE",       // errorCode
        "system",         // category
        "System message", // voice
        color             // color
    };
}





// Response history tracking to avoid repetition
static std::unordered_map<std::string, std::deque<size_t>> responseHistory;
static std::mutex responseHistoryMutex;
static const size_t MAX_HISTORY_SIZE = 3;

// Simple random picker with history awareness
static std::string pickRandom(const std::string& key, const std::vector<std::string>& options) {
    if (options.empty()) return "";
    if (options.size() == 1) return options[0];

    static std::random_device rd;
    static std::mt19937 gen(rd());
    
    std::lock_guard<std::mutex> lock(responseHistoryMutex);
    
    // Get history for this key
    auto& history = responseHistory[key];
    
    // Try to pick a response not in recent history
    std::vector<size_t> availableIndices;
    for (size_t i = 0; i < options.size(); ++i) {
        bool inHistory = false;
        for (size_t histIdx : history) {
            if (histIdx == i) {
                inHistory = true;
                break;
            }
        }
        if (!inHistory) {
            availableIndices.push_back(i);
        }
    }
    
    // If all responses are in history, clear it and use all options
    if (availableIndices.empty()) {
        availableIndices.clear();
        for (size_t i = 0; i < options.size(); ++i) {
            availableIndices.push_back(i);
        }
        history.clear();
    }
    
    // Pick a random index from available options
    std::uniform_int_distribution<> dist(0, static_cast<int>(availableIndices.size()) - 1);
    size_t chosenIdx = availableIndices[dist(gen)];
    
    // Update history
    history.push_back(chosenIdx);
    if (history.size() > MAX_HISTORY_SIZE) {
        history.pop_front();
    }
    
    return options[chosenIdx];
}

// Response database
static std::unordered_map<std::string, std::vector<std::string>> responses = {
    // --- General ---
    { "unrecognized", {
        "Sorry, I didn’t understand: ",
        "Hmm, that didn’t sound like a command: ",
        "I’m not sure what you meant by: "
    }},
    { "no_match", {
        "No matching command found.",
        "That doesn’t match anything I know.",
        "I couldn’t map that to a command."
    }},

    // --- App / Web ---
    { "open_app_success", {
        "Opened ",
        "Launching ",
        "Here we go — opening "
    }},
    { "open_app_fail", {
        "Failed to open ",
        "Couldn’t launch ",
        "I wasn’t able to start "
    }},
    { "open_app_no_name", {
        "No application name detected.",
        "I need an app name for that.",
        "Couldn’t tell which app to open."
    }},
    { "search_web", {
        "Searching the web for ",
        "Looking that up online: ",
        "On it, searching for "
    }},

    // --- Timers ---
    { "timer", {
        "Timer set for ",
        "Alright, I’ll count down ",
        "Got it — timer started for "
    }},

    // --- Console ---
    { "clean", {
        "History cleared.",
        "Console wiped clean.",
        "All previous entries removed."
    }},
    { "help", {
        "Here are the available commands.",
        "These are the commands you can use.",
        "Listing all supported commands now."
    }},

    // --- Filesystem ---
    { "pwd", {
        "Current directory is ",
        "You’re currently in ",
        "Working directory: "
    }},
    { "change_dir_success", {
        "Changed directory to ",
        "Now working in ",
        "Switched folder to "
    }},
    { "change_dir_fail", {
        "Failed to change directory: ",
        "Couldn’t move into that folder: ",
        "Unable to switch directory: "
    }},
    { "mkdir", {
        "Created directory ",
        "New folder created: ",
        "Made a directory at "
    }},
    { "mkdir_fail", {
        "Failed to create directory ",
        "Couldn’t make folder: ",
        "Unable to create directory: "
    }},
    { "rm", {
        "Removed ",
        "Deleted ",
        "Successfully removed "
    }},
    { "rm_fail", {
        "Failed to remove ",
        "Couldn’t delete ",
        "Unable to remove "
    }},

    // --- NLP / AI ---
    { "reload_nlp", {
        "NLP rules reloaded.",
        "Language rules refreshed.",
        "Rule set reloaded successfully."
    }},
    { "reload_nlp_fail", {
        "Reload failed: ",
        "Couldn’t reload NLP rules: ",
        "Rule reload error: "
    }},
    { "grim_ai_no_response", {
        "I didn’t generate a response.",
        "No reply came through this time.",
        "I wasn’t able to respond."
    }},
    { "grim_ai_no_query", {
        "No query provided.",
        "I didn’t catch a question to answer.",
        "Nothing to respond to."
    }},

    // --- Memory ---
    { "remember", {
        "Remembered: ",
        "Got it — I’ll remember ",
        "Saved to memory: "
    }},
    { "remember_fail", {
        "Missing key or value for remember.",
        "Couldn’t save — key or value is missing.",
        "I need both a key and a value to remember."
    }},
    { "recall", {
        "I recall ",
        "From memory: ",
        "I’ve got this saved: "
    }},
    { "recall_unknown", {
        "I don’t know ",
        "Nothing saved for ",
        "I couldn’t find anything about "
    }},
    { "recall_no_key", {
        "No key provided for recall.",
        "You didn’t tell me what to recall.",
        "I need a key to look up."
    }},
    { "forget", {
        "Forgotten: ",
        "I’ve removed ",
        "No longer remembering "
    }},
    { "forget_unknown", {
        "I didn’t know ",
        "That wasn’t in memory: ",
        "Couldn’t forget — nothing stored for "
    }},
    { "forget_no_key", {
        "No key provided for forget.",
        "I need a key to remove from memory.",
        "Can’t forget without a name."
    }},

    // --- Voice ---
    { "voice", {
        "Starting a 5-second recording…",
        "Listening now… go ahead.",
        "I’m ready, start speaking."
    }},
    { "voice_heard", {
        "I heard you say: ",
        "Got it, you said: ",
        "Recognized speech: "
    }},
    { "voice_none", {
        "I didn’t catch that.",
        "No speech detected.",
        "Hmm, I couldn’t hear anything."
    }},

    // --- Voice Stream ---
    { "voice_stream", {
        "Starting live microphone stream…",
        "Live voice stream active now.",
        "Okay, streaming microphone input."
    }},
    { "voice_stream_stop", {
        "Stopping live microphone stream…",
        "Live voice stream halted.",
        "Mic stream stopped."
    }},

    // --- Startup ---
    { "startup", {
        "GRIM is ready to go!",
        "All systems online.",
        "Boot complete. Let’s roll."
    }},

    // --- Greetings (Time-based) ---
    { "greeting_morning", {
        "Good morning!",
        "Morning! Ready to assist.",
        "Rise and shine! What can I do for you?"
    }},
    { "greeting_afternoon", {
        "Good afternoon!",
        "Afternoon! How can I help?",
        "Hey there! What's on the agenda?"
    }},
    { "greeting_evening", {
        "Good evening!",
        "Evening! What do you need?",
        "Hello! Ready for the evening."
    }},
    { "greeting_night", {
        "Still up? I'm here if you need me.",
        "Late night session? Let's do this.",
        "Working late? I've got your back."
    }},

    // --- Acknowledgments ---
    { "ack_understood", {
        "Got it.",
        "Understood.",
        "Okay.",
        "Sure thing."
    }},
    { "ack_working", {
        "Working on it...",
        "Give me a moment...",
        "One second...",
        "Processing..."
    }},
    { "ack_done", {
        "Done.",
        "All set.",
        "Complete.",
        "Finished."
    }},
};

std::string ResponseManager::get(const std::string& keyOrMessage) {
    auto it = responses.find(keyOrMessage);
    if (it != responses.end() && !it->second.empty()) {
        return pickRandom(keyOrMessage, it->second);
    }

    // ✅ FIX: If it already looks like a full message, return it as-is
    // Check for: starts with [, has newlines, contains spaces (likely a sentence), or doesn't start with lowercase
    if (!keyOrMessage.empty()) {
        // Already formatted messages
        if (keyOrMessage[0] == '[' || keyOrMessage.find('\n') != std::string::npos) {
            return keyOrMessage;
        }
        
        // ✅ NEW: Check if it's a full sentence (contains multiple words or ends with punctuation)
        bool hasSpaces = (keyOrMessage.find(' ') != std::string::npos);
        bool endsWithPunct = (!keyOrMessage.empty() && 
                             (keyOrMessage.back() == '.' || 
                              keyOrMessage.back() == '!' || 
                              keyOrMessage.back() == '?'));
        
        // If it has spaces or ends with punctuation, it's probably a complete message
        if (hasSpaces || endsWithPunct) {
            return keyOrMessage;
        }
    }

    // Otherwise, treat as an unknown intent key and fallback gracefully
    return ErrorManager::getUserMessage("ERR_CORE_UNKNOWN_COMMAND") + " (" + keyOrMessage + ")";
}

// Get a response with parameter substitution
std::string ResponseManager::getWithParams(const std::string& key,
                                          const std::unordered_map<std::string, std::string>& params) {
    std::string response = get(key);
    
    // Replace {param_name} with actual values
    for (const auto& [paramName, paramValue] : params) {
        std::string placeholder = "{" + paramName + "}";
        size_t pos = 0;
        while ((pos = response.find(placeholder, pos)) != std::string::npos) {
            response.replace(pos, placeholder.length(), paramValue);
            pos += paramValue.length();
        }
    }
    
    return response;
}

// Get contextual greeting based on time of day
std::string ResponseManager::getGreeting() {
    std::time_t now = std::time(nullptr);
    
    // Thread-safe time conversion
    std::tm localTimeBuf;
#ifdef _WIN32
    localtime_s(&localTimeBuf, &now);
#else
    localtime_r(&now, &localTimeBuf);
#endif
    
    int hour = localTimeBuf.tm_hour;
    
    std::string key;
    if (hour >= 5 && hour < 12) {
        key = "greeting_morning";
    } else if (hour >= 12 && hour < 17) {
        key = "greeting_afternoon";
    } else if (hour >= 17 && hour < 22) {
        key = "greeting_evening";
    } else {
        key = "greeting_night";
    }
    
    return get(key);
}

// Clear response history
void ResponseManager::clearHistory() {
    std::lock_guard<std::mutex> lock(responseHistoryMutex);
    responseHistory.clear();
}

