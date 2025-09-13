#include "response_manager.hpp"
#include <unordered_map>
#include <random>

// Simple random picker
static std::string pickRandom(const std::vector<std::string>& options) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, static_cast<int>(options.size()) - 1);
    return options[dist(gen)];
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
    { "set_timer", {
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
    { "show_help", {
        "Here are the available commands.",
        "These are the commands you can use.",
        "Listing all supported commands now."
    }},

    // --- Filesystem ---
    { "show_pwd", {
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
    { "make_dir_success", {
        "Created directory ",
        "New folder created: ",
        "Made a directory at "
    }},
    { "make_dir_fail", {
        "Failed to create directory ",
        "Couldn’t make folder: ",
        "Unable to create directory: "
    }},
    { "remove_file_success", {
        "Removed ",
        "Deleted ",
        "Successfully removed "
    }},
    { "remove_file_fail", {
        "Failed to remove ",
        "Couldn’t delete ",
        "Unable to remove "
    }},

    // --- NLP / AI ---
    { "reload_nlp_success", {
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
    { "voice_start", {
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
    { "voice_stream_start", {
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
};

std::string ResponseManager::get(const std::string& intent) {
    auto it = responses.find(intent);
    if (it != responses.end() && !it->second.empty()) {
        return pickRandom(it->second);
    }
    // fallback default
    return "[no response configured]";
}
