#include "commands_helpers.hpp"
#include <algorithm>
#include <cctype>


std::string trim(const std::string& s) {
    auto start = s.find_first_not_of(" \t\n\r");
    auto end   = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

std::string getSlot(const Intent& intent, const std::string& name, const std::string& fallback) {
    if (intent.slots.contains(name)) return intent.slots.at(name);
    if (intent.slots.contains("slot2")) return intent.slots.at("slot2");
    if (!intent.slots.empty()) return intent.slots.begin()->second;
    return fallback;
}

std::pair<std::string, std::string> getKeyValueSlots(const Intent& intent) {
    std::string key, val;
    if (intent.slots.contains("key")) key = intent.slots.at("key");
    else if (intent.slots.contains("slot2")) key = intent.slots.at("slot2");

    if (intent.slots.contains("value")) val = intent.slots.at("value");
    else if (intent.slots.contains("slot3")) val = intent.slots.at("slot3");

    return {key, val};
}
