#include "commands.hpp"
#include "ai.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <algorithm>
#include <system_error>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

namespace fs = std::filesystem;

// External memory
std::deque<std::string> contextMemory;
const size_t kMaxContext = 10;
nlohmann::json longTermMemory;

// Normalize a memory key
std::string normalizeKey(std::string key) {
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c){ return std::tolower(c); });
    key.erase(key.begin(), std::find_if(key.begin(), key.end(), [](unsigned char ch){ return !std::isspace(ch); }));
    key.erase(std::find_if(key.rbegin(), key.rend(), [](unsigned char ch){ return !std::isspace(ch); }).base(), key.end());
    std::replace(key.begin(), key.end(), '_', ' ');
    if (key.rfind("my ", 0) == 0) key = key.substr(3);
    return key;
}

// Save/load memory
void loadMemory() {
    std::ifstream in("grim_memory.json");
    if (in.is_open()) in >> longTermMemory;
    else longTermMemory = nlohmann::json::object();
}
void saveMemory() {
    std::ofstream out("grim_memory.json");
    out << longTermMemory.dump(4);
}
void saveToMemory(const std::string& line) {
    if (contextMemory.size() >= kMaxContext) contextMemory.pop_front();
    contextMemory.push_back(line);
}
std::string buildContextPrompt(const std::string& query) {
    std::ostringstream oss;
    for (auto& entry : contextMemory) oss << entry << "\n";
    oss << "User: " << query << "\nGRIM:";
    return oss.str();
}

// ========================= HANDLERS =========================
void handleCommand(
    const Intent& intent,
    std::string& buffer,
    fs::path& currentDir,
    std::vector<Timer>& timers,
    nlohmann::json& longTermMemory,
    NLP& nlp,
    ConsoleHistory& history
) {
    auto addHistory = [&](const std::string& s, sf::Color c = sf::Color::White) {
        history.push(s, c);        // add to UI history
        std::cout << s << std::endl; // also log to console
    };

    if (!intent.matched) {
        addHistory("[NLP] No intent matched.", sf::Color(255,200,140));
        return;
    }

    if (intent.name == "open_app") {
        auto it = intent.slots.find("app");
        if (it != intent.slots.end()) {
            std::string app = it->second;
            std::string resolved = resolveAlias(app);
            if (resolved.empty()) resolved = app;
#ifdef _WIN32
            STARTUPINFOA si = { sizeof(si) };
            PROCESS_INFORMATION pi;
            BOOL success = CreateProcessA(resolved.c_str(), NULL, NULL, NULL, FALSE,
                                          DETACHED_PROCESS, NULL, NULL, &si, &pi);
            if (!success) addHistory("Failed to open app: " + resolved, sf::Color(255,140,140));
            else { CloseHandle(pi.hProcess); CloseHandle(pi.hThread); }
#else
            int ret = system(resolved.c_str());
            if (ret != 0) addHistory("Failed to open app: " + resolved, sf::Color(255,140,140));
#endif
        }
    }
    else if (intent.name == "search_web") {
        auto it = intent.slots.find("query");
        if (it != intent.slots.end()) {
            try {
                std::string prompt = "Search the web and summarize results for: " + it->second;
                std::string reply = callAI(prompt);
                addHistory("[Web Result] " + reply, sf::Color(180,200,255));
            } catch (const std::exception& e) {
                addHistory(std::string("[Web] Error: ") + e.what(), sf::Color(255,140,140));
            }
        }
    }
    else if (intent.name == "set_timer") {
    history.push("[Debug] Slots captured:", sf::Color(200,200,200));
    for (auto& kv : intent.slots) {
        history.push(" - " + kv.first + " = " + kv.second, sf::Color(200,200,200));
    }

    try {
        int value = std::stoi(intent.slots.at("value"));
        std::string unit = intent.slots.at("unit");

        int seconds = value;
        if (unit == "minute" || unit == "minutes" || unit == "min" || unit == "m") {
            seconds = value * 60;
        }
        else if (unit == "hour" || unit == "hours" || unit == "hr" || unit == "h") {
            seconds = value * 3600;
        }
        // else assume "seconds" or "s" → already in seconds

        Timer t;
        t.seconds = seconds;
        timers.push_back(t);

        history.push("[Timer] Set a timer for " + std::to_string(value) + " " + unit, sf::Color(255,200,0));
    }
    catch (const std::exception& e) {
        history.push(std::string("[Timer] Error: ") + e.what(), sf::Color::Red);
    }
}



    else if (intent.name == "clean") {
        addHistory("[Clean] Preview/confirm/purge logic goes here.", sf::Color(200,200,255));
    }
    else if (intent.name == "show_help") {
        addHistory("GRIM Command Reference", sf::Color(200,200,255));
        addHistory("------------------------", sf::Color(120,120,135));
        addHistory("help, pwd, cd, ls, mkdir, rm, clean, grim, search, timer, reloadnlp, quit", sf::Color(180,220,255));
    }
    else if (intent.name == "show_pwd") {
        addHistory(currentDir.string());
    }
    else if (intent.name == "change_dir") {
        auto it=intent.slots.find("path");
        if(it!=intent.slots.end()) {
            fs::path newPath = currentDir / it->second;
            std::error_code ec;
            if(fs::exists(newPath,ec) && fs::is_directory(newPath,ec)) {
                currentDir = newPath;
                fs::current_path(currentDir, ec);
                addHistory("[cd] Changed directory to " + currentDir.string());
            } else {
                addHistory("[cd] Directory not found: " + it->second, sf::Color(255,140,140));
            }
        }
    }
    else if (intent.name == "list_dir") {
        std::error_code ec;
        addHistory("[ls] Listing contents of: " + currentDir.string());
        for(auto& e : fs::directory_iterator(currentDir, ec)) {
            addHistory("  " + e.path().filename().string());
        }
    }
    else if (intent.name == "make_dir") {
        auto it=intent.slots.find("dirname");
        if(it!=intent.slots.end()) {
            std::error_code ec;
            fs::path newDir = currentDir / it->second;
            if(fs::create_directory(newDir, ec)) {
                addHistory("[mkdir] Created: " + newDir.string());
            } else {
                addHistory("[mkdir] Failed to create: " + newDir.string(), sf::Color(255,140,140));
            }
        }
    }
    else if (intent.name == "remove_file") {
        auto it=intent.slots.find("target");
        if(it!=intent.slots.end()) {
            std::error_code ec;
            fs::path target = currentDir / it->second;
            if(fs::remove_all(target, ec) > 0) {
                addHistory("[rm] Removed: " + target.string());
            } else {
                addHistory("[rm] Failed to remove: " + target.string(), sf::Color(255,140,140));
            }
        }
    }
    else if (intent.name == "reload_nlp") {
        std::string err;
        bool ok = nlp.load_rules("nlp_rules.json", &err);
        if(ok) addHistory("[NLP] Reloaded rules.");
        else   addHistory("[NLP] Reload failed: " + err, sf::Color(255,140,140));
    }
    else if (intent.name == "grim_ai") {
        auto it=intent.slots.find("query");
        if(it!=intent.slots.end()) {
            try {
                std::string prompt = buildContextPrompt(it->second);
                std::string reply = callAI(prompt);
                saveToMemory("User: " + it->second);
                saveToMemory("GRIM: " + reply);
                addHistory("[AI] " + reply, sf::Color(180,200,255));
            } catch(const std::exception& e) {
                addHistory(std::string("[AI] Error: ") + e.what(), sf::Color(255,140,140));
            }
        }
    }
    else if (intent.name == "remember") {
        auto itKey = intent.slots.find("key");
        auto itVal = intent.slots.find("value");
        if (itKey != intent.slots.end() && itVal != intent.slots.end()) {
            std::string key = normalizeKey(itKey->second);
            longTermMemory[key] = itVal->second;
            saveMemory();
            addHistory("🧠 Remembered: " + key + " = " + itVal->second, sf::Color(180,255,180));
        }
    }
    else if (intent.name == "recall") {
        auto it = intent.slots.find("key");
        if (it != intent.slots.end()) {
            std::string key = normalizeKey(it->second);
            if (longTermMemory.contains(key)) {
                addHistory("Your " + key + " is " + longTermMemory[key].dump(), sf::Color(180,200,255));
            } else {
                addHistory("I don’t remember anything about " + key, sf::Color(255,200,120));
            }
        } else {
            addHistory("Here's what I remember:", sf::Color(200,200,255));
            for (auto& [k,v] : longTermMemory.items()) {
                addHistory(" - " + k + " = " + v.dump(), sf::Color(180,200,255));
            }
        }
    }
    else if (intent.name == "forget") {
        auto it = intent.slots.find("key");
        if (it != intent.slots.end()) {
            std::string key = normalizeKey(it->second);
            if (longTermMemory.contains(key)) {
                longTermMemory.erase(key);
                saveMemory();
                addHistory("🗑️ Forgot: " + key, sf::Color(255,180,180));
            } else {
                addHistory("⚠️ I don’t remember anything about " + key, sf::Color(255,200,120));
            }
        }
    }
}
