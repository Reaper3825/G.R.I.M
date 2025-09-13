#include "commands.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#include "ai.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "voice_speak.hpp"
#include "resources.hpp"
#include "system_detect.hpp"
#include "response_manager.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <fstream>


#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <psapi.h>
#include <winternl.h>
#endif

namespace fs = std::filesystem;

// ---------------- Helpers ----------------

static void parseAndDispatch(const std::string& text,
                             std::string& buffer,
                             fs::path& currentDir,
                             std::vector<Timer>& timers,
                             nlohmann::json& longTermMemory,
                             NLP& nlp,
                             ConsoleHistory& history) {
    Intent intent = nlp.parse(text);
    if (intent.matched) {
        handleCommand(intent, buffer, currentDir, timers, longTermMemory, nlp, history);
    } else {
        history.push(ResponseManager::get("unrecognized") + text, sf::Color::Red);
        speak(ResponseManager::get("unrecognized") + text, "routine");
    }
}

#ifdef _WIN32
// --- CPU usage ---
static ULARGE_INTEGER lastIdleTime   = {};
static ULARGE_INTEGER lastKernelTime = {};
static ULARGE_INTEGER lastUserTime   = {};

static double getCPUUsage() {
    FILETIME idleTime, kernelTime, userTime;
    if (!GetSystemTimes(&idleTime, &kernelTime, &userTime))
        return -1.0;

    ULARGE_INTEGER idle, kernel, user;
    idle.LowPart   = idleTime.dwLowDateTime;
    idle.HighPart  = idleTime.dwHighDateTime;
    kernel.LowPart = kernelTime.dwLowDateTime;
    kernel.HighPart= kernelTime.dwHighDateTime;
    user.LowPart   = userTime.dwLowDateTime;
    user.HighPart  = userTime.dwHighDateTime;

    ULONGLONG idleDiff   = idle.QuadPart   - lastIdleTime.QuadPart;
    ULONGLONG kernelDiff = kernel.QuadPart - lastKernelTime.QuadPart;
    ULONGLONG userDiff   = user.QuadPart   - lastUserTime.QuadPart;
    ULONGLONG sysTotal   = kernelDiff + userDiff;

    lastIdleTime   = idle;
    lastKernelTime = kernel;
    lastUserTime   = user;

    if (sysTotal == 0) return 0.0;
    return (double)(sysTotal - idleDiff) * 100.0 / (double)sysTotal;
}

// --- Memory usage ---
static double getMemoryUsagePercent(DWORDLONG &usedMB, DWORDLONG &totalMB) {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);

    totalMB = memInfo.ullTotalPhys / (1024 * 1024);
    usedMB  = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024 * 1024);

    return (double)usedMB * 100.0 / (double)totalMB;
}

// --- Resolve exe on PATH ---
static std::string findOnPath(const std::string& app) {
    char buffer[MAX_PATH];
    DWORD result = SearchPathA(NULL, app.c_str(), ".exe", MAX_PATH, buffer, NULL);
    if (result > 0 && result < MAX_PATH) {
        return std::string(buffer);
    }
    return app; // fallback
}
#endif // _WIN32

bool handleCommand(const Intent& intent,
                   std::string& buffer,
                   fs::path& currentDir,
                   std::vector<Timer>& timers,
                   nlohmann::json& longTermMemory,
                   NLP& nlp,
                   ConsoleHistory& history) {
    if (!intent.matched) {
        history.push(ResponseManager::get("no_match"), sf::Color(200,200,200));
        speak(ResponseManager::get("no_match"), "routine");
        std::cout << "[DEBUG][Command] No intent matched for input: '" << buffer
                  << "' (rules loaded=" << nlp.rule_count() << ")\n";
        return true;
    }

    const std::string& name = intent.name;

    // ---- Intent Debug ----
    std::cout << "[DEBUG][Command] Intent received: '" << name
              << "' (score=" << intent.score
              << ", slots=" << intent.slots.size() << ")\n";

    for (const auto& [k, v] : intent.slots) {
        std::cout << "    slot[" << k << "] = '" << v << "'\n";
    }

    // ---- App / Web ----
    if (name == "open_app") {
        std::cout << "[DEBUG][Command] Dispatch: open_app\n";

        std::string appName;
        if (intent.slots.contains("app")) appName = intent.slots.at("app");
        else if (intent.slots.contains("slot2")) appName = intent.slots.at("slot2");
        else if (!intent.slots.empty()) appName = intent.slots.begin()->second;

        if (!appName.empty()) {
            std::string resolved = resolveAlias(appName);
            if (!resolved.empty()) appName = resolved;

    #ifdef _WIN32
            if (appName.find(':') == std::string::npos &&
                appName.find('/') == std::string::npos) {
                appName = findOnPath(appName);
            }

            if (appName.ends_with(".exe")) {
                STARTUPINFOA si{}; PROCESS_INFORMATION pi{}; si.cb = sizeof(si);
                if (CreateProcessA(appName.c_str(), NULL, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
                    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
                    history.push(ResponseManager::get("open_app_success") + appName, sf::Color::Green);
                    speak(ResponseManager::get("open_app_success") + appName, "routine");
                } else {
                    history.push(ResponseManager::get("open_app_fail") + appName, sf::Color::Red);
                    speak(ResponseManager::get("open_app_fail") + appName, "routine");
                }
            } else {
                HINSTANCE result = ShellExecuteA(NULL, "open", appName.c_str(), NULL, NULL, SW_SHOWNORMAL);
                if ((INT_PTR)result <= 32) {
                    history.push(ResponseManager::get("open_app_fail") + appName, sf::Color::Red);
                    speak(ResponseManager::get("open_app_fail") + appName, "routine");
                } else {
                    history.push(ResponseManager::get("open_app_success") + appName, sf::Color::Green);
                    speak(ResponseManager::get("open_app_success") + appName, "routine");
                }
            }
    #else
            int rc = system(appName.c_str());
            if (rc != 0) {
                history.push(ResponseManager::get("open_app_fail") + appName, sf::Color::Red);
                speak(ResponseManager::get("open_app_fail") + appName, "routine");
            } else {
                history.push(ResponseManager::get("open_app_success") + appName, sf::Color::Green);
                speak(ResponseManager::get("open_app_success") + appName, "routine");
            }
    #endif
        } else {
            history.push(ResponseManager::get("open_app_no_name"), sf::Color::Red);
            speak(ResponseManager::get("open_app_no_name"), "routine");
        }
        return true;
    }

    if (name == "search_web") {
        std::cout << "[DEBUG][Command] Dispatch: search_web\n";
        std::string q;
        if (intent.slots.contains("query")) q = intent.slots.at("query");
        else if (intent.slots.contains("slot2")) q = intent.slots.at("slot2");
        else if (!intent.slots.empty()) q = intent.slots.begin()->second;

        if (!q.empty()) {
    #ifdef _WIN32
            std::string url = "https://www.google.com/search?q=" + q;
            ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
    #endif
            history.push(ResponseManager::get("search_web") + q, sf::Color::Cyan);
            speak(ResponseManager::get("search_web") + q, "routine");
        }
        return true;
    }

    // ---- Timers ----
    if (name == "set_timer") {
        std::cout << "[DEBUG][Command] Dispatch: set_timer\n";

        int value = 0;
        std::string unit = "s";
        if (intent.slots.contains("value")) value = std::stoi(intent.slots.at("value"));
        else if (intent.slots.contains("slot2")) value = std::stoi(intent.slots.at("slot2"));

        if (intent.slots.contains("unit")) unit = intent.slots.at("unit");
        else if (intent.slots.contains("slot3")) unit = intent.slots.at("slot3");

        int seconds = value;
        if (unit.starts_with("m")) seconds *= 60;
        else if (unit.starts_with("h")) seconds *= 3600;

        timers.push_back(Timer{seconds, sf::Clock(), false});
        history.push(ResponseManager::get("set_timer") + std::to_string(seconds) + " seconds.", sf::Color(255,200,0));
        speak(ResponseManager::get("set_timer") + std::to_string(seconds) + " seconds", "reminder");
        return true;
    }

    // ---- Console ----
    if (name == "clean") {
        std::cout << "[DEBUG][Command] Dispatch: clean\n";
        history.clear();
        history.push(ResponseManager::get("clean"), sf::Color::Green);
        speak(ResponseManager::get("clean"), "routine");
        return true;
    }

    if (name == "show_help") {
        std::cout << "[DEBUG][Command] Dispatch: show_help\n";
        history.push(ResponseManager::get("show_help"), sf::Color::Cyan);
        speak(ResponseManager::get("show_help"), "routine");
        return true;
    }

    // ---- Filesystem ----
    if (name == "show_pwd") {
        std::cout << "[DEBUG][Command] Dispatch: show_pwd\n";
        std::string msg = fs::current_path().string();
        history.push(ResponseManager::get("show_pwd") + msg, sf::Color::Yellow);
        speak(ResponseManager::get("show_pwd") + msg, "routine");
        return true;
    }

    if (name == "change_dir") {
        std::cout << "[DEBUG][Command] Dispatch: change_dir\n";
        std::string path;
        if (intent.slots.contains("path")) path = intent.slots.at("path");
        else if (intent.slots.contains("slot2")) path = intent.slots.at("slot2");

        if (!path.empty()) {
            try {
                fs::current_path(path);
                history.push(ResponseManager::get("change_dir_success") + fs::current_path().string(), sf::Color::Green);
                speak(ResponseManager::get("change_dir_success") + fs::current_path().string(), "routine");
            } catch (const std::exception& e) {
                history.push(ResponseManager::get("change_dir_fail") + e.what(), sf::Color::Red);
                speak(ResponseManager::get("change_dir_fail") + e.what(), "routine");
            }
        }
        return true;
    }

    if (name == "list_dir") {
        std::cout << "[DEBUG][Command] Dispatch: list_dir\n";
        for (auto& e : fs::directory_iterator(fs::current_path())) {
            std::string entry = e.path().filename().string();
            history.push(entry, sf::Color::White);
            speak(entry, "routine"); // might get verbose
        }
        return true;
    }

    if (name == "make_dir") {
        std::cout << "[DEBUG][Command] Dispatch: make_dir\n";
        std::string path;
        if (intent.slots.contains("path")) path = intent.slots.at("path");
        else if (intent.slots.contains("slot2")) path = intent.slots.at("slot2");

        if (!path.empty()) {
            try {
                fs::create_directory(path);
                history.push(ResponseManager::get("make_dir_success") + path, sf::Color::Green);
                speak(ResponseManager::get("make_dir_success") + path, "routine");
            } catch (...) {
                history.push(ResponseManager::get("make_dir_fail") + path, sf::Color::Red);
                speak(ResponseManager::get("make_dir_fail") + path, "routine");
            }
        }
        return true;
    }

    if (name == "remove_file") {
        std::cout << "[DEBUG][Command] Dispatch: remove_file\n";
        std::string path;
        if (intent.slots.contains("path")) path = intent.slots.at("path");
        else if (intent.slots.contains("slot2")) path = intent.slots.at("slot2");

        if (!path.empty()) {
            try {
                fs::remove_all(path);
                history.push(ResponseManager::get("remove_file_success") + path, sf::Color::Green);
                speak(ResponseManager::get("remove_file_success") + path, "routine");
            } catch (...) {
                history.push(ResponseManager::get("remove_file_fail") + path, sf::Color::Red);
                speak(ResponseManager::get("remove_file_fail") + path, "routine");
            }
        }
        return true;
    }

    // ---- NLP / AI ----
    if (name == "reload_nlp") {
        std::cout << "[DEBUG][Command] Dispatch: reload_nlp\n";

        std::string path = getResourcePath() + "/nlp_rules.json";
        std::cout << "[NLP] Reload requested. Looking for rules at: " << path << std::endl;

        std::string err;
        if (!nlp.load_rules(path, &err)) {
            history.push(ResponseManager::get("reload_nlp_fail") + err, sf::Color::Red);
            speak(ResponseManager::get("reload_nlp_fail") + err, "routine");
            std::cerr << "[NLP] Failed to reload rules from: " << path
                      << " (" << err << ")" << std::endl;
        } else {
            history.push(ResponseManager::get("reload_nlp_success"), sf::Color::Green);
            speak(ResponseManager::get("reload_nlp_success"), "routine");
            std::cout << "[NLP] Rules successfully reloaded from: " << path << std::endl;
        }
        return true;
    }

    else if (name == "ai_backend") {
    std::cout << "[DEBUG][Command] Dispatch: ai_backend\n";

    std::string prompt;

    if (intent.slots.contains("prompt")) {
        prompt = intent.slots.at("prompt");
    }

    // Trim leading/trailing whitespace
    prompt.erase(0, prompt.find_first_not_of(" \t\n\r"));
    prompt.erase(prompt.find_last_not_of(" \t\n\r") + 1);

    if (prompt.empty()) {
        // Empty input → Jarvis-style default
        std::string defaultResponses[] = {
            "Yes?",
            "How can I assist you?",
            "I'm listening.",
            "What can I do for you today?"
        };
        std::string response = defaultResponses[rand() % 4];

        history.push("[AI] " + response, sf::Color::Green);
        Voice::speakText(response, "summary");
    } 
    else if (prompt == "auto" || prompt == "ollama" ||
             prompt == "localai" || prompt == "openai") 
    {
        // Valid backend → switch
        aiConfig["backend"] = prompt;

        std::ofstream f("ai_config.json");
        f << aiConfig.dump(4);
        f.close();

        std::string msg = "[AI] Backend switched to " + prompt;
        history.push(msg, sf::Color::Green);
        Voice::speakText(msg, "summary");
    } 
    else 
    {
        // General chat → AI response
        history.push("[AI] Thinking about: " + prompt, sf::Color::Cyan);

        std::string aiResponse = ai_process(prompt, longTermMemory);

        history.push("[AI] " + aiResponse, sf::Color::Green);
        Voice::speakText(aiResponse, "summary");
    }

    return true;
}


    return true;
}











    // ---- Memory ----
    if (name == "remember") {
        std::cout << "[DEBUG][Command] Dispatch: remember\n";
        std::string key, val;

        if (intent.slots.contains("key")) key = intent.slots.at("key");
        else if (intent.slots.contains("slot2")) key = intent.slots.at("slot2");

        if (intent.slots.contains("value")) val = intent.slots.at("value");
        else if (intent.slots.contains("slot3")) val = intent.slots.at("slot3");

        if (!key.empty() && !val.empty()) {
            longTermMemory[key] = val;
            saveMemory();
            history.push(ResponseManager::get("remember") + key + " = " + val, sf::Color::Green);
            speak(ResponseManager::get("remember") + key + " = " + val, "routine");
        } else {
            history.push(ResponseManager::get("remember_fail"), sf::Color::Red);
            speak(ResponseManager::get("remember_fail"), "routine");
        }
        return true;
    }

    if (name == "recall") {
        std::cout << "[DEBUG][Command] Dispatch: recall\n";
        std::string key;
        if (intent.slots.contains("key")) key = intent.slots.at("key");
        else if (intent.slots.contains("slot2")) key = intent.slots.at("slot2");

        if (!key.empty()) {
            if (longTermMemory.contains(key)) {
                history.push(ResponseManager::get("recall") + key + " = " + longTermMemory[key].get<std::string>(), sf::Color::Yellow);
                speak(ResponseManager::get("recall") + key + " = " + longTermMemory[key].get<std::string>(), "routine");
            } else {
                history.push(ResponseManager::get("recall_unknown") + key, sf::Color::Red);
                speak(ResponseManager::get("recall_unknown") + key, "routine");
            }
        } else {
            history.push(ResponseManager::get("recall_no_key"), sf::Color::Red);
            speak(ResponseManager::get("recall_no_key"), "routine");
        }
        return true;
    }

    if (name == "forget") {
        std::cout << "[DEBUG][Command] Dispatch: forget\n";
        std::string key;
        if (intent.slots.contains("key")) key = intent.slots.at("key");
        else if (intent.slots.contains("slot2")) key = intent.slots.at("slot2");

        if (!key.empty()) {
            if (longTermMemory.contains(key)) {
                longTermMemory.erase(key);
                saveMemory();
                history.push(ResponseManager::get("forget") + key, sf::Color::Yellow);
                speak(ResponseManager::get("forget") + key, "routine");
            } else {
                history.push(ResponseManager::get("forget_unknown") + key, sf::Color::Red);
                speak(ResponseManager::get("forget_unknown") + key, "routine");
            }
        } else {
            history.push(ResponseManager::get("forget_no_key"), sf::Color::Red);
            speak(ResponseManager::get("forget_no_key"), "routine");
        }
        return true;
    }

    // ---- System ----
    else if (name == "system_info") {
        std::cout << "[DEBUG][Command] Dispatch: system_info\n";

        SystemInfo sys = detectSystem();

        history.push("[System Info]", sf::Color::Cyan);
        history.push("OS         : " + sys.osName + " (" + sys.arch + ")", sf::Color::Yellow);
        history.push("CPU Cores  : " + std::to_string(sys.cpuCores), sf::Color::Yellow);
        history.push("RAM        : " + std::to_string(sys.ramMB) + " MB", sf::Color::Green);

        if (sys.hasGPU) {
            std::string gpuLine = sys.gpuName + " (" + std::to_string(sys.gpuCount) + " device(s))";
            history.push("GPU        : " + gpuLine, sf::Color(180, 255, 180));

            if (sys.hasCUDA)  history.push("CUDA       : Supported", sf::Color::Green);
            if (sys.hasMetal) history.push("Metal      : Supported", sf::Color::Green);
            if (sys.hasROCm)  history.push("ROCm       : Supported", sf::Color::Green);
        } else {
            history.push("GPU        : None detected", sf::Color::Red);
        }

        history.push("Suggested Whisper model: " + sys.suggestedModel, sf::Color::Cyan);
        return true;
    }

    // ---- Voice ----
    if (name == "voice") {
        std::cout << "[DEBUG][Command] Dispatch: voice\n";
        history.push(ResponseManager::get("voice_start"), sf::Color::Cyan);

        std::string transcript = Voice::runVoiceDemo(longTermMemory);

        if (!transcript.empty()) {
            history.push(ResponseManager::get("voice_heard") + transcript, sf::Color::Yellow);
            speak(ResponseManager::get("voice_heard") + transcript, "routine");
            parseAndDispatch(transcript, buffer, currentDir, timers, longTermMemory, nlp, history);
        } else {
            history.push(ResponseManager::get("voice_none"), sf::Color::Red);
            speak(ResponseManager::get("voice_none"), "routine");
        }
        return true;
    }

    // ---- Voice Stream ----
    if (name == "voice_stream") {
        std::cout << "[DEBUG][Command] Dispatch: voice_stream\n";

        if (!VoiceStream::isRunning()) {
            history.push(ResponseManager::get("voice_stream_start"), sf::Color(0, 200, 255));
            if (VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory, nlp)) {
                std::cout << "[VoiceStream] Stream started.\n";
            } else {
                std::cerr << "[VoiceStream] Failed to start.\n";
            }
        } else {
            history.push(ResponseManager::get("voice_stream_stop"), sf::Color(0, 200, 255));
            VoiceStream::stop();
            std::cout << "[VoiceStream] Stream stopped.\n";
        }

        return true;
    }

    return true; // fallback if no command matched above
}
