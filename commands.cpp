#include "commands.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#include "ai.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "resources.hpp"

#include <iostream>
#include <filesystem>
#include <string>

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
        history.push("[Voice] Sorry, I didn’t understand: " + text, sf::Color::Red);
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
    history.push("[WARN] No command matched.", sf::Color(200,200,200));
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

    // Prefer named "app" slot
    if (intent.slots.contains("app")) {
        appName = intent.slots.at("app");
    }
    // Fallback: use slot2 if present
    else if (intent.slots.contains("slot2")) {
        appName = intent.slots.at("slot2");
    }
    // Absolute fallback: take first available slot
    else if (!intent.slots.empty()) {
        appName = intent.slots.begin()->second;
    }

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
                history.push("[INFO] Opened: " + appName, sf::Color::Green);
            } else {
                history.push("[ERROR] Failed to open: " + appName, sf::Color::Red);
            }
        } else {
            HINSTANCE result = ShellExecuteA(NULL, "open", appName.c_str(), NULL, NULL, SW_SHOWNORMAL);
            if ((INT_PTR)result <= 32) {
                history.push("[ERROR] Failed to open: " + appName, sf::Color::Red);
            } else {
                history.push("[INFO] Opened: " + appName, sf::Color::Green);
            }
        }
#else
        int rc = system(appName.c_str());
        if (rc != 0) {
            history.push("[ERROR] Failed to launch: " + appName, sf::Color::Red);
        } else {
            history.push("[INFO] Opened: " + appName, sf::Color::Green);
        }
#endif
    } else {
        history.push("[ERROR] No app name detected in intent.", sf::Color::Red);
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
        history.push("[INFO] Searching web: " + q, sf::Color::Cyan);
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
    history.push("[Timer] Set for " + std::to_string(seconds) + " seconds.", sf::Color(255,200,0));
    return true;
}


    // ---- Console ----
    if (name == "clean") {
        std::cout << "[DEBUG][Command] Dispatch: clean\n";
        history.clear();
        history.push("[INFO] History cleared.", sf::Color::Green);
        return true;
    }

    if (name == "show_help") {
        std::cout << "[DEBUG][Command] Dispatch: show_help\n";
        history.push("Commands: help, open <app>, search <q>, timer, pwd, cd <path>, ls, mkdir <path>, rm <path>, reloadnlp, grim <msg>, remember <k> is <v>, recall <k>, forget <k>, system, voice, voice_stream, quit", sf::Color::Cyan);
        return true;
    }

    // ---- Filesystem ----
    if (name == "show_pwd") {
        std::cout << "[DEBUG][Command] Dispatch: show_pwd\n";
        history.push(fs::current_path().string(), sf::Color::Yellow);
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
            history.push("[INFO] Changed directory to " + fs::current_path().string(), sf::Color::Green);
        } catch (const std::exception& e) {
            history.push(std::string("[ERROR] ") + e.what(), sf::Color::Red);
        }
    }
    return true;
}


    if (name == "list_dir") {
        std::cout << "[DEBUG][Command] Dispatch: list_dir\n";
        for (auto& e : fs::directory_iterator(fs::current_path())) {
            history.push(e.path().filename().string(), sf::Color::White);
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
            history.push("[INFO] Created directory: " + path, sf::Color::Green);
        } catch (...) {
            history.push("[ERROR] Failed to create: " + path, sf::Color::Red);
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
            history.push("[INFO] Removed: " + path, sf::Color::Green);
        } catch (...) {
            history.push("[ERROR] Failed to remove: " + path, sf::Color::Red);
        }
    }
    return true;
}


    // ---- NLP / AI ----
    if (name == "reload_nlp") {
        std::cout << "[DEBUG][Command] Dispatch: reload_nlp\n";
        std::string err;
        if (!nlp.load_rules("nlp_rules.json", &err)) {
            history.push("[ERROR] Reload failed: " + err, sf::Color::Red);
        } else {
            history.push("[INFO] NLP rules reloaded.", sf::Color::Green);
        }
        return true;
    }

    if (name == "grim_ai") {
    std::cout << "[DEBUG][Command] Dispatch: grim_ai\n";
    std::string q;
    if (intent.slots.contains("query")) q = intent.slots.at("query");
    else if (intent.slots.contains("slot1")) q = intent.slots.at("slot1"); // fallback
    else if (intent.slots.contains("slot2")) q = intent.slots.at("slot2");

    if (!q.empty()) {
        std::string reply = callAI(q);
        if (!reply.empty()) {
            history.push("AI: " + reply, sf::Color(160,200,255));
        } else {
            history.push("[AI] No response generated.", sf::Color::Red);
        }
    } else {
        history.push("[AI] No query provided.", sf::Color::Red);
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
        history.push("[INFO] Remembered: " + key + " = " + val, sf::Color::Green);
    } else {
        history.push("[ERROR] Missing key or value for remember.", sf::Color::Red);
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
            history.push("[Recall] " + key + " = " + longTermMemory[key].get<std::string>(), sf::Color::Yellow);
        } else {
            history.push("[Recall] I don’t know " + key, sf::Color::Red);
        }
    } else {
        history.push("[ERROR] No key provided for recall.", sf::Color::Red);
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
            history.push("[Forget] Removed: " + key, sf::Color::Yellow);
        } else {
            history.push("[Forget] I didn’t know " + key, sf::Color::Red);
        }
    } else {
        history.push("[ERROR] No key provided for forget.", sf::Color::Red);
    }
    return true;
}


    // ---- System ----
    if (name == "system_info") {
        std::cout << "[DEBUG][Command] Dispatch: system_info\n";
#ifdef _WIN32
        double cpuLoad = getCPUUsage();
        DWORDLONG totalMB = 0, usedMB = 0;
        double memPercent = getMemoryUsagePercent(usedMB, totalMB);

        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        int cpuCount = sysInfo.dwNumberOfProcessors;

        std::string gpuName = "[Unavailable]";
        DISPLAY_DEVICEA dd;
        dd.cb = sizeof(dd);
        if (EnumDisplayDevicesA(NULL, 0, &dd, 0)) {
            gpuName = dd.DeviceString;
        }

        history.push("[System Info]", sf::Color::Cyan);
        history.push("CPU Cores : " + std::to_string(cpuCount), sf::Color::Yellow);
        history.push("CPU Usage : " + (cpuLoad >= 0.0 ? std::to_string((int)cpuLoad) + "%" : "[Unavailable]"), sf::Color::Yellow);
        history.push("Memory    : " + std::to_string(usedMB) + " MB / " + std::to_string(totalMB) + " MB (" + std::to_string((int)memPercent) + "%)", sf::Color::Green);
        history.push("GPU       : " + gpuName, sf::Color(180, 255, 180));
#else
        history.push("[System Info]", sf::Color::Cyan);
        history.push("System info not implemented on this platform yet.", sf::Color::Red);
#endif
        return true;
    }

    // ---- Voice ----
    if (name == "voice") {
        std::cout << "[DEBUG][Command] Dispatch: voice\n";
        history.push("[Voice] Starting 5-second recording...", sf::Color::Cyan);
        fs::path modelPath = fs::path(getResourcePath()).parent_path() / "ggml-small.bin";
        std::string transcript = runVoiceDemo(modelPath.string());

        if (!transcript.empty()) {
            history.push("[Voice] Heard: " + transcript, sf::Color::Yellow);
            parseAndDispatch(transcript, buffer, currentDir, timers, longTermMemory, nlp, history);
        } else {
            history.push("[Voice] No speech detected.", sf::Color::Red);
        }
        return true;
    }

    if (name == "voice_stream") {
        std::cout << "[DEBUG][Command] Dispatch: voice_stream\n";
        history.push("[VoiceStream] Starting live microphone stream...", sf::Color(0, 200, 255));
        runVoiceStream(nullptr, &history, timers, longTermMemory, nlp);
        history.push("[VoiceStream] Stopped.", sf::Color(0, 200, 255));
        return true;
    }

    return true;
}
