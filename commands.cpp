#include "commands.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#include "ai.hpp"

#include <iostream>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <psapi.h>     // GlobalMemoryStatusEx, MEMORYSTATUSEX
#include <winternl.h>  // ULONGLONG, ULARGE_INTEGER

// ---------------- Windows Helpers ----------------

// --- CPU usage helper ---
static ULARGE_INTEGER lastIdleTime   = {};
static ULARGE_INTEGER lastKernelTime = {};
static ULARGE_INTEGER lastUserTime   = {};

double getCPUUsage() {
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

// --- Memory usage helper ---
double getMemoryUsagePercent(DWORDLONG &usedMB, DWORDLONG &totalMB) {
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);

    totalMB = memInfo.ullTotalPhys / (1024 * 1024);
    usedMB  = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024 * 1024);

    return (double)usedMB * 100.0 / (double)totalMB;
}

// --- Helper: resolve exe name on PATH ---
static std::string findOnPath(const std::string& app) {
    char buffer[MAX_PATH];
    DWORD result = SearchPathA(
        NULL, app.c_str(), ".exe", MAX_PATH, buffer, NULL
    );
    if (result > 0 && result < MAX_PATH) {
        return std::string(buffer);
    }
    return app; // fallback
}
#endif // _WIN32


namespace fs = std::filesystem;

// ---------------- Command Dispatcher ----------------
bool handleCommand(const Intent& intent,
                   std::string& /*buffer*/,
                   fs::path& /*currentDir*/,
                   std::vector<Timer>& timers,
                   nlohmann::json& longTermMemory,
                   NLP& nlp,
                   ConsoleHistory& history)
{
    if (!intent.matched) {
        history.push("[WARN] No command matched.", sf::Color(200,200,200));
        return true;
    }

    // ---- open_app ----
    if (intent.name == "open_app") {
        auto it = intent.slots.find("app");
        if (it != intent.slots.end()) {
            std::string appName = it->second;
            std::string resolved = resolveAlias(appName);
            if (resolved.empty()) resolved = appName;

#ifdef _WIN32
            if (resolved.find(':') == std::string::npos &&
                resolved.find('/') == std::string::npos) {
                resolved = findOnPath(resolved);
            }

            // If it looks like an .exe, run it directly with CreateProcess
            if (resolved.ends_with(".exe")) {
                STARTUPINFOA si{};
                PROCESS_INFORMATION pi{};
                si.cb = sizeof(si);

                if (CreateProcessA(
                        resolved.c_str(),   // application
                        NULL,               // command line
                        NULL, NULL, FALSE,
                        0, NULL, NULL,
                        &si, &pi))
                {
                    CloseHandle(pi.hProcess);
                    CloseHandle(pi.hThread);
                    history.push("[INFO] Opened: " + resolved, sf::Color::Green);
                } else {
                    history.push("[ERROR] Failed to open: " + resolved, sf::Color::Red);
                }
            } else {
                // Assume it's a URL or non-.exe, let ShellExecute handle it
                HINSTANCE result = ShellExecuteA(
                    NULL, "open", resolved.c_str(),
                    NULL, NULL, SW_SHOWNORMAL
                );
                if ((INT_PTR)result <= 32) {
                    history.push("[ERROR] Failed to open: " + resolved, sf::Color::Red);
                } else {
                    history.push("[INFO] Opened: " + resolved, sf::Color::Green);
                }
            }
#else
            int rc = system(resolved.c_str());
            if (rc != 0) history.push("[ERROR] Failed to launch: " + resolved, sf::Color::Red);
            else history.push("[INFO] Opened: " + resolved, sf::Color::Green);
#endif
        }
        return true;
    }

    // ---- search_web ----
    if (intent.name == "search_web") {
        auto it = intent.slots.find("query");
        if (it != intent.slots.end()) {
            std::string q = it->second;
#ifdef _WIN32
            std::string url = "https://www.google.com/search?q=" + q;
            ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
            history.push("[INFO] Searching web: " + q, sf::Color::Cyan);
        }
        return true;
    }

    // ---- set_timer ----
    if (intent.name == "set_timer") {
        int value = 0;
        std::string unit = "s";
        if (auto it = intent.slots.find("value"); it != intent.slots.end())
            value = std::stoi(it->second);
        if (auto it = intent.slots.find("unit"); it != intent.slots.end())
            unit = it->second;

        int seconds = value;
        if (unit.starts_with("m")) seconds = value * 60;
        else if (unit.starts_with("h")) seconds = value * 3600;

        timers.push_back(Timer{seconds, sf::Clock(), false});
        history.push("[Timer] Set for " + std::to_string(seconds) + " seconds.", sf::Color(255,200,0));
        return true;
    }

    // ---- clean ----
    if (intent.name == "clean") {
        history.clear();
        history.push("[INFO] History cleared.", sf::Color::Green);
        return true;
    }

    // ---- show_help ----
    if (intent.name == "show_help") {
        history.push("Commands: help, open <app>, search <q>, timer, pwd, cd <path>, ls, mkdir <path>, rm <path>, reloadnlp, grim <msg>, remember <k> is <v>, recall <k>, forget <k>, quit", sf::Color::Cyan);
        return true;
    }

    // ---- show_pwd ----
    if (intent.name == "show_pwd") {
        history.push(fs::current_path().string(), sf::Color::Yellow);
        return true;
    }

    // ---- change_dir ----
    if (intent.name == "change_dir") {
        auto it = intent.slots.find("path");
        if (it != intent.slots.end()) {
            std::string path = it->second;
            try {
                fs::current_path(path);
                history.push("[INFO] Changed directory to " + fs::current_path().string(), sf::Color::Green);
            } catch (const std::exception& e) {
                history.push(std::string("[ERROR] ") + e.what(), sf::Color::Red);
            }
        }
        return true;
    }

    // ---- list_dir ----
    if (intent.name == "list_dir") {
        for (auto& e : fs::directory_iterator(fs::current_path())) {
            history.push(e.path().filename().string(), sf::Color::White);
        }
        return true;
    }

    // ---- make_dir ----
    if (intent.name == "make_dir") {
        auto it = intent.slots.find("path");
        if (it != intent.slots.end()) {
            try {
                fs::create_directory(it->second);
                history.push("[INFO] Created directory: " + it->second, sf::Color::Green);
            } catch (...) {
                history.push("[ERROR] Failed to create: " + it->second, sf::Color::Red);
            }
        }
        return true;
    }

    // ---- remove_file ----
    if (intent.name == "remove_file") {
        auto it = intent.slots.find("path");
        if (it != intent.slots.end()) {
            try {
                fs::remove_all(it->second);
                history.push("[INFO] Removed: " + it->second, sf::Color::Green);
            } catch (...) {
                history.push("[ERROR] Failed to remove: " + it->second, sf::Color::Red);
            }
        }
        return true;
    }

    // ---- reload_nlp ----
    if (intent.name == "reload_nlp") {
        std::string err;
        if (!nlp.load_rules("nlp_rules.json", &err)) {
            history.push("[ERROR] Reload failed: " + err, sf::Color::Red);
        } else {
            history.push("[INFO] NLP rules reloaded.", sf::Color::Green);
        }
        return true;
    }

    // ---- grim_ai ----
    if (intent.name == "grim_ai") {
        auto it = intent.slots.find("query");
        if (it != intent.slots.end()) {
            std::string reply = callAI(it->second);
            history.push("AI: " + reply, sf::Color(160,200,255));
        }
        return true;
    }

    // ---- remember ----
    if (intent.name == "remember") {
        auto key = intent.slots.find("key");
        auto val = intent.slots.find("value");
        if (key != intent.slots.end() && val != intent.slots.end()) {
            longTermMemory[key->second] = val->second;
            saveMemory();
            history.push("[INFO] Remembered: " + key->second + " = " + val->second, sf::Color::Green);
        }
        return true;
    }

    // ---- recall ----
    if (intent.name == "recall") {
        auto key = intent.slots.find("key");
        if (key != intent.slots.end()) {
            if (longTermMemory.contains(key->second)) {
                history.push("[Recall] " + key->second + " = " + longTermMemory[key->second].get<std::string>(), sf::Color::Yellow);
            } else {
                history.push("[Recall] I don’t know " + key->second, sf::Color::Red);
            }
        }
        return true;
    }

    // ---- forget ----
    if (intent.name == "forget") {
        auto key = intent.slots.find("key");
        if (key != intent.slots.end()) {
            if (longTermMemory.contains(key->second)) {
                longTermMemory.erase(key->second);
                saveMemory();
                history.push("[Forget] Removed: " + key->second, sf::Color::Yellow);
            } else {
                history.push("[Forget] I didn’t know " + key->second, sf::Color::Red);
            }
        }
        return true;
    }
            // ---- system_info ----
    if (intent.name == "system_info") {
#ifdef _WIN32
        // --- CPU usage ---
        double cpuLoad = getCPUUsage();

        // --- Memory usage ---
        DWORDLONG totalMB = 0, usedMB = 0;
        double memPercent = getMemoryUsagePercent(usedMB, totalMB);

        // --- CPU cores ---
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        int cpuCount = sysInfo.dwNumberOfProcessors;

        // --- GPU info (basic adapter name) ---
        std::string gpuName = "[Unavailable]";
        DISPLAY_DEVICEA dd;
        dd.cb = sizeof(dd);
        if (EnumDisplayDevicesA(NULL, 0, &dd, 0)) {
            gpuName = dd.DeviceString;
        }

        // --- Push results to history ---
        history.push("[System Info]", sf::Color::Cyan);
        history.push("CPU Cores : " + std::to_string(cpuCount), sf::Color::Yellow);
        if (cpuLoad >= 0.0) {
            history.push("CPU Usage : " + std::to_string((int)cpuLoad) + "%", sf::Color::Yellow);
        } else {
            history.push("CPU Usage : [Unavailable]", sf::Color::Red);
        }
        history.push("Memory    : " + std::to_string(usedMB) + " MB / " +
                     std::to_string(totalMB) + " MB (" + std::to_string((int)memPercent) + "%)",
                     sf::Color::Green);
        history.push("GPU       : " + gpuName, sf::Color(180, 255, 180));
#else
        history.push("[System Info]", sf::Color::Cyan);
        history.push("System info not implemented on this platform yet.", sf::Color::Red);
#endif
        return true;
    }


    return true;
}
