#include "voice_speak.hpp"
#include "resources.hpp"
#include "ai.hpp"

#include <SFML/Audio.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

#ifdef _WIN32
    #include <objbase.h>
    #include <windows.h>
    #include <sapi.h>
    #include <sphelper.h>
#endif

// =========================================================
// Persistent Coqui bridge state
// =========================================================
#ifdef _WIN32
static HANDLE g_hChildStdinRd  = NULL;
static HANDLE g_hChildStdinWr  = NULL;
static HANDLE g_hChildStdoutRd = NULL;
static HANDLE g_hChildStdoutWr = NULL;
static PROCESS_INFORMATION g_piProcInfo;
#endif

static bool g_ttsReady = false;

static nlohmann::json getVoiceConfig() {
    return aiConfig["voice"];
}

namespace Voice {

// =========================================================
// Audio Playback (SFML, async)
// =========================================================
void playAudio(const std::string& path) {
    std::thread([path]() {
        sf::SoundBuffer buffer;
        if (!buffer.loadFromFile(path)) {
            std::cerr << "[Voice] Failed to load audio: " << path << "\n";
            return;
        }

        sf::Sound sound(buffer);
        sound.play();

        std::cout << "[Voice] Playing audio: " << path
                  << " (duration " << buffer.getDuration().asSeconds() << "s)\n";

       while (sound.getStatus() == sf::Sound::Status::Playing) {
        sf::sleep(sf::milliseconds(100));
        }


        std::cout << "[Voice] Finished playback\n";
    }).detach();
}

// =========================================================
// Local Speech (Windows SAPI)
// =========================================================
bool speakLocal(const std::string& text, const std::string& /*voiceModel*/) {
#ifdef _WIN32
    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) {
        std::cerr << "[Voice][SAPI] CoInitializeEx failed: 0x" << std::hex << hr << "\n";
        return false;
    }

    ISpVoice* pVoice = nullptr;
    hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void**)&pVoice);
    if (FAILED(hr) || !pVoice) {
        std::cerr << "[Voice][SAPI] Failed to create ISpVoice\n";
        CoUninitialize();
        return false;
    }

    int wlen = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, NULL, 0);
    std::wstring wtext(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &wtext[0], wlen);

    hr = pVoice->Speak(wtext.c_str(), SPF_ASYNC, NULL);
    if (FAILED(hr)) {
        std::cerr << "[Voice][SAPI] Speak() failed\n";
        pVoice->Release();
        CoUninitialize();
        return false;
    }

    pVoice->WaitUntilDone(INFINITE);
    pVoice->Release();
    CoUninitialize();
    return true;
#else
    (void)text;
    return false;
#endif
}

// =========================================================
// Cloud speech (stub)
// =========================================================
bool speakCloud(const std::string& text, const std::string& engine) {
    std::cerr << "[Voice] speakCloud not implemented (engine=" << engine << ")\n";
    (void)text;
    return false;
}

// =========================================================
// Init Coqui Bridge (persistent subprocess) with READY handshake
// =========================================================
bool initTTS() {
#ifdef _WIN32
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create stdout pipe
    if (!CreatePipe(&g_hChildStdoutRd, &g_hChildStdoutWr, &saAttr, 0)) {
        std::cerr << "[Voice] Stdout pipe failed\n";
        return false;
    }
    SetHandleInformation(g_hChildStdoutRd, HANDLE_FLAG_INHERIT, 0);

    // Create stdin pipe
    if (!CreatePipe(&g_hChildStdinRd, &g_hChildStdinWr, &saAttr, 0)) {
        std::cerr << "[Voice] Stdin pipe failed\n";
        return false;
    }
    SetHandleInformation(g_hChildStdinWr, HANDLE_FLAG_INHERIT, 0);

    // Paths
    fs::path resourcesPath = getResourcePath();
    fs::path rootPath = resourcesPath.parent_path().parent_path().parent_path();
    fs::path pythonExe = rootPath / ".venv" / "Scripts" / "python.exe";

    // Prefer source Resources/python/tts_bridge.py, fallback to build/resources/python/tts_bridge.py
    fs::path script = "D:/G.R.I.M/Resources/python/tts_bridge.py";
    if (fs::exists(script)) {
        std::cout << "[Voice] Using source bridge: " << script.string() << "\n";
    } else {
        script = resourcesPath / "python" / "tts_bridge.py";
        std::cout << "[Voice] Using fallback bridge: " << script.string() << "\n";
    }

    if (!fs::exists(pythonExe) || !fs::exists(script)) {
        std::cerr << "[Voice] ERROR: Missing python.exe or tts_bridge.py\n";
        return false;
    }

    std::string cmd = "\"" + pythonExe.string() + "\" \"" + script.string() + "\"";
    std::replace(cmd.begin(), cmd.end(), '/', '\\');

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdError = g_hChildStdoutWr;
    si.hStdOutput = g_hChildStdoutWr;
    si.hStdInput  = g_hChildStdinRd;
    si.dwFlags |= STARTF_USESTDHANDLES;

    ZeroMemory(&g_piProcInfo, sizeof(g_piProcInfo));
    std::vector<char> cmdVec(cmd.begin(), cmd.end());
    cmdVec.push_back('\0');

    if (!CreateProcessA(NULL, cmdVec.data(), NULL, NULL, TRUE, CREATE_NO_WINDOW,
                        NULL, NULL, &si, &g_piProcInfo)) {
        std::cerr << "[Voice] CreateProcess failed (" << GetLastError() << ")\n";
        return false;
    }

    // Wait for READY
    std::string respLine;
    char buffer[1];
    DWORD bytesRead;
    while (true) {
        if (!ReadFile(g_hChildStdoutRd, buffer, 1, &bytesRead, NULL) || bytesRead == 0)
            break;
        if (buffer[0] == '\n') {
            try {
                auto resp = nlohmann::json::parse(respLine);
                if (resp.contains("status") && resp["status"] == "READY") {
                    std::cout << "[Voice] Bridge READY (model="
                              << resp.value("model", "?") << ")\n";
                    g_ttsReady = true;
                    return true;
                }
            } catch (...) {}
            respLine.clear();
        } else respLine.push_back(buffer[0]);
    }

    std::cerr << "[Voice] Bridge failed to send READY\n";
    return false;
#else
    return false;
#endif
}


// =========================================================
// Send text → receive .wav file path
// =========================================================
std::string coquiSpeak(const std::string& text,
                       const std::string& speaker,
                       double speed) {
#ifdef _WIN32
    if (!g_ttsReady) {
        std::cerr << "[Voice] coquiSpeak called before initTTS\n";
        return "";
    }

    nlohmann::json req = {{"text", text}, {"speaker", speaker}, {"speed", speed}};
    std::string line = req.dump() + "\n";

    DWORD written;
    if (!WriteFile(g_hChildStdinWr, line.c_str(), (DWORD)line.size(), &written, NULL)) {
        std::cerr << "[Voice] WriteFile failed\n";
        return "";
    }

    // Read one JSON response
    std::string respLine;
    char buffer[1];
    DWORD bytesRead;
    while (true) {
        if (!ReadFile(g_hChildStdoutRd, buffer, 1, &bytesRead, NULL) || bytesRead == 0)
            break;
        if (buffer[0] == '\n') {
            try {
                auto resp = nlohmann::json::parse(respLine);
                if (resp.contains("file")) return resp["file"];
                if (resp.contains("error")) {
                    std::cerr << "[Voice][Bridge] " << resp["error"] << "\n";
                    return "";
                }
            } catch (...) {}
            respLine.clear();
            break;
        } else respLine.push_back(buffer[0]);
    }
#endif
    return "";
}

// =========================================================
// Unified Entry Point
// =========================================================
void speak(const std::string& text, const std::string& category) {
    auto cfg = getVoiceConfig();
    std::string engine = cfg.value("engine", "sapi");
    if (cfg.contains("rules") && cfg["rules"].contains(category))
        engine = cfg["rules"][category];

    std::string speaker = cfg.value("speaker", "p225");
    double speed = cfg.value("speed", 1.0);

    if (engine == "coqui") {
        std::string wav = coquiSpeak(text, speaker, speed);
        if (!wav.empty()) playAudio(wav);
    } else if (engine == "sapi") {
        speakLocal(text, cfg.value("local_engine", ""));
    }
}

// =========================================================
// Simplified Helper
// =========================================================
bool speakText(const std::string& text, bool preferOnline) {
    auto cfg = getVoiceConfig();
    std::string engine = preferOnline ? cfg.value("engine","sapi") : "sapi";
    if (engine == "coqui") {
        std::string wav = coquiSpeak(text, cfg.value("speaker","p225"),
                                     cfg.value("speed",1.0));
        if (!wav.empty()) { playAudio(wav); return true; }
        return false;
    } else if (engine == "sapi") {
        return speakLocal(text, cfg.value("local_engine",""));
    }
    return false;
}

} // namespace Voice
