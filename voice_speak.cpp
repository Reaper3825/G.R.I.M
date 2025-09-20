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
static HANDLE g_hChildStdinRd = NULL;
static HANDLE g_hChildStdinWr = NULL;
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
// Audio Playback (SFML, async) – used for Coqui WAVs
// =========================================================
void playAudio(const std::string& path) {
    std::thread([path]() {
        sf::SoundBuffer buffer;
        if (!buffer.loadFromFile(path)) {
            std::cerr << "[Voice] Failed to load audio: " << path << "\n";
            return;
        }

        sf::Sound sound(buffer); // ✅ construct with buffer
        sound.play();

        std::cout << "[Voice] Playing audio: " << path
                  << " (duration " << buffer.getDuration().asSeconds() << "s)"
                  << std::endl;

        sf::Clock clock;
        while (sound.getStatus() == sf::Sound::Status::Playing) {
            sf::sleep(sf::milliseconds(100));
        }

        std::cout << "[Voice] Finished playback at "
                  << clock.getElapsedTime().asSeconds()
                  << "s" << std::endl;
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
        std::cerr << "[Voice][SAPI] Failed to create ISpVoice: 0x" << std::hex << hr << "\n";
        CoUninitialize();
        return false;
    }

    // Convert UTF-8 string to wide string
    int wlen = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, NULL, 0);
    std::wstring wtext(wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, &wtext[0], wlen);

    // Speak async
    hr = pVoice->Speak(wtext.c_str(), SPF_ASYNC, NULL);
    if (FAILED(hr)) {
        std::cerr << "[Voice][SAPI] Speak() failed: 0x" << std::hex << hr << "\n";
        pVoice->Release();
        CoUninitialize();
        return false;
    }

    // Wait until finished
    pVoice->WaitUntilDone(INFINITE);

    pVoice->Release();
    CoUninitialize();
    return true;
#else
    (void)text;
    (void)voiceModel;
    return false;
#endif
}

// =========================================================
// Init Coqui Bridge (persistent subprocess)
// =========================================================
bool initTTS() {
#ifdef _WIN32
    SECURITY_ATTRIBUTES saAttr;
    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create pipes for STDOUT
    if (!CreatePipe(&g_hChildStdoutRd, &g_hChildStdoutWr, &saAttr, 0)) {
        std::cerr << "[Voice] Stdout pipe failed\n";
        return false;
    }
    if (!SetHandleInformation(g_hChildStdoutRd, HANDLE_FLAG_INHERIT, 0)) {
        std::cerr << "[Voice] Stdout SetHandleInformation failed\n";
        return false;
    }

    // Create pipes for STDIN
    if (!CreatePipe(&g_hChildStdinRd, &g_hChildStdinWr, &saAttr, 0)) {
        std::cerr << "[Voice] Stdin pipe failed\n";
        return false;
    }
    if (!SetHandleInformation(g_hChildStdinWr, HANDLE_FLAG_INHERIT, 0)) {
        std::cerr << "[Voice] Stdin SetHandleInformation failed\n";
        return false;
    }

    // Build command
    std::string cmd = "resources/.venv/Scripts/python.exe resources/python/tts_bridge.py";

    STARTUPINFOA siStartInfo;
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFOA));
    siStartInfo.cb = sizeof(STARTUPINFOA);
    siStartInfo.hStdError = g_hChildStdoutWr;
    siStartInfo.hStdOutput = g_hChildStdoutWr;
    siStartInfo.hStdInput = g_hChildStdinRd;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    ZeroMemory(&g_piProcInfo, sizeof(PROCESS_INFORMATION));

    if (!CreateProcessA(
            NULL,
            cmd.data(),
            NULL,
            NULL,
            TRUE,
            CREATE_NO_WINDOW,
            NULL,
            NULL,
            &siStartInfo,
            &g_piProcInfo)) {
        std::cerr << "[Voice] CreateProcess failed (" << GetLastError() << ")\n";
        return false;
    }

    g_ttsReady = true;
    return true;
#else
    std::cerr << "[Voice] initTTS not implemented on this platform yet.\n";
    return false;
#endif
}

// =========================================================
// Shutdown Coqui Bridge
// =========================================================
void shutdownTTS() {
#ifdef _WIN32
    if (g_piProcInfo.hProcess) {
        TerminateProcess(g_piProcInfo.hProcess, 0);
        CloseHandle(g_piProcInfo.hProcess);
        CloseHandle(g_piProcInfo.hThread);
    }
    if (g_hChildStdinRd) CloseHandle(g_hChildStdinRd);
    if (g_hChildStdinWr) CloseHandle(g_hChildStdinWr);
    if (g_hChildStdoutRd) CloseHandle(g_hChildStdoutRd);
    if (g_hChildStdoutWr) CloseHandle(g_hChildStdoutWr);

    g_hChildStdinRd = g_hChildStdinWr = NULL;
    g_hChildStdoutRd = g_hChildStdoutWr = NULL;
#endif
    g_ttsReady = false;
}

// =========================================================
// Coqui Speak (send JSON → receive JSON)
// =========================================================
std::string coquiSpeak(const std::string& text,
                       const std::string& speaker,
                       double speed) {
    if (!g_ttsReady) {
        std::cerr << "[Voice] Coqui not initialized\n";
        return "";
    }

#ifdef _WIN32
    // Build JSON
    nlohmann::json req = {{"text", text}, {"speaker", speaker}, {"speed", speed}};
    std::string reqStr = req.dump() + "\n";

    DWORD written;
    if (!WriteFile(g_hChildStdinWr, reqStr.c_str(),
                   (DWORD)reqStr.size(), &written, NULL)) {
        std::cerr << "[Voice] Failed to write to bridge\n";
        return "";
    }

    // Read response line
    std::string respLine;
    char buffer[1];
    DWORD bytesRead;
    while (true) {
        if (!ReadFile(g_hChildStdoutRd, buffer, 1, &bytesRead, NULL) || bytesRead == 0) {
            break;
        }
        if (buffer[0] == '\n') break;
        respLine.push_back(buffer[0]);
    }

    try {
        auto resp = nlohmann::json::parse(respLine);
        if (resp.contains("file")) return resp["file"].get<std::string>();
        if (resp.contains("error")) {
            std::cerr << "[Voice] Bridge error: " << resp["error"] << "\n";
        }
    } catch (std::exception& e) {
        std::cerr << "[Voice] Parse error: " << e.what() << "\n";
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
    if (cfg.contains("rules") && cfg["rules"].contains(category)) {
        engine = cfg["rules"][category].get<std::string>();
    }

    std::string speaker = cfg.value("speaker", "p225");
    double speed = cfg.value("speed", 1.0);

    if (engine == "coqui") {
        std::string wav = coquiSpeak(text, speaker, speed);
        if (!wav.empty()) playAudio(wav);
    } else if (engine == "sapi") {
        speakLocal(text, cfg.value("local_engine", ""));
    }
}

} // namespace Voice
