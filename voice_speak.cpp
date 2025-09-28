#include "voice_speak.hpp"
#include "logger.hpp"
#include "popup_ui/popup_ui.hpp" 

#include <SFML/Audio.hpp>
#include <thread>
#include <chrono>
#include <filesystem>
#include <random>
#include <fstream>
#include <unordered_map>
#include <nlohmann/json.hpp>

#ifdef _WIN32
    #include <windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace Voice {
    static std::vector<std::unique_ptr<sf::SoundBuffer>> activeBuffers;
    static std::vector<std::unique_ptr<sf::Sound>> activeSounds;

    static std::string g_engine     = "coqui";
    static std::string g_speaker    = "p225";
    static double      g_speed      = 1.0;
    static fs::path    g_outputDir  = "D:/G.R.I.M/resources/tts_out";
    static std::unordered_map<std::string, std::string> g_rules;

    // 🔹 Flag that tracks if TTS bridge is ready
    static bool g_ttsReady = false;

#ifdef _WIN32
    static HANDLE hChildStdinWr = nullptr;
    static HANDLE hChildStdoutRd = nullptr;
    static PROCESS_INFORMATION piProcInfo{};
#endif

    // =========================================================
    // Helpers
    // =========================================================
    static std::string randomString(size_t length) {
        static const char charset[] =
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";
        static thread_local std::mt19937 rg{std::random_device{}()};
        static thread_local std::uniform_int_distribution<size_t> dist(0, sizeof(charset) - 2);

        std::string result;
        result.reserve(length);
        for (size_t i = 0; i < length; i++) {
            result.push_back(charset[dist(rg)]);
        }
        return result;
    }

    static void cleanupSounds() {
        activeSounds.erase(
            std::remove_if(activeSounds.begin(), activeSounds.end(),
                [](const std::unique_ptr<sf::Sound>& s) {
                    return s->getStatus() == sf::SoundSource::Status::Stopped;
                }),
            activeSounds.end()
        );
    }

    // Return true if any active sound is currently playing
    bool isPlaying() {
        for (const auto& s : activeSounds) {
            if (s && s->getStatus() == sf::SoundSource::Status::Playing) return true;
        }
        return false;
    }

#ifdef _WIN32
    static std::string readLineFromBridge() {
        std::string result;
        char ch;
        DWORD read = 0, avail = 0;

        while (true) {
            if (!PeekNamedPipe(hChildStdoutRd, nullptr, 0, nullptr, &avail, nullptr)) {
                LOG_ERROR("Voice/Bridge", "PeekNamedPipe failed");
                break;
            }
            if (avail == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (!ReadFile(hChildStdoutRd, &ch, 1, &read, nullptr) || read == 0) {
                break; // EOF or error
            }
            if (ch == '\r') continue;
            if (ch == '\n') break;
            result.push_back(ch);
        }
        return result;
    }

    static std::string readJsonLineFromBridge(int timeoutMs = 30000) {
        auto start = std::chrono::steady_clock::now();

        while (true) {
            std::string line = readLineFromBridge();
            if (!line.empty()) {
                if (line[0] == '{')
                    return line;

                LOG_DEBUG("Voice/Bridge", "Skipped non-JSON: " + line);
            }
            if (std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start).count() > timeoutMs) {
                LOG_ERROR("Voice/Bridge", "Handshake timeout");
                return "";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
#endif

    // =========================================================
    // Init / Shutdown
    // =========================================================
    bool initTTS() {
        try {
            fs::path cfgPath = fs::path("D:/G.R.I.M/resources/ai_config.json");
            if (fs::exists(cfgPath)) {
                std::ifstream in(cfgPath);
                json cfg;
                in >> cfg;
                if (cfg.contains("voice")) {
                    auto& v = cfg["voice"];
                    if (v.contains("engine"))      g_engine    = v["engine"].get<std::string>();
                    if (v.contains("speaker"))     g_speaker   = v["speaker"].get<std::string>();
                    if (v.contains("speed"))       g_speed     = v["speed"].get<double>();
                    if (v.contains("output_dir"))  g_outputDir = v["output_dir"].get<std::string>();
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Voice/Init", std::string("Error reading ai_config.json: ") + e.what());
        }

#ifdef _WIN32
        if (g_engine == "coqui") {
            SECURITY_ATTRIBUTES saAttr{};
            saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
            saAttr.bInheritHandle = TRUE;
            saAttr.lpSecurityDescriptor = nullptr;

            HANDLE hChildStdinRdTmp = nullptr, hChildStdoutWrTmp = nullptr;
            CreatePipe(&hChildStdinRdTmp, &hChildStdinWr, &saAttr, 0);
            SetHandleInformation(hChildStdinWr, HANDLE_FLAG_INHERIT, 0);
            CreatePipe(&hChildStdoutRd, &hChildStdoutWrTmp, &saAttr, 0);
            SetHandleInformation(hChildStdoutRd, HANDLE_FLAG_INHERIT, 0);

            STARTUPINFOA si{};
            ZeroMemory(&si, sizeof(si));
            si.cb = sizeof(STARTUPINFOA);
            si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
            si.hStdOutput = hChildStdoutWrTmp;
            si.hStdInput  = hChildStdinRdTmp;
            si.dwFlags |= STARTF_USESTDHANDLES;

            std::string cmd = "\"C:/Program Files/Python310/python.exe\" -u D:/G.R.I.M/resources/python/coqui_bridge.py --persistent";
            std::vector<char> mutableCmd(cmd.begin(), cmd.end());
            mutableCmd.push_back('\0');

            ZeroMemory(&piProcInfo, sizeof(piProcInfo));
            CreateProcessA(nullptr, mutableCmd.data(), nullptr, nullptr, TRUE, 0,
                           nullptr, "D:/G.R.I.M/resources/python", &si, &piProcInfo);

            CloseHandle(hChildStdoutWrTmp);
            CloseHandle(hChildStdinRdTmp);

            std::string response = readJsonLineFromBridge();
            try {
                auto resp = json::parse(response);
                if (resp.value("status", "") == "ready") {
                    g_ttsReady = true; // 🔹 set ready flag
                    LOG_PHASE("Voice bridge ready", true);
                }
            } catch (const std::exception& e) {
                LOG_ERROR("Voice/Init", std::string("Parsing handshake failed: ") + e.what() +
                                         " raw=" + response);
            }
        }
#endif
        return true;
    }

    void shutdownTTS() {
#ifdef _WIN32
        if (hChildStdinWr) {
            std::string exitCmd = R"({"command":"exit"})" "\n";
            DWORD written;
            WriteFile(hChildStdinWr, exitCmd.c_str(), (DWORD)exitCmd.size(), &written, nullptr);
        }
        if (piProcInfo.hProcess) {
            WaitForSingleObject(piProcInfo.hProcess, 2000);
            CloseHandle(piProcInfo.hProcess);
            CloseHandle(piProcInfo.hThread);
        }
#endif
        LOG_PHASE("Voice shutdownTTS complete", true);
        g_ttsReady = false; // 🔹 reset flag
    }

    // =========================================================
    // Query ready state
    // =========================================================
    bool isReady() {
        return g_ttsReady;
    }

    // =========================================================
    // Playback
    // =========================================================
    void playAudio(const std::string& path) {
    try {
        auto buffer = std::make_unique<sf::SoundBuffer>();
        if (!buffer->loadFromFile(path)) {
            LOG_ERROR("Voice/Audio", "Could not load file: " + path);
            return;
        }

        auto sound = std::make_unique<sf::Sound>(*buffer);
        sound->setVolume(100.f);

    // 🔹 Trigger popup before playing and ensure activity is refreshed
    // after we actually start playback.
    notifyPopupActivity();

    sound->play();

        LOG_DEBUG("Voice/Audio", "Playing: " + path +
            " (duration=" + std::to_string(buffer->getDuration().asSeconds()) + "s)");

        activeBuffers.push_back(std::move(buffer));
        activeSounds.push_back(std::move(sound));

    cleanupSounds();

    } catch (const std::exception& e) {
        LOG_ERROR("Voice/Audio", std::string("Exception: ") + e.what());
    }
}

    // =========================================================
    // Coqui Speak
    // =========================================================
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed) {
#ifdef _WIN32
        if (!hChildStdinWr || !hChildStdoutRd) {
            LOG_ERROR("Voice/Coqui", "Bridge not running");
            return "";
        }

        fs::create_directories(g_outputDir);
        std::string outFile = (g_outputDir / (randomString(32) + ".wav")).string();

        json req = {
            {"command", "speak"},
            {"text", text},
            {"speaker", speaker},
            {"speed", speed},
            {"out", outFile}
        };
        std::string line = req.dump() + "\n";

        DWORD written = 0;
        BOOL ok = WriteFile(hChildStdinWr, line.c_str(), (DWORD)line.size(), &written, nullptr);
        LOG_DEBUG("Voice/Coqui", "Sent request (" + std::to_string(written) + " bytes): " + line);
        if (!ok) {
            LOG_ERROR("Voice/Coqui", "WriteFile failed");
        }

        std::string response = readJsonLineFromBridge();
        LOG_DEBUG("Voice/Coqui", "Got response: " + response);

        try {
            auto resp = json::parse(response);
            if (resp.contains("file")) {
                LOG_DEBUG("Voice/Coqui", "Bridge returned file: " + resp["file"].get<std::string>());
                return resp["file"].get<std::string>();
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Voice/Coqui", std::string("Parse error: ") + e.what() +
                                    " raw=" + response);
        }
#endif
        return "";
    }

    // =========================================================
    // High-level Speak
    // =========================================================
    void speak(const std::string& text, const std::string& category) {
        std::thread([text, category]() {
            LOG_DEBUG("Voice", "speak(text=\"" + text + "\", category=\"" + category + "\")");

            // 🔹 Default to Coqui
            std::string engine = "coqui";

            // 🔹 Only override if the rule explicitly exists AND is valid
            auto it = g_rules.find(category);
            if (it != g_rules.end()) {
                if (it->second == "coqui" || it->second == "sapi") {
                    engine = it->second;
                } else {
                    LOG_ERROR("Voice", "Invalid engine override: " + it->second + " (falling back to Coqui)");
                }
            }

            LOG_DEBUG("Voice", "Engine selected: " + engine);

            if (engine == "coqui") {
                std::string wavPath = coquiSpeak(text, g_speaker, g_speed);
                if (!wavPath.empty()) {
                    playAudio(wavPath);
                } else {
                    LOG_ERROR("Voice", "coquiSpeak returned empty path");
                }
                return;
            }

#ifdef _WIN32
            if (engine == "sapi") {
                LOG_DEBUG("Voice", "Routing speech to Windows SAPI");
                std::string command = "powershell -Command "
                                      "\"Add-Type -AssemblyName System.Speech; "
                                      "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                                      ".Speak([Console]::In.ReadToEnd())\"";
                FILE* pipe = _popen(command.c_str(), "w");
                if (pipe) {
                    fwrite(text.c_str(), 1, text.size(), pipe);
                    _pclose(pipe);
                }
            }
#endif
        }).detach();
    }
}
