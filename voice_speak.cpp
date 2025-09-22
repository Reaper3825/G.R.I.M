#include "voice_speak.hpp"

#include <SFML/Audio.hpp>
#include <iostream>
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
    // =========================================================
    // Globals
    // =========================================================
    static std::vector<std::unique_ptr<sf::SoundBuffer>> activeBuffers;
    static std::vector<std::unique_ptr<sf::Sound>> activeSounds;

    static std::string g_engine     = "coqui";
    static std::string g_speaker    = "p225";
    static double      g_speed      = 1.0;
    static fs::path    g_outputDir  = "D:/G.R.I.M/resources/tts_out";
    static std::unordered_map<std::string, std::string> g_rules;

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

#ifdef _WIN32
    static std::string readLineFromBridge() {
        std::string result;
        char ch;
        DWORD read;
        while (true) {
            if (!ReadFile(hChildStdoutRd, &ch, 1, &read, nullptr) || read == 0) {
                break; // EOF or error
            }
            if (ch == '\n') break;
            result.push_back(ch);
        }
        return result;
    }

    static std::string readJsonLineFromBridge() {
        while (true) {
            std::string line = readLineFromBridge();
            if (line.empty()) return "";
            if (line[0] != '{') {
                std::cerr << "[Voice][Bridge][LOG] " << line << std::endl;
                continue; // skip non-JSON
            }
            return line;
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

                    if (v.contains("rules") && v["rules"].is_object()) {
                        g_rules.clear();
                        for (auto& [k, val] : v["rules"].items()) {
                            g_rules[k] = val.get<std::string>();
                        }
                    }
                }
                std::cout << "[Voice][Init] Loaded config: engine=" << g_engine
                          << ", speaker=" << g_speaker
                          << ", speed=" << g_speed
                          << ", output_dir=" << g_outputDir
                          << ", rules=" << g_rules.size() << std::endl;
            } else {
                std::cerr << "[Voice][Init] WARNING: ai_config.json not found, using defaults." << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[Voice][Init] ERROR reading ai_config.json: " << e.what()
                      << " (using defaults)" << std::endl;
        }

#ifdef _WIN32
        if (g_engine == "coqui") {
            SECURITY_ATTRIBUTES saAttr{};
            saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
            saAttr.bInheritHandle = TRUE;

            HANDLE hChildStdinRd, hChildStdoutWr;
            CreatePipe(&hChildStdinRd, &hChildStdinWr, &saAttr, 0);
            CreatePipe(&hChildStdoutRd, &hChildStdoutWr, &saAttr, 0);

            STARTUPINFOA siStartInfo{};
            siStartInfo.cb = sizeof(STARTUPINFOA);
            siStartInfo.hStdError = hChildStdoutWr;
            siStartInfo.hStdOutput = hChildStdoutWr;
            siStartInfo.hStdInput = hChildStdinRd;
            siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

            std::string cmd = "python D:/G.R.I.M/resources/python/coqui_bridge.py --persistent";

            if (!CreateProcessA(
                nullptr,
                cmd.data(),
                nullptr, nullptr, TRUE, 0,
                nullptr, nullptr,
                &siStartInfo, &piProcInfo))
            {
                std::cerr << "[Voice][Init] ERROR: Failed to start Coqui bridge" << std::endl;
                return false;
            }

            // Wait for {"status":"ready"}
            std::string response = readJsonLineFromBridge();
            try {
                auto resp = json::parse(response);
                if (resp.value("status", "") == "ready") {
                    std::cout << "[Voice] Bridge ready." << std::endl;
                }
            } catch (...) {
                std::cerr << "[Voice] ERROR: Bridge did not return valid JSON startup" << std::endl;
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

            std::string response = readJsonLineFromBridge();
            std::cout << "[Voice] Bridge shutdown response: " << response << std::endl;
        }
        if (piProcInfo.hProcess) {
            WaitForSingleObject(piProcInfo.hProcess, 2000);
            CloseHandle(piProcInfo.hProcess);
            CloseHandle(piProcInfo.hThread);
        }
#endif
        std::cout << "[Voice] shutdownTTS complete" << std::endl;
    }

    // =========================================================
    // Playback
    // =========================================================
    void playAudio(const std::string& path) {
        try {
            auto buffer = std::make_unique<sf::SoundBuffer>();
            if (!buffer->loadFromFile(path)) {
                std::cerr << "[Voice][Audio] ERROR: Could not load file: " << path << std::endl;
                return;
            }

            auto sound = std::make_unique<sf::Sound>(*buffer);
            sound->play();

            activeBuffers.push_back(std::move(buffer));
            activeSounds.push_back(std::move(sound));

            std::cout << "[Voice][Audio] Playing: " << path << std::endl;

            cleanupSounds();
        } catch (const std::exception& e) {
            std::cerr << "[Voice][Audio] Exception: " << e.what() << std::endl;
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
            std::cerr << "[Voice][Coqui] Bridge not running" << std::endl;
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

        DWORD written;
        WriteFile(hChildStdinWr, line.c_str(), (DWORD)line.size(), &written, nullptr);

        std::string response = readJsonLineFromBridge();

        try {
            auto resp = json::parse(response);
            if (resp.contains("file"))
                return resp["file"].get<std::string>();
            std::cerr << "[Voice][Coqui] ERROR: " << resp.dump() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[Voice][Coqui] Parse error: " << e.what()
                      << " (raw=" << response << ")" << std::endl;
        }
#endif
        return "";
    }

    // =========================================================
    // High-level Speak
    // =========================================================
    void speak(const std::string& text, const std::string& category) {
        std::thread([text, category]() {
            std::cout << "[Voice] speak(text=\"" << text
                      << "\", category=\"" << category << "\")" << std::endl;

            std::string engine = g_engine;
            auto it = g_rules.find(category);
            if (it != g_rules.end()) {
                engine = it->second;
            }

            if (engine == "coqui") {
                std::string wavPath = coquiSpeak(text, g_speaker, g_speed);
                if (!wavPath.empty()) {
                    playAudio(wavPath);
                }
                return;
            }

#ifdef _WIN32
            if (engine == "sapi") {
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
