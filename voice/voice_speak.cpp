#include "voice_speak.hpp"
#include "tts_cache.hpp"  // ? Added
#include "logger.hpp"
#include "popup_ui/popup_ui.hpp" 

#include "core/audio_core.hpp"
#include <thread>
#include <chrono>
#include <filesystem>
#include <random>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <nlohmann/json.hpp>

#ifdef _WIN32
    #include <windows.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace Voice {

    // =========================================================
    // Audio state
    // =========================================================


    static std::string g_engine     = "coqui";
    static std::string g_speaker    = "p225";
    static double      g_speed      = 1.0;
    static fs::path    g_outputDir  = "D:/G.R.I.M/resources/tts_out";
    static std::unordered_map<std::string, std::string> g_rules;

    // =========================================================
    // Bridge state
    // =========================================================
    static bool g_ttsReady = false;

    #ifdef _WIN32
    static HANDLE hChildStdinWr = nullptr;
    static HANDLE hChildStdoutRd = nullptr;
    static PROCESS_INFORMATION piProcInfo{};
    #endif

    // =========================================================
    // Queue state
    // =========================================================
    static std::queue<std::pair<std::string,std::string>> speakQueue;
    static std::mutex queueMutex;
    static std::condition_variable queueCV;
    static bool workerRunning = false;
    static std::thread workerThread;

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
                break;
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
        // ? Initialize TTS cache first
        TTSCache::init();
        
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
            // =========================================================
            // Integrated pipe + process creation snippet
            // =========================================================
            SECURITY_ATTRIBUTES saAttr{};
            saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
            saAttr.bInheritHandle = TRUE;
            saAttr.lpSecurityDescriptor = nullptr;

            HANDLE hChildStdoutRdLocal = nullptr, hChildStdoutWr = nullptr;
            HANDLE hChildStdinRd = nullptr, hChildStdinWrLocal = nullptr;

            // Create pipe for child process's STDOUT
            if (!CreatePipe(&hChildStdoutRdLocal, &hChildStdoutWr, &saAttr, 0))
                LOG_ERROR("Voice/Init", "Failed to create stdout pipe");

            SetHandleInformation(hChildStdoutRdLocal, HANDLE_FLAG_INHERIT, 0);

            // Create pipe for child process's STDIN
            if (!CreatePipe(&hChildStdinRd, &hChildStdinWrLocal, &saAttr, 0))
                LOG_ERROR("Voice/Init", "Failed to create stdin pipe");

            SetHandleInformation(hChildStdinWrLocal, HANDLE_FLAG_INHERIT, 0);

            // Set up STARTUPINFOA
            STARTUPINFOA si{};
            ZeroMemory(&piProcInfo, sizeof(piProcInfo));
            si.cb = sizeof(STARTUPINFOA);
            si.hStdError  = hChildStdoutWr;
            si.hStdOutput = hChildStdoutWr;
            si.hStdInput  = hChildStdinRd;
            si.dwFlags |= STARTF_USESTDHANDLES;

            // Launch Python bridge
            std::string cmd = "python D:/G.R.I.M/resources/python/coqui_bridge.py --persistent";
            std::vector<char> mutableCmd(cmd.begin(), cmd.end());
            mutableCmd.push_back('\0');

            BOOL success = CreateProcessA(
                nullptr,
                mutableCmd.data(),
                nullptr,
                nullptr,
                TRUE,
                0,
                nullptr,
                "D:/G.R.I.M/resources/python",
                &si,
                &piProcInfo
            );

            if (!success) {
                LOG_ERROR("Voice/Init", "CreateProcessA failed: " + std::to_string(GetLastError()));
                return false;
            }

            // Close the unneeded ends in parent
            CloseHandle(hChildStdoutWr);
            CloseHandle(hChildStdinRd);

            // Store pipe ends we need
            hChildStdoutRd = hChildStdoutRdLocal;
            hChildStdinWr = hChildStdinWrLocal;

            // =========================================================
            // Extended handshake + ready logic
            // =========================================================
            LOG_DEBUG("Voice", "Launching Coqui TTS bridge and waiting up to 60s for handshake...");

            std::string response = readJsonLineFromBridge(60000);

            if (response.empty()) {
                LOG_ERROR("Voice/Init", "No handshake received from Coqui after 60s. "
                                        "Process kept alive but marked not ready.");
                return false;
            }

            try {
                auto resp = json::parse(response);
                if (resp.value("status", "") == "ready") {
                    g_ttsReady = true;
                    LOG_PHASE("Voice bridge ready", true);
                } else {
                    LOG_ERROR("Voice/Init", "Unexpected handshake payload: " + response);
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
        // ? Shutdown cache (saves index + cleanup)
        TTSCache::shutdown();
        
        LOG_PHASE("Voice shutdownTTS complete", true);
        g_ttsReady = false;
    }

    // =========================================================
    // Queue worker
    // =========================================================
    static void speakWorker() {
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            queueCV.wait(lock, [] { return !speakQueue.empty() || !workerRunning; });
            if (!workerRunning) break;

            auto [text, category] = speakQueue.front();
            speakQueue.pop();
            lock.unlock();

            LOG_DEBUG("Voice/Worker", "Processing: " + text);

            std::string engine = "coqui";
            auto it = g_rules.find(category);
            if (it != g_rules.end()) {
                if (it->second == "coqui" || it->second == "sapi")
                    engine = it->second;
            }

            if (engine == "coqui") {
                std::string wavPath = coquiSpeak(text, g_speaker, g_speed);
                if (!wavPath.empty()) {
                    playAudio(wavPath);
                    while (isPlaying()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }
            }
#ifdef _WIN32
            else if (engine == "sapi") {
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
        }
    }

    void initQueue() {
        workerRunning = true;
        workerThread = std::thread(speakWorker);
    }

    void shutdownQueue() {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            workerRunning = false;
        }
        queueCV.notify_all();
        if (workerThread.joinable()) workerThread.join();
    }

    bool isReady() {
        return g_ttsReady;
    }

    // =========================================================
    // Audio state and playback integration (AudioCore-based)
    // =========================================================
    // Tracks whether GRIM is currently speaking (via AudioCore)
    std::atomic<bool> g_isSpeaking{false};

    bool isSpeaking() {
        // Query AudioCore playback status
        return Audio::isPlaying() || g_isSpeaking.load();
    }

    bool isPlaying() {
        // Mirror AudioCore's tracking state
        return Audio::isPlaying();
    }

    // =========================================================
    // Voice playback bridge
    // =========================================================
    void playAudio(const std::string& path) {
    if (!Audio::init()) {
        LOG_ERROR("Voice/Audio", "PortAudio init failed.");
        return;
    }

    if (!std::filesystem::exists(path)) {
        LOG_ERROR("Voice/Audio", "File not found: " + path);
        return;
    }

    notifyPopupActivity();
    g_isSpeaking = true;

    LOG_DEBUG("Voice/Audio", "Playing via AudioCore: " + path);
    if (!Audio::playWav(path)) {
        LOG_ERROR("Voice/Audio", "AudioCore playback failed for: " + path);
    }

    // Block until playback finishes (non-threaded mode)
    while (Audio::isPlaying()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    g_isSpeaking = false;
    LOG_DEBUG("Voice/Audio", "Playback complete for: " + path);
}



    // =========================================================
    // Coqui Speak
    // =========================================================
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed) {
#ifdef _WIN32
        // ? Check cache first
        std::string cached = TTSCache::getCached(text, speaker, speed);
        if (!cached.empty() && fs::exists(cached)) {
            LOG_DEBUG("Voice/Coqui", "Using cached audio: " + cached);
            return cached;
        }

        if (!hChildStdinWr || !hChildStdoutRd) {
            LOG_ERROR("Voice/Coqui", "Bridge not running");
            return "";
        }

        // Generate to temp directory
        fs::path tempDir = fs::path("D:/G.R.I.M/resources/tts_out/temp");
        fs::create_directories(tempDir);
        std::string outFile = (tempDir / (randomString(32) + ".wav")).string();

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
            return "";
        }

        std::string response = readJsonLineFromBridge();

        if (response.empty()) {
            LOG_ERROR("Voice/Bridge", "Timeout waiting for Coqui response (30s)");
            return "";
        }

        try {
            auto resp = json::parse(response);

            if (resp.value("status", "") == "ok" && resp.contains("file")) {
                std::string generatedFile = resp["file"].get<std::string>();
                LOG_DEBUG("Voice/Bridge", "Received response: " + response);
                
                // ? Store in cache
                TTSCache::store(text, speaker, speed, generatedFile);
                
                return generatedFile;
            } else {
                LOG_DEBUG("Voice/Bridge", "Unexpected JSON from Coqui: " + response);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Voice/Bridge",
                      std::string("Failed to parse Coqui JSON: ") + e.what() +
                      " raw=" + response);
        }
#endif
        return "";
    }

    // =========================================================
    // High-level Speak (enqueue)
    // =========================================================
    void speak(const std::string& text, const std::string& category) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            speakQueue.emplace(text, category);
        }
        queueCV.notify_one();
    }

} // namespace Voice
