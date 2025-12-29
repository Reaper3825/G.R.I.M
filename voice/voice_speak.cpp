#include "voice_speak.hpp"
#include "tts_cache.hpp"  // ? Added
#include "logger.hpp"
#include "popup_ui/popup_ui.hpp"
#include "core/audio_core.hpp"
#include "ai/ai.hpp"  // ✅ NEW: For AI fallback message generation
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
    static std::string g_speaker    = "default";  // ? Changed from "p225" to "default"
    static double      g_speed      = 1.0;
    static std::string g_language   = "en";  // ? Added language support for XTTS v2
    static fs::path    g_outputDir  = "D:/G.R.I.M/resources/tts_out";
    static std::unordered_map<std::string, std::string> g_rules;

    // =========================================================
    // Bridge state
    // =========================================================
    static bool g_ttsReady = false;
    static bool g_xttsV2Enabled = false;  // ? Track if XTTS v2 is loaded

    #ifdef _WIN32
    static HANDLE hChildStdinWr = nullptr;
    static HANDLE hChildStdoutRd = nullptr;
    static HANDLE hChildStderrRd = nullptr;
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
    static std::mutex ttsGenerationMutex;  // ? ADD: Prevent concurrent TTS generation
    static std::mutex playbackMutex;  // ? NEW: Prevent concurrent playback

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

    #ifdef _WIN32
    // Thread to read child's stderr so it doesn't get mixed into stdout JSON stream
    static std::thread stderrReaderThread;

    static void stderrReaderLoop() {
        if (!hChildStderrRd) return;

        std::string line;
        char ch;
        DWORD read = 0, avail = 0;

        while (true) {
            if (!PeekNamedPipe(hChildStderrRd, nullptr, 0, nullptr, &avail, nullptr)) {
                break;
            }
            if (avail == 0) {
                // If process exited, stop
                DWORD exitCode = 0;
                if (piProcInfo.hProcess && GetExitCodeProcess(piProcInfo.hProcess, &exitCode) && exitCode != STILL_ACTIVE) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            if (!ReadFile(hChildStderrRd, &ch, 1, &read, nullptr) || read == 0) {
                break;
            }
            if (ch == '\r') continue;
            if (ch == '\n') {
                if (!line.empty()) {
                    LOG_DEBUG("Voice/Bridge/Stderr", line);
                    line.clear();
                }
                continue;
            }
            line.push_back(ch);
        }

        if (!line.empty()) {
            LOG_DEBUG("Voice/Bridge/Stderr", line);
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
            fs::path cfgPath = fs::path("D:/G.R.I.M/ai_config.json");
            if (fs::exists(cfgPath)) {
                std::ifstream in(cfgPath);
                json cfg;
                in >> cfg;
                if (cfg.contains("voice")) {
                    auto& v = cfg["voice"];
                    if (v.contains("engine"))      g_engine    = v["engine"].get<std::string>();
                    if (v.contains("speaker"))     g_speaker   = v["speaker"].get<std::string>();
                    if (v.contains("speed"))       g_speed     = v["speed"].get<double>();
                    if (v.contains("language"))    g_language  = v["language"].get<std::string>();  // ? Load language
                    if (v.contains("output_dir"))  g_outputDir = v["output_dir"].get<std::string>();
                    
                    LOG_DEBUG("Voice/Init", "Loaded config: speaker=" + g_speaker + ", language=" + g_language + ", engine=" + g_engine);
                }
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Voice/Init", std::string("Error reading ai_config.json: ") + e.what());
        }

#ifdef _WIN32
        if (g_engine == "coqui" && !g_ttsReady) {
            // =========================================================
            // Launch Coqui XTTS v2 Python Bridge (Optimized with FP16 + torch.compile)
            // =========================================================
            SECURITY_ATTRIBUTES saAttr{};
            saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
            saAttr.bInheritHandle = TRUE;
            saAttr.lpSecurityDescriptor = nullptr;

            HANDLE hChildStdoutRdLocal = nullptr, hChildStdoutWr = nullptr;
            HANDLE hChildStdinRd = nullptr, hChildStdinWrLocal = nullptr;
            HANDLE hChildStderrRdLocal = nullptr, hChildStderrWr = nullptr;

            // Create pipe for child process's STDOUT
            if (!CreatePipe(&hChildStdoutRdLocal, &hChildStdoutWr, &saAttr, 0))
                LOG_ERROR("Voice/Init", "Failed to create stdout pipe");

            SetHandleInformation(hChildStdoutRdLocal, HANDLE_FLAG_INHERIT, 0);

            // Create pipe for child process's STDIN
            if (!CreatePipe(&hChildStdinRd, &hChildStdinWrLocal, &saAttr, 0))
                LOG_ERROR("Voice/Init", "Failed to create stdin pipe");

            SetHandleInformation(hChildStdinWrLocal, HANDLE_FLAG_INHERIT, 0);

            // Create pipe for child process's STDERR (separate from STDOUT)
            if (!CreatePipe(&hChildStderrRdLocal, &hChildStderrWr, &saAttr, 0))
                LOG_ERROR("Voice/Init", "Failed to create stderr pipe");

            SetHandleInformation(hChildStderrRdLocal, HANDLE_FLAG_INHERIT, 0);

            // Set up STARTUPINFOA
            STARTUPINFOA si{};
            ZeroMemory(&piProcInfo, sizeof(piProcInfo));
            si.cb = sizeof(STARTUPINFOA);
            si.hStdError  = hChildStderrWr;
            si.hStdOutput = hChildStdoutWr;
            si.hStdInput  = hChildStdinRd;
            si.dwFlags |= STARTF_USESTDHANDLES;

            // ? Launch XTTS v2 bridge with GPU support and configured speaker
            // Use the virtual environment Python interpreter to ensure dependencies are available
            std::string pythonExe = "D:/G.R.I.M/.venv/Scripts/python.exe";
            std::string scriptPath = "D:/G.R.I.M/resources/python/coqui_bridge.py";
            std::string cmd = pythonExe + " " + scriptPath + " --persistent --model tts_models/multilingual/multi-dataset/xtts_v2 --gpu --speaker " + g_speaker + " --language " + g_language;
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
            CloseHandle(hChildStderrWr);

            // Store pipe ends we need
            hChildStdoutRd = hChildStdoutRdLocal;
            hChildStdinWr = hChildStdinWrLocal;
            hChildStderrRd = hChildStderrRdLocal;

            // Launch stderr reader thread to log child's stderr separately
            try {
                stderrReaderThread = std::thread(stderrReaderLoop);
            } catch (...) {
                LOG_ERROR("Voice/Init", "Failed to start stderr reader thread");
            }

            // =========================================================
            // Extended handshake + ready logic for XTTS v2
            // =========================================================
            LOG_DEBUG("Voice", "Launching Coqui XTTS v2 bridge and waiting up to 120s for model load...");

            std::string response = readJsonLineFromBridge(120000);  // ? Increased timeout for XTTS v2 model load

            if (response.empty()) {
                LOG_ERROR("Voice/Init", "No handshake received from Coqui XTTS v2 after 120s. "
                                        "Process kept alive but marked not ready.");
                return false;
            }

            try {
                auto resp = json::parse(response);
                if (resp.value("status", "") == "ready") {
                    g_ttsReady = true;
                    g_xttsV2Enabled = resp.value("xtts_v2", false);  // ? Check if XTTS v2
                    std::string device = resp.value("device", "cpu");
                    std::string model = resp.value("model", "unknown");
                    
                    LOG_PHASE("Coqui XTTS v2 bridge ready", true);
                    LOG_DEBUG("Voice", "Model: " + model + " | Device: " + device + " | XTTS v2: " + 
                             (g_xttsV2Enabled ? "Yes" : "No"));
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
        if (hChildStderrRd) {
            CloseHandle(hChildStderrRd);
            hChildStderrRd = nullptr;
        }

        if (stderrReaderThread.joinable()) {
            // Give the stderr thread a brief moment to finish
            stderrReaderThread.join();
        }
#endif
        // ? Shutdown cache (saves index + cleanup)
        TTSCache::shutdown();
        
        LOG_PHASE("Voice shutdownTTS complete", true);
        g_ttsReady = false;
    }
    
    // =========================================================
    // ? NEW: Initialize Pre-cache in Background
    // =========================================================
    void initPreCache() {
        if (g_ttsReady) {
            // Start pre-caching in background thread
            std::thread([]() {
                // Wait for model warmup to complete
                std::this_thread::sleep_for(std::chrono::seconds(3));
                preCacheCommonPhrases();
            }).detach();
        }
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

    // =========================================================
    // ? NEW: Pre-cache Common Phrases for Instant Playback
    // =========================================================
    void preCacheCommonPhrases() {
        std::vector<std::string> commonPhrases = {
            "Yes",
            "No",
            "Okay",
            "I'm listening",
            "How can I help you?",
            "Understood",
            "Processing",
            "Done",
            "Ready",
            "Error occurred",
            "Please wait",
            "Command executed",
            "I'm here",
            "Go ahead",
            "Affirmative",
            "Negative",
            "Acknowledged"
        };
        
        LOG_DEBUG("Voice", "Pre-caching common phrases...");
        
        int cached = 0;
        int alreadyCached = 0;
        
        for (const auto& phrase : commonPhrases) {
            // Check if already cached
            std::string existing = TTSCache::getCached(phrase, "default", 1.0);
            
            if (existing.empty()) {
                // Generate and cache
                std::string wav = coquiSpeak(phrase, "default", 1.0);
                if (!wav.empty()) {
                    TTSCache::store(phrase, "default", 1.0, wav);
                    cached++;
                }
            } else {
                alreadyCached++;
            }
        }
        
        LOG_DEBUG("Voice", "Pre-cache complete: " + std::to_string(cached) + 
                 " generated, " + std::to_string(alreadyCached) + " already cached");
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
    // ? LOCK: Only one audio playback at a time
    std::lock_guard<std::mutex> playbackLock(playbackMutex);
    
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
        g_isSpeaking = false;
        return;
    }

    // Block until playback finishes (non-threaded mode)
    while (Audio::isPlaying()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    g_isSpeaking = false;
    LOG_DEBUG("Voice/Audio", "Playback complete for: " + path);
}



    // =========================================================
    // Coqui Speak (XTTS v2 Enhanced)
    // =========================================================
    std::string coquiSpeak(const std::string& text,
                           const std::string& speaker,
                           double speed) {
#ifdef _WIN32
        // ? LOCK: Prevent concurrent TTS generation
        std::lock_guard<std::mutex> generationLock(ttsGenerationMutex);
        
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

        // ? Enhanced request with XTTS v2 parameters
        json req = {
            {"command", "speak"},
            {"text", text},
            {"speaker", speaker},
            {"speed", speed},
            {"language", g_language},  // ? Include language for XTTS v2
            {"out", outFile}
        };
        std::string line = req.dump() + "\n";

        DWORD written = 0;
        BOOL ok = WriteFile(hChildStdinWr, line.c_str(), (DWORD)line.size(), &written, nullptr);
        
        if (g_xttsV2Enabled) {
            LOG_DEBUG("Voice/Coqui", "XTTS v2 request (" + std::to_string(written) + " bytes) | Speaker: " + speaker + " | Lang: " + g_language);
        } else {
            LOG_DEBUG("Voice/Coqui", "Sent request (" + std::to_string(written) + " bytes): " + line);
        }
        
        if (!ok) {
            LOG_ERROR("Voice/Coqui", "WriteFile failed");
            return "";
        }

        // ? Increased timeout for XTTS v2 (can be slower but higher quality)
        std::string response = readJsonLineFromBridge(60000);  // 60s timeout

        if (response.empty()) {
            LOG_ERROR("Voice/Bridge", "Timeout waiting for Coqui response (60s)");
            return "";
        }

        try {
            auto resp = json::parse(response);

            if (resp.value("status", "") == "ok" && resp.contains("file")) {
                std::string generatedFile = resp["file"].get<std::string>();
                LOG_DEBUG("Voice/Bridge", "Received response: " + response);
                
                // ? FIX: Store in cache and get the FINAL path (might be moved)
                std::string finalPath = TTSCache::store(text, speaker, speed, generatedFile);
                
                // ? Return the final path (from cache, not temp)
                if (!finalPath.empty() && fs::exists(finalPath)) {
                    LOG_DEBUG("Voice/Coqui", "Returning final cached path: " + finalPath);
                    return finalPath;
                } else {
                    // Fallback: if cache failed, use original file
                    LOG_DEBUG("Voice/Coqui", "Cache storage failed, using temp file: " + generatedFile);
                    return generatedFile;
                }
            } else if (resp.value("status", "") == "fallback_notice") {
                // ✅ NEW: Handle fallback notice - have AI generate a "hold on" message
                std::string fallbackMsg = resp.value("message", "Processing...");
                LOG_DEBUG("Voice/Bridge", "Fallback notice received: " + fallbackMsg);
                
                // Generate an AI response for "give me a moment" type message
                std::string aiPrompt = "Generate a brief, in-character message saying you need a moment to process something. Keep it under 15 words and make it sound natural for your personality.";
                
                // Call AI backend asynchronously to generate the message
                auto aiFuture = callAIAsync(aiPrompt);
                
                // Wait for AI response with timeout
                if (aiFuture.wait_for(std::chrono::seconds(3)) == std::future_status::ready) {
                    std::string aiResponse = aiFuture.get();
                    if (!aiResponse.empty()) {
                        LOG_DEBUG("Voice/Fallback", "AI generated fallback message: " + aiResponse);
                        // Speak the AI-generated message immediately while processing continues
                        speak(aiResponse, "system");
                    }
                } else {
                    LOG_DEBUG("Voice/Fallback", "AI timeout, using default message");
                    speak("Give me just a moment...", "system");
                }
                
                // Continue processing - wait for the actual TTS result
                response = readJsonLineFromBridge(60000);
                if (!response.empty()) {
                    resp = json::parse(response);
                    if (resp.value("status", "") == "ok" && resp.contains("file")) {
                        std::string generatedFile = resp["file"].get<std::string>();
                        std::string finalPath = TTSCache::store(text, speaker, speed, generatedFile);
                        return !finalPath.empty() ? finalPath : generatedFile;
                    }
                }
                return "";
            } else {
                std::string errMsg = resp.value("message", "unknown error");
                LOG_ERROR("Voice/Bridge", "Coqui TTS error: " + errMsg);
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
        // ? FIX: Strip "GRIM:" or "GRIM :" prefix if present
        std::string cleanedText = text;
        
        // Check for "GRIM:" or "GRIM :" at the start (case-insensitive)
        std::string lowerText = text;
        std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(), ::tolower);
        
        if (lowerText.find("grim:") == 0) {
            cleanedText = text.substr(5);  // Remove "grim:"
        } else if (lowerText.find("grim :") == 0) {
            cleanedText = text.substr(6);  // Remove "grim :"
        }
        
        // Trim leading/trailing whitespace
        cleanedText.erase(0, cleanedText.find_first_not_of(" \t\n\r"));
        cleanedText.erase(cleanedText.find_last_not_of(" \t\n\r") + 1);
        
        // If cleaning resulted in empty text, use original
        if (cleanedText.empty()) {
            cleanedText = text;
        }
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            speakQueue.emplace(cleanedText, category);
        }
        queueCV.notify_one();
    }
    
    // =========================================================
    // XTTS v2 Utility Functions
    // =========================================================
    bool isXTTSv2Enabled() {
        return g_xttsV2Enabled;
    }
    
    void setLanguage(const std::string& lang) {
        g_language = lang;
        LOG_DEBUG("Voice", "Language set to: " + lang);
    }
    
    std::string getLanguage() {
        return g_language;
    }
    
    void setSpeaker(const std::string& speaker) {
        g_speaker = speaker;
        LOG_DEBUG("Voice", "Speaker updated to: " + speaker);
    }
    
    std::string getSpeaker() {
        return g_speaker;
    }

} // namespace Voice
