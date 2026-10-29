#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include "MMO/Core/SessionContextManager.hpp"
#include "voice/voice.hpp"
#include "voice/voice_stream.hpp"
#include "voice/voice_speak.hpp"
#include "voice/voice_platform_queries.hpp"
#include "voice/tts_cache.hpp"
#include "core/audio_core.hpp"
#include "resources.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

nlohmann::json aiConfig, longTermMemory;
namespace {
std::string transcript, wavPath, lastText, lastPlayed, language = "fr";
bool xtts = true, streamSucceeds = true, streamRunning = false, speaking = false, wavSucceeds = true, cacheActive = true;
int captures = 0, starts = 0, syntheses = 0, audioPlays = 0, wavPlays = 0, precaches = 0, failureMode = 0;
std::vector<std::string> cacheEvents;
nlohmann::json fixturePaths, sapiVoices, sapiDevice;
void maybeFail() {
    if (failureMode == 1) throw std::runtime_error("fixture voice failure");
    if (failureMode == 2) throw 42;
}
}
namespace Voice {
State g_state;
std::string captureAndTranscribeSpeech(nlohmann::json& config) {
    ++captures; maybeFail();
    if (&config != &aiConfig) throw std::runtime_error("Capture used copied config");
    config["capture_used"] = true; return transcript;
}
bool isXTTSv2Enabled() { return xtts; }
std::string getLanguage() { return language; }
std::string coquiSpeak(const std::string& text, const std::string& speaker, double speed, double pitch) {
    ++syntheses; maybeFail(); lastText = text;
    if (speaker != "default" || speed != 1.0 || pitch != 1.0) throw std::runtime_error("Synthesis defaults changed");
    return wavPath;
}
void playAudio(const std::string& path) { ++audioPlays; lastPlayed = path; maybeFail(); }
bool isSpeaking() { return speaking; }
void preCacheCommonPhrases() { ++precaches; cacheEvents.push_back("precache"); maybeFail(); }
namespace TTSCache {
void init() { cacheEvents.push_back("init"); cacheActive = true; maybeFail(); }
void shutdown() { cacheEvents.push_back("shutdown"); cacheActive = false; maybeFail(); }
}
}
namespace VoiceStream {
bool start(whisper_context* context, ConsoleHistory* history, std::vector<Timer>& pendingTimers, nlohmann::json& memory) {
    ++starts; maybeFail();
    if (context != Voice::g_state.ctx || history != &GRIM::MMO::SessionContextManager::instance().displayHistory("default") ||
        &pendingTimers != &timers || &memory != &longTermMemory) throw std::runtime_error("Streaming host state was copied/replaced");
    if (streamSucceeds) streamRunning = true;
    return streamSucceeds;
}
}
namespace Audio {
bool playWav(const std::string& path) { ++wavPlays; lastPlayed = path; maybeFail(); return wavSucceeds; }
}
namespace GRIM::VoiceCommandServices {
nlohmann::json paths() { return fixturePaths; }
nlohmann::json listSapiVoices() { return sapiVoices; }
nlohmann::json sapiOutputDevice() { return sapiDevice; }
}

void runVoicePluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance(); const auto baseline = registry.toolCount();
    const auto folder = fs::temp_directory_path() / ("grim-voice-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create voice fixture");
    struct Cleanup {
        fs::path folder;
        GrimPluginAPI* api = nullptr;
        ~Cleanup() {
            PluginManager::unloadAll(); if (api) PluginAPI::cleanupPluginContext(api);
            Voice::g_state.ctx = nullptr; failureMode = 0;
            std::error_code error;
            // Exact fixture paths only; no recursive removal or real cache paths.
            for (const auto* name : {"voice_plugin.dll", "test.wav", "embeddings/demo.npz", "embeddings/keep.txt", "embeddings",
                "temp/a.wav", "temp/keep.txt", "temp/blocked.wav/keep.txt", "temp/blocked.wav", "temp", "cache/b.wav", "cache", "index.json"})
                fs::remove(folder / name, error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    fixturePaths = {{"test_wav", (folder / "test.wav").string()}, {"embeddings", (folder / "embeddings").string()},
        {"temp", (folder / "temp").string()}, {"cache", (folder / "cache").string()}, {"cache_index", (folder / "index.json").string()}};
    captures = starts = syntheses = audioPlays = wavPlays = precaches = failureMode = 0;
    transcript.clear(); wavPath.clear(); cacheEvents.clear(); Voice::g_state.ctx = nullptr;
    auto* denied = PluginAPI::createPluginAPI("voice-denied", "", 0);
    check(denied->voice_service(denied, GRIM_VOICE_CAPTURE, nullptr) == nullptr && captures == 0, "Voice permission bypass");
    check(denied->voice_service(nullptr, GRIM_VOICE_CAPTURE, nullptr) == nullptr, "Voice accepted invalid API identity");
    PluginAPI::cleanupPluginContext(denied);
    cleanup.api = PluginAPI::createPluginAPI("voice-reader", "", GRIM_PERM_VOICE);
    check(cleanup.api->voice_service(cleanup.api, static_cast<GrimVoiceOperation>(99), nullptr) == nullptr &&
          cleanup.api->voice_service(cleanup.api, GRIM_VOICE_CAPTURE, "invalid") == nullptr &&
          cleanup.api->voice_service(cleanup.api, GRIM_VOICE_CAPTURE, "[]") == nullptr && captures == 0, "Invalid voice request executed");
    check(cleanup.api->voice_service(cleanup.api, GRIM_VOICE_CACHE_SHUTDOWN, nullptr) == nullptr && cacheEvents.empty(), "Cache operation bypassed filesystem permission");
    fs::copy_file(pluginDirectory / "voice_plugin.dll", folder / "voice_plugin.dll");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 11 && captures == 0 && starts == 0 && cacheEvents.empty(), "Voice load changed host state or registration count");
    const char* ids[] = {"voice", "voice_stream", "test_tts", "list_voices", "test_sapi", "tts_device", "nevermind", "create_embedding", "list_embeddings", "clear_tts_cache", "reset_tts_cache"};
    for (const auto* id : ids) {
        const auto d = registry.getTool(id); const std::string name = id;
        const bool informational = name == "list_voices" || name == "list_embeddings";
        check(d && d->provider_name == "Voice Plugin" && d->provider_type == ToolProviderType::Plugin &&
              (d->permission_bits & GRIM_PERM_VOICE) && d->is_informational == informational && d->needs_confirmation != informational,
              "Voice metadata/provider/policy incorrect");
    }
    const auto invoke = [&](const char* id, const std::string& argument = "") { return registry.invoke(id, argument); };
    const auto matches = [&](const CommandResult& r, bool success, const std::string& message, const char* code, const char* category,
                             const std::string& voice, Color color) {
        check(r.success == success && r.message == message && r.errorCode == code && r.category == category && r.voice == voice &&
              r.color.toUInt() == color.toUInt() && r.structuredOutputJson.empty(), "Voice command result differs from original");
    };
    matches(invoke("voice", "ignored"), false, "configured:ERR_VOICE_NO_SPEECH", "ERR_VOICE_NO_SPEECH", "error", "No speech detected", Colors::Red);
    transcript = "literal transcript";
    matches(invoke("voice"), true, "> literal transcript", "ERR_NONE", "routine", "", Colors::Cyan);
    check(aiConfig.at("capture_used") == true, "Capture did not receive live config");
    matches(invoke("voice_stream"), false, "configured:ERR_VOICE_NO_CONTEXT", "ERR_VOICE_NO_CONTEXT", "error", "Voice context missing", Colors::Red);
    check(starts == 0, "Missing stream context attempted start");
    Voice::g_state.ctx = reinterpret_cast<whisper_context*>(uintptr_t{1}); streamSucceeds = false;
    matches(invoke("voice_stream"), false, "configured:ERR_VOICE_STREAM_FAIL", "ERR_VOICE_STREAM_FAIL", "error", "Voice streaming failed", Colors::Red);
    streamSucceeds = true;
    matches(invoke("voice_stream", "ignored"), true, "[Voice] Streaming started.", "ERR_NONE", "routine", "Voice streaming started", Colors::Green);
    check(streamRunning, "Stream did not remain host owned");
    matches(invoke("test_tts"), false, "[Voice][Test] ERROR: Coqui XTTS v2 TTS failed.", "", "", "", Colors::Red);
    check(lastText == "This is a Coqui XTTS version two voice test." && audioPlays == 0, "TTS defaults/failure playback changed");
    wavPath = (folder / "generated.wav").string();
    matches(invoke("test_tts", "Custom text"), true, "[Voice][Test] Coqui XTTS v2 playback requested.", "", "", "", Colors::Green);
    check(lastText == "Custom text" && lastPlayed == wavPath && audioPlays == 1, "TTS argument/playback wiring changed");
    aiConfig["voice"] = {{"engine", "coqui"}, {"speaker", "fixture"}, {"speed", 1.25}}; xtts = true;
    matches(invoke("list_voices"), true, "[Voice] Current Coqui XTTS v2 configuration:\n - Model: tts_models/multilingual/multi-dataset/xtts_v2\n - Engine: XTTS v2 (Voice Cloning)\n - Language: fr\n - Supported Languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi\n - Speaker: fixture\n - Speed: 1.25\n", "ERR_NONE", "debug", "Coqui voices listed", Colors::Yellow);
    xtts = false;
    matches(invoke("list_voices"), true, "[Voice] Current Coqui XTTS v2 configuration:\n - Model: Legacy Coqui TTS (VITS)\n - Speaker: fixture\n - Speed: 1.25\n", "ERR_NONE", "debug", "Coqui voices listed", Colors::Yellow);
    aiConfig["voice"]["engine"] = "sapi";
    sapiVoices = {{"com_ok", false}};
    matches(invoke("list_voices"), false, "[Voice][Error] Failed to initialize COM.", "ERR_TTS_COM", "debug", "COM init failed", Colors::Red);
    sapiVoices = {{"com_ok", true}, {"enumerated", false}};
    matches(invoke("list_voices"), false, "[Voice][Error] Failed to enumerate SAPI voices.", "ERR_TTS_ENUM", "debug", "Failed to list SAPI voices", Colors::Red);
    sapiVoices = {{"com_ok", true}, {"enumerated", true}, {"count", 2}, {"voices", {"One", "Two"}}};
    matches(invoke("list_voices", "ignored"), true, "[Voice] Found 2 installed SAPI voices:\n - One\n - Two\n", "ERR_NONE", "debug", "SAPI voices listed", Colors::Yellow);
    sapiDevice = {{"com_ok", false}};
    matches(invoke("tts_device"), false, "[Voice][Error] Failed to initialize COM.", "ERR_TTS_COM", "debug", "COM init failed", Colors::Red);
    sapiDevice = {{"com_ok", true}, {"voice_created", false}};
    matches(invoke("tts_device"), false, "[Voice][Error] Failed to create SAPI voice instance.", "ERR_TTS_INIT", "debug", "SAPI init failed", Colors::Red);
    sapiDevice = {{"com_ok", true}, {"voice_created", true}, {"has_output", false}};
    matches(invoke("tts_device"), true, "[Voice] Could not retrieve current output device.\n", "ERR_NONE", "debug", "Device info", Colors::Yellow);
    sapiDevice["has_output"] = true; sapiDevice["description"] = nullptr;
    matches(invoke("tts_device"), true, "", "ERR_NONE", "debug", "Device info", Colors::Yellow);
    sapiDevice["description"] = "Fixture speakers";
    matches(invoke("tts_device", "ignored"), true, "[Voice] Current SAPI output device: Fixture speakers\n", "ERR_NONE", "debug", "Device info", Colors::Yellow);
    matches(invoke("test_sapi"), false, "[Audio] Test file not found.", "ERR_AUDIO_MISSING", "error", "Missing audio file", Colors::Red);
    { std::ofstream file(folder / "test.wav"); file << "fixture"; }
    wavSucceeds = false;
    matches(invoke("test_sapi"), false, "[Audio] Playback failed.", "ERR_AUDIO_FAIL", "error", "Audio playback failed", Colors::Red);
    wavSucceeds = true;
    matches(invoke("test_sapi", "ignored"), true, "[Audio] Test file played successfully.", "ERR_NONE", "routine", "Audio playback succeeded", Colors::Green);
    speaking = true;
    matches(invoke("nevermind"), true, "Alright, cancelled.", "ERR_NONE", "cancellation", "Alright, cancelled.", Color(128,128,255));
    check(speaking && streamRunning, "Cancellation placeholder unexpectedly stopped host state");
    check(invoke("create_embedding", "speaker-only").errorCode == "ERR_CMD_EXCEPTION", "Existing missing-path embedding exception changed");
    matches(invoke("create_embedding", "demo   C:/voice reference.wav"), true, "[Voice] Embedding creation queued for: demo", "ERR_NONE", "routine", "Creating voice embedding", Colors::Green);
    matches(invoke("list_embeddings"), true, "[Voice] Cached Speaker Embeddings:\n\nNo embeddings found. Create one with: create_embedding <speaker_id> <audio_file>\n", "ERR_NONE", "debug", "No embeddings found", Colors::Yellow);
    fs::create_directory(folder / "embeddings");
    matches(invoke("list_embeddings"), true, "[Voice] Cached Speaker Embeddings:\n\nNo embeddings found.\n", "ERR_NONE", "debug", "No embeddings found", Colors::Cyan);
    { std::ofstream file(folder / "embeddings/demo.npz", std::ios::binary); file << std::string(2048, 'x'); }
    matches(invoke("list_embeddings"), true, "[Voice] Cached Speaker Embeddings:\n\n - demo (2 KB)\n\nTotal: 1 embeddings\n\nUsage: Set speaker in ai_config.json to use cached embedding.\n", "ERR_NONE", "debug", "Embeddings listed", Colors::Cyan);
    const auto prepareCache = [&] {
        fs::create_directories(folder / "temp"); fs::create_directories(folder / "cache");
        for (const auto* path : {"temp/a.wav", "cache/b.wav", "temp/keep.txt", "index.json"}) { std::ofstream file(folder / path); file << "fixture"; }
        cacheEvents.clear();
    };
    prepareCache();
    matches(invoke("clear_tts_cache"), true, "TTS cache cleared. Removed 2 cached audio files. Cache will rebuild as needed.", "ERR_NONE", "routine", "Cache cleared successfully", Colors::Green);
    check(cacheEvents == std::vector<std::string>{"shutdown", "init"} && fs::exists(folder / "temp/keep.txt") &&
          !fs::exists(folder / "temp/a.wav") && !fs::exists(folder / "cache/b.wav") && !fs::exists(folder / "index.json"), "Cache clear lifecycle/files changed");
    prepareCache();
    matches(invoke("reset_tts_cache"), true, "TTS cache reset complete. Removed 2 old files and regenerated common phrases.", "ERR_NONE", "routine", "Cache reset with fresh audio", Colors::Green);
    check(cacheEvents == std::vector<std::string>{"shutdown", "init", "precache"} && precaches == 1 && cacheActive, "Cache reset ordering changed");
    failureMode = 1;
    const auto cacheFailure = invoke("clear_tts_cache");
    matches(cacheFailure, false, "Failed to clear cache: fixture voice failure", "ERR_CACHE_CLEAR_FAILED", "error", "Cache clear failed", Colors::Red);
    check(invoke("voice").errorCode == "ERR_CMD_EXCEPTION", "Voice standard exception did not propagate");
    failureMode = 2;
    check(invoke("voice").message == "[Error] Exception while running command: voice", "Unknown voice exception changed");
    failureMode = 0; transcript = std::string(70000, 'x');
    matches(invoke("voice"), true, "> " + transcript, "ERR_NONE", "routine", "", Colors::Cyan);
    const auto beforeCapture = captures, beforeStart = starts, beforeSynthesis = syntheses, beforePlayback = audioPlays + wavPlays;
    const auto events = cacheEvents;
    fs::remove(folder / "voice_plugin.dll"); PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && captures == beforeCapture && starts == beforeStart && syntheses == beforeSynthesis &&
          audioPlays + wavPlays == beforePlayback && cacheEvents == events && streamRunning && speaking,
          "Voice unload changed host capture/stream/audio/cache state");
    for (const auto* id : ids) check(!registry.isRegistered(id), "Voice registration survived DLL removal");
    std::cout << "Voice plugin: 11 commands, host state, exact results, native-query fixtures, cache files, placeholders, and unload tests passed\n";
}
