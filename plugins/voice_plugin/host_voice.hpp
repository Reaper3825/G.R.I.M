#pragma once
#include "plugins/runtime_service_client.hpp"
#include <filesystem>
#include <sstream>

namespace hostVoice {
inline nlohmann::json service(GrimVoiceOperation operation, const nlohmann::json& arguments = nlohmann::json::object()) {
    const auto text = arguments.dump();
    return grim::runtime::response(grim::runtime::host->voice_service(grim::runtime::host, operation, text.c_str()));
}
inline std::string captureAndTranscribeSpeech() { return service(GRIM_VOICE_CAPTURE).get<std::string>(); }
inline bool hasContext() { return service(GRIM_VOICE_HAS_STREAM_CONTEXT).get<bool>(); }
inline bool startStream() { return service(GRIM_VOICE_START_STREAM).get<bool>(); }
inline bool isXTTSv2Enabled() { return service(GRIM_VOICE_XTTS_ENABLED).get<bool>(); }
inline std::string getLanguage() { return service(GRIM_VOICE_LANGUAGE).get<std::string>(); }
inline std::string coquiSpeak(const std::string& text, const std::string& speaker, double speed) {
    return service(GRIM_VOICE_SYNTHESIZE_COQUI, {{"text", text}, {"speaker", speaker}, {"speed", speed}}).get<std::string>();
}
inline void playAudio(const std::string& path) { service(GRIM_VOICE_PLAY_AUDIO, {{"path", path}}); }
inline bool playWav(const std::string& path) { return service(GRIM_VOICE_PLAY_WAV, {{"path", path}}).get<bool>(); }
inline bool isSpeaking() { return service(GRIM_VOICE_IS_SPEAKING).get<bool>(); }
inline nlohmann::json config() { return service(GRIM_VOICE_CONFIG); }
inline nlohmann::json listSapiVoices() { return service(GRIM_VOICE_LIST_SAPI); }
inline nlohmann::json sapiOutputDevice() { return service(GRIM_VOICE_SAPI_DEVICE); }
inline nlohmann::json paths() { return service(GRIM_VOICE_PATHS); }
inline void preCacheCommonPhrases() { service(GRIM_VOICE_PRECACHE_COMMON); }
inline std::string errorMessage(const std::string& code) {
    return grim::runtime::take(grim::runtime::host->get_user_error_message(grim::runtime::host, code.c_str()));
}
namespace TTSCache {
inline void shutdown() { service(GRIM_VOICE_CACHE_SHUTDOWN); }
inline void init() { service(GRIM_VOICE_CACHE_INIT); }
}
}
CommandResult cmdVoice(const std::string&);
CommandResult cmdVoiceStream(const std::string&);
CommandResult cmd_testTTS(const std::string&);
CommandResult cmd_listVoices(const std::string&);
CommandResult cmd_testSAPI(const std::string&);
CommandResult cmd_ttsDevice(const std::string&);
CommandResult cmdNevermind(const std::string&);
CommandResult cmd_createEmbedding(const std::string&);
CommandResult cmd_listEmbeddings(const std::string&);
CommandResult cmdClearCache(const std::string&);
CommandResult cmdResetCache(const std::string&);
