#include "voice_command_services.hpp"
#include "voice_platform_queries.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"
#include "voice_speak.hpp"
#include "tts_cache.hpp"
#include "core/audio_core.hpp"
#include "MMO/Core/SessionContextManager.hpp"
#include "resources.hpp"
#include <stdexcept>

namespace GRIM::VoiceCommandServices {
nlohmann::json execute(GrimVoiceOperation operation, const nlohmann::json& request) {
    switch (operation) {
        case GRIM_VOICE_CAPTURE: return Voice::captureAndTranscribeSpeech(aiConfig);
        case GRIM_VOICE_HAS_STREAM_CONTEXT: return Voice::g_state.ctx != nullptr;
        case GRIM_VOICE_START_STREAM: {
            auto& history = MMO::SessionContextManager::instance().displayHistory("default");
            return VoiceStream::start(Voice::g_state.ctx, &history, timers, longTermMemory);
        }
        case GRIM_VOICE_XTTS_ENABLED: return Voice::isXTTSv2Enabled();
        case GRIM_VOICE_LANGUAGE: return Voice::getLanguage();
        case GRIM_VOICE_SYNTHESIZE_COQUI:
            return Voice::coquiSpeak(request.at("text").get<std::string>(), request.at("speaker").get<std::string>(), request.at("speed").get<double>());
        case GRIM_VOICE_PLAY_AUDIO: Voice::playAudio(request.at("path").get<std::string>()); return nullptr;
        case GRIM_VOICE_CONFIG: return aiConfig["voice"];
        case GRIM_VOICE_LIST_SAPI: return listSapiVoices();
        case GRIM_VOICE_SAPI_DEVICE: return sapiOutputDevice();
        case GRIM_VOICE_PLAY_WAV: return Audio::playWav(request.at("path").get<std::string>());
        case GRIM_VOICE_IS_SPEAKING: return Voice::isSpeaking();
        case GRIM_VOICE_CACHE_SHUTDOWN: Voice::TTSCache::shutdown(); return nullptr;
        case GRIM_VOICE_CACHE_INIT: Voice::TTSCache::init(); return nullptr;
        case GRIM_VOICE_PRECACHE_COMMON: Voice::preCacheCommonPhrases(); return nullptr;
        case GRIM_VOICE_PATHS: return paths();
    }
    throw std::invalid_argument("Unknown voice operation");
}
}
