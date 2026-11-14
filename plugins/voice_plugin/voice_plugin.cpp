#include "core/plugin_api.hpp"
#include "plugins/plugin_logging.hpp"
#include "runtime/runtime_state.hpp"
#include "runtime/voice.hpp"
#include "runtime/voice_stream.hpp"
#include "runtime/voice_speak.hpp"
#include "runtime/wake_voice.hpp"
#include "runtime/audio_core.hpp"
#include "runtime/tts_cache.hpp"
#include <mutex>
#include <filesystem>
#include <stdexcept>
namespace {
const GrimPluginAPI* host=nullptr;
std::mutex operationMutex;
bool configured=false, tts=false;
constexpr uint32_t permissions=GRIM_PERM_VOICE|GRIM_PERM_FILESYSTEM|GRIM_PERM_PROCESS;
const char* ids[]={"voice.status","voice.configure","voice.capture","voice.stream.start","voice.stream.stop",
 "voice.tts.start","voice.speak","voice.play","voice.stop","voice.precache","voice.wake.start","voice.wake.stop","voice.devices","voice.synthesize","voice.cache.status"};
nlohmann::json status() {
 auto result=VoiceRuntime::snapshot();
 result["configured"]=configured; result["streaming"]=VoiceStream::isRunning();
 result["wake_listening"]=WakeVoice::isRunning(); result["tts_ready"]=Voice::isReady();
 result["playing"]=Voice::isPlaying(); result["speaking"]=Voice::isSpeaking(); result["provider"]=Voice::activeTTSProviderId();
 return result;
}
void publish() {
 auto json=nlohmann::json::array({{{"resource_id","status"},{"kind","voice.status"},{"name","Voice status and transcripts"},{"runtime_id",1},{"state",status()}}}).dump();
 if(host->replace_resources(host,"voice.status",json.c_str(),json.size())!=GRIM_OK) throw std::runtime_error("Unable to publish voice status");
}
void stop() {
 VoiceRuntime::cancel=true;
 WakeVoice::stop(); VoiceStream::stop();
 Audio::stopAll(); Voice::shutdownQueue(); Voice::shutdownTTS(); Voice::shutdown(); Audio::shutdown();
 tts=false; VoiceRuntime::cancel=false;
}
GrimResult call(const GrimToolCall* request,GrimToolResultWriter* writer,void* context) {
 std::lock_guard<std::mutex> lock(operationMutex);
 GrimToolResult result; std::string message,output; nlohmann::json detail;
 try {
  auto args=request->arguments_size ? nlohmann::json::parse(request->arguments,request->arguments+request->arguments_size) : nlohmann::json::object();
  if(!args.is_object()) throw std::invalid_argument("Expected JSON object");
  const std::string op=static_cast<const char*>(context);
  if(op=="voice.configure") {
   const auto root=std::filesystem::path(args.at("root").get<std::string>());
   if(!root.is_absolute() || !std::filesystem::is_directory(root)) throw std::invalid_argument("root must name an existing absolute directory");
   if(!args.contains("voice")) args["voice"]=nlohmann::json::object();
   if(!args.contains("whisper")) args["whisper"]=nlohmann::json::object();
   if(!args["voice"].is_object() || !args["whisper"].is_object()) throw std::invalid_argument("voice and whisper must be objects");
   if(configured) stop();
   VoiceRuntime::config=args; configured=true;
  } else if(op=="voice.stop") stop();
  else if(op=="voice.stream.stop") VoiceStream::stop();
  else if(op=="voice.wake.stop") WakeVoice::stop();
  else if(op!="voice.status") {
   if(!configured) throw std::runtime_error("Configure voice before starting audio or models");
   if(op=="voice.devices") {
    detail=nlohmann::json::array();
    for(const auto& device:Audio::listDevices()) detail.push_back({{"name",device.name},{"default",device.isDefault},{"input",device.isInput},{"output",device.isOutput}});
   } else if(op=="voice.cache.status") detail=Voice::TTSCache::getStats();
   else if(op=="voice.synthesize") {
    if(!tts) throw std::runtime_error("Start TTS first");
    Voice::TTSSynthesisRequest synthesis; synthesis.text=args.at("text").get<std::string>();
    synthesis.voice_id=args.value("speaker",Voice::getSpeaker()); synthesis.language=args.value("language",Voice::getLanguage());
    synthesis.rate=args.value("rate",1.0); synthesis.pitch=args.value("pitch",1.0);
    Voice::shutdownQueue();
    Voice::TTSSynthesisResult audio;
    try { audio=Voice::synthesize(synthesis); } catch(...) { Voice::initQueue(); throw; }
    Voice::initQueue();
    if(!audio.success) throw std::runtime_error(audio.error_code+": "+audio.error_message);
    detail={{"audio_path",audio.audio_path}};
   } else if(op=="voice.capture") {
    if(VoiceStream::isRunning() || WakeVoice::isRunning()) throw std::runtime_error("Stop streaming/wake capture first");
    if(!Voice::loadWhisper(VoiceRuntime::config)) throw std::runtime_error("Whisper model unavailable");
    VoiceRuntime::transcript(Voice::captureAndTranscribeSpeech(VoiceRuntime::config));
   } else if(op=="voice.stream.start") {
    if(WakeVoice::isRunning()) throw std::runtime_error("Stop wake capture first");
    if(!Voice::loadWhisper(VoiceRuntime::config) || !VoiceStream::start(Voice::getWhisperContext())) throw std::runtime_error("Unable to start transcription");
   } else if(op=="voice.tts.start") {
    if(!tts) { if(!Voice::initTTS() || !Voice::isReady()) { Voice::shutdownTTS(); throw std::runtime_error("TTS provider unavailable"); } Voice::initQueue(); tts=true; }
   } else if(op=="voice.speak") {
    if(!tts) throw std::runtime_error("Start TTS first");
    Voice::speak(args.at("text").get<std::string>(),"plugin");
   } else if(op=="voice.play") {
    if(!Audio::playWav(args.at("path").get<std::string>())) throw std::runtime_error("Playback failed");
   } else if(op=="voice.precache") {
    if(!tts) throw std::runtime_error("Start TTS first");
    Voice::shutdownQueue();
    try { Voice::preCacheCommonPhrases(); } catch(...) { Voice::initQueue(); throw; }
    Voice::initQueue();
   } else if(op=="voice.wake.start") {
    if(VoiceStream::isRunning()) throw std::runtime_error("Stop streaming first");
    WakeVoice::start(); if(!WakeVoice::isRunning()) throw std::runtime_error("Wake detector unavailable");
   }
  }
  publish(); auto response=status(); if(!detail.is_null()) response["result"]=detail; output=response.dump(); message="Voice operation completed";
  result.success=true; result.error_code="ERR_NONE";
 } catch(const std::exception& error) { message=error.what(); result.error_code="ERR_VOICE"; }
 catch(...) { message="Unknown voice failure"; result.error_code="ERR_VOICE"; }
 result.message=message.c_str(); result.category="voice"; result.structured_output_json=output.empty()?nullptr:output.c_str();
 return writer->complete(writer,&result);
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
 static const GrimPluginInfo info{GRIM_API_VERSION,"Voice Plugin","2.0.0","GRIM Project","Provider-owned capture, transcription, wake detection, synthesis and playback",permissions}; return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
 if(!api || api->api_version!=GRIM_API_VERSION || !api->register_tools || !api->replace_resources) return GRIM_ERROR_API_VERSION_MISMATCH;
 std::lock_guard<std::mutex> lock(operationMutex);
 host=api; configured=false; tts=false; VoiceRuntime::reset(); grim::plugin::setLoggingAPI(api);
 try {
  GrimToolRegistration registrations[15];
  const char* descriptions[]={"Read voice status and the latest 64 transcript/wake/error events","Configure provider root, voice and whisper settings without starting devices",
   "Capture and transcribe speech without executing it","Start continuous transcription without conversation dispatch","Stop streaming transcription",
   "Explicitly initialize the configured TTS provider","Queue text for speech","Play a WAV file","Stop voice workers, models and playback",
   "Generate and cache common phrases synchronously","Start wake detection without command dispatch","Stop wake detection","Enumerate audio devices","Synthesize text to an audio file without playback","Read provider cache statistics"};
  for(size_t i=0;i<15;++i) {
   auto& r=registrations[i]; auto& d=r.descriptor; r.handler=call; r.user_data=const_cast<char*>(ids[i]);
   r.usage_type=GRIM_TOOL_USAGE_DEBUG; r.short_description=descriptions[i];
   d.tool_id=d.display_name=d.usage=ids[i]; d.version="2.0.0"; d.description=descriptions[i]; d.category="voice";
   d.argument_format="json"; d.argument_schema_json=R"({"type":"object"})";
   d.required_permissions=permissions; d.is_informational=i==0 || i==12 || i==14; d.needs_confirmation=!d.is_informational;
  }
  publish(); const auto registered=api->register_tools(api,registrations,15);
  if(registered!=GRIM_OK) { host=nullptr; grim::plugin::setLoggingAPI(nullptr); }
  return registered;
 } catch(...) { host=nullptr; grim::plugin::setLoggingAPI(nullptr); return GRIM_ERROR_UNKNOWN; }
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
 VoiceRuntime::cancel=true;
 std::lock_guard<std::mutex> lock(operationMutex);
 if(!host) return;
 stop(); configured=false; VoiceRuntime::reset(); host=nullptr; grim::plugin::setLoggingAPI(nullptr);
}
