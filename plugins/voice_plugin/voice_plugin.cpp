#include "host_voice.hpp"
#include "plugins/plugin_logging.hpp"
namespace {
template<CommandResult (*Handler)(const std::string&)>
GrimResult call(const GrimToolCall* request, GrimToolResultWriter* writer, void*) {
    return grim::runtime::deliver(writer, Handler(std::string(request->arguments, request->arguments_size)));
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Voice Plugin", "1.0.0", "GRIM Project",
        "Voice commands with host-owned capture, streaming, synthesis, playback, and cache lifecycle", GRIM_PERM_VOICE | GRIM_PERM_FILESYSTEM};
    return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->voice_service) return GRIM_ERROR_API_VERSION_MISMATCH;
    grim::runtime::host = api; grim::plugin::setLoggingAPI(api);
    GrimToolRegistration tools[11];
    const char* ids[] = {"voice", "voice_stream", "test_tts", "list_voices", "test_sapi", "tts_device", "nevermind",
        "create_embedding", "list_embeddings", "clear_tts_cache", "reset_tts_cache"};
    const char* descriptions[] = {"Capture and transcribe a one-shot voice command", "Start host voice streaming",
        "Test text-to-speech output", "List available TTS voices", "Test playback of the existing test WAV",
        "Query the current SAPI output device", "Return the existing cancellation acknowledgment (playback stop remains a placeholder)",
        "Return the existing embedding-queued acknowledgment (bridge submission remains a placeholder)",
        "List available voice embedding files", "Clear TTS audio cache", "Reset TTS cache and regenerate common phrases"};
    const GrimToolHandler handlers[] = {call<cmdVoice>, call<cmdVoiceStream>, call<cmd_testTTS>, call<cmd_listVoices>,
        call<cmd_testSAPI>, call<cmd_ttsDevice>, call<cmdNevermind>, call<cmd_createEmbedding>, call<cmd_listEmbeddings>,
        call<cmdClearCache>, call<cmdResetCache>};
    const char* usages[] = {"voice", "voice_stream", "test_tts [text]", "list_voices", "test_sapi", "tts_device", "nevermind",
        "create_embedding <speaker_id> <audio_file_path>", "list_embeddings", "clear_tts_cache", "reset_tts_cache"};
    GrimToolParameter text;
    text.name = "text"; text.type = "string"; text.description = "Test text; omitted uses the existing Coqui test phrase";
    GrimToolParameter embeddingArguments;
    embeddingArguments.name = "arguments"; embeddingArguments.type = "string"; embeddingArguments.required = 1;
    embeddingArguments.description = "Speaker ID followed by reference audio path; preserves the existing placeholder and parsing behavior";
    for (size_t i = 0; i < 11; ++i) {
        tools[i].handler = handlers[i]; auto& d = tools[i].descriptor;
        d.tool_id = d.display_name = ids[i]; d.version = "1.0.0"; d.description = descriptions[i]; d.usage = usages[i];
        const bool informational = i == 3 || i == 8;
        d.category = informational ? "information" : (i == 6 ? "action" : "system");
        d.argument_format = "raw_text"; d.is_informational = informational; d.needs_confirmation = !informational;
        const bool filesystem = i == 4 || i == 8 || i == 9 || i == 10;
        d.required_permissions = GRIM_PERM_VOICE | (filesystem ? GRIM_PERM_FILESYSTEM : 0);
        d.examples = {&usages[i], 1};
    }
    tools[2].descriptor.parameters = &text; tools[2].descriptor.parameter_count = 1;
    tools[7].descriptor.parameters = &embeddingArguments; tools[7].descriptor.parameter_count = 1;
    return api->register_tools(api, tools, 11);
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    grim::plugin::setLoggingAPI(nullptr); grim::runtime::host = nullptr;
}
