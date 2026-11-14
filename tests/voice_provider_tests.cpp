#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <iostream>
#include <map>
#include <filesystem>
#include <chrono>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace {
struct Tool { GrimToolHandler handler; void* context; };
std::map<std::string,Tool> registered;
std::string output;
bool success=false;
int timers=0, publications=0;
void check(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
GrimResult registerTools(const GrimPluginAPI*,const GrimToolRegistration* tools,size_t count) {
    for(size_t i=0;i<count;++i) registered.emplace(tools[i].descriptor.tool_id,Tool{tools[i].handler,tools[i].user_data});
    return GRIM_OK;
}
GrimResult publish(const GrimPluginAPI*,const char*,const char* json,size_t size) {
    const auto value=nlohmann::json::parse(json,json+size);
    check(value.is_array(),"Resources were not structured"); ++publications; return GRIM_OK;
}
GrimTimerHandle timer(const GrimPluginAPI*,uint64_t,bool,GrimTimerCallback,void*) { ++timers;return reinterpret_cast<void*>(1); }
GrimResult cancel(const GrimPluginAPI*,GrimTimerHandle) { --timers;return GRIM_OK; }
void log(const GrimPluginAPI*,GrimLogLevel,const char*) {}
GrimResult complete(GrimToolResultWriter*,const GrimToolResult* result) {
    success=result->success;output=result->structured_output_json ? result->structured_output_json : "";return GRIM_OK;
}
void invoke(const char* id,const std::string& args) {
    const auto& t=registered.at(id);GrimToolCall call;call.arguments=args.data();call.arguments_size=args.size();
    GrimToolResultWriter writer;writer.complete=complete;
    check(t.handler(&call,&writer,t.context)==GRIM_OK,"Handler failed transport");
}
}
int main(int argc,char** argv) try {
    check(argc==2,"Expected provider library path");
#ifdef _WIN32
    auto lib=LoadLibraryA(argv[1]);check(lib!=nullptr,"Cannot load voice DLL");
    auto symbol=[&](const char* name){ return GetProcAddress(lib,name); };
#else
    auto lib=dlopen(argv[1],RTLD_NOW);check(lib!=nullptr,"Cannot load voice library");
    auto symbol=[&](const char* name){ return dlsym(lib,name); };
#endif
    auto init=reinterpret_cast<GrimResult(*)(const GrimPluginAPI*)>(symbol("grim_plugin_init"));
    auto stop=reinterpret_cast<void(*)()>(symbol("grim_plugin_shutdown"));
    check(init&&stop,"Missing exports");
    GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;api.register_tools=registerTools;
    api.replace_resources=publish;api.create_timer=timer;api.cancel_timer=cancel;api.log=log;
    auto old=api;old.api_version=9<<16;check(init(&old)==GRIM_ERROR_API_VERSION_MISMATCH,"Old ABI accepted");
    const auto testRoot=std::filesystem::temp_directory_path() / ("grim-voice-owned-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(std::filesystem::create_directory(testRoot),"Cannot create private cache fixture");
    struct Cleanup { std::filesystem::path root; ~Cleanup() { std::error_code error; std::filesystem::remove_all(root,error); } } cleanup{testRoot};
    for(int pass=0;pass<2;++pass) {
        registered.clear();check(init(&api)==GRIM_OK,"Initialization failed");
        check(registered.size()==15,"Tool ownership incorrect");
        check(!registered.count("nevermind")&&!registered.count("voice"),"Legacy wrappers retained");
        invoke("voice.status","{}");check(success,"Idle status failed");
        auto state=nlohmann::json::parse(output);
        check(!state.at("configured").get<bool>()&&!state.at("streaming").get<bool>()&&!state.at("tts_ready").get<bool>()&&!state.at("wake_listening").get<bool>(),"Init started runtime");
        check(state.at("events").empty(),"Idle provider fabricated observations");
        for(const auto* id:{"voice.capture","voice.stream.start","voice.tts.start","voice.precache","voice.wake.start"}) {
            invoke(id,"{}");check(!success,"Unconfigured runtime operation accepted");
        }
        invoke("voice.configure",R"({"root":"relative"})");check(!success,"Relative root accepted");
        invoke("voice.configure",nlohmann::json({{"root",testRoot.string()}}).dump());check(success,"Explicit configuration failed");
        invoke("voice.status","{}");state=nlohmann::json::parse(output);
        check(state.at("configured").get<bool>()&&!state.at("tts_ready").get<bool>()&&!state.at("streaming").get<bool>(),"Configure started models/devices");
        invoke("voice.precache","{}");check(!success,"Precache accepted without TTS");
        invoke("voice.speak",R"({"text":"test"})");check(!success,"Speak accepted without TTS");
        invoke("voice.status","[]");check(!success,"Nonobject accepted");
        invoke("voice.stop","{}");check(success,"Idle stop failed");
#ifdef _WIN32
        invoke("voice.configure",nlohmann::json({{"root",testRoot.string()},{"voice",{{"engine","sapi"}}}}).dump());check(success,"SAPI configuration failed");
        invoke("voice.tts.start","{}");check(success,"Native speech worker did not start");
        check(nlohmann::json::parse(output).at("tts_ready").get<bool>(),"TTS readiness missing");
        invoke("voice.precache","{}");check(!success,"Unsupported SAPI file synthesis claimed successful precaching");
        invoke("voice.stop","{}");check(success,"Worker shutdown failed");
        invoke("voice.tts.start","{}");check(success,"Speech worker did not restart");
#endif
        stop();stop();
    }
    api.replace_resources=[](const GrimPluginAPI*,const char*,const char*,size_t){return GRIM_ERROR_UNKNOWN;};
    check(init(&api)!=GRIM_OK,"Resource failure accepted");stop();
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    std::cout<<"Voice provider ABI, idle initialization, configuration, validation and lifecycle passed\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
