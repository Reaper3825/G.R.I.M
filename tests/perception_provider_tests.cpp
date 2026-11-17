#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
namespace {
GrimToolDispatch dispatch = nullptr;
std::set<std::string> declared;
std::string output;
bool success=false;
GrimSystemTickCallback systemTick = nullptr;
void* tickContext = nullptr;
int publications=0;
void check(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }

GrimResult publish(GrimPluginHandle,const char*,const char* json,size_t size) {
    const auto value=nlohmann::json::parse(json,json+size);
    check(value.is_array(),"Resources were not structured"); ++publications; return GRIM_OK;
}
GrimResult subscribe(GrimPluginHandle, GrimSystemTickCallback callback, void* context) {
    if (systemTick) return GRIM_ERROR_ALREADY_EXISTS;
    systemTick = callback;
    tickContext = context;
    return GRIM_OK;
}
void log(GrimPluginHandle,GrimLogLevel,const char*) {}
GrimResult complete(GrimToolResultWriter*,const GrimToolResult* result) {
    success=result->success;output=result->structured_output_json ? result->structured_output_json : "";return GRIM_OK;
}
void invoke(const char* id,const std::string& args) {
    check(declared.count(id) == 1, "Tool missing from manifest");
    GrimToolCall call;call.arguments=args.data();call.arguments_size=args.size();
    GrimToolResultWriter writer;writer.complete=complete;
    check(dispatch(id, &call, &writer)==GRIM_OK,"Handler failed transport");
}
}
int main(int argc,char** argv) try {
    check(argc==2,"Expected provider library path");
#ifdef _WIN32
    auto lib=LoadLibraryA(argv[1]);check(lib!=nullptr,"Cannot load perception DLL");
    auto symbol=[&](const char* name){ return GetProcAddress(lib,name); };
#else
    auto lib=dlopen(argv[1],RTLD_NOW);check(lib!=nullptr,"Cannot load perception library");
    auto symbol=[&](const char* name){ return dlsym(lib,name); };
#endif
    auto init=reinterpret_cast<GrimResult(*)(const GrimPluginAPI*, GrimPluginHandle)>(symbol("grim_plugin_init"));
    auto stop=reinterpret_cast<void(*)()>(symbol("grim_plugin_shutdown"));
    dispatch = reinterpret_cast<GrimToolDispatch>(symbol("grim_plugin_dispatch_tool"));
    check(dispatch != nullptr, "Missing dispatcher export");
    auto sidecar = std::filesystem::path(argv[1]).replace_extension(".tools.json");
    std::ifstream manifestFile(sidecar);
    check(static_cast<bool>(manifestFile), "Missing tool manifest");
    const auto manifest = nlohmann::json::parse(manifestFile);
    for (const auto& tool : manifest.at("tools")) declared.insert(tool.at("tool_id").get<std::string>());
    check(init&&stop,"Missing exports");
    GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;
    api.replace_resources=publish;api.subscribe_system_tick=subscribe;api.log=log;
    auto old=api;old.api_version=8<<16;check(init(&old, 1)==GRIM_ERROR_API_VERSION_MISMATCH,"Old ABI accepted");
    for(int pass=0;pass<2;++pass) {
        check(init(&api, 1)==GRIM_OK,"Initialization failed");
        check(declared.size()==7&&systemTick&&tickContext,"Tool/system tick ownership incorrect");
        systemTick(0, 0, tickContext);
        check(!declared.count("perception_read_text"),"Legacy tool was retained");
        invoke("perception.observe","{}");check(success,"Idle observation failed");
        const auto state=nlohmann::json::parse(output);
        check(!state.at("digital_running").get<bool>()&&!state.at("physical_running").get<bool>(),"Init started acquisition");
        check(!state.at("digital").at("available").get<bool>(),"Idle provider fabricated observation");
        invoke("perception.digital.start",R"({"interval_ms":0})");check(!success,"Invalid cadence accepted");
        invoke("perception.configure",R"({"operator":"missing","config":{}})");check(!success,"Unknown operator accepted");
        invoke("perception.calibrate","{}");check(!success,"Calibration without camera accepted");
        // The host closes the callback gate before calling DLL shutdown.
        systemTick = nullptr;
        tickContext = nullptr;
        stop();stop();
    }
    api.subscribe_system_tick=[](GrimPluginHandle,GrimSystemTickCallback,void*){return GRIM_ERROR_UNKNOWN;};
    check(init(&api, 1)!=GRIM_OK,"System tick subscription failure accepted");stop();
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    std::cout<<"Perception provider ABI, idle initialization, structured observations, validation and lifecycle passed\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
