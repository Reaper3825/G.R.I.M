#include "core/plugin_api.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <iostream>
#include <map>
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
    auto lib=LoadLibraryA(argv[1]);check(lib!=nullptr,"Cannot load perception DLL");
    auto symbol=[&](const char* name){ return GetProcAddress(lib,name); };
#else
    auto lib=dlopen(argv[1],RTLD_NOW);check(lib!=nullptr,"Cannot load perception library");
    auto symbol=[&](const char* name){ return dlsym(lib,name); };
#endif
    auto init=reinterpret_cast<GrimResult(*)(const GrimPluginAPI*)>(symbol("grim_plugin_init"));
    auto stop=reinterpret_cast<void(*)()>(symbol("grim_plugin_shutdown"));
    check(init&&stop,"Missing exports");
    GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;api.register_tools=registerTools;
    api.replace_resources=publish;api.create_timer=timer;api.cancel_timer=cancel;api.log=log;
    auto old=api;old.api_version=8<<16;check(init(&old)==GRIM_ERROR_API_VERSION_MISMATCH,"Old ABI accepted");
    for(int pass=0;pass<2;++pass) {
        registered.clear();check(init(&api)==GRIM_OK,"Initialization failed");
        check(registered.size()==7&&timers==1,"Tool/timer ownership incorrect");
        check(!registered.count("perception_read_text"),"Legacy tool was retained");
        invoke("perception.observe","{}");check(success,"Idle observation failed");
        const auto state=nlohmann::json::parse(output);
        check(!state.at("digital_running").get<bool>()&&!state.at("physical_running").get<bool>(),"Init started acquisition");
        check(!state.at("digital").at("available").get<bool>(),"Idle provider fabricated observation");
        invoke("perception.digital.start",R"({"interval_ms":0})");check(!success,"Invalid cadence accepted");
        invoke("perception.configure",R"({"operator":"missing","config":{}})");check(!success,"Unknown operator accepted");
        invoke("perception.calibrate","{}");check(!success,"Calibration without camera accepted");
        stop();check(timers==0,"Shutdown retained timer");stop();
    }
    api.create_timer=[](const GrimPluginAPI*,uint64_t,bool,GrimTimerCallback,void*)->GrimTimerHandle{return nullptr;};
    check(init(&api)!=GRIM_OK,"Timer failure accepted");stop();
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    std::cout<<"Perception provider ABI, idle initialization, structured observations, validation and lifecycle passed\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
