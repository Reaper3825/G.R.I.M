#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "MMO/Core/Descriptors/RegisteredResource.hpp"
#include "commands/command_result.hpp"
#include "ui/calendar_provider_client.hpp"
#include <windows.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <iostream>
extern std::vector<std::string> pluginNotifications;
void runProviderSchedulingTests(const std::filesystem::path& timerDirectory,const std::filesystem::path& scheduleDirectory) {
    namespace fs=std::filesystem;
    using Json=nlohmann::json;
    const auto check=[](bool ok,const char* text){if(!ok) throw std::runtime_error(text);};
    if(timerDirectory.empty() || scheduleDirectory.empty()) return;
    const auto root=fs::temp_directory_path()/("grim-provider-scheduling-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(root),"Cannot create scheduling fixture");
    const DWORD length=GetEnvironmentVariableW(L"LOCALAPPDATA",nullptr,0);
    std::wstring previous(length,L'\0');
    if(length){GetEnvironmentVariableW(L"LOCALAPPDATA",previous.data(),length);previous.resize(length-1);}
    struct Cleanup {
        fs::path root;std::wstring previous;
        ~Cleanup(){
            PluginManager::unloadAll();SetEnvironmentVariableW(L"LOCALAPPDATA",previous.empty()?nullptr:previous.c_str());
            std::error_code e;
            for(const auto* p:{"timers_plugin.dll","schedule_plugin.dll","GRIM/calendar/events.json", "GRIM/calendar/events.json.lock", "GRIM/calendar/events.json.tmp", "GRIM/calendar", "GRIM"}) fs::remove(root/p,e);
            fs::remove(root,e);
        }
    } cleanup{root,previous};
    SetEnvironmentVariableW(L"LOCALAPPDATA",root.c_str());
    fs::copy_file(timerDirectory/"timers_plugin.dll",root/"timers_plugin.dll");
    fs::copy_file(scheduleDirectory/"schedule_plugin.dll",root/"schedule_plugin.dll");
    auto& registry=GRIM::MMO::ToolRegistry::instance();const auto baseline=registry.toolCount();
    const auto resource=[&](const char* name,const char* id){
        for(const auto& group:GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if(group.provider.name==name) for(const auto& r:group.resources) if(r.resource_id==id) return GRIM::MMO::RegisteredResource{r,group.provider};
        return GRIM::MMO::RegisteredResource{};
    };
    PluginManager::initialize(root.string());
    check(registry.toolCount()==baseline+7,"Scheduling plugins missing tools");
    check(!fs::exists(root/"GRIM/calendar/events.json"),"Initialization accessed durable records before publication");
    check(resource("Timers Plugin","queue").state.at("pending")==0,"Initial timer queue missing");
    check(resource("Schedule Plugin","items").state.at("ready")==false,"Schedule candidate unexpectedly ready");
    for(const auto* input:{"","invalid","99999999999999999999999"})
        check(registry.invoke("set_timer",input).errorCode=="ERR_TIMER_INVALID","Invalid timer accepted");
    for(const auto* input:{"0","-3"}) check(registry.invoke("set_timer",input).errorCode=="ERR_TIMER_NONPOSITIVE","Nonpositive timer accepted");
    check(registry.invoke("set_timer"," 1suffix").success,"Countdown prefix parsing changed");
    check(resource("Timers Plugin","queue").state.at("pending")==1,"Countdown resource not updated");
    // The calendar client uses exactly the same provider tools, never a raw scheduler pointer.
    const auto invoke=[&](const char* op,Json args){return GRIM::UI::callSchedule(op,args);};
    auto created=invoke("create",{{"id","due"},{"kind","timer"},{"title","Due notification"},{"duration_seconds",1}});
    check(created.at("success")==true,"Saved timer creation failed");
    const auto before=pluginNotifications.size();
    PluginAPI::processTimers();check(pluginNotifications.size()==before,"Notification fired before deadline");
    std::this_thread::sleep_for(std::chrono::milliseconds(1150));PluginAPI::processTimers();
    check(pluginNotifications.size()==before+2,"Plugin countdown and schedule did not each notify once");
    PluginAPI::processTimers();check(pluginNotifications.size()==before+2,"Notification was duplicated");
    check(resource("Timers Plugin","queue").state.at("pending")==0,"Expired countdown retained");
    check(invoke("get",{{"id","due"}}).at("item").at("delivery_status")=="dispatched","Saved dispatch state missing");
    auto cancelled=invoke("create",{{"id","cancel"},{"kind","timer"},{"title","Cancelled"},{"duration_seconds",1}});
    invoke("cancel",{{"id","cancel"},{"revision",cancelled.at("item").at("revision")}});
    invoke("create",{{"id","persist"},{"kind","timer"},{"title","Survives reload"},{"duration_seconds",1}});
    check(registry.invoke("set_timer","5").success,"Countdown before reload failed");
    const auto old=resource("Schedule Plugin","items").provider.instance;
    {std::ofstream broken(root/"schedule_plugin.dll",std::ios::binary|std::ios::trunc);broken<<"broken";}
    PluginManager::checkForHotReload();
    check(resource("Schedule Plugin","items").provider.instance==old && invoke("get",{{"id","persist"}}).at("success")==true,
          "Failed replacement lost active schedule");
    fs::copy_file(scheduleDirectory/"schedule_plugin.dll",root/"schedule_plugin.dll",fs::copy_options::overwrite_existing);
    for(const auto* name:{"schedule_plugin.dll","timers_plugin.dll"}) fs::last_write_time(root/name,fs::last_write_time(root/name)+std::chrono::seconds(2));
    PluginManager::checkForHotReload();
    check(resource("Schedule Plugin","items").provider.instance!=old,"Schedule generation did not replace");
    check(resource("Timers Plugin","queue").state.at("pending")==0,"Reload retained cancelled countdown state");
    check(invoke("get",{{"id","persist"}}).at("item").at("delivery_status")=="pending","Schedule did not reopen saved state");
    fs::remove(root/"timers_plugin.dll");PluginManager::checkForHotReload();
    std::this_thread::sleep_for(std::chrono::milliseconds(1150));PluginAPI::processTimers();
    check(pluginNotifications.size()==before+3,"Reload duplicated delivery or cancelled timer fired");
    check(invoke("get",{{"id","persist"}}).at("item").at("delivery_status")=="dispatched","Reload dispatch not persisted");
    fs::remove(root/"schedule_plugin.dll");PluginManager::checkForHotReload();
    PluginAPI::processTimers();
    check(registry.toolCount()==baseline && resource("Schedule Plugin","items").resource_id.empty() &&
          resource("Timers Plugin","queue").resource_id.empty(),"Unload retained scheduling surface");
    bool unavailable=false;try{invoke("list",Json::object());}catch(...){unavailable=true;}
    check(unavailable,"Calendar client called unloaded provider");
    std::cout<<"Plugin-owned timers/scheduling: delivery, cancellation, resources, persisted reload, failed replacement and UI routing passed\n";
}
