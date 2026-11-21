#include "core/plugin_manager.hpp"
#include "plugin_deployment_fixture.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "MMO/Core/Descriptors/RegisteredResource.hpp"
#include "commands/command_result.hpp"
#ifdef GRIM_NATIVE_UI_TEST_HOST
#include "ui/ui_root.hpp"
#endif
#include <windows.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <iostream>
extern std::vector<std::string> pluginNotifications;
void runProviderSchedulingTests(const std::filesystem::path& scheduleDirectory) {
    namespace fs=std::filesystem;
    using Json=nlohmann::json;
    const auto check=[](bool ok,const char* text){if(!ok) throw std::runtime_error(text);};
    if(scheduleDirectory.empty()) return;
    const auto root=fs::temp_directory_path()/("grim-provider-scheduling-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(root),"Cannot create scheduling fixture");
    const DWORD length=GetEnvironmentVariableW(L"LOCALAPPDATA",nullptr,0);
    std::wstring previous(length,L'\0');
    if(length){GetEnvironmentVariableW(L"LOCALAPPDATA",previous.data(),length);previous.resize(length-1);}
    struct Cleanup {
        fs::path root;std::wstring previous;
        ~Cleanup(){
            PluginManager::shutdownBatch();SetEnvironmentVariableW(L"LOCALAPPDATA",previous.empty()?nullptr:previous.c_str());
            std::error_code e;
            for(const auto* p:{"schedule_plugin/schedule_plugin.dll","schedule_plugin/schedule_plugin.tools.json","schedule_plugin/.deployment","schedule_plugin","GRIM/calendar/events.json", "GRIM/calendar/events.json.lock", "GRIM/calendar/events.json.tmp", "GRIM/calendar", "GRIM"}) fs::remove(root/p,e);
            fs::remove(root,e);
        }
    } cleanup{root,previous};
    SetEnvironmentVariableW(L"LOCALAPPDATA",root.c_str());
    fs::create_directory(root / "schedule_plugin");
    fs::copy_file(scheduleDirectory/"schedule_plugin.dll",root/"schedule_plugin/schedule_plugin.dll");
    fs::copy_file(scheduleDirectory/"schedule_plugin.tools.json",root/"schedule_plugin/schedule_plugin.tools.json");
    writePluginFixtureRevision(root / "schedule_plugin");
    auto& registry=GRIM::MMO::ToolRegistry::instance();const auto baseline=registry.toolCount();
    const auto resource=[&](const char* name,const char* id){
        for(const auto& group:GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if(group.provider.name==name) for(const auto& r:group.resources) if(r.resource_id==id) return GRIM::MMO::RegisteredResource{r,group.provider};
        return GRIM::MMO::RegisteredResource{};
    };
    PluginManager::loadBatch(root.string());
#ifdef GRIM_NATIVE_UI_TEST_HOST
    UIRoot::get().processPendingTasks();
    check(UIRoot::get().getPanelByKey("calendar") != nullptr, "Native Calendar did not attach through loader");
    check(resource("Schedule Plugin", "panel:calendar").kind == "ui.panel", "Native panel resource missing");
#endif
    check(registry.toolCount()==baseline+6,"Schedule provider did not register six tools");
    check(!fs::exists(root/"GRIM/calendar/events.json"),"Initialization accessed durable records before the first poll");
    check(resource("Schedule Plugin","items").state.at("ready")==false,"Schedule ready before first poll");
    const auto invoke=[&](const char* op,Json args){
        const auto result=registry.invoke("schedule."+std::string(op),args.dump());
        check(result.at("status") == "completed",result.at("message").get_ref<const std::string&>().c_str());
        return result.at("payload");
    };
    auto created=invoke("create",{{"id","due"},{"kind","timer"},{"title","Due notification"},{"duration_seconds",1}});
    check(created.at("success")==true,"Saved timer creation failed");
    const auto before=pluginNotifications.size();
    const auto firstFrame = std::chrono::steady_clock::now();
    auto previousFrame = firstFrame;
    const auto pumpFrame = [&] {
        const auto now = std::chrono::steady_clock::now();
        PluginAPI::dispatchSystemTick(std::chrono::duration<double>(now - previousFrame).count(),
                                      std::chrono::duration<double>(now - firstFrame).count());
        previousFrame = now;
    };
    pumpFrame();check(pluginNotifications.size()==before,"Notification fired before deadline");
    std::this_thread::sleep_for(std::chrono::milliseconds(1150));pumpFrame();
    check(pluginNotifications.size()==before+1,"Schedule did not notify once");
    pumpFrame();check(pluginNotifications.size()==before+1,"Notification was duplicated");
    check(invoke("get",{{"id","due"}}).at("item").at("delivery_status")=="dispatched","Saved dispatch state missing");
    auto cancelled=invoke("create",{{"id","cancel"},{"kind","timer"},{"title","Cancelled"},{"duration_seconds",1}});
    invoke("cancel",{{"id","cancel"},{"revision",cancelled.at("item").at("revision")}});
    invoke("create",{{"id","persist"},{"kind","timer"},{"title","Survives reload"},{"duration_seconds",1}});
    const auto old=resource("Schedule Plugin","items").provider.instance;
    PluginManager::shutdownBatch();
    {std::ofstream broken(root/"schedule_plugin/schedule_plugin.dll",std::ios::binary|std::ios::trunc);broken<<"broken";}
    PluginManager::loadBatch(root);
    check(registry.toolCount() == baseline && resource("Schedule Plugin", "items").resource_id.empty(),
          "Failed native load retained schedule tools or resources");
    PluginManager::shutdownBatch();
    fs::copy_file(scheduleDirectory/"schedule_plugin.dll",root/"schedule_plugin/schedule_plugin.dll",fs::copy_options::overwrite_existing);
    PluginManager::loadBatch(root);
#ifdef GRIM_NATIVE_UI_TEST_HOST
    UIRoot::get().processPendingTasks();
    check(UIRoot::get().getPanelByKey("calendar") != nullptr, "Replacement native panel missing");
#endif
    check(resource("Schedule Plugin","items").provider.instance!=old,"Schedule generation did not replace");
    check(invoke("get",{{"id","persist"}}).at("item").at("delivery_status")=="pending","Schedule did not reopen saved state");
    std::this_thread::sleep_for(std::chrono::milliseconds(1150));pumpFrame();
    check(pluginNotifications.size()==before+2,"Reload duplicated delivery or cancelled timer fired");
    check(invoke("get",{{"id","persist"}}).at("item").at("delivery_status")=="dispatched","Reload dispatch not persisted");
    PluginManager::shutdownBatch();
    fs::remove(root/"schedule_plugin/schedule_plugin.dll");
    fs::remove(root/"schedule_plugin/.deployment");
    fs::remove(root/"schedule_plugin/schedule_plugin.tools.json");
    fs::remove(root/"schedule_plugin");
    pumpFrame();
#ifdef GRIM_NATIVE_UI_TEST_HOST
    check(!UIRoot::get().getPanelByKey("calendar"), "DLL unload left native Calendar attached");
    UIRoot::get().processPendingTasks();
#endif
    check(registry.toolCount()==baseline && resource("Schedule Plugin","items").resource_id.empty(),"Unload retained scheduling surface");
    bool unavailable=false;try{invoke("list",Json::object());}catch(...){unavailable=true;}
    check(unavailable,"Calendar client called unloaded provider");
    std::cout<<"Schedule provider: delivery, cancellation, resources, persisted reload, failed replacement and UI routing passed\n";
}
