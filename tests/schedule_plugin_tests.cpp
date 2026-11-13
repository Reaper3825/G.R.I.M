#include "plugins/schedule_plugin/scheduling_service.hpp"
#include "plugins/schedule_plugin/schedule_plugin.cpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "commands/command_result.hpp"
#include <cassert>
#include <iostream>
void logError(const std::string&,const std::string&) {}
using Json=nlohmann::json;
static int64_t now=grim::scheduling::scheduleNow();
static std::vector<std::string> notifications;
static GrimResult registerTools(const GrimPluginAPI*,const GrimToolRegistration* tools,size_t count) {
    assert(count==6);
    std::vector<GRIM::MMO::ToolRegistration> registrations;
    for(size_t i=0;i<count;++i) {
        const auto& t=tools[i];assert(t.usage_type==GRIM_TOOL_USAGE_MODEL);
        auto schema=Json::parse(t.descriptor.argument_schema_json);assert(schema["type"]=="object" && schema["additionalProperties"]==false);
        GRIM::MMO::ToolDescriptor d;d.tool_id=t.descriptor.tool_id;d.display_name=d.tool_id;
        d.argument_format="json";d.argument_schema_json=t.descriptor.argument_schema_json;
        d.description=t.descriptor.description;d.short_description=t.short_description;d.usage_type=GRIM::MMO::ToolUsage::Model;
        d.needs_confirmation=t.descriptor.needs_confirmation;d.is_informational=t.descriptor.is_informational;
        auto handler=t.handler;auto context=t.user_data;
        registrations.push_back({d,[handler,context](const std::string& args){
            CommandResult result;GrimToolResultWriter writer;writer.host_context=&result;
            writer.complete=[](GrimToolResultWriter* w,const GrimToolResult* r){auto& value=*static_cast<CommandResult*>(w->host_context);
                value.success=r->success;value.message=r->message;value.structuredOutputJson=r->structured_output_json;return GRIM_OK;};
            GrimToolCall request;request.arguments=args.data();request.arguments_size=args.size();assert(handler(&request,&writer,context)==GRIM_OK);return result;
        }});
    }
    return GRIM::MMO::ProviderRegistry::instance().publish(
        {GRIM::MMO::ToolProviderType::Plugin, "schedule_test", "fixture-1", {}}, registrations)
        ? GRIM_OK : GRIM_ERROR_ALREADY_EXISTS;
}
static Json invoke(const std::string& operation,Json args) {
    auto& registry=GRIM::MMO::ToolRegistry::instance();
    return Json::parse(registry.invoke(
        "schedule." + operation, args.dump(),
        {.expected_registry_version = registry.version(), .require_canonical_model = true}).structuredOutputJson);
}
int main() {
    const auto root=std::filesystem::temp_directory_path()/("grim-schedule-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {

        GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;api.register_tools=registerTools;
        api.replace_resources=[](const GrimPluginAPI*,const char*,const char*,size_t){return GRIM_OK;};
        api.create_timer=[](const GrimPluginAPI*,uint64_t,bool,GrimTimerCallback,void*){return reinterpret_cast<GrimTimerHandle>(1);};
        api.cancel_timer=[](const GrimPluginAPI*,GrimTimerHandle){return GRIM_OK;};
        api.show_notification=[](const GrimPluginAPI*,const char*,const char* text){notifications.emplace_back(text);};
        api.string_free=[](const GrimPluginAPI*,const char* p){delete[] p;};
        assert(grim_plugin_init(&api)==GRIM_OK);
        provider=std::make_unique<grim::scheduling::Provider>(&api,root/"events.json");
        provider->prepare();
        assert(GRIM::MMO::ToolRegistry::instance().getModelTools().size()==6);
        auto event=invoke("create",{{"id","event"},{"kind","event"},{"title","Meeting"},{"start_utc_ms",now+1000000},{"end_utc_ms",now+2000000}});
        assert(event["success"]==true);
        assert(invoke("create",{{"id","event"},{"kind","event"},{"title","Retry"}})["error"]=="conflict");
        assert(invoke("get",{{"id","event"}})["item"]["title"]=="Meeting");
        auto update=invoke("update",{{"id","event"},{"revision",event["item"]["revision"]},{"title","Updated"}});
        assert(update["success"]==true);
        assert(invoke("cancel",{{"id","event"},{"revision",event["item"]["revision"]}})["error"]=="conflict");
        assert(invoke("list",{{"from_utc_ms",now+2000000},{"until_utc_ms",now+3000000}})["items"].empty());
        for(const auto* kind:{"timer","alarm","reminder"}) {
            Json args={{"id",kind},{"kind",kind},{"title","Notify"}};
            if(std::string(kind)=="timer") args["duration_seconds"]=10;else args["start_utc_ms"]=now+10000;
            assert(invoke("create",args)["success"]==true);
        }
        provider->poll(now+9999);assert(notifications.empty());
        provider->poll(now+11000);assert(notifications.size()==3);
        provider->poll(now+12000);assert(notifications.size()==3);
        auto alarm=invoke("status",{{"id","alarm"}});assert(alarm["item"]["delivery_status"]=="dispatched");
        assert(invoke("cancel",{{"id","alarm"},{"revision",alarm["item"]["revision"]}})["already_dispatched"]==true);
        assert(invoke("status",Json::object())["background_delivery"]==false);
        assert(invoke("create",{{"kind","timer"},{"title","Bad"},{"duration_seconds",0}})["success"]==false);
        assert(invoke("update",{{"id","event"},{"revision",-1},{"title","Bad"}})["success"]==false);
        assert(invoke("get",{{"id","missing"}})["error"]=="not_found");
        assert(invoke("list",{{"typo",1}})["success"]==false);
        grim_plugin_shutdown();
    }
    {
        grim::scheduling::SchedulingService reopened(root/"events.json");
        auto item=reopened.execute("get",{{"id","event"}},now);assert(item["item"]["title"]=="Updated");
        assert(reopened.takeDue(now+20000).empty());
    }
    GRIM::MMO::ProviderRegistry::instance().retire("schedule_test", "fixture-1");
    std::filesystem::remove_all(root);
    std::cout<<"Schedule service, plugin handlers and model ToolRegistry tests passed\n";
}
