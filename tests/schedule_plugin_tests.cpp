#include "plugins/schedule_plugin/scheduling_service.hpp"
#include "plugins/schedule_plugin/schedule_plugin.cpp"
#include "tests/provider_panel_fixture.hpp"
#include <cassert>
#include <iostream>
using Json=nlohmann::json;
static std::vector<std::string> notifications;
static Json invoke(const std::string& operation, Json args) {
    const auto id = "schedule." + operation;
    const auto payload = args.dump();
    GrimToolCall request{sizeof(GrimToolCall), payload.data(), payload.size()};
    Json result;
    GrimToolResultWriter writer;
    writer.host_context = &result;
    writer.complete = [](GrimToolResultWriter* current, const GrimToolResult* value) {
        *static_cast<Json*>(current->host_context) = value->structured_output_json
            ? Json::parse(value->structured_output_json) : Json();
        return GRIM_OK;
    };
    assert(grim_plugin_dispatch_tool(id.c_str(), &request, &writer) == GRIM_OK);
    return result;
}
void runSchedulePluginTests() {
    const auto now = grim::scheduling::scheduleNow();
    const auto root=std::filesystem::temp_directory_path()/("grim-schedule-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {

        GrimPluginAPI api{};api.api_version=GRIM_API_VERSION;
        api.replace_resources=[](const GrimPluginAPI*,const char*,const char*,size_t){return GRIM_OK;};
        api.subscribe_system_tick=[](const GrimPluginAPI*,GrimSystemTickCallback,void*){return GRIM_OK;};
        api.show_notification=[](const GrimPluginAPI*,const char*,const char* text){notifications.emplace_back(text);};
        api.log=[](const GrimPluginAPI*,GrimLogLevel,const char*){};
        auto lease = std::make_shared<GRIM::UI::ProviderPanelLease>();
        lease->owner = "fixture-1";
        prepareProviderPanelFixture(api, lease, true);
        auto oldApi=api;oldApi.api_version=12u<<16;
        assert(grim_plugin_init(&oldApi)==GRIM_ERROR_API_VERSION_MISMATCH);
        assert(grim_plugin_init(&api)==GRIM_OK);
        calendarPanel.reset();
        provider=std::make_unique<grim::scheduling::Provider>(&api,root/"events.json");
        provider->prepare();
        auto event=invoke("create",{{"id","event"},{"kind","event"},{"title","Meeting"},{"start_utc_ms",now+1000000},{"end_utc_ms",now+2000000}});
        assert(event["success"]==true);
        assert(invoke("create",{{"id","event"},{"kind","event"},{"title","Retry"}})["error"]=="conflict");
        assert(invoke("get",{{"id","event"}})["item"]["title"]=="Meeting");
        auto update=invoke("update",{{"id","event"},{"revision",event["item"]["revision"]},{"title","Updated"}});
        assert(update["success"]==true);
        assert(invoke("cancel",{{"id","event"},{"revision",event["item"]["revision"]}})["error"]=="conflict");
        assert(invoke("list",{{"from_utc_ms",now+2000000},{"until_utc_ms",now+3000000}})["items"].empty());
        auto lastDue = now;
        for(const auto* kind:{"timer","alarm","reminder"}) {
            Json args={{"id",kind},{"kind",kind},{"title","Notify"}};
            if(std::string(kind)=="timer") args["duration_seconds"]=10;else args["start_utc_ms"]=now+10000;
            const auto created = invoke("create",args);
            assert(created["success"]==true);
            lastDue = std::max(lastDue, created.at("item").at("start_utc_ms").get<int64_t>());
        }
        provider->poll(now+9999);assert(notifications.empty());
        provider->poll(lastDue);assert(notifications.size()==3);
        provider->poll(lastDue+1000);assert(notifications.size()==3);
        auto alarm=invoke("status",{{"id","alarm"}});assert(alarm["item"]["delivery_status"]=="dispatched");
        assert(invoke("cancel",{{"id","alarm"},{"revision",alarm["item"]["revision"]}})["already_dispatched"]==true);
        assert(invoke("status",Json::object())["background_delivery"]==false);
        assert(invoke("create",{{"kind","timer"},{"title","Bad"},{"duration_seconds",0}})["success"]==false);
        assert(invoke("update",{{"id","event"},{"revision",-1},{"title","Bad"}})["success"]==false);
        assert(invoke("get",{{"id","missing"}})["error"]=="not_found");
        assert(invoke("list",{{"typo",1}})["success"]==false);
        GRIM::UI::retireProviderPanels(lease);
        grim_plugin_shutdown();
    }
    {
        grim::scheduling::SchedulingService reopened(root/"events.json");
        auto item=reopened.execute("get",{{"id","event"}},now);assert(item["item"]["title"]=="Updated");
        assert(reopened.takeDue(now+20000).empty());
    }
    std::filesystem::remove_all(root);
    std::cout<<"Schedule service and plugin dispatcher tests passed\n";
}
