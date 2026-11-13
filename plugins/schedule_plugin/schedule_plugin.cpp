#include "core/plugin_api.hpp"
#include "schedule_provider.hpp"
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <array>
namespace {
std::unique_ptr<grim::scheduling::Provider> provider;
const char* operations[]={"create","get","list","update","cancel","status"};
GrimResult call(const GrimToolCall* request,GrimToolResultWriter* writer,void* context) {
    if(!writer || !writer->complete) return GRIM_ERROR_INVALID_PARAM;
    nlohmann::json value;
    try {
        if(!provider || !request || !request->arguments || request->arguments_size>65536) throw std::runtime_error("Invalid scheduling request");
        value=provider->execute(static_cast<const char*>(context),
            nlohmann::json::parse(request->arguments,request->arguments+request->arguments_size));
    } catch(const std::exception& error) {value={{"success",false},{"error","service_error"},{"message",error.what()}};}
    const std::string output=value.dump();
    const bool success=value.value("success",false);
    const std::string message=success?"Scheduling operation completed":value.value("message",std::string("Scheduling operation failed"));
    const std::string code=success?"ERR_NONE":value.value("error",std::string("service_error"));
    GrimToolResult result;result.success=success;result.message=message.c_str();result.error_code=code.c_str();
    result.category=success?"schedule":"error";result.structured_output_json=output.c_str();result.color_rgb=success?0x66CC99:0xFF6464;
    return writer->complete(writer,&result);
}
}
extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION,"Schedule Plugin","1.0.0","GRIM Project",
        "Manage saved calendar items and in-process local notifications",GRIM_PERM_SYSTEM|GRIM_PERM_FILESYSTEM|GRIM_PERM_UI};return &info;
}
extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if(!api || (api->api_version>>16)!=GRIM_API_VERSION_MAJOR || api->api_version<GRIM_API_VERSION ||
        !api->create_timer || !api->cancel_timer || !api->replace_resources || !api->show_notification || !api->register_tools) return GRIM_ERROR_API_VERSION_MISMATCH;
    provider=std::make_unique<grim::scheduling::Provider>(api);
    using Json=nlohmann::json;
    const Json id={{"type","string"},{"minLength",1},{"maxLength",4096}};
    const Json revision={{"type","integer"},{"minimum",1}};
    const Json time={{"type","integer"},{"description","UTC milliseconds since Unix epoch; resolve the user's timezone before calling"}};
    const Json kind={{"type","string"},{"enum",{"event","timer","alarm","reminder"}}};
    const Json fields={{"id",id},{"kind",kind},{"title",{{"type","string"},{"minLength",1},{"maxLength",120}}},{"description",{{"type","string"},{"maxLength",4096}}},
        {"start_utc_ms",time},{"end_utc_ms",time},{"duration_seconds",{{"type","integer"},{"minimum",1},{"maximum",31536000}}}};
    std::array<Json,6> schemas;
    for(auto& schema:schemas) schema={{"type","object"},{"additionalProperties",false},{"properties",Json::object()}};
    schemas[0]["properties"]=fields;schemas[0]["required"]={"kind","title"};
    schemas[0]["oneOf"]={
        {{"properties",{{"kind",{{"const","event"}}}}},{"required",{"start_utc_ms","end_utc_ms"}}},
        {{"properties",{{"kind",{{"const","timer"}}}}},{"required",{"duration_seconds"}}},
        {{"properties",{{"kind",{{"enum",{"alarm","reminder"}}}}}},{"required",{"start_utc_ms"}}}};
    schemas[1]["properties"]={{"id",id}};schemas[1]["required"]={"id"};
    schemas[2]["properties"]={{"kind",kind},{"from_utc_ms",time},{"until_utc_ms",time},
        {"limit",{{"type","integer"},{"minimum",1},{"maximum",200},{"default",50}}},{"offset",{{"type","integer"},{"minimum",0},{"default",0}}}};
    schemas[3]["properties"]=fields;schemas[3]["properties"].erase("kind");schemas[3]["properties"]["revision"]=revision;schemas[3]["required"]={"id","revision"};
    schemas[4]["properties"]={{"id",id},{"revision",revision}};schemas[4]["required"]={"id","revision"};
    schemas[5]["properties"]={{"id",id}};
    const char* summaries[]={"Create a saved event or local notification","Retrieve a scheduled item by its identifier","Find scheduled items by time and kind","Modify a scheduled item using its revision","Cancel a scheduled item using its revision","Inspect scheduler capabilities and item delivery status"};
    const char* descriptions[]={
        "Create an event (start/end UTC), timer (duration_seconds), alarm or reminder (start UTC). Optional caller id supports safe retries: on conflict use get. Local notifications require GRIM running; overdue items dispatch on restart. No native alarm or remote-device delivery.",
        "Get the complete saved item, current revision and delivery status by id.",
        "List items overlapping [from_utc_ms,until_utc_ms), ordered by start then id. Optional kind filter; limit defaults to 50, maximum 200; offset pagination is not a snapshot.",
        "Patch fields on an item with matching revision. Kind cannot change. Timer duration restarts from now; changing alarm/reminder start rearms delivery. Title-only changes do not rearm. Fetch again on conflict.",
        "Delete an item with matching revision. Prevents future dispatch; cannot recall a notification already handed to the host. Missing ids return not_found.",
        "With id, inspect the item and delivery state. Without id, inspect supported kinds and device delivery capabilities. Dispatched means handed to the host, not acknowledged or delivered by a native device."};
    const char* examples[]={R"(schedule.create {"kind":"timer","title":"Check oven","duration_seconds":600})",
        R"(schedule.get {"id":"example"})",R"(schedule.list {"kind":"event","limit":20})",
        R"(schedule.update {"id":"example","revision":1,"title":"Updated title"})",
        R"(schedule.cancel {"id":"example","revision":1})",R"(schedule.status {})"};
    const char* tags[]={"scheduling","calendar"};
    std::array<std::string,6> ids,schemaText,usage;
    std::array<GrimToolRegistration,6> registrations;
    for(size_t i=0;i<6;++i) {
        ids[i]="schedule."+std::string(operations[i]);schemaText[i]=schemas[i].dump();usage[i]=ids[i]+" <json>";
        auto& tool=registrations[i];tool.handler=call;tool.user_data=const_cast<char*>(operations[i]);tool.usage_type=GRIM_TOOL_USAGE_MODEL;tool.short_description=summaries[i];
        auto& d=tool.descriptor;d.tool_id=d.display_name=ids[i].c_str();d.version="1.0.0";d.description=descriptions[i];d.usage=usage[i].c_str();
        d.category="scheduling";d.argument_format="json";d.argument_schema_json=schemaText[i].c_str();
        d.required_permissions=GRIM_PERM_SYSTEM|GRIM_PERM_FILESYSTEM|GRIM_PERM_UI;d.is_informational=i==1 || i==2 || i==5;
        d.needs_confirmation=!d.is_informational;d.examples={&examples[i],1};d.capability_tags={tags,2};
    }
    const auto result=api->register_tools(api,registrations.data(),registrations.size());
    if(result!=GRIM_OK) return result;
    try {provider->prepare();return GRIM_OK;} catch(...) {return GRIM_ERROR_UNKNOWN;}
}
extern "C" PLUGIN_EXPORT void grim_plugin_shutdown(){provider.reset();}
