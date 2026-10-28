// Exercise the real API implementation and registry with host service sinks.
#include "core/plugin_api_impl.hpp"
#include "core/plugin_manager.hpp"
#include <future>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <thread>
#include "commands/commands_core.hpp"
#include "MMO/Core/HardwareInventory.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

GRIM::MMO::HardwareInventory g_hardwareInventory;
static std::vector<std::string> logs;
void logDebug(const std::string&, const std::string& message) { logs.push_back(message); }
void logError(const std::string&, const std::string& message) { logs.push_back(message); }
void notifyPopupActivity() {}
void showPopup() {}
namespace Voice {
void speak(const std::string&, const std::string&) {}
bool isSpeaking() { return false; }
}
std::string normalizeWord(const std::string& word) { return word; }
namespace GRIMInput {
std::pair<std::string, std::string> parseInput(const std::string& text) {
    const auto space = text.find(' ');
    return {text.substr(0, space), space == std::string::npos ? "" : text.substr(space + 1)};
}
}
void ensureCorePluginsRegistered() {}
CommandResult cmdOpenApp(const std::string&) { return CommandResult{}; }

namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
GrimResult identify(const GrimToolCall*, GrimToolResultWriter* writer, void* data) {
    const auto* api = static_cast<const GrimPluginAPI*>(data);
    api->log(api, GRIM_LOG_INFO, "callback");
    std::string message = "owned local message";
    GrimToolResult result;
    result.success = api->has_permission(api, GRIM_PERM_MEMORY);
    result.message = message.c_str();
    result.structured_output_json = "{\"answer\":42}";
    auto status = writer->complete(writer, &result);
    message.assign(200, 'x');
    return status;
}
GrimToolRegistration definition(const char* id, GrimToolHandler handler = identify, void* data = nullptr) {
    GrimToolRegistration tool;
    tool.handler = handler;
    tool.user_data = data;
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = id;
    d.version = "1.2.3";
    d.description = "A complete test tool";
    d.usage = "test [arguments]";
    d.category = "test";
    d.argument_format = "raw_text";
    return tool;
}
GrimResult registerOne(GrimPluginAPI* api, const char* id) {
    auto tool = definition(id, identify, api);
    tool.descriptor.required_permissions = api->get_permissions(api);
    return api->register_tools(api, &tool, 1);
}
GrimResult missingResult(const GrimToolCall*, GrimToolResultWriter*, void*) { return GRIM_OK; }
GrimResult doubleResult(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    GrimToolResult result;
    writer->complete(writer, &result);
    writer->complete(writer, &result);
    return GRIM_OK;
}
GrimResult invalidResult(const GrimToolCall*, GrimToolResultWriter* writer, void*) {
    GrimToolResult result;
    result.structured_output_json = "bad json";
    writer->complete(writer, &result); // Even ignored writer errors must fail the call.
    return GRIM_OK;
}
struct Blocking {
    std::promise<void> entered, release;
    std::shared_future<void> released = release.get_future().share();
};
GrimResult blocking(const GrimToolCall*, GrimToolResultWriter* writer, void* data) {
    auto& state = *static_cast<Blocking*>(data);
    state.entered.set_value();
    state.released.wait();
    GrimToolResult result;
    result.success = 1;
    return writer->complete(writer, &result);
}
void observe(const char*, const void*, size_t, void* data) {
    const auto* api = static_cast<const GrimPluginAPI*>(data);
    api->log(api, GRIM_LOG_INFO, "event");
}
void tick(uint64_t, void* data) { ++*static_cast<int*>(data); }
void checkName(const GrimPluginAPI* api, const char* expected) {
    const auto* name = api->get_plugin_name(api);
    check(std::string(name) == expected, "Plugin identity changed after another load");
    api->string_free(api, name);
}
}

void runPluginReconcileTests(const std::filesystem::path&, const std::filesystem::path&);

int main(int argc, char** argv) try {
    auto* alpha = PluginAPI::createPluginAPI("alpha", "alpha.dll", GRIM_PERM_MEMORY);
    check(registerOne(alpha, "alpha_tool") == GRIM_OK, "Alpha registration failed");
    auto* beta = PluginAPI::createPluginAPI("beta", "beta.dll", GRIM_PERM_UI);
    check(alpha != beta, "Plugins share an API table");
    checkName(alpha, "alpha");
    checkName(beta, "beta");
    check(alpha->has_permission(alpha, GRIM_PERM_MEMORY), "Alpha lost permissions");
    check(!beta->has_permission(beta, GRIM_PERM_MEMORY), "Beta inherited Alpha permissions");
    check(!alpha->has_permission(alpha, static_cast<GrimPermission>(GRIM_PERM_MEMORY | GRIM_PERM_UI)), "Combined permissions partially accepted");
    check(beta->store_memory(beta, "key", "value", "test") == GRIM_ERROR_PERMISSION_DENIED, "API used another owner's permissions");
    check(dispatchCommand("alpha_tool", "").success, "Command callback used wrong owner");
    check(logs.back() == "[Plugin: alpha] callback", "Callback logging used wrong owner");
    check(registerOne(alpha, "alpha_later") == GRIM_OK, "Later registration failed");
    auto tool = GRIM::MMO::ToolRegistry::instance().getTool("alpha_later");
    check(tool && tool->provider_name == "alpha" && tool->permission_bits == GRIM_PERM_MEMORY, "Later registration used last loaded plugin");
    check(beta->unregister_command(beta, "alpha_tool") == GRIM_ERROR_PERMISSION_DENIED, "Plugin removed another owner's tool");
    GRIM::MMO::ToolRegistry::instance().registerSimple("builtin", "Builtin test", "test",
        [](const std::string&) { return CommandResult{}; });
    check(registerOne(beta, "builtin") == GRIM_ERROR_ALREADY_EXISTS, "Plugin overwrote builtin");
    check(registerOne(beta, "") == GRIM_ERROR_INVALID_PARAM, "Empty tool name accepted");

    auto event = alpha->subscribe_event(alpha, "test", observe, alpha);
    check(beta->unsubscribe_event(beta, event) == GRIM_ERROR_PERMISSION_DENIED, "Plugin removed another owner's event");
    beta->emit_event(beta, "test", nullptr, 0);
    check(logs.back() == "[Plugin: alpha] event", "Event callback used wrong owner");
    int alphaTicks = 0, betaTicks = 0;
    auto timer = alpha->create_timer(alpha, 0, false, tick, &alphaTicks);
    check(beta->cancel_timer(beta, timer) == GRIM_ERROR_PERMISSION_DENIED, "Plugin cancelled another owner's timer");
    beta->create_timer(beta, 0, false, tick, &betaTicks);
    bool duplicateRejected = false;
    try { PluginAPI::createPluginAPI("alpha", "duplicate.dll", 0); }
    catch (const std::invalid_argument&) { duplicateRejected = true; }
    check(duplicateRejected, "Duplicate identity replaced a retained API");


    std::string executionMessage;
    GrimToolResultWriter recipient;
    recipient.host_context = &executionMessage;
    recipient.complete = [](GrimToolResultWriter* writer, const GrimToolResult* result) {
        *static_cast<std::string*>(writer->host_context) = result->message;
        return GRIM_OK;
    };
    check(alpha->execute_command(alpha, "alpha_tool", &recipient) == GRIM_OK &&
          executionMessage == "owned local message", "Command execution writer failed");
    check(alpha->execute_command(alpha, nullptr, &recipient) == GRIM_ERROR_INVALID_PARAM,
          "Null execution input accepted");
    auto copiedResult = dispatchCommand("alpha_tool", "");
    check(copiedResult.message == "owned local message" &&
          copiedResult.structuredOutputJson == R"({"answer":42})", "Host did not copy result data immediately");
    for (auto handler : {missingResult, doubleResult, invalidResult}) {
        auto broken = definition("broken", handler);
        check(beta->register_tools(beta, &broken, 1) == GRIM_OK, "Broken fixture registration failed");
        check(dispatchCommand("broken", "").errorCode == "ERR_PLUGIN_RESULT", "Invalid result accepted");
        beta->unregister_command(beta, "broken");
    }
    auto* staged = PluginAPI::createPluginAPI("staged", "staged.dll", 0, true);
    std::string description = "copied metadata";
    std::string schema = R"({"type":"object","properties":{"path":{"type":"string"}}})";
    const char* examples[] = {"test example"};
    const char* alias[] = {"staged_alias"};
    GrimToolParameter parameter;
    parameter.name = "path"; parameter.type = "path"; parameter.description = "Destination"; parameter.required = 1;
    auto complete = definition("staged_tool", missingResult);
    complete.descriptor.description = description.c_str();
    complete.descriptor.argument_format = "json";
    complete.descriptor.argument_schema_json = schema.c_str();
    complete.descriptor.examples = {examples, 1};
    complete.descriptor.aliases = {alias, 1};
    complete.descriptor.parameters = &parameter;
    complete.descriptor.parameter_count = 1;
    check(staged->register_tools(staged, &complete, 1) == GRIM_OK, "Staging failed");
    description.assign(200, 'x');
    schema.assign(200, 'x');
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("staged_tool"), "Uncommitted tool leaked");
    check(PluginAPI::commitPluginTools("staged"), "Commit failed");
    auto snapshot = GRIM::MMO::ToolRegistry::instance().getTool("staged_alias");
    check(snapshot && snapshot->description == "copied metadata" && snapshot->version == "1.2.3" &&
          snapshot->parameters[0].required && snapshot->provider_name == "staged" &&
          snapshot->provider_path == "staged.dll" && snapshot->argument_format == "json" &&
          snapshot->argument_schema_json.find("properties") != std::string::npos &&
          snapshot->examples == std::vector<std::string>{"test example"}, "Descriptor copy lost fields");
    PluginAPI::cleanupPluginContext("staged");

    staged = PluginAPI::createPluginAPI("staged", "staged.dll", 0, true);
    GrimToolRegistration collision[] = {definition("new_tool"), definition("builtin")};
    check(staged->register_tools(staged, collision, 2) == GRIM_OK, "Staging collision failed early");
    check(!PluginAPI::commitPluginTools("staged") && !GRIM::MMO::ToolRegistry::instance().isRegistered("new_tool"),
          "Failed commit left a partial batch");
    PluginAPI::cleanupPluginContext("staged");
    staged = PluginAPI::createPluginAPI("staged", "staged.dll", 0, true);
    complete = definition("invalid");
    complete.descriptor.struct_size = 0;
    check(staged->register_tools(staged, &complete, 1) == GRIM_ERROR_INVALID_PARAM &&
          !PluginAPI::commitPluginTools("staged"), "Ignored registration error did not reject initialization");
    PluginAPI::cleanupPluginContext("staged");
    complete = definition("invalid");
    complete.descriptor.required_permissions = GRIM_PERM_MEMORY;
    check(beta->register_tools(beta, &complete, 1) == GRIM_ERROR_INVALID_PARAM, "Undeclared permissions accepted");
    complete.descriptor.required_permissions = 0;
    complete.descriptor.argument_format = "json";
    complete.descriptor.argument_schema_json = "[]";
    check(beta->register_tools(beta, &complete, 1) == GRIM_ERROR_INVALID_PARAM, "Invalid schema accepted");

    Blocking blocked;
    auto waitTool = definition("blocking", blocking, &blocked);
    check(beta->register_tools(beta, &waitTool, 1) == GRIM_OK, "Blocking tool registration failed");
    auto running = std::async(std::launch::async, [] { return dispatchCommand("blocking", ""); });
    blocked.entered.get_future().wait();
    auto stopping = std::async(std::launch::async, [] { PluginAPI::beginPluginShutdown("beta"); });
    // Once unregistered, no new invocation can enter; shutdown must still await the old call.
    while (GRIM::MMO::ToolRegistry::instance().isRegistered("blocking")) std::this_thread::yield();
    check(stopping.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "Shutdown did not wait for an active call");
    check(!dispatchCommand("blocking", "").success, "Stopping provider accepted a new call");
    blocked.release.set_value();
    check(running.get().success, "Active call was interrupted");
    stopping.get();
    PluginAPI::cleanupPluginContext("alpha");
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("alpha_tool") && !GRIM::MMO::ToolRegistry::instance().getTool("alpha_later"), "Owner cleanup retained tools");
    PluginAPI::processTimers();
    check(alphaTicks == 0 && betaTicks == 0, "Owner cleanup retained timer or removed another owner's timer");
    checkName(beta, "beta");
    const auto before = logs.size();
    beta->emit_event(beta, "test", nullptr, 0);
    check(logs.size() == before, "Owner cleanup retained event callback");
    alpha = PluginAPI::createPluginAPI("alpha", "new-alpha.dll", 0);
    checkName(alpha, "alpha");
    check(!alpha->has_permission(alpha, GRIM_PERM_MEMORY), "Reload inherited old permissions");
    PluginAPI::cleanupPluginContext("alpha");
    PluginAPI::cleanupPluginContext("beta");

    check(argc >= 4, "Expected valid/rejected/replacement DLL fixture folders");
    const auto count = GRIM::MMO::ToolRegistry::instance().toolCount();
    PluginManager::initialize(argv[1]);
    check(GRIM::MMO::ToolRegistry::instance().toolCount() == count + 2, "Folder load did not populate registry");
    check(dispatchCommand("Declared-alpha", "").message == "fixture result", "Loaded DLL handler failed");
    PluginManager::unloadAll();
    check(GRIM::MMO::ToolRegistry::instance().toolCount() == count, "DLL unload left registry entries");
    PluginManager::initialize(argv[2]);
    check(GRIM::MMO::ToolRegistry::instance().toolCount() == count, "Rejected DLL left registrations");
    PluginManager::unloadAll();
    runPluginReconcileTests(argv[3], argc > 4 ? argv[4] : "");
    std::cout << "Plugin API ownership tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
