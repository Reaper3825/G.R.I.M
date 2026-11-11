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
static std::mutex logsMutex;
void logDebug(const std::string&, const std::string& message) { std::lock_guard lock(logsMutex); logs.push_back(message); }
void logError(const std::string&, const std::string& message) { std::lock_guard lock(logsMutex); logs.push_back(message); }
void notifyPopupActivity() {}
void showPopup() {}
namespace Voice {
void speak(const std::string&, const std::string&) {}
}
std::string normalizeWord(const std::string& word) { return word; }
namespace GRIMInput {
std::pair<std::string, std::string> parseInput(const std::string& text) {
    const auto space = text.find(' ');
    return {text.substr(0, space), space == std::string::npos ? "" : text.substr(space + 1)};
}
}

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
void runFilesystemPluginTests(const std::filesystem::path&);
void runUtilityPluginTests(const std::filesystem::path&);
void runApplicationPluginTests(const std::filesystem::path&);
void runToolInspectionPluginTests(const std::filesystem::path&);
void runHostAliasServiceTests();
void runSystemInfoPluginTests(const std::filesystem::path&);
void runRuntimePluginTests(const std::filesystem::path&, const std::filesystem::path&);
void runPerceptionPluginTests(const std::filesystem::path&);
void runVoicePluginTests(const std::filesystem::path&);
void runUIPluginTests(const std::filesystem::path&);
void runInputPluginTests(const std::filesystem::path&);

int main(int argc, char** argv) try {
    {
        auto& registry = GRIM::MMO::ProviderRegistry::instance();
        const auto count = registry.snapshot().providers.size();
        auto* empty = PluginAPI::createPluginAPI("zero-tool-provider", "empty.dll", 0, true);
        check(registry.snapshot().providers.size() == count, "Staged provider leaked before commit");
        check(PluginAPI::commitPluginTools(empty), "Zero-tool provider commit failed");
        check(registry.snapshot().providers.size() == count + 1, "Zero-tool provider was not published");
        PluginAPI::cleanupPluginContext(empty);
        check(registry.snapshot().providers.size() == count, "Zero-tool provider survived unload");
    }
    auto* alpha = PluginAPI::createPluginAPI("alpha", "alpha.dll", GRIM_PERM_MEMORY, true);
    check(registerOne(alpha, "alpha_tool") == GRIM_OK &&
          registerOne(alpha, "alpha_second") == GRIM_OK &&
          PluginAPI::commitPluginTools(alpha), "Alpha publication failed");
    Blocking blocked;
    auto waitTool = definition("blocking", blocking, &blocked);
    auto* beta = PluginAPI::createPluginAPI("beta", "beta.dll", GRIM_PERM_UI, true);
    check(beta->register_tools(beta, &waitTool, 1) == GRIM_OK &&
          PluginAPI::commitPluginTools(beta), "Beta publication failed");
    check(alpha != beta, "Plugins share an API table");
    checkName(alpha, "alpha");
    checkName(beta, "beta");
    check(alpha->has_permission(alpha, GRIM_PERM_MEMORY), "Alpha lost permissions");
    check(!beta->has_permission(beta, GRIM_PERM_MEMORY), "Beta inherited Alpha permissions");
    check(!alpha->has_permission(alpha, static_cast<GrimPermission>(GRIM_PERM_MEMORY | GRIM_PERM_UI)), "Combined permissions partially accepted");
    check(beta->store_memory(beta, "key", "value", "test") == GRIM_ERROR_PERMISSION_DENIED, "API used another owner's permissions");
    check(dispatchCommand("alpha_tool", "").success, "Command callback used wrong owner");
    check(logs.back() == "[Plugin: alpha] callback", "Callback logging used wrong owner");
    check(registerOne(alpha, "alpha_later") == GRIM_ERROR_UNSUPPORTED &&
          !GRIM::MMO::ToolRegistry::instance().isRegistered("alpha_later"),
          "Published provider accepted a new tool");
    auto tool = GRIM::MMO::ToolRegistry::instance().getTool("alpha_second");
    check(tool && tool->provider.name == "alpha" && tool->permission_bits == GRIM_PERM_MEMORY,
          "Staged registration lost provider ownership");
    check(beta->unregister_command(beta, "alpha_tool") == GRIM_ERROR_PERMISSION_DENIED, "Plugin removed another owner's tool");
    using namespace GRIM::MMO;
    const ProviderDescriptor collisionOwner{ToolProviderType::Plugin, "Collision fixture", "fixture-1", {}};
    ToolDescriptor collisionTool;
    collisionTool.tool_id = "reserved_fixture_tool";
    check(ProviderRegistry::instance().publish(collisionOwner, {
        {collisionTool, [](const std::string&) { return CommandResult{}; }}
    }), "Collision fixture publication failed");

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
        auto* brokenProvider = PluginAPI::createPluginAPI("broken-fixture", "", 0, true);
        check(brokenProvider->register_tools(brokenProvider, &broken, 1) == GRIM_OK &&
              PluginAPI::commitPluginTools(brokenProvider), "Broken fixture publication failed");
        check(dispatchCommand("broken", "").errorCode == "ERR_PLUGIN_RESULT", "Invalid result accepted");
        PluginAPI::cleanupPluginContext(brokenProvider);
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
    complete.usage_type = GRIM_TOOL_USAGE_SYSTEM;
    std::string shortDescription = "Inspect information supplied by a selected entity";
    complete.short_description = shortDescription.c_str();
    std::string bindingName = "path";
    std::string bindingType = "t";
    GrimToolBinding binding{bindingName.c_str(), bindingType.c_str()};
    complete.descriptor.usage = "staged_tool {path}";
    complete.bindings = &binding; complete.binding_count = 1;
    check(staged->register_tools(staged, &complete, 1) == GRIM_OK, "Staging failed");
    shortDescription.assign(200, 'x');
    bindingName.assign(200, 'x'); bindingType.assign(200, 'x');
    description.assign(200, 'x');
    schema.assign(200, 'x');
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("staged_tool"), "Uncommitted tool leaked");
    check(PluginAPI::commitPluginTools("staged"), "Commit failed");
    auto snapshot = GRIM::MMO::ToolRegistry::instance().getTool("staged_alias");
    check(snapshot && snapshot->short_description=="Inspect information supplied by a selected entity",
          "Short description was not copied during registration");
    check(snapshot && snapshot->bindings == std::map<std::string, std::string>{{"path", "t"}},
          "Binding labels were not copied before registration returned");
    check(snapshot && snapshot->usage_type == GRIM::MMO::ToolUsage::System &&
          snapshot->description == "copied metadata" && snapshot->version == "1.2.3" &&
          snapshot->parameters[0].required && snapshot->provider.name == "staged" &&
          snapshot->provider.path == "staged.dll" && snapshot->argument_format == "json" &&
          snapshot->argument_schema_json.find("properties") != std::string::npos &&
          snapshot->aliases == std::vector<std::string>{"staged_alias"}, "Descriptor copy lost fields");
    PluginAPI::cleanupPluginContext("staged");

    auto first = definition("batch_first");
    auto second = definition("batch_second");
    second.usage_type = GRIM_TOOL_USAGE_DEBUG;
    GrimToolRegistration batch[] = {first, second};
    auto* invalidProvider = PluginAPI::createPluginAPI("invalid-fixture", "", GRIM_PERM_UI, true);
    check(registerOne(invalidProvider, "") == GRIM_ERROR_INVALID_PARAM, "Empty tool name accepted");
    --batch[1].struct_size;
    check(invalidProvider->register_tools(invalidProvider, batch, 2) == GRIM_ERROR_INVALID_PARAM &&
          !GRIM::MMO::ToolRegistry::instance().isRegistered("batch_first"),
          "Invalid registration size partially committed a batch");
    batch[1] = second;
    ++batch[1].descriptor.struct_size;
    check(invalidProvider->register_tools(invalidProvider, batch, 2) == GRIM_ERROR_INVALID_PARAM &&
          !GRIM::MMO::ToolRegistry::instance().isRegistered("batch_first"),
          "Invalid descriptor size partially committed a batch");
    batch[1] = second;
    auto* batchProvider = PluginAPI::createPluginAPI("batch-fixture", "", 0, true);
    check(batchProvider->register_tools(batchProvider, batch, 2) == GRIM_OK &&
          PluginAPI::commitPluginTools(batchProvider) &&
          GRIM::MMO::ToolRegistry::instance().getTool("batch_first")->usage_type == GRIM::MMO::ToolUsage::Model &&
          GRIM::MMO::ToolRegistry::instance().getTool("batch_second")->usage_type == GRIM::MMO::ToolUsage::Debug,
          "v4 batch registration lost tool metadata");
    PluginAPI::cleanupPluginContext(batchProvider);
    auto invalidSummary=definition("invalid_summary");invalidSummary.short_description="Too short";
    check(invalidProvider->register_tools(invalidProvider, &invalidSummary,1)==GRIM_ERROR_INVALID_PARAM,"Invalid short summary registered");
    auto badBinding = definition("bad_binding");
    badBinding.descriptor.usage = "bad_binding {target}";
    const GrimToolBinding duplicateBindings[] = {{"target", "t"}, {"target", "t"}};
    badBinding.bindings = duplicateBindings; badBinding.binding_count = 2;
    check(invalidProvider->register_tools(invalidProvider, &badBinding, 1) == GRIM_ERROR_INVALID_PARAM,
          "Duplicate binding placeholder accepted");
    badBinding.bindings = nullptr; badBinding.binding_count = 1;
    check(invalidProvider->register_tools(invalidProvider, &badBinding, 1) == GRIM_ERROR_INVALID_PARAM,
          "Null binding array accepted");
    GrimToolRegistration invalidUsage[] = {definition("usage_valid"), definition("usage_invalid")};
    invalidUsage[1].usage_type = static_cast<GrimToolUsage>(99);
    check(invalidProvider->register_tools(invalidProvider, invalidUsage, 2) == GRIM_ERROR_INVALID_PARAM &&
          !GRIM::MMO::ToolRegistry::instance().isRegistered("usage_valid"), "Invalid usage published partial batch");

    staged = PluginAPI::createPluginAPI("staged", "staged.dll", 0, true);
    GrimToolRegistration collision[] = {definition("new_tool"), definition("reserved_fixture_tool")};
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
    check(invalidProvider->register_tools(invalidProvider, &complete, 1) == GRIM_ERROR_INVALID_PARAM, "Undeclared permissions accepted");
    complete.descriptor.required_permissions = 0;
    complete.descriptor.argument_format = "json";
    complete.descriptor.argument_schema_json = "[]";
    check(invalidProvider->register_tools(invalidProvider, &complete, 1) == GRIM_ERROR_INVALID_PARAM, "Invalid schema accepted");

    check(!PluginAPI::commitPluginTools(invalidProvider), "Invalid declarations did not poison publication");
    PluginAPI::cleanupPluginContext(invalidProvider);
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
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("alpha_tool") && !GRIM::MMO::ToolRegistry::instance().getTool("alpha_second"), "Owner cleanup retained tools");
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
    if (argc > 5 && *argv[5]) runFilesystemPluginTests(argv[5]);
    if (argc > 6 && *argv[6]) runUtilityPluginTests(argv[6]);
    if (argc > 7 && *argv[7]) runApplicationPluginTests(argv[7]);
    if (argc > 8 && *argv[8]) runToolInspectionPluginTests(argv[8]);
    runHostAliasServiceTests();
    if (argc > 9 && *argv[9]) runSystemInfoPluginTests(argv[9]);
    if ((argc > 10 && *argv[10]) || (argc > 11 && *argv[11])) runRuntimePluginTests(argc > 10 ? argv[10] : "", argc > 11 ? argv[11] : "");
    if (argc > 12 && *argv[12]) runPerceptionPluginTests(argv[12]);
    if (argc > 13 && *argv[13]) runVoicePluginTests(argv[13]);
    runUIPluginTests(argc > 14 ? argv[14] : "");
    runInputPluginTests(argc > 15 ? argv[15] : "");
    ProviderRegistry::instance().retire(collisionOwner.name, collisionOwner.instance);
    std::cout << "Plugin API ownership tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
