// Exercise the real API implementation and registry with host service sinks.
#include "loaded_provider_fixture.hpp"
#include "core/plugin_manager.hpp"
#include <future>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <thread>
#include "commands/command_result.hpp"
#include "MMO/Core/HardwareInventory.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

GRIM::MMO::HardwareInventory g_hardwareInventory;
static std::vector<std::string> logs;
static std::mutex logsMutex;
void logDebug(const std::string&, const std::string& message) { std::lock_guard lock(logsMutex); logs.push_back(message); }
void logError(const std::string&, const std::string& message) { std::lock_guard lock(logsMutex); logs.push_back(message); }
std::vector<std::string> pluginTestLogSnapshot() {
    std::lock_guard lock(logsMutex);
    return logs;
}
void notifyPopupActivity() {}
void showPopup() {}
std::vector<std::string> pluginNotifications;
void publishPluginNotification(const std::string& title,const std::string& message) { pluginNotifications.push_back(title+": "+message); }
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
    const auto* api = static_cast<const LoadedProviderFixture*>(data);
    (*api)->log(api->handle(), GRIM_LOG_INFO, "callback");
    std::string message = "owned local message";
    GrimToolResult result;
    result.success = (*api)->has_permission(api->handle(), GRIM_PERM_MEMORY);
    result.message = message.c_str();
    result.structured_output_json = "{\"answer\":42}";
    auto status = writer->complete(writer, &result);
    message.assign(200, 'x');
    return status;
}
using Json = nlohmann::json;
Json definition(const char* id, Json permissions = Json::array()) {
    return {{"tool_id", id}, {"description", "A complete test tool"}, {"usage", "test [arguments]"},
        {"category", "test"}, {"argument_format", "raw_text"}, {"permissions", std::move(permissions)},
        {"is_informational", false}, {"needs_confirmation", false}};
}

LoadedProviderFixture* alphaAPI = nullptr;
GrimResult dispatchIdentify(const char*, const GrimToolCall* call, GrimToolResultWriter* writer) {
    return identify(call, writer, alphaAPI);
}
GrimResult dispatchEmpty(const char*, const GrimToolCall*, GrimToolResultWriter*) { return GRIM_OK; }

void loadTools(LoadedProviderFixture& provider, std::initializer_list<Json> tools, GrimToolDispatch dispatch = dispatchEmpty) {
    provider.load(Json(tools), dispatch);
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
Blocking* blockedCall = nullptr;
GrimResult dispatchBlocking(const char*, const GrimToolCall* call, GrimToolResultWriter* writer) {
    return blocking(call, writer, blockedCall);
}
GrimResult (*brokenHandler)(const GrimToolCall*, GrimToolResultWriter*, void*) = nullptr;
GrimResult dispatchBroken(const char*, const GrimToolCall* call, GrimToolResultWriter* writer) {
    return brokenHandler(call, writer, nullptr);
}

void checkName(const LoadedProviderFixture* api, const char* expected) {
    const auto* name = (*api)->get_plugin_name(api->handle());
    check(std::string(name) == expected, "Plugin identity changed after another load");
    (*api)->string_free(api->handle(), name);
}


}

void runPluginReconcileTests(const std::filesystem::path&, const std::filesystem::path&);
void runFilesystemPluginTests(const std::filesystem::path&);
void runUtilityPluginTests(const std::filesystem::path&);
void runApplicationPluginTests(const std::filesystem::path&);
void runToolInspectionPluginTests(const std::filesystem::path&);
void runSystemInfoPluginTests(const std::filesystem::path&);
void runRuntimePluginTests(const std::filesystem::path&);
void runUIPluginTests(const std::filesystem::path&);
void runInputPluginTests(const std::filesystem::path&);

// Exercise the generic resource ABI with no domain implementation linked into the host.
void runProviderResourceTests() {
    using namespace GRIM::MMO;
    auto& registry = ProviderRegistry::instance();
    const auto baseline = registry.snapshot().resourceCount();
    const auto payload = nlohmann::json::array({{{"resource_id", "item"}, {"kind", "fixture"},
        {"name", "Fixture item"}, {"runtime_id", uint64_t{1}}, {"state", {{"value", 42}}}}}).dump();
    auto publish = [&](GrimPluginHandle instance, const std::string& text) {
        return PluginAPI::hostAPI().replace_resources(instance, "fixture", text.data(), text.size());
    };
    const ProviderDescriptor descriptor{.name = "resource-provider", .path = "fixture",
        .version = "1.2.3", .required_permissions = 0};
    LoadedProviderFixture first(descriptor);
    first.load(Json::array({definition("generation_tool")}), dispatchEmpty);
    const auto oldHandle = first.handle();
    auto copied = payload;
    check(publish(oldHandle, copied) == GRIM_OK, "Live provider could not publish resources");
    copied.assign(copied.size(), 'x');
    check(registry.snapshot().resourceCount() == baseline + 1, "Initial resource missing");
    for (const auto& group : registry.snapshot().providers)
        if (group.provider.name == descriptor.name)
            check(group.tools.size() == 1 && group.resources.at(0).state.at("value") == 42,
                  "Tools and copied resources did not share the provider record");
    const auto revision = registry.resourceVersion();
    check(publish(oldHandle, "[{}]") == GRIM_ERROR_INVALID_PARAM && registry.resourceVersion() == revision,
          "Malformed update changed live resources");
    auto duplicate = Json::parse(payload);
    duplicate.push_back(duplicate[0]);
    check(publish(oldHandle, duplicate.dump()) == GRIM_ERROR_INVALID_PARAM && registry.resourceVersion() == revision,
          "Duplicate update partially committed");

    LoadedProviderFixture second(descriptor);
    check(publish(second.handle(), payload) == GRIM_ERROR_PLUGIN_NOT_LOADED,
          "Incoming instance published resources while the old instance was live");
    first.unload();
    check(registry.snapshot().resourceCount() == baseline && !ToolRegistry::instance().isRegistered("generation_tool"),
          "Unload retained owned tools or resources");
    second.load(Json::array({definition("generation_tool")}), dispatchEmpty);
    check(second.handle() != oldHandle && publish(second.handle(), payload) == GRIM_OK,
          "Replacement did not load with its own identity and resources");
    check(publish(oldHandle, "[]") == GRIM_ERROR_PLUGIN_NOT_LOADED && registry.snapshot().resourceCount() == baseline + 1,
          "Stale handle mutated the replacement");
    check(publish(second.handle(), "[]") == GRIM_OK && registry.snapshot().resourceCount() == baseline,
          "Empty resource batch did not remove the kind");

}

void runProviderSchedulingTests(const std::filesystem::path&);

static void runSystemTickTests() {
    struct Tick {
        int calls = 0;
        double delta = 0;
        double offset = 0;
    } firstTick, replacementTick;
    const auto callback = [](double delta, double offset, void* data) {
        auto& tick = *static_cast<Tick*>(data);
        ++tick.calls;
        tick.delta = delta;
        tick.offset = offset;
    };

    const GRIM::MMO::ProviderDescriptor descriptor{.name = "tick-provider", .path = "tick.dll",
        .version = "1.2.3", .required_permissions = 0};
    LoadedProviderFixture first(descriptor);
    first.load();
    check(first->subscribe_system_tick(first.handle(), callback, &firstTick) == GRIM_OK, "Tick subscription failed");
    check(first->subscribe_system_tick(first.handle(), callback, &firstTick) == GRIM_ERROR_ALREADY_EXISTS,
          "Duplicate tick subscription accepted");
    PluginAPI::dispatchSystemTick(0.125, 1.5);
    check(firstTick.calls == 1 && firstTick.delta == 0.125 && firstTick.offset == 1.5,
          "Provider did not receive frame timing");
    first.unload();

    LoadedProviderFixture replacement(descriptor);
    replacement.load();
    check(replacement->subscribe_system_tick(replacement.handle(), callback, &replacementTick) == GRIM_OK,
          "Replacement tick subscription failed");
    PluginAPI::dispatchSystemTick(0.25, 1.75);
    check(firstTick.calls == 1 && replacementTick.calls == 1, "Tick dispatched to the wrong instance");
    replacement.unload();
    PluginAPI::dispatchSystemTick(0.25, 2.0);
    check(replacementTick.calls == 1, "Unloaded provider received a tick");

}

int main(int argc, char** argv) try {
    runProviderResourceTests();
    runSystemTickTests();
    {
        auto& registry = GRIM::MMO::ProviderRegistry::instance();
        const auto count = registry.snapshot().providers.size();
        LoadedProviderFixture empty({.name = "zero-tool-provider", .path = "empty.dll", .version = "1.2.3", .required_permissions = 0});
        empty.load();
        check(registry.snapshot().providers.size() == count + 1, "Zero-tool provider was not published");
        empty.unload();
        check(registry.snapshot().providers.size() == count, "Zero-tool provider survived unload");
    }
    LoadedProviderFixture alpha({.name = "alpha", .path = "alpha.dll", .version = "1.2.3", .required_permissions = GRIM_PERM_MEMORY});
    alphaAPI = &alpha;
    loadTools(alpha, {definition("alpha_tool", {"memory"}), definition("alpha_second", {"memory"})}, dispatchIdentify);
    Blocking blocked;
    blockedCall = &blocked;
    LoadedProviderFixture beta({.name = "beta", .path = "beta.dll", .version = "1.2.3", .required_permissions = GRIM_PERM_UI});
    loadTools(beta, {definition("blocking")}, dispatchBlocking);
    check(alpha.handle() != beta.handle() && alpha.operator->() == beta.operator->(),
          "Providers do not share one API table with distinct handles");
    checkName(&alpha, "alpha");
    checkName(&beta, "beta");
    check(alpha->has_permission(alpha.handle(), GRIM_PERM_MEMORY), "Alpha lost permissions");
    check(!beta->has_permission(beta.handle(), GRIM_PERM_MEMORY), "Beta inherited Alpha permissions");
    check(!alpha->has_permission(alpha.handle(), static_cast<GrimPermission>(GRIM_PERM_MEMORY | GRIM_PERM_UI)), "Combined permissions partially accepted");
    check(GRIM::MMO::ToolRegistry::instance().invoke("alpha_tool", "").success, "Command callback used wrong owner");
    check(logs.back() == "[Plugin: alpha] callback", "Callback logging used wrong owner");
    auto tool = GRIM::MMO::ToolRegistry::instance().getTool("alpha_second");
    check(tool && tool->provider.name == "alpha" && tool->permission_bits == GRIM_PERM_MEMORY,
          "Staged registration lost provider ownership");
    check(beta->unregister_command(beta.handle(), "alpha_tool") == GRIM_ERROR_PERMISSION_DENIED, "Plugin removed another owner's tool");
    using namespace GRIM::MMO;
    const ProviderDescriptor collisionOwner{ToolProviderType::Plugin, "Collision fixture", "fixture-1", {}};
    ToolDescriptor collisionTool;
    collisionTool.tool_id = "reserved_fixture_tool";
    check(ProviderRegistry::instance().publish(collisionOwner, {
        {collisionTool, [](const std::string&) { return CommandResult{}; }}
    }), "Collision fixture publication failed");

    bool duplicateRejected = false;
    try { LoadedProviderFixture duplicate({.name = "alpha", .path = "duplicate.dll", .version = "1.2.3", .required_permissions = 0}); duplicate.load(); }
    catch (const std::runtime_error&) { duplicateRejected = true; }
    check(duplicateRejected, "Duplicate identity replaced a retained API");


    std::string executionMessage;
    GrimToolResultWriter recipient;
    recipient.host_context = &executionMessage;
    recipient.complete = [](GrimToolResultWriter* writer, const GrimToolResult* result) {
        *static_cast<std::string*>(writer->host_context) = result->message;
        return GRIM_OK;
    };
    check(alpha->execute_command(alpha.handle(), "alpha_tool", &recipient) == GRIM_OK &&
          executionMessage == "owned local message", "Command execution writer failed");
    check(alpha->execute_command(alpha.handle(), nullptr, &recipient) == GRIM_ERROR_INVALID_PARAM,
          "Null execution input accepted");
    auto copiedResult = GRIM::MMO::ToolRegistry::instance().invoke("alpha_tool", "");
    check(copiedResult.message == "owned local message" &&
          copiedResult.structuredOutputJson == R"({"answer":42})", "Host did not copy result data immediately");
    for (auto handler : {missingResult, doubleResult, invalidResult}) {
        brokenHandler = handler;
        LoadedProviderFixture brokenProvider({.name = "broken-fixture", .path = "", .version = "1.2.3", .required_permissions = 0});
        loadTools(brokenProvider, {definition("broken")}, dispatchBroken);
        check(GRIM::MMO::ToolRegistry::instance().invoke("broken", "").errorCode == "ERR_PLUGIN_RESULT", "Invalid result accepted");
        brokenProvider.unload();
    }
    LoadedProviderFixture metadata({.name = "metadata", .path = "metadata.dll", .version = "1.2.3", .required_permissions = 0});
    auto complete = definition("metadata_tool");
    complete.update({
        {"description", "copied metadata"}, {"argument_format", "json"},
        {"argument_schema", Json::parse(R"({"type":"object","properties":{"path":{"type":"string"}}})")},
        {"aliases", {"metadata_alias"}}, {"parameters", {{{"name", "path"}, {"type", "path"},
            {"description", "Destination"}, {"required", true}}}},
        {"usage_type", 2}, {"short_description", "Inspect information supplied by a selected entity"},
        {"usage", "metadata_tool {path}"}, {"bindings", {{"path", "t"}}}, {"usage_count", 5},
        {"success_count", 3}, {"failure_count", 2}, {"success_rate", 0.6}
    });
    loadTools(metadata, {complete});
    auto snapshot = GRIM::MMO::ToolRegistry::instance().getTool("metadata_alias");
    check(snapshot && snapshot->short_description=="Inspect information supplied by a selected entity",
          "Short description was not copied during registration");
    check(snapshot && snapshot->bindings == std::map<std::string, std::string>{{"path", "t"}},
          "Binding labels were not copied before registration returned");
    check(snapshot && snapshot->usage_type == GRIM::MMO::ToolUsage::System &&
          snapshot->description == "copied metadata" && snapshot->version == "1.2.3" &&
          snapshot->parameters[0].required && snapshot->provider.name == "metadata" &&
          snapshot->provider.path == "metadata.dll" && snapshot->argument_format == "json" &&
          snapshot->argument_schema_json.find("properties") != std::string::npos &&
          snapshot->aliases == std::vector<std::string>{"metadata_alias"}, "Descriptor copy lost fields");
    check(snapshot && snapshot->usage_count == 0 && snapshot->success_count == 0 &&
          snapshot->failure_count == 0 && snapshot->success_rate == 1.0f,
          "Plugin registration supplied host execution statistics");
    metadata.unload();

    auto first = definition("batch_first");
    auto second = definition("batch_second");
    second["usage_type"] = 0;
    LoadedProviderFixture batchProvider({.name = "batch-fixture", .path = "", .version = "1.2.3", .required_permissions = 0});
    loadTools(batchProvider, {first, second});
    check(ToolRegistry::instance().getTool("batch_first")->usage_type == ToolUsage::Model &&
          ToolRegistry::instance().getTool("batch_second")->usage_type == ToolUsage::Debug,
          "Manifest lost tool metadata");
    batchProvider.unload();

    LoadedProviderFixture invalidProvider({.name = "invalid-fixture", .path = "", .version = "1.2.3", .required_permissions = GRIM_PERM_UI});
    const auto rejects = [&](const auto& read, const char* reason) {
        try { read(); } catch (const std::exception&) { return; }
        throw std::runtime_error(reason);
    };
    rejects([&] { invalidProvider.load(Json::object(), dispatchEmpty); },
            "Non-array manifest tools accepted");
    rejects([&] { loadTools(invalidProvider, {first}, nullptr); }, "Missing dispatcher accepted");
    auto invalid = definition("");
    rejects([&] { loadTools(invalidProvider, {first, invalid}); }, "Empty tool ID accepted");
    invalid = definition("invalid");
    invalid["short_description"] = "Too short";
    rejects([&] { loadTools(invalidProvider, {invalid}); }, "Invalid summary accepted");
    invalid.erase("short_description");
    invalid["usage"] = "invalid {target}";
    invalid["bindings"] = {{"target", "invalid_type"}};
    rejects([&] { loadTools(invalidProvider, {invalid}); }, "Invalid binding accepted");
    invalid["bindings"] = {{"missing", "t"}};
    rejects([&] { loadTools(invalidProvider, {invalid}); }, "Missing binding placeholder accepted");
    invalid.erase("bindings");
    invalid["usage_type"] = 99;
    rejects([&] { loadTools(invalidProvider, {first, invalid}); }, "Invalid usage type accepted");
    invalid.erase("usage_type");
    invalid["permissions"] = {"memory"};
    rejects([&] { loadTools(invalidProvider, {invalid}); }, "Undeclared permissions accepted");
    invalid["permissions"] = Json::array();
    invalid["argument_format"] = "json";
    invalid["argument_schema"] = Json::array();
    rejects([&] { loadTools(invalidProvider, {invalid}); }, "Invalid schema accepted");
    check(!ToolRegistry::instance().isRegistered("batch_first"), "Invalid manifest partially published");
    invalidProvider.unload();

    LoadedProviderFixture conflict({.name = "conflict", .path = "conflict.dll", .version = "1.2.3", .required_permissions = 0});
    rejects([&] { loadTools(conflict, {definition("new_tool"), definition("reserved_fixture_tool")}); },
            "Tool collision accepted");
    check(!ToolRegistry::instance().isRegistered("new_tool"), "Failed load left a partial manifest");
    auto running = std::async(std::launch::async, [] { return GRIM::MMO::ToolRegistry::instance().invoke("blocking", ""); });
    blocked.entered.get_future().wait();
    auto stopping = std::async(std::launch::async, [&] { beta.unload(); });
    // Once unregistered, no new invocation can enter; shutdown must still await the old call.
    while (GRIM::MMO::ToolRegistry::instance().isRegistered("blocking")) std::this_thread::yield();
    check(stopping.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
          "Shutdown did not wait for an active call");
    check(!GRIM::MMO::ToolRegistry::instance().invoke("blocking", "").success, "Stopping provider accepted a new call");
    blocked.release.set_value();
    check(running.get().success, "Active call was interrupted");
    stopping.get();
    alpha.unload();
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("alpha_tool") && !GRIM::MMO::ToolRegistry::instance().getTool("alpha_second"), "Owner cleanup retained tools");
    LoadedProviderFixture replacement({.name = "alpha", .path = "new-alpha.dll", .version = "1.2.3", .required_permissions = 0});
    replacement.load();
    checkName(&replacement, "alpha");
    check(!replacement->has_permission(replacement.handle(), GRIM_PERM_MEMORY), "Reload inherited old permissions");
    replacement.unload();

    check(argc >= 4, "Expected valid/rejected/replacement DLL fixture folders");
    const auto count = GRIM::MMO::ToolRegistry::instance().toolCount();
    PluginManager::initialize(argv[1]);
    check(GRIM::MMO::ToolRegistry::instance().toolCount() == count + 2, "Folder load did not populate registry");
    check(GRIM::MMO::ToolRegistry::instance().invoke("Declared-alpha", "").message == "fixture result", "Loaded DLL handler failed");
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
    if (argc > 9 && *argv[9]) runSystemInfoPluginTests(argv[9]);
    if (argc > 10 && *argv[10]) runRuntimePluginTests(argv[10]);
    runUIPluginTests(argc > 13 ? argv[13] : "");
    runInputPluginTests(argc > 14 ? argv[14] : "");
    runProviderSchedulingTests(argc > 15 ? argv[15] : "");
    ProviderRegistry::instance().retire(collisionOwner.name, collisionOwner.instance);
    std::cout << "Plugin API ownership tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
