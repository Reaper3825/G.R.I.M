// Exercise the real API implementation and registry with host service sinks.
#include "core/plugin_api_impl.hpp"
#include "core/plugin_manager.hpp"
#include <future>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <condition_variable>
#include <chrono>
#include <atomic>
#include <thread>
#include "commands/commands_core.hpp"
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
using Json = nlohmann::json;
Json definition(const char* id, Json permissions = Json::array()) {
    return {{"tool_id", id}, {"description", "A complete test tool"}, {"usage", "test [arguments]"},
        {"category", "test"}, {"argument_format", "raw_text"}, {"permissions", std::move(permissions)},
        {"is_informational", false}, {"needs_confirmation", false}};
}

const GrimPluginAPI* alphaAPI = nullptr;
GrimResult dispatchIdentify(const char*, const GrimToolCall* call, GrimToolResultWriter* writer) {
    return identify(call, writer, const_cast<GrimPluginAPI*>(alphaAPI));
}
GrimResult dispatchEmpty(const char*, const GrimToolCall*, GrimToolResultWriter*) { return GRIM_OK; }

std::vector<GRIM::MMO::ToolRegistration> manifestTools(
    const GrimPluginAPI* api, std::initializer_list<Json> tools, GrimToolDispatch dispatch = dispatchEmpty) {
    return PluginAPI::readManifestTools(Json(std::vector<Json>(tools)), api, dispatch);
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

void observe(const char*, const void*, size_t, void* data) {
    const auto* api = static_cast<const GrimPluginAPI*>(data);
    api->log(api, GRIM_LOG_INFO, "event");
}
void checkName(const GrimPluginAPI* api, const char* expected) {
    const auto* name = api->get_plugin_name(api);
    check(std::string(name) == expected, "Plugin identity changed after another load");
    api->string_free(api, name);
}

struct GenerationObserver : GRIM::MMO::ToolRegistryListener {
    std::optional<GRIM::MMO::RegisteredTool> tool;
    GRIM::MMO::ProviderRegistrySnapshot snapshot;

    void onToolRegistered(const std::string&) override {}
    void onToolUnregistered(const std::string&) override {}
    void onToolUpdated(const std::string& id) override {
        if (id != "generation_tool") return;
        tool = GRIM::MMO::ToolRegistry::instance().getTool(id);
        snapshot = GRIM::MMO::ProviderRegistry::instance().snapshot();
    }
};
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
    auto publish = [&](const GrimPluginAPI* api, const std::string& text) {
        return api->replace_resources(api, "fixture", text.data(), text.size());
    };
    auto* first = PluginAPI::createPluginAPI({.name = "resource-provider", .path = "fixture", .version = "1.2.3", .required_permissions = 0}, true);
    check(!first->is_provider_current(first), "Staging provider is current");
    auto copiedPayload = payload;
    check(publish(first, copiedPayload) == GRIM_OK, "Cannot stage resources without tools");
    copiedPayload.assign(copiedPayload.size(), 'x');
    check(registry.snapshot().resourceCount() == baseline, "Staged resource leaked");
    check(PluginAPI::commitPluginGeneration(first), "Resource-only provider commit failed");
    check(first->is_provider_current(first) && registry.snapshot().resourceCount() == baseline + 1,
          "Initial resources missing at commit");
    for (const auto& group : registry.snapshot().providers)
        if (group.provider.name == "resource-provider")
            check(group.resources.at(0).state.at("value") == 42, "Resource payload was not copied");
    const auto revision = registry.resourceVersion();
    check(publish(first, "[{}]") == GRIM_ERROR_INVALID_PARAM && registry.resourceVersion() == revision,
          "Malformed update changed live resources");
    auto duplicate = nlohmann::json::parse(payload); duplicate.push_back(duplicate[0]);
    check(publish(first, duplicate.dump()) == GRIM_ERROR_INVALID_PARAM && registry.resourceVersion() == revision,
          "Duplicate update partially committed");
    auto* failed = PluginAPI::createPluginAPI({.name = "resource-provider", .path = "replacement", .version = "1.2.3", .required_permissions = 0}, true, first);
    check(publish(failed, payload) == GRIM_OK && publish(failed, "[{}]") == GRIM_ERROR_INVALID_PARAM,
          "Invalid staged resources accepted");
    check(!PluginAPI::commitPluginGeneration(failed, first), "Ignored resource error committed replacement");
    PluginAPI::cleanupPluginContext(failed);
    check(first->is_provider_current(first) && registry.snapshot().resourceCount() == baseline + 1,
          "Failed replacement removed current resources");
    auto* second = PluginAPI::createPluginAPI({.name = "resource-provider", .path = "replacement", .version = "1.2.3", .required_permissions = 0}, true, first);
    check(publish(second, payload) == GRIM_OK && PluginAPI::commitPluginGeneration(second, first), "Resource replacement failed");
    check(!first->is_provider_current(first) && publish(first, "[]") == GRIM_ERROR_PLUGIN_NOT_LOADED,
          "Retired generation could mutate replacement");
    PluginAPI::cleanupPluginContext(first);
    check(registry.snapshot().resourceCount() == baseline + 1, "Old cleanup erased replacement resources");
    check(publish(second, "[]") == GRIM_OK && registry.snapshot().resourceCount() == baseline,
          "Empty resource batch did not remove kind");
    PluginAPI::cleanupPluginContext(second);

    auto* old = PluginAPI::createPluginAPI({.name = "generation-provider", .path = "old.dll", .version = "1.2.3", .required_permissions = 0}, true);
    check(publish(old, payload) == GRIM_OK && PluginAPI::commitPluginGeneration(old, nullptr,
          manifestTools(old, {definition("generation_tool")})), "Initial mixed generation failed");
    bool rejectedRename = false;
    try {
        auto* renamed = PluginAPI::createPluginAPI({.name = "renamed-provider", .path = "new.dll", .version = "1.2.3", .required_permissions = 0}, true, old);
        PluginAPI::cleanupPluginContext(renamed);
    } catch (const std::invalid_argument&) {
        rejectedRename = true;
    }
    check(rejectedRename, "Replacement changed the declared provider name");
    auto* replacement = PluginAPI::createPluginAPI({.name = "generation-provider", .path = "new.dll", .version = "1.2.3", .required_permissions = 0}, true, old);
    auto newPayload = nlohmann::json::parse(payload);
    newPayload[0]["state"]["value"] = 43;
    check(publish(replacement, newPayload.dump()) == GRIM_OK, "Staging mixed generation failed");

    auto& tools = ToolRegistry::instance();
    const auto oldInstance = tools.getTool("generation_tool").value().provider.instance;
    GenerationObserver observer;
    tools.addListener(&observer);
    const bool committed = PluginAPI::commitPluginGeneration(replacement, old,
        manifestTools(replacement, {definition("generation_tool")}));
    tools.removeListener(&observer);
    check(committed && observer.tool && observer.tool->provider.instance != oldInstance,
          "Update listener did not see the replacement tool");
    bool matchingResources = false;
    for (const auto& group : observer.snapshot.providers) {
        if (group.provider.instance == observer.tool->provider.instance) {
            matchingResources = group.tools.size() == 1 && group.tools[0].tool_id == "generation_tool" &&
                group.resources.size() == 1 && group.resources[0].state.at("value") == 43;
        }
    }
    check(matchingResources, "Update listener did not see matching replacement resources");
    PluginAPI::beginPluginShutdown(old);
    const char* oldIdentityText = old->get_provider_identity(old);
    check(oldIdentityText != nullptr, "Retired generation lost its identity before shutdown");
    const std::string oldIdentityJson = oldIdentityText;
    old->string_free(old, oldIdentityText);
    const auto oldIdentity = nlohmann::json::parse(oldIdentityJson);
    check(oldIdentity.at("name") == "generation-provider" && oldIdentity.at("instance") == oldInstance &&
          oldIdentity.at("path") == "old.dll", "Retired API returned the wrong generation identity");
    PluginAPI::cleanupPluginContext(old);
    PluginAPI::cleanupPluginContext(replacement);
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

    auto* first = PluginAPI::createPluginAPI({.name = "tick-provider", .path = "first.dll", .version = "1.2.3", .required_permissions = 0}, true);
    check(first->subscribe_system_tick(first, callback, &firstTick) == GRIM_OK, "Tick subscription failed");
    check(first->subscribe_system_tick(first, callback, &firstTick) == GRIM_ERROR_ALREADY_EXISTS,
          "Duplicate tick subscription accepted");
    PluginAPI::dispatchSystemTick(0, 0);
    check(firstTick.calls == 0, "Unpublished generation received a tick");
    check(PluginAPI::commitPluginGeneration(first), "Tick generation publication failed");
    PluginAPI::dispatchSystemTick(0.125, 1.5);
    check(firstTick.calls == 1 && firstTick.delta == 0.125 && firstTick.offset == 1.5,
          "Published generation did not receive frame timing");

    auto* replacement = PluginAPI::createPluginAPI({.name = "tick-provider", .path = "second.dll", .version = "1.2.3", .required_permissions = 0}, true, first);
    check(replacement->subscribe_system_tick(replacement, callback, &replacementTick) == GRIM_OK &&
          PluginAPI::commitPluginGeneration(replacement, first), "Tick replacement failed");
    PluginAPI::dispatchSystemTick(0.25, 1.75);
    check(firstTick.calls == 1 && replacementTick.calls == 1, "Tick dispatched to the wrong generation");
    PluginAPI::cleanupPluginContext(first);
    PluginAPI::beginPluginShutdown(replacement);
    PluginAPI::dispatchSystemTick(0.25, 2.0);
    check(replacementTick.calls == 1, "Stopping generation received a tick");
    PluginAPI::cleanupPluginContext(replacement);
}

int main(int argc, char** argv) try {
    runProviderResourceTests();
    runSystemTickTests();
    {
        auto& registry = GRIM::MMO::ProviderRegistry::instance();
        const auto count = registry.snapshot().providers.size();
        auto* empty = PluginAPI::createPluginAPI({.name = "zero-tool-provider", .path = "empty.dll", .version = "1.2.3", .required_permissions = 0}, true);
        check(registry.snapshot().providers.size() == count, "Staged provider leaked before commit");
        check(PluginAPI::commitPluginGeneration(empty), "Zero-tool provider commit failed");
        check(registry.snapshot().providers.size() == count + 1, "Zero-tool provider was not published");
        PluginAPI::cleanupPluginContext(empty);
        check(registry.snapshot().providers.size() == count, "Zero-tool provider survived unload");
    }
    auto* alpha = PluginAPI::createPluginAPI({.name = "alpha", .path = "alpha.dll", .version = "1.2.3", .required_permissions = GRIM_PERM_MEMORY}, true);
    alphaAPI = alpha;
    check(PluginAPI::commitPluginGeneration(alpha, nullptr,
        manifestTools(alpha, {definition("alpha_tool", {"memory"}), definition("alpha_second", {"memory"})},
                      dispatchIdentify)), "Alpha publication failed");
    Blocking blocked;
    blockedCall = &blocked;
    auto* beta = PluginAPI::createPluginAPI({.name = "beta", .path = "beta.dll", .version = "1.2.3", .required_permissions = GRIM_PERM_UI}, true);
    check(PluginAPI::commitPluginGeneration(beta, nullptr,
        manifestTools(beta, {definition("blocking")}, dispatchBlocking)), "Beta publication failed");
    check(alpha != beta, "Plugins share an API table");
    checkName(alpha, "alpha");
    checkName(beta, "beta");
    check(alpha->has_permission(alpha, GRIM_PERM_MEMORY), "Alpha lost permissions");
    check(!beta->has_permission(beta, GRIM_PERM_MEMORY), "Beta inherited Alpha permissions");
    check(!alpha->has_permission(alpha, static_cast<GrimPermission>(GRIM_PERM_MEMORY | GRIM_PERM_UI)), "Combined permissions partially accepted");
    check(beta->search_memory(beta, "key") == nullptr, "API used another owner's permissions");
    check(dispatchCommand("alpha_tool", "").success, "Command callback used wrong owner");
    check(logs.back() == "[Plugin: alpha] callback", "Callback logging used wrong owner");
    check(!PluginAPI::commitPluginGeneration(alpha, nullptr,
        manifestTools(alpha, {definition("alpha_later")})) &&
          !GRIM::MMO::ToolRegistry::instance().isRegistered("alpha_later"),
          "Published provider accepted a second manifest");
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
    bool duplicateRejected = false;
    try { PluginAPI::createPluginAPI({.name = "alpha", .path = "duplicate.dll", .version = "1.2.3", .required_permissions = 0}); }
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
        brokenHandler = handler;
        auto* brokenProvider = PluginAPI::createPluginAPI({.name = "broken-fixture", .path = "", .version = "1.2.3", .required_permissions = 0}, true);
        check(PluginAPI::commitPluginGeneration(brokenProvider, nullptr,
            manifestTools(brokenProvider, {definition("broken")}, dispatchBroken)), "Broken fixture publication failed");
        check(dispatchCommand("broken", "").errorCode == "ERR_PLUGIN_RESULT", "Invalid result accepted");
        PluginAPI::cleanupPluginContext(brokenProvider);
    }
    auto* staged = PluginAPI::createPluginAPI({.name = "staged", .path = "staged.dll", .version = "1.2.3", .required_permissions = 0}, true);
    auto complete = definition("staged_tool");
    complete.update({
        {"description", "copied metadata"}, {"argument_format", "json"},
        {"argument_schema", Json::parse(R"({"type":"object","properties":{"path":{"type":"string"}}})")},
        {"aliases", {"staged_alias"}}, {"parameters", {{{"name", "path"}, {"type", "path"},
            {"description", "Destination"}, {"required", true}}}},
        {"usage_type", 2}, {"short_description", "Inspect information supplied by a selected entity"},
        {"usage", "staged_tool {path}"}, {"bindings", {{"path", "t"}}}, {"usage_count", 5},
        {"success_count", 3}, {"failure_count", 2}, {"success_rate", 0.6}
    });
    const auto stagedTools = manifestTools(staged, {complete});
    check(!GRIM::MMO::ToolRegistry::instance().isRegistered("staged_tool"), "Uncommitted tool leaked");
    check(PluginAPI::commitPluginGeneration(staged, nullptr, stagedTools), "Commit failed");
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
    check(snapshot && snapshot->usage_count == 0 && snapshot->success_count == 0 &&
          snapshot->failure_count == 0 && snapshot->success_rate == 1.0f,
          "Plugin registration supplied host execution statistics");
    PluginAPI::cleanupPluginContext("staged");

    auto first = definition("batch_first");
    auto second = definition("batch_second");
    second["usage_type"] = 0;
    auto* batchProvider = PluginAPI::createPluginAPI({.name = "batch-fixture", .path = "", .version = "1.2.3", .required_permissions = 0}, true);
    check(PluginAPI::commitPluginGeneration(batchProvider, nullptr, manifestTools(batchProvider, {first, second})) &&
          ToolRegistry::instance().getTool("batch_first")->usage_type == ToolUsage::Model &&
          ToolRegistry::instance().getTool("batch_second")->usage_type == ToolUsage::Debug,
          "Manifest lost tool metadata");
    PluginAPI::cleanupPluginContext(batchProvider);

    auto* invalidProvider = PluginAPI::createPluginAPI({.name = "invalid-fixture", .path = "", .version = "1.2.3", .required_permissions = GRIM_PERM_UI}, true);
    const auto rejects = [&](const auto& read, const char* reason) {
        try { read(); } catch (const std::exception&) { return; }
        throw std::runtime_error(reason);
    };
    rejects([&] { PluginAPI::readManifestTools(Json::object(), invalidProvider, dispatchEmpty); },
            "Non-array manifest tools accepted");
    rejects([&] { manifestTools(invalidProvider, {first}, nullptr); }, "Missing dispatcher accepted");
    auto invalid = definition("");
    rejects([&] { manifestTools(invalidProvider, {first, invalid}); }, "Empty tool ID accepted");
    invalid = definition("invalid");
    invalid["short_description"] = "Too short";
    rejects([&] { manifestTools(invalidProvider, {invalid}); }, "Invalid summary accepted");
    invalid.erase("short_description");
    invalid["usage"] = "invalid {target}";
    invalid["bindings"] = {{"target", "invalid_type"}};
    rejects([&] { manifestTools(invalidProvider, {invalid}); }, "Invalid binding accepted");
    invalid["bindings"] = {{"missing", "t"}};
    rejects([&] { manifestTools(invalidProvider, {invalid}); }, "Missing binding placeholder accepted");
    invalid.erase("bindings");
    invalid["usage_type"] = 99;
    rejects([&] { manifestTools(invalidProvider, {first, invalid}); }, "Invalid usage type accepted");
    invalid.erase("usage_type");
    invalid["permissions"] = {"memory"};
    rejects([&] { manifestTools(invalidProvider, {invalid}); }, "Undeclared permissions accepted");
    invalid["permissions"] = Json::array();
    invalid["argument_format"] = "json";
    invalid["argument_schema"] = Json::array();
    rejects([&] { manifestTools(invalidProvider, {invalid}); }, "Invalid schema accepted");
    check(!ToolRegistry::instance().isRegistered("batch_first"), "Invalid manifest partially published");
    PluginAPI::cleanupPluginContext(invalidProvider);

    staged = PluginAPI::createPluginAPI({.name = "staged", .path = "staged.dll", .version = "1.2.3", .required_permissions = 0}, true);
    check(!PluginAPI::commitPluginGeneration(staged, nullptr,
          manifestTools(staged, {definition("new_tool"), definition("reserved_fixture_tool")})) &&
          !ToolRegistry::instance().isRegistered("new_tool"), "Failed commit left a partial manifest");
    PluginAPI::cleanupPluginContext(staged);
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
    checkName(beta, "beta");
    const auto before = logs.size();
    beta->emit_event(beta, "test", nullptr, 0);
    check(logs.size() == before, "Owner cleanup retained event callback");
    alpha = PluginAPI::createPluginAPI({.name = "alpha", .path = "new-alpha.dll", .version = "1.2.3", .required_permissions = 0});
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
