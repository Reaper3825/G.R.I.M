// ============================================================================
// G.R.I.M Plugin API Implementation
// ============================================================================
// This file implements the host-side functions that populate the plugin API
// function table. These are the actual implementations that plugins call.
// ============================================================================

#include "plugin_api_impl.hpp"
#include "plugin_notifications.hpp"
#include "logger.hpp"
#include "error_manager.hpp"
#include "ui/ui_panel_service.hpp"
#include "../MMO/Core/Registries/ToolRegistry.hpp"  // Phase 0.25: pluginâ†”ToolRegistry sync
#include "../MMO/Core/Descriptors/ToolDescriptorJson.hpp"
#include "popup_ui/popup_ui.hpp"
#include "helpers/grim_input.hpp"  // For GRIMInput::parseInput
#include <unordered_map>
#include <vector>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <mutex>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace PluginAPI {

// ============================================================================
// Internal Data Structures
// ============================================================================

using GRIM::MMO::ProviderRegistry;
using CallGate = ProviderRegistry::CallGate;
using ActiveCall = CallGate::ActiveCall;

static std::unordered_map<void*, std::string> s_allocated_strings;
static std::atomic<uint64_t> s_next_panel_request = 1;
static std::mutex s_stringMutex;

// ============================================================================
// Helper Functions
// ============================================================================

static auto providerFor(GrimPluginHandle handle) {
    return ProviderRegistry::instance().nativeProvider(handle);
}

static const char* allocateString(const std::string& str) {
    std::lock_guard lock(s_stringMutex);
    auto owned=std::make_unique<char[]>(str.size()+1);
    char* buffer=owned.get();
    std::strcpy(buffer, str.c_str());
    s_allocated_strings[buffer] = str;
    return owned.release();
}

// ============================================================================
// Logging Implementation
// ============================================================================

static void api_log(GrimPluginHandle self, GrimLogLevel level, const char* message) {
    const auto owner = providerFor(self);
    if (!owner || !message) return;
    std::string prefix = "[Plugin: " + owner->identity.name + "] ";
    std::string full_msg = prefix + message;
    
    switch (level) {
        case GRIM_LOG_DEBUG:    LOG_DEBUG("Plugin", full_msg); break;
        case GRIM_LOG_INFO: LOG_DEBUG("Plugin", full_msg); break;  // Map INFO to DEBUG
    case GRIM_LOG_WARNING:  LOG_ERROR("Plugin", full_msg); break;  // Map WARNING to ERROR
        case GRIM_LOG_ERROR:    LOG_ERROR("Plugin", full_msg); break;
        case GRIM_LOG_CRITICAL: LOG_ERROR("Plugin", full_msg); break;  // Map CRITICAL to ERROR
    }
}

// ============================================================================
// Command Registration Implementation
// ============================================================================

// Registration/lifecycle calls are serialized by the host. Tool calls may overlap.
static std::string copyText(const char* text, bool required = false, size_t maxBytes = 65536) {
    if (!text) {
        if (required) throw std::invalid_argument("Missing required text");
        return {};
    }
    size_t size = 0;
    while (size <= maxBytes && text[size]) ++size;
    if (size > maxBytes || (required && size == 0)) throw std::invalid_argument("Invalid text size");
    return {text, size};
}
static GRIM::MMO::ToolDescriptor decodeManifestTool(const nlohmann::json& entry,
                                                     const char* version) {
    using namespace GRIM::MMO;
    ToolDescriptor out;
    readToolDescriptorMetadata(entry, out, version);

    if (out.argument_format == "json") {
        const auto& schema = entry.at("argument_schema");
        if (!schema.is_object()) throw std::invalid_argument("Tool argument schema must be an object");
        out.argument_schema_json = schema.dump();
        if (out.parameters.empty()) {
            const auto properties = schema.value("properties", nlohmann::json::object());
            if (!properties.is_object()) throw std::invalid_argument("Invalid argument schema properties");
            const auto required = schema.value("required", std::unordered_set<std::string>{});
            for (const auto& [name, property] : properties.items()) {
                if (!property.is_object()) throw std::invalid_argument("Invalid argument schema property");
                const auto type = property.value("type", nlohmann::json());
                out.parameters.push_back({name, type.is_string() ? type.get<std::string>() : "any",
                    property.value("description", name), required.count(name) != 0});
            }
        }
    }
    const auto validateText = [](const std::string& value, bool required = true) {
        if ((required && value.empty()) || value.size() > 65536 || value.find('\0') != std::string::npos)
            throw std::invalid_argument("Invalid tool text");
    };
    for (const auto* field : {&out.tool_id, &out.display_name, &out.version, &out.description,
                              &out.usage, &out.category})
        validateText(*field);
    if (out.argument_format != "raw_text" && out.argument_format != "json")
        throw std::invalid_argument("Unsupported argument format");
    validateText(out.argument_schema_json, false);
    if (out.parameters.size() > 256) throw std::invalid_argument("Too many tool parameters");
    std::unordered_set<std::string> parameters;
    for (const auto& parameter : out.parameters) {
        validateText(parameter.name);
        validateText(parameter.type);
        validateText(parameter.description);
        if (!parameters.insert(parameter.name).second) throw std::invalid_argument("Duplicate parameter");
    }
    for (const auto* list : {&out.aliases, &out.keywords, &out.capability_tags, &out.preconditions}) {
        if (list->size() > 256) throw std::invalid_argument("Too many tool metadata entries");
        for (const auto& value : *list) validateText(value);
    }
    return out;
}

static std::vector<GRIM::MMO::ToolDescriptor> readManifestTools(
    const nlohmann::json& entries, const GRIM::MMO::ProviderDescriptor& identity,
    GrimToolDispatch dispatch) {
    if (!entries.is_array() || entries.size() > 256 || (!entries.empty() && !dispatch))
        throw std::invalid_argument("Invalid tool manifest entries or missing dispatcher");
    std::vector<GRIM::MMO::ToolDescriptor> tools;
    tools.reserve(entries.size());
    for (const auto& entry : entries) {
        tools.push_back(decodeManifestTool(entry, identity.version.c_str()));
    }
    return tools;
}

// Provider-neutral bridge: copied descriptions only; implementation and state stay in the DLL.
static GrimResult api_replace_resources(GrimPluginHandle self, const char* kind,
                                       const char* json, size_t size) noexcept {
    try {
        const auto owner = providerFor(self);
        if (!owner) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        ActiveCall active(owner->gate);
        if (!active) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        if (!json || !size || size > 64 * 1024 * 1024) throw std::invalid_argument("Invalid resource payload");
        const auto resourceKind = copyText(kind, true, 256);
        const auto values = nlohmann::json::parse(json, json + size);
        if (!values.is_array() || values.size() > 10000) throw std::invalid_argument("Invalid resource array");
        std::vector<GRIM::MMO::ResourceDescriptor> resources;
        for (const auto& value : values) {
            GRIM::MMO::ResourceDescriptor resource;
            resource.resource_id = value.at("resource_id").get<std::string>();
            resource.kind = value.at("kind").get<std::string>();
            resource.name = value.at("name").get<std::string>();
            resource.parent_id = value.value("parent_id", std::string{});
            if (!value.at("runtime_id").is_number_unsigned()) throw std::invalid_argument("Invalid runtime identity");
            resource.runtime_id = value.at("runtime_id").get<uint64_t>();
            resource.state = value.at("state");
            resources.push_back(std::move(resource));
        }
        return GRIM::MMO::ProviderRegistry::instance().replaceResources(
            owner->identity, resourceKind, resources)
            ? GRIM_OK : GRIM_ERROR_PLUGIN_NOT_LOADED;
    } catch (...) {
        return GRIM_ERROR_INVALID_PARAM;
    }
}

static GrimResult api_unregister_command(GrimPluginHandle self, const char* command_name) {
    if (!command_name || !*command_name) return GRIM_ERROR_INVALID_PARAM;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(command_name);
    if (!tool || tool->tool_id != command_name) return GRIM_ERROR_NOT_FOUND;
    const auto owner = providerFor(self);
    if (!owner) return GRIM_ERROR_PLUGIN_NOT_LOADED;
    if (tool->provider.type != GRIM::MMO::ToolProviderType::Plugin || tool->provider.name != owner->identity.name || tool->provider.instance != owner->identity.instance)
        return GRIM_ERROR_PERMISSION_DENIED;
    return registry.unregisterTool(command_name, owner->identity.name, owner->identity.instance) ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
}

static const char* api_execute_command(GrimPluginHandle self, const char* command_text) noexcept {
    if (!command_text) return nullptr;
    try {
        const auto owner = providerFor(self);
        if (!owner) return nullptr;
        ActiveCall active(owner->gate);
        if (!active) return nullptr;
        auto [cmd, arg] = GRIMInput::parseInput(command_text);
        return allocateString(GRIM::MMO::ToolRegistry::instance().invoke(cmd, arg).dump());
    } catch (...) {
        return nullptr;
    }
}

// ============================================================================
// Notification Implementation
// ============================================================================

static void api_show_notification(GrimPluginHandle self, const char* title, const char* message) {
    if (!providerFor(self)) return;
    
    if (title && message) {
        std::string full_msg = std::string(title) + ": " + message;
        LOG_DEBUG("Plugin", full_msg);
        
        publishPluginNotification(title, message);
    }
}

static const char* api_get_tool_snapshot(GrimPluginHandle self, const char* tool_id) noexcept {
    try {
        if (!providerFor(self)) return nullptr;
        auto& registry = GRIM::MMO::ToolRegistry::instance();
        auto tools = nlohmann::json::array();
        if (tool_id) {
            if (const auto tool = registry.getTool(tool_id)) tools.push_back(*tool);
        } else {
            for (const auto& tool : registry.getRegisteredTools()) tools.push_back(tool);
        }
        return allocateString(nlohmann::json{{"format", "grim-tool-snapshot"},
            {"version", 1}, {"tools", std::move(tools)}}.dump());
    } catch (...) { return nullptr; }
}

static const char* api_get_user_error_message(GrimPluginHandle, const char* code) noexcept {
    if (!code) return nullptr;
    try { return allocateString(ErrorManager::getUserMessage(code)); }
    catch (...) { return nullptr; }
}

// ============================================================================
// System tick subscriptions
// ============================================================================

static GrimResult api_subscribe_system_tick(GrimPluginHandle self,
                                            GrimSystemTickCallback callback, void* userData) noexcept {
    if (!callback) return GRIM_ERROR_INVALID_PARAM;
    try {
        const auto owner = providerFor(self);
        if (!owner) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        ActiveCall active(owner->gate);
        if (!active) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        std::lock_guard lock(owner->gate.mutex);
        if (owner->tick) return GRIM_ERROR_ALREADY_EXISTS;
        owner->tick = callback;
        owner->tickData = userData;
        return GRIM_OK;
    } catch (const std::bad_alloc&) {
        return GRIM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return GRIM_ERROR_INVALID_PARAM;
    }
}

// ============================================================================
// Plugin Lifecycle Implementation
// ============================================================================

static const char* api_get_plugin_name(GrimPluginHandle self) {
    const auto owner = providerFor(self);
    return owner ? allocateString(owner->identity.name) : nullptr;
}

static const char* api_get_plugin_path(GrimPluginHandle self) {
    const auto owner = providerFor(self);
    return owner ? allocateString(owner->identity.path) : nullptr;
}

// ============================================================================
// Utility Implementation
// ============================================================================

static void api_string_free(GrimPluginHandle self, const char* str) {
    if (!str) return;
    
    std::lock_guard lock(s_stringMutex);
    auto it = s_allocated_strings.find(const_cast<void*>(static_cast<const void*>(str)));
    if (it != s_allocated_strings.end()) {
        delete[] str;
        s_allocated_strings.erase(it);
    }
}

// ============================================================================
// API Table Construction
// ============================================================================

static const char* api_ui_panel_service(GrimPluginHandle self, GrimUIPanelOperation operation,
                                      const char* panel_key, uint64_t request_id) noexcept {
    try {
        const auto owner = providerFor(self);
        if (!owner) return nullptr;
        ActiveCall active(owner->gate, ActiveCall::Kind::Dispatch);
        if (!active) return nullptr;
        if (operation > GRIM_UI_HIDE_CURRENT) return nullptr;
        if (operation == GRIM_UI_POLL || operation == GRIM_UI_CANCEL) {
            if (panel_key || !request_id) return nullptr;
        } else if (operation == GRIM_UI_SHOW_CURRENT || operation == GRIM_UI_HIDE_CURRENT) {
            if (!request_id || !panel_key || !*panel_key) return nullptr;
        } else if (request_id || (operation == GRIM_UI_LIST ? panel_key != nullptr : !panel_key || !*panel_key)) {
            return nullptr;
        }
        const auto key = copyText(panel_key, false, 256);
        std::lock_guard lock(owner->panelRequestsMutex);
        nlohmann::json response = {{"format", "grim-ui-panel-response"}, {"version", 1},
                                  {"request_id", request_id}, {"status", "pending"}};
        if (operation == GRIM_UI_POLL || operation == GRIM_UI_CANCEL) {
            // Resolve requests only within the calling plugin instance.
            const auto found = owner->panelRequests.find(request_id);
            if (found == owner->panelRequests.end()) {
                response["status"] = "error"; response["error"] = "request_not_found";
                return allocateString(response.dump());
            }
            if (operation == GRIM_UI_CANCEL) {
                found->second.cancelled->store(true);
                owner->panelRequests.erase(found);
                response["status"] = "completed";
                response["result"] = {{"success", true}, {"released", true}};
                return allocateString(response.dump());
            }
            if (found->second.result.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                return allocateString(response.dump());
            response["status"] = "completed";
            try { response["result"] = found->second.result.get(); }
            catch (const std::exception& error) {
                response["result"] = {{"success", false}, {"error", "host_exception"}, {"message", error.what()}};
            } catch (...) { response["result"] = {{"success", false}, {"error", "host_exception"}}; }
            owner->panelRequests.erase(found);
            return allocateString(response.dump());
        }
        if (owner->panelRequests.size() >= 64) {
            response["status"] = "error"; response["error"] = "too_many_requests";
            return allocateString(response.dump());
        }
        const auto id = s_next_panel_request.fetch_add(1);
        response["request_id"] = id;
        const bool guarded = operation == GRIM_UI_SHOW_CURRENT || operation == GRIM_UI_HIDE_CURRENT;
        auto request = GRIM::UI::submitPanelRequest(
            operation == GRIM_UI_SHOW_CURRENT ? GRIM_UI_SHOW :
                (operation == GRIM_UI_HIDE_CURRENT ? GRIM_UI_HIDE : operation), key,
            guarded ? request_id : 0);
        owner->panelRequests.emplace(id, std::move(request));
        return allocateString(response.dump());
    } catch (...) { return nullptr; }
}


const GrimPluginAPI& hostAPI() {
    static const GrimPluginAPI table = [] {
        GrimPluginAPI api{};
        // Fill in API table
        api.api_version = GRIM_API_VERSION;

        // Logging
        api.log = api_log;

        // Commands
        api.unregister_command = api_unregister_command;
        api.execute_command = api_execute_command;

        // Notifications
        api.show_notification = api_show_notification;

        // Plugin Lifecycle
        api.get_plugin_name = api_get_plugin_name;
        api.get_plugin_path = api_get_plugin_path;

        // Utilities
        api.string_free = api_string_free;
        api.get_user_error_message = api_get_user_error_message;
        api.get_tool_snapshot = api_get_tool_snapshot;
        api.ui_panel_service = api_ui_panel_service;
        api.replace_resources = api_replace_resources;
        api.subscribe_system_tick = api_subscribe_system_tick;

        return api;
    }();
    return table;
}

ProviderBatch::Record ProviderBatch::stage(PluginModule plugin) {
    auto& registry = Registry::instance();
    auto record = std::make_shared<Registry::ProviderRecord>();
    GrimPluginHandle handle;
    {
        std::lock_guard lock(registry.tools_.mutex_);
        handle = registry.nextHandle_++;
    }
    plugin.descriptor.instance = plugin.descriptor.name + "#" + std::to_string(handle);
    record->descriptor = std::make_shared<const GRIM::MMO::ProviderDescriptor>(std::move(plugin.descriptor));
    auto& state = record->native.emplace(*record->descriptor);
    state.handle = handle;
    state.module = std::move(plugin.library);
    state.dispatch = plugin.dispatch;
    state.panelLease->owner = record->descriptor->instance;
    Registry::prepareTools(*record, readManifestTools(plugin.tools, *record->descriptor, state.dispatch));
    {
        std::lock_guard lock(registry.tools_.mutex_);
        registry.validateNativeCandidateLocked(*record);
        registry.records_.emplace(record->descriptor->name, record);
    }
    return record;
}

void ProviderBatch::initialize(const Record& record, decltype(&grim_plugin_init) init,
                               decltype(&grim_plugin_shutdown) shutdown) {
    auto& state = *record->native;
    GRIM::UI::registerProviderPanelLease(state.handle, state.panelLease);
    state.shutdown = shutdown;
    const auto result = init ? init(&hostAPI(), state.handle) : GRIM_OK;
    if (result != GRIM_OK)
        throw std::runtime_error("Plugin initialization failed with result " + std::to_string(result));
}

std::vector<PluginLoadOutcome> ProviderBatch::load(const std::vector<std::filesystem::path>& libraries,
                                                  OpenPlugin open) {
    auto& registry = Registry::instance();
    std::vector<PluginLoadOutcome> outcomes;
    std::vector<Record> prepared;
    std::vector<std::size_t> preparedOutcomes;
    outcomes.reserve(libraries.size());
    prepared.reserve(libraries.size());
    preparedOutcomes.reserve(libraries.size());

    try {
        for (const auto& library : libraries) {
            PluginLoadOutcome outcome{library, PluginLoadStatus::InvalidBundle, {}};
            Record record;
            try {
                auto plugin = open(library, outcome.status);
                const auto init = plugin.init;
                const auto shutdown = plugin.shutdown;
                outcome.status = PluginLoadStatus::RegistrationRejected;
                record = stage(std::move(plugin));
                outcome.status = PluginLoadStatus::InitializationFailed;
                initialize(record, init, shutdown);
                preparedOutcomes.push_back(outcomes.size());
                prepared.push_back(std::move(record));
            } catch (const std::exception& error) {
                if (record) stop({record});
                outcome.detail = error.what();
            } catch (...) {
                if (record) stop({record});
                outcome.detail = "Plugin raised an unknown exception";
            }
            outcomes.push_back(std::move(outcome));
        }

        registry.publishPrepared(prepared);
        for (std::size_t i = 0; i < prepared.size(); ++i) {
            auto& outcome = outcomes[preparedOutcomes[i]];
            if (prepared[i]->published) {
                outcome.status = PluginLoadStatus::Loaded;
            } else {
                stop({prepared[i]});
                outcome.status = PluginLoadStatus::RegistrationRejected;
                outcome.detail = "Duplicate tool ID or alias";
            }
        }
    } catch (...) {
        stop(prepared);
        throw;
    }
    return outcomes;
}

void ProviderBatch::stop(const std::vector<Record>& records) {
    auto& registry = Registry::instance();
    for (const auto& record : records) {
        auto& gate = record->native->gate;
        std::lock_guard lock(gate.mutex);
        gate.state = CallGate::State::Stopping;
    }
    registry.unpublishPrepared(records);

    for (auto it = records.rbegin(); it != records.rend(); ++it) {
        const auto& record = *it;
        auto& state = *record->native;
        {
            std::unique_lock lock(state.gate.mutex);
            state.gate.idle.wait(lock, [&] { return state.gate.active == 0; });
        }
        GRIM::UI::retireProviderPanels(state.panelLease);
        {
            std::lock_guard lock(state.panelRequestsMutex);
            for (auto& [id, request] : state.panelRequests) request.cancelled->store(true);
            state.panelRequests.clear();
        }
        if (const auto shutdown = std::exchange(state.shutdown, nullptr)) {
            try { shutdown(); }
            catch (...) { LOG_ERROR("PluginAPI", "Plugin shutdown raised an exception: " + record->descriptor->name); }
        }
        {
            std::lock_guard lock(registry.tools_.mutex_);
            const auto found = registry.records_.find(record->descriptor->name);
            if (found != registry.records_.end() && found->second == record)
                registry.records_.erase(found);
        }
        state.module.reset();
    }
}

std::size_t ProviderBatch::shutdown() {
    auto& registry = Registry::instance();
    std::vector<Record> records;
    {
        std::lock_guard lock(registry.tools_.mutex_);
        for (const auto& [name, record] : registry.records_) {
            if (record->native && record->native->module) records.push_back(record);
        }
    }
    std::sort(records.begin(), records.end(), [](const Record& a, const Record& b) {
        return a->native->handle < b->native->handle;
    });
    stop(records);
    return records.size();
}

#ifdef GRIM_PLUGIN_BATCH_TESTS
GrimPluginHandle ProviderBatch::loadFixture(const GRIM::MMO::ProviderDescriptor& descriptor,
                                           const nlohmann::json& tools, GrimToolDispatch dispatch) {
    auto record = stage(PluginModule{.descriptor = descriptor, .tools = tools, .dispatch = dispatch});
    try {
        initialize(record, nullptr, nullptr);
        Registry::instance().publishPrepared({record});
        if (!record->published) throw std::runtime_error("Fixture registration rejected");
        return record->native->handle;
    } catch (...) {
        stop({record});
        throw;
    }
}

void ProviderBatch::stopFixture(GrimPluginHandle handle) {
    auto& registry = Registry::instance();
    Record fixture;
    {
        std::lock_guard lock(registry.tools_.mutex_);
        for (const auto& [name, record] : registry.records_) {
            if (record->native && !record->native->module && record->native->handle == handle) {
                fixture = record;
                break;
            }
        }
    }
    if (fixture) stop({fixture});
}
#endif

void dispatchSystemTick(double deltaSeconds, double offsetSeconds) {
    for (const auto& owner : ProviderRegistry::instance().readyNativeProviders()) {
        auto& state = *owner;
        ActiveCall active(state.gate, ActiveCall::Kind::Dispatch);
        if (!active) continue;
        GrimSystemTickCallback callback;
        void* data;
        {
            std::lock_guard lock(state.gate.mutex);
            callback = state.tick;
            data = state.tickData;
        }
        if (!callback) continue;
        try { callback(deltaSeconds, offsetSeconds, data); }
        catch (...) { LOG_ERROR("PluginAPI", "System tick callback raised an exception"); }
    }
}

} // namespace PluginAPI
