// ============================================================================
// G.R.I.M Plugin API Implementation
// ============================================================================
// This file implements the host-side functions that populate the plugin API
// function table. These are the actual implementations that plugins call.
// ============================================================================

#include "core/plugin_tool_result.hpp"
#include "plugin_api_impl.hpp"
#include "plugin_manager.hpp"
#include "plugin_manifest.hpp"
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
#include <condition_variable>
#include <mutex>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace PluginAPI {

// ============================================================================
// Internal Data Structures
// ============================================================================

struct CallGate;
struct EventSubscription {
    std::string event_name;
    GrimEventCallback callback;
    void* user_data;
    std::string plugin_name;
    std::shared_ptr<CallGate> gate;
};

struct CallGate {
    std::mutex mutex;
    std::condition_variable idle;
    bool accepting = true;
    bool dispatchEnabled = false;
    size_t active = 0;
};

struct SystemTickSubscription {
    GrimSystemTickCallback callback;
    void* userData;
    std::shared_ptr<CallGate> gate;
};

struct LoadedProvider {
    explicit LoadedProvider(GRIM::MMO::ProviderDescriptor descriptor)
        : identity(std::move(descriptor)) {}

    const GRIM::MMO::ProviderDescriptor identity;
    GrimPluginHandle handle = 0;
    std::shared_ptr<CallGate> gate = std::make_shared<CallGate>();
    std::shared_ptr<GRIM::UI::ProviderPanelLease> panelLease =
        std::make_shared<GRIM::UI::ProviderPanelLease>();
    std::mutex panelRequestsMutex;
    std::unordered_map<uint64_t, GRIM::UI::PanelServiceRequest> panelRequests;
};

// Static storage
static std::unordered_map<uint64_t, EventSubscription> s_event_subscriptions;
static uint64_t s_next_event_id = 1;
static std::unordered_map<std::string, SystemTickSubscription> s_systemTickSubscriptions;
static std::unordered_map<void*, std::string> s_allocated_strings;
static std::unordered_map<GrimPluginHandle, std::weak_ptr<LoadedProvider>> s_instances;
static std::mutex s_instanceMutex;
static uint64_t s_next_instance = 1;
static std::atomic<uint64_t> s_next_panel_request = 1;
static std::mutex s_resourceMutex;
static std::mutex s_stringMutex;

// ============================================================================
// Helper Functions
// ============================================================================

static std::shared_ptr<LoadedProvider> providerFor(GrimPluginHandle handle) {
    std::lock_guard lock(s_instanceMutex);
    const auto found = s_instances.find(handle);
    return found == s_instances.end() ? nullptr : found->second.lock();
}

static bool checkPermission(GrimPluginHandle self, GrimPermission perm) {
    const auto owner = providerFor(self);
    return owner && (owner->identity.required_permissions & perm) == static_cast<uint32_t>(perm);
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
struct ResultCapture {
    CommandResult value;
    bool completed = false, invalid = false;
};
static GrimResult completeResult(GrimToolResultWriter* writer, const GrimToolResult* input) noexcept {
    auto& capture = *static_cast<ResultCapture*>(writer->host_context);
    try {
        if (!input || input->struct_size != sizeof(GrimToolResult) || input->success > 1 || capture.completed)
            throw std::invalid_argument("Invalid or repeated result");
        CommandResult result;
        result.success = input->success != 0;
        // Catalog listings can legitimately exceed an individual metadata field.
        result.message = copyText(input->message, false, 16 * 1024 * 1024);
        result.errorCode = copyText(input->error_code);
        result.category = copyText(input->category);
        // Existing perception commands use the complete observation as voice.
        result.voice = copyText(input->voice, false, 16 * 1024 * 1024);
        result.structuredOutputJson = copyText(input->structured_output_json);
        if (!result.structuredOutputJson.empty()) {
            const auto parsed = nlohmann::json::parse(result.structuredOutputJson);
            (void)parsed;
        }
        const auto rgb = input->color_rgb;
        result.color = Color{static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8), static_cast<uint8_t>(rgb)};
        capture.value = std::move(result);
        capture.completed = true;
        return GRIM_OK;
    } catch (...) {
        capture.invalid = true;
        return GRIM_ERROR_INVALID_PARAM;
    }
}
struct ActiveCall {
    std::shared_ptr<CallGate> gate;
    bool entered = false;
    explicit ActiveCall(std::shared_ptr<CallGate> value, bool requireDispatch = false) : gate(std::move(value)) {
        std::lock_guard lock(gate->mutex);
        if (gate->accepting && (!requireDispatch || gate->dispatchEnabled)) { ++gate->active; entered = true; }
    }
    ~ActiveCall() {
        if (!entered) return;
        std::lock_guard lock(gate->mutex);
        if (--gate->active == 0) gate->idle.notify_all();
    }
};
static GRIM::MMO::ToolDescriptor decodeManifestTool(const nlohmann::json& entry,
                                                     const char* version, uint32_t providerPermissions) {
    using namespace GRIM::MMO;
    ToolDescriptor out;
    out.permission_bits = readManifestPermissions(entry.at("permissions"));
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
    if ((out.permission_bits & providerPermissions) != out.permission_bits)
        throw std::invalid_argument("Tool requires undeclared permissions");

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

static std::vector<GRIM::MMO::ToolRegistration> readManifestTools(
    const nlohmann::json& entries, const LoadedProvider& owner, GrimToolDispatch dispatch) {
    if (!entries.is_array() || entries.size() > 256 || (!entries.empty() && !dispatch))
        throw std::invalid_argument("Invalid tool manifest entries or missing dispatcher");
    std::vector<GRIM::MMO::ToolRegistration> tools;
    tools.reserve(entries.size());
    for (const auto& entry : entries) {
        auto descriptor = decodeManifestTool(entry, owner.identity.version.c_str(), owner.identity.required_permissions);
        auto invoke = [gate = owner.gate, dispatch, id = descriptor.tool_id](const std::string& arguments) -> CommandResult {
            ActiveCall active(gate, true);
            if (!active.entered) return {false, "Plugin is stopping", "ERR_TOOL_UNAVAILABLE"};
            ResultCapture capture;
            GrimToolResultWriter writer{sizeof(GrimToolResultWriter), &capture, completeResult};
            GrimToolCall call{sizeof(GrimToolCall), arguments.c_str(), arguments.size()};
            const auto delivered = dispatch(id.c_str(), &call, &writer);
            if (delivered != GRIM_OK || !capture.completed || capture.invalid)
                return {false, "Plugin did not deliver a valid result", "ERR_PLUGIN_RESULT"};
            return std::move(capture.value);
        };
        tools.push_back({std::move(descriptor), std::move(invoke)});
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
        if (!active.entered) return GRIM_ERROR_PLUGIN_NOT_LOADED;
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

static GrimResult api_execute_command(GrimPluginHandle self, const char* command_text, GrimToolResultWriter* writer) {
    if (!command_text || !writer || writer->struct_size != sizeof(*writer) || !writer->complete)
        return GRIM_ERROR_INVALID_PARAM;
    const auto owner = providerFor(self);
    if (!owner) return GRIM_ERROR_PLUGIN_NOT_LOADED;
    ActiveCall active(owner->gate);
    if (!active.entered) return GRIM_ERROR_PLUGIN_NOT_LOADED;
    auto [cmd, arg] = GRIMInput::parseInput(command_text);
    const auto value = GRIM::MMO::ToolRegistry::instance().invoke(cmd, arg);
    return grim::completeToolResult(writer, value);
}

// ============================================================================
// Event System Implementation
// ============================================================================

static GrimEventHandle api_subscribe_event(GrimPluginHandle self, const char* event_name, GrimEventCallback callback, void* user_data) {
    if (!event_name || !callback) {
        return nullptr;
    }
    
    const auto owner = providerFor(self);
    if (!owner) return nullptr;
    ActiveCall active(owner->gate);
    if (!active.entered) return nullptr;
    std::lock_guard lock(s_resourceMutex);
    uint64_t handle = s_next_event_id++;
    
    EventSubscription sub;
    sub.event_name = event_name;
    sub.callback = callback;
    sub.user_data = user_data;
    sub.plugin_name = owner->identity.instance;
    sub.gate = owner->gate;
    
    s_event_subscriptions[handle] = sub;
    
    LOG_DEBUG("PluginAPI", "Event subscription: " + std::string(event_name));
    return reinterpret_cast<GrimEventHandle>(handle);
}

static GrimResult api_unsubscribe_event(GrimPluginHandle self, GrimEventHandle handle) {
    if (!handle) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    uint64_t id = reinterpret_cast<uint64_t>(handle);
    std::lock_guard lock(s_resourceMutex);
    auto it = s_event_subscriptions.find(id);
    
    if (it == s_event_subscriptions.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    const auto owner = providerFor(self);
    if (!owner) return GRIM_ERROR_PLUGIN_NOT_LOADED;
    if (it->second.plugin_name != owner->identity.instance) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    s_event_subscriptions.erase(it);
    return GRIM_OK;
}

static void dispatchEvent(const char* event_name, const void* data, size_t size) {
    std::vector<EventSubscription> callbacks;
    {
        std::lock_guard lock(s_resourceMutex);
        for (const auto& [id, sub] : s_event_subscriptions)
            if (sub.event_name == event_name) callbacks.push_back(sub);
    }
    for (const auto& sub : callbacks) {
        ActiveCall active(sub.gate, true);
        if (active.entered) sub.callback(event_name, data, size, sub.user_data);
    }
}
static GrimResult api_emit_event(GrimPluginHandle, const char* event_name, const void* data, size_t data_size) {
    if (!event_name) return GRIM_ERROR_INVALID_PARAM;
    dispatchEvent(event_name, data, data_size);
    return GRIM_OK;
}

// ============================================================================
// Notification Implementation
// ============================================================================

static void api_show_notification(GrimPluginHandle self, const char* title, const char* message) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
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
        if (!active.entered) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        std::lock_guard lock(s_resourceMutex);
        const auto inserted = s_systemTickSubscriptions.try_emplace(
            owner->identity.instance, SystemTickSubscription{callback, userData, owner->gate}).second;
        return inserted ? GRIM_OK : GRIM_ERROR_ALREADY_EXISTS;
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

static uint32_t api_get_permissions(GrimPluginHandle self) {
    const auto owner = providerFor(self);
    return owner ? owner->identity.required_permissions : 0;
}

static bool api_has_permission(GrimPluginHandle self, GrimPermission perm) {
    return checkPermission(self, perm);
}

static void api_request_reload(GrimPluginHandle self) {
    const auto owner = providerFor(self);
    if (!owner) return;
    ActiveCall active(owner->gate);
    if (!active.entered) return;
    LOG_DEBUG("PluginAPI", "Plugin requested reload: " + owner->identity.name);
    PluginManager::requestReload(self);
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
        ActiveCall active(owner->gate, true);
        if (!active.entered || !checkPermission(self, GRIM_PERM_UI)) return nullptr;
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

        // Events
        api.subscribe_event = api_subscribe_event;
        api.unsubscribe_event = api_unsubscribe_event;
        api.emit_event = api_emit_event;

        // Notifications
        api.show_notification = api_show_notification;

        // Plugin Lifecycle
        api.get_plugin_name = api_get_plugin_name;
        api.get_plugin_path = api_get_plugin_path;
        api.get_permissions = api_get_permissions;
        api.has_permission = api_has_permission;
        api.request_reload = api_request_reload;

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

std::shared_ptr<LoadedProvider> registerInstance(GRIM::MMO::ProviderDescriptor provider) {
    std::lock_guard lock(s_instanceMutex);
    const auto handle = s_next_instance++;
    provider.instance = provider.name + "#" + std::to_string(handle);
    auto owner = std::make_shared<LoadedProvider>(std::move(provider));
    owner->handle = handle;
    owner->panelLease->owner = owner->identity.instance;
    owner->panelLease->uiPermitted = (owner->identity.required_permissions & GRIM_PERM_UI) != 0;
    s_instances.emplace(handle, owner);
    return owner;
}

GrimPluginHandle instanceHandle(const LoadedProvider& provider) { return provider.handle; }

void validateManifestTools(const nlohmann::json& entries, const GRIM::MMO::ProviderDescriptor& provider) {
    if (!entries.is_array() || entries.size() > 256)
        throw std::invalid_argument("Invalid tool manifest entries");
    for (const auto& entry : entries)
        (void)decodeManifestTool(entry, provider.version.c_str(), provider.required_permissions);
}

void registerProvider(LoadedProvider& provider, const nlohmann::json& entries, GrimToolDispatch dispatch) {
    const auto tools = readManifestTools(entries, provider, dispatch);
    if (!GRIM::MMO::ProviderRegistry::instance().publish(provider.identity, tools))
        throw std::runtime_error("Provider registration rejected");
    GRIM::UI::registerProviderPanelLease(provider.handle, provider.panelLease);
}

void enableDispatch(LoadedProvider& provider) {
    std::lock_guard lock(provider.gate->mutex);
    provider.gate->dispatchEnabled = true;
}

void retireProvider(LoadedProvider& provider) {
    {
        std::lock_guard lock(provider.gate->mutex);
        provider.gate->accepting = false;
    }
    GRIM::MMO::ProviderRegistry::instance().retire(provider.identity.name, provider.identity.instance);
    GRIM::UI::retireProviderPanels(provider.panelLease);
    {
        std::unique_lock lock(provider.gate->mutex);
        provider.gate->idle.wait(lock, [&] { return provider.gate->active == 0; });
    }
    {
        std::lock_guard lock(provider.panelRequestsMutex);
        for (auto& [id, request] : provider.panelRequests) request.cancelled->store(true);
        provider.panelRequests.clear();
    }
    std::lock_guard lock(s_resourceMutex);
    const auto& instance = provider.identity.instance;
    s_systemTickSubscriptions.erase(instance);
    std::erase_if(s_event_subscriptions, [&](const auto& entry) { return entry.second.plugin_name == instance; });
}

void releaseInstance(LoadedProvider& provider) {
    std::lock_guard lock(s_instanceMutex);
    s_instances.erase(provider.handle);
}

void dispatchSystemTick(double deltaSeconds, double offsetSeconds) {
    std::vector<SystemTickSubscription> subscribers;
    {
        std::lock_guard lock(s_resourceMutex);
        subscribers.reserve(s_systemTickSubscriptions.size());
        for (const auto& entry : s_systemTickSubscriptions) subscribers.push_back(entry.second);
    }
    for (const auto& subscriber : subscribers) {
        ActiveCall active(subscriber.gate, true);
        if (!active.entered) continue;
        try {
            subscriber.callback(deltaSeconds, offsetSeconds, subscriber.userData);
        } catch (...) {
            LOG_ERROR("PluginAPI", "System tick callback raised an exception");
        }
    }
}

void emitGlobalEvent(const char* event_name, const void* data, size_t data_size) {
    if (event_name) dispatchEvent(event_name, data, data_size);
}

} // namespace PluginAPI
