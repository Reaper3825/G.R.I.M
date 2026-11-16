// ============================================================================
// G.R.I.M Plugin API Implementation
// ============================================================================
// This file implements the host-side functions that populate the plugin API
// function table. These are the actual implementations that plugins call.
// ============================================================================

#include "plugin_api_impl.hpp"
#include "plugin_manager.hpp"
#include "plugin_notifications.hpp"
#include "logger.hpp"
#include "commands/commands_core.hpp"
#include "error_manager.hpp"
#include "memory/command_memory_services.hpp"
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
#include <fstream>
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
    bool published = true;
    size_t active = 0;
};

struct SystemTickSubscription {
    GrimSystemTickCallback callback;
    void* userData;
    std::shared_ptr<CallGate> gate;
};

struct PluginContext {
    explicit PluginContext(std::shared_ptr<const GRIM::MMO::ProviderDescriptor> descriptor)
        : identity(std::move(descriptor)) {}

    const std::shared_ptr<const GRIM::MMO::ProviderDescriptor> identity;
    uint32_t permissions;
    GrimPluginAPI api{};
    std::shared_ptr<CallGate> gate = std::make_shared<CallGate>();
    bool staging = false;
    bool resource_staging_failed = false;
    std::shared_ptr<GRIM::UI::ProviderPanelLease> panelLease =
        std::make_shared<GRIM::UI::ProviderPanelLease>();
    std::mutex panelRequestsMutex;
    std::unordered_map<uint64_t, GRIM::UI::PanelServiceRequest> panelRequests;
};

// Static storage
static std::unordered_map<uint64_t, EventSubscription> s_event_subscriptions;
static uint64_t s_next_event_id = 1;
static std::unordered_map<std::string, SystemTickSubscription> s_systemTickSubscriptions;
static std::unordered_map<std::string, std::string> s_context_storage;
static std::unordered_map<void*, std::string> s_allocated_strings;
static std::unordered_map<std::string, std::unique_ptr<PluginContext>> s_plugin_contexts;
static uint64_t s_next_instance = 1;
static std::atomic<uint64_t> s_next_panel_request = 1;
static std::mutex s_resourceMutex;
static std::mutex s_stringMutex;

// ============================================================================
// Helper Functions
// ============================================================================

static const PluginContext& pluginContext(const GrimPluginAPI* self) {
    if (!self || !self->host_context) {
        throw std::invalid_argument("Plugin API requires its owning API object");
    }
    return *static_cast<const PluginContext*>(self->host_context);
}

static bool checkPermission(const GrimPluginAPI* self, GrimPermission perm) {
    return (pluginContext(self).permissions & perm) == static_cast<uint32_t>(perm);
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

static void api_log(const GrimPluginAPI* self, GrimLogLevel level, const char* message) {
    std::string prefix = "[Plugin: " + pluginContext(self).identity->name + "] ";
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
    explicit ActiveCall(std::shared_ptr<CallGate> value, bool requirePublished = false) : gate(std::move(value)) {
        std::lock_guard lock(gate->mutex);
        if (gate->accepting && (!requirePublished || gate->published)) { ++gate->active; entered = true; }
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
    static const std::unordered_map<std::string, uint32_t> permissionFlags{
        {"filesystem", GRIM_PERM_FILESYSTEM}, {"process", GRIM_PERM_PROCESS},
        {"network", GRIM_PERM_NETWORK}, {"memory", GRIM_PERM_MEMORY}, {"voice", GRIM_PERM_VOICE},
        {"ui", GRIM_PERM_UI}, {"system", GRIM_PERM_SYSTEM}, {"input", GRIM_PERM_INPUT}
    };
    ToolDescriptor out;
    for (const auto& name : entry.at("permissions").get<std::vector<std::string>>())
        out.permission_bits |= permissionFlags.at(name);
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

std::vector<GRIM::MMO::ToolRegistration> readToolManifest(
    const std::filesystem::path& path, const GrimPluginAPI* api, const char* version, GrimToolDispatch dispatch) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open tool manifest: " + path.string());
    const auto manifest = nlohmann::json::parse(input);
    const auto& owner = pluginContext(api);
    const auto& schemaVersion = manifest.at("schema_version");
    if (!schemaVersion.is_number_integer() || schemaVersion != 1 ||
        manifest.at("provider") != owner.identity->name || manifest.at("version") != version)
        throw std::invalid_argument("Tool manifest version or provider identity mismatch");
    const auto& entries = manifest.at("tools");
    if (!entries.is_array() || entries.size() > 256 || (!entries.empty() && !dispatch))
        throw std::invalid_argument("Invalid tool manifest entries or missing dispatcher");
    std::vector<GRIM::MMO::ToolRegistration> tools;
    tools.reserve(entries.size());
    for (const auto& entry : entries) {
        auto descriptor = decodeManifestTool(entry, version, owner.permissions);
        auto invoke = [gate = owner.gate, dispatch, id = descriptor.tool_id](const std::string& arguments) -> CommandResult {
            ActiveCall active(gate);
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
static const char* api_get_provider_identity(const GrimPluginAPI* self) noexcept {
    try {
        const auto& identity = *pluginContext(self).identity;
        return allocateString(nlohmann::json{{"name", identity.name}, {"instance", identity.instance},
                                            {"path", identity.path}}.dump());
    } catch (...) { return nullptr; }
}
static bool api_is_provider_current(const GrimPluginAPI* self) noexcept {
    try {
        const auto& owner = pluginContext(self);
        ActiveCall active(owner.gate, true);
        if (!active.entered) return false;
        const auto snapshot = GRIM::MMO::ProviderRegistry::instance().snapshot();
        return std::any_of(snapshot.providers.begin(), snapshot.providers.end(), [&](const auto& group) {
            return group.provider.name == owner.identity->name && group.provider.instance == owner.identity->instance;
        });
    } catch (...) { return false; }
}
static GrimResult api_replace_resources(const GrimPluginAPI* self, const char* kind,
                                       const char* json, size_t size) noexcept {
    PluginContext* context = nullptr;
    try {
        auto& owner = const_cast<PluginContext&>(pluginContext(self));
        context = &owner;
        ActiveCall active(owner.gate);
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
            *owner.identity, resourceKind, resources)
            ? GRIM_OK : GRIM_ERROR_PLUGIN_NOT_LOADED;
    } catch (...) {
        if (context && context->staging) context->resource_staging_failed = true;
        return GRIM_ERROR_INVALID_PARAM;
    }
}

static GrimResult api_unregister_command(const GrimPluginAPI* self, const char* command_name) {
    if (!command_name || !*command_name) return GRIM_ERROR_INVALID_PARAM;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(command_name);
    if (!tool || tool->tool_id != command_name) return GRIM_ERROR_NOT_FOUND;
    const auto& owner = pluginContext(self);
    if (tool->provider.type != GRIM::MMO::ToolProviderType::Plugin || tool->provider.name != owner.identity->name || tool->provider.instance != owner.identity->instance)
        return GRIM_ERROR_PERMISSION_DENIED;
    return registry.unregisterTool(command_name, owner.identity->name, owner.identity->instance) ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
}

static GrimResult api_execute_command(const GrimPluginAPI* self, const char* command_text, GrimToolResultWriter* writer) {
    if (!command_text || !writer || writer->struct_size != sizeof(*writer) || !writer->complete)
        return GRIM_ERROR_INVALID_PARAM;
    auto [cmd, arg] = GRIMInput::parseInput(command_text);
    const auto value = dispatchCommand(cmd, arg);
    GrimToolResult result;
    result.success = value.success;
    result.message = value.message.c_str();
    result.error_code = value.errorCode.c_str();
    result.category = value.category.c_str();
    result.voice = value.voice.c_str();
    result.structured_output_json = value.structuredOutputJson.c_str();
    result.color_rgb = (value.color.r << 16) | (value.color.g << 8) | value.color.b;
    return writer->complete(writer, &result);
}

// ============================================================================
// Event System Implementation
// ============================================================================

static GrimEventHandle api_subscribe_event(const GrimPluginAPI* self, const char* event_name, GrimEventCallback callback, void* user_data) {
    if (!event_name || !callback) {
        return nullptr;
    }
    
    std::lock_guard lock(s_resourceMutex);
    uint64_t handle = s_next_event_id++;
    
    EventSubscription sub;
    sub.event_name = event_name;
    sub.callback = callback;
    sub.user_data = user_data;
    sub.plugin_name = pluginContext(self).identity->instance;
    sub.gate = pluginContext(self).gate;
    
    s_event_subscriptions[handle] = sub;
    
    LOG_DEBUG("PluginAPI", "Event subscription: " + std::string(event_name));
    return reinterpret_cast<GrimEventHandle>(handle);
}

static GrimResult api_unsubscribe_event(const GrimPluginAPI* self, GrimEventHandle handle) {
    if (!handle) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    uint64_t id = reinterpret_cast<uint64_t>(handle);
    std::lock_guard lock(s_resourceMutex);
    auto it = s_event_subscriptions.find(id);
    
    if (it == s_event_subscriptions.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    if (it->second.plugin_name != pluginContext(self).identity->instance) {
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
static GrimResult api_emit_event(const GrimPluginAPI*, const char* event_name, const void* data, size_t data_size) {
    if (!event_name) return GRIM_ERROR_INVALID_PARAM;
    dispatchEvent(event_name, data, data_size);
    return GRIM_OK;
}

// ============================================================================
// Context Implementation
// ============================================================================

static const char* api_get_context(const GrimPluginAPI* self, const char* key) {
    if (!key) return nullptr;
    
    auto it = s_context_storage.find(key);
    if (it == s_context_storage.end()) {
        return nullptr;
    }
    
    return allocateString(it->second);
}

static GrimResult api_set_context(const GrimPluginAPI* self, const char* key, const char* value) {
    if (!key || !value) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    s_context_storage[key] = value;
    return GRIM_OK;
}

static GrimResult api_delete_context(const GrimPluginAPI* self, const char* key) {
    if (!key) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    auto it = s_context_storage.find(key);
    if (it == s_context_storage.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    s_context_storage.erase(it);
    return GRIM_OK;
}

static bool api_has_context(const GrimPluginAPI* self, const char* key) {
    if (!key) return false;
    return s_context_storage.find(key) != s_context_storage.end();
}

// ============================================================================
// Notification Implementation
// ============================================================================

static void api_show_notification(const GrimPluginAPI* self, const char* title, const char* message) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (title && message) {
        std::string full_msg = std::string(title) + ": " + message;
        LOG_DEBUG("Plugin", full_msg);
        
        publishPluginNotification(title, message);
    }
}

template<typename Callback>
static const char* runtimeServiceResponse(const GrimPluginAPI* self, GrimPermission permission, Callback callback) noexcept {
    try {
        if (!checkPermission(self, permission)) return nullptr;
        nlohmann::json response{{"format", "grim-runtime-service"}, {"version", 1}, {"value", nullptr}, {"error", nullptr}};
        try { response["value"] = callback(); }
        catch (const std::exception& error) { response["error"] = {{"kind", "exception"}, {"message", error.what()}}; }
        catch (...) { response["error"] = {{"kind", "unknown"}, {"message", ""}}; }
        return allocateString(response.dump());
    } catch (...) { return nullptr; }
}
static const char* api_stage_manual_memory(const GrimPluginAPI* self, const char* raw) noexcept {
    if (!raw || !*raw) return nullptr;
    return runtimeServiceResponse(self, GRIM_PERM_MEMORY, [&] {
        GRIM::CommandMemory::stageManual(raw);
        return nlohmann::json(nullptr);
    });
}
static const char* api_search_memory(const GrimPluginAPI* self, const char* query) noexcept {
    if (!query) return nullptr;
    return runtimeServiceResponse(self, GRIM_PERM_MEMORY, [&] {
        auto hits = nlohmann::json::array();
        for (const auto& hit : GRIM::CommandMemory::search(query)) hits.push_back({{"id", hit.id}, {"raw", hit.raw}});
        return hits;
    });
}
static const char* api_flush_memory_record_if_present(const GrimPluginAPI* self, uint64_t id) noexcept {
    return runtimeServiceResponse(self, GRIM_PERM_MEMORY, [&] {
        return nlohmann::json(GRIM::CommandMemory::flushIfPresent(id));
    });
}

static const char* api_get_tool_snapshot(const GrimPluginAPI* self, const char* tool_id) noexcept {
    try {
        (void)pluginContext(self);
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

static const char* api_get_user_error_message(const GrimPluginAPI*, const char* code) noexcept {
    if (!code) return nullptr;
    try { return allocateString(ErrorManager::getUserMessage(code)); }
    catch (...) { return nullptr; }
}

// ============================================================================
// System tick subscriptions
// ============================================================================

static GrimResult api_subscribe_system_tick(const GrimPluginAPI* self,
                                            GrimSystemTickCallback callback, void* userData) noexcept {
    if (!callback) return GRIM_ERROR_INVALID_PARAM;
    try {
        const auto& owner = pluginContext(self);
        ActiveCall active(owner.gate);
        if (!active.entered) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        std::lock_guard lock(s_resourceMutex);
        const auto inserted = s_systemTickSubscriptions.try_emplace(
            owner.identity->instance, SystemTickSubscription{callback, userData, owner.gate}).second;
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

static const char* api_get_plugin_name(const GrimPluginAPI* self) {
    return allocateString(pluginContext(self).identity->name);
}

static const char* api_get_plugin_path(const GrimPluginAPI* self) {
    return allocateString(pluginContext(self).identity->path);
}

static uint32_t api_get_permissions(const GrimPluginAPI* self) {
    return pluginContext(self).permissions;
}

static bool api_has_permission(const GrimPluginAPI* self, GrimPermission perm) {
    return checkPermission(self, perm);
}

static void api_request_reload(const GrimPluginAPI* self) {
    LOG_DEBUG("PluginAPI", "Plugin requested reload: " + pluginContext(self).identity->name);
    PluginManager::requestReload(self);
}

// ============================================================================
// Utility Implementation
// ============================================================================

static void api_string_free(const GrimPluginAPI* self, const char* str) {
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

static const char* api_ui_panel_service(const GrimPluginAPI* self, GrimUIPanelOperation operation,
                                      const char* panel_key, uint64_t request_id) noexcept {
    try {
        auto& owner = const_cast<PluginContext&>(pluginContext(self));
        ActiveCall active(owner.gate, true);
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
        std::lock_guard lock(owner.panelRequestsMutex);
        nlohmann::json response = {{"format", "grim-ui-panel-response"}, {"version", 1},
                                  {"request_id", request_id}, {"status", "pending"}};
        if (operation == GRIM_UI_POLL || operation == GRIM_UI_CANCEL) {
            const auto found = owner.panelRequests.find(request_id);
            if (found == owner.panelRequests.end()) {
                response["status"] = "error"; response["error"] = "request_not_found";
                return allocateString(response.dump());
            }
            if (operation == GRIM_UI_CANCEL) {
                found->second.cancelled->store(true);
                owner.panelRequests.erase(found);
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
            owner.panelRequests.erase(found);
            return allocateString(response.dump());
        }
        if (owner.panelRequests.size() >= 64) {
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
        owner.panelRequests.emplace(id, std::move(request));
        return allocateString(response.dump());
    } catch (...) { return nullptr; }
}


GrimPluginAPI* createPluginAPI(const std::string& plugin_name, const std::string& plugin_path, uint32_t permissions, bool staging, const GrimPluginAPI* replacing) {
    if (plugin_name.empty()) throw std::invalid_argument("Empty plugin identity");
    for (const auto& [id, context] : s_plugin_contexts) {
        if (context->identity->name == plugin_name && &context->api != replacing)
            throw std::invalid_argument("Duplicate plugin identity: " + plugin_name);
    }
    if (replacing && pluginContext(replacing).identity->name != plugin_name)
        throw std::invalid_argument("Replacement must retain provider identity");
    if (replacing && !staging)
        throw std::invalid_argument("Replacement must be staged");

    const auto instance = plugin_name + "#" + std::to_string(s_next_instance++);
    auto identity = std::make_shared<const GRIM::MMO::ProviderDescriptor>(
        GRIM::MMO::ProviderDescriptor{GRIM::MMO::ToolProviderType::Plugin, plugin_name, instance, plugin_path});
    auto context = std::make_unique<PluginContext>(identity);
    context->panelLease->owner = instance;
    context->panelLease->uiPermitted = (permissions & GRIM_PERM_UI) != 0;
    context->permissions = permissions;
    context->staging = staging;
    context->gate->published = !staging;
    auto& api = context->api;
    api.host_context = context.get();
    
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
    
    // Context
    api.get_context = api_get_context;
    api.set_context = api_set_context;
    api.delete_context = api_delete_context;
    api.has_context = api_has_context;
    
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
    api.stage_manual_memory = api_stage_manual_memory;
    api.search_memory = api_search_memory;
    api.flush_memory_record_if_present = api_flush_memory_record_if_present;
    api.ui_panel_service = api_ui_panel_service;
    api.get_provider_identity = api_get_provider_identity;
    api.replace_resources = api_replace_resources;
    api.is_provider_current = api_is_provider_current;
    api.subscribe_system_tick = api_subscribe_system_tick;
    
    auto* result = &api;
    auto [entry, inserted] = s_plugin_contexts.emplace(instance, std::move(context));
    if (!inserted) throw std::logic_error("Duplicate plugin instance: " + instance);
    bool generationCreated = false;
    try {
        if (staging) {
            GRIM::MMO::ProviderRegistry::instance().beginGeneration(identity);
            generationCreated = true;
            GRIM::UI::registerProviderPanelLease(entry->second->panelLease);
        }
    } catch (...) {
        if (generationCreated) GRIM::MMO::ProviderRegistry::instance().retire(plugin_name, instance);
        s_plugin_contexts.erase(entry);
        throw;
    }
    return result;
}

// Name-based wrappers are for host callers with exactly one loaded generation.
static GrimPluginAPI* uniquePlugin(const std::string& name) {
    GrimPluginAPI* result = nullptr;
    for (const auto& [id, context] : s_plugin_contexts) if (context->identity->name == name) {
        if (result) throw std::invalid_argument("Ambiguous plugin generation");
        result = &context->api;
    }
    if (!result) throw std::invalid_argument("Missing plugin context");
    return result;
}
bool commitPluginGeneration(const GrimPluginAPI* api, const GrimPluginAPI* replacing,
                            const std::vector<GRIM::MMO::ToolRegistration>& tools) {
    auto& owner = const_cast<PluginContext&>(pluginContext(api));
    if (owner.resource_staging_failed || !owner.staging ||
        !GRIM::UI::validateProviderPanels(owner.panelLease, replacing ? pluginContext(replacing).panelLease : nullptr)) return false;
    auto& providers = GRIM::MMO::ProviderRegistry::instance();
    // Handlers may become reachable as soon as the registry swaps its entries,
    // including from listeners. Late registrations must no longer be staged.
    owner.staging = false;
    owner.panelLease->acceptingPanels = false;
    bool committed;
    try {
        committed = providers.publish(
            *owner.identity,
            tools, replacing ? pluginContext(replacing).identity->instance : std::string{});
    } catch (...) {
        owner.staging = true;
        owner.panelLease->acceptingPanels = true;
        throw;
    }
    if (!committed) {
        owner.staging = true;
        owner.panelLease->acceptingPanels = true;
        return false;
    }
    // Queue retirement before replacement creation. Generation-specific leases
    // prevent a later shutdown of the old DLL from removing its replacement.
    if (replacing) GRIM::UI::retireProviderPanels(pluginContext(replacing).panelLease);
    if (!owner.panelLease->nativePanels.empty())
        GRIM::UI::activateProviderPanels(owner.panelLease);
    if (replacing) {
        const auto& old = pluginContext(replacing);
        std::lock_guard lock(old.gate->mutex);
        old.gate->accepting = false;
    }
    owner.staging = false;
    { std::lock_guard lock(owner.gate->mutex); owner.gate->published = true; }
    return true;
}
bool commitPluginGeneration(const std::string& name) { return commitPluginGeneration(uniquePlugin(name), nullptr); }
void beginPluginShutdown(const GrimPluginAPI* api) {
    auto& owner = const_cast<PluginContext&>(pluginContext(api));
    {
        std::lock_guard lock(owner.gate->mutex);
        owner.gate->accepting = false;
    }
    GRIM::MMO::ProviderRegistry::instance().retire(owner.identity->name, owner.identity->instance);
    GRIM::UI::retireProviderPanels(owner.panelLease);
    std::unique_lock lock(owner.gate->mutex);
    owner.gate->idle.wait(lock, [&]{ return owner.gate->active == 0; });
    lock.unlock();
    std::lock_guard requestsLock(owner.panelRequestsMutex);
    for (auto& [id, request] : owner.panelRequests)
        request.cancelled->store(true);
    owner.panelRequests.clear();
}
void beginPluginShutdown(const std::string& name) { beginPluginShutdown(uniquePlugin(name)); }
void cleanupPluginContext(const GrimPluginAPI* api) {
    beginPluginShutdown(api);
    const auto instance = pluginContext(api).identity->instance;
    std::lock_guard lock(s_resourceMutex);
    s_systemTickSubscriptions.erase(instance);
    for (auto it = s_event_subscriptions.begin(); it != s_event_subscriptions.end();) {
        if (it->second.plugin_name == instance) it = s_event_subscriptions.erase(it);
        else ++it;
    }
    s_plugin_contexts.erase(instance);
}
void cleanupPluginContext(const std::string& name) { cleanupPluginContext(uniquePlugin(name)); }

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
