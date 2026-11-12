// ============================================================================
// G.R.I.M Plugin API Implementation
// ============================================================================
// This file implements the host-side functions that populate the plugin API
// function table. These are the actual implementations that plugins call.
// ============================================================================

#include "plugin_api.hpp"
#include "local_calendar.hpp"
#include "scheduling_service.hpp"
#include "plugin_manager.hpp"
#include "logger.hpp"
#include "commands/commands_core.hpp"
#include "error_manager.hpp"
#include "aliases.hpp"
#include "commands/commands_timers.hpp"
#include "memory/command_memory_services.hpp"
#include "perception/perception_queries.hpp"
#include "voice/voice_command_services.hpp"
#include "ui/ui_panel_service.hpp"
#include "perception/input_command_service.hpp"
#include "voice/voice_speak.hpp"
#include "../MMO/Core/Registries/ToolRegistry.hpp"  // Phase 0.25: plugin↔ToolRegistry sync
#include "../MMO/Core/Descriptors/ToolDescriptorJson.hpp"
#include "popup_ui/popup_ui.hpp"
#include "helpers/grim_input.hpp"  // For GRIMInput::parseInput
#include "synonyms.hpp"            // For normalizeWord() synonym resolution
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstdarg>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <unordered_set>
#include <nlohmann/json.hpp>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>  // For CreateProcess, psapi, tlhelp32, etc.
    #include <psapi.h>
    #include <tlhelp32.h>
#else
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <signal.h>
    #include <dirent.h>
    #include <fstream>
    #include <cstdlib>
#endif

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

struct PluginTimer {
    uint64_t id;
    uint64_t interval_ms;
    bool repeat;
    GrimTimerCallback callback;
    void* user_data;
    std::chrono::steady_clock::time_point next_fire;
    bool active;
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
struct PluginContext {
    std::string instance;
    std::string name;
    std::string path;
    uint32_t permissions;
    GrimPluginAPI api{};
    std::shared_ptr<CallGate> gate = std::make_shared<CallGate>();
    bool staging = false;
    bool registration_failed = false;
    std::vector<GRIM::MMO::ToolRegistration> pending;
    std::vector<std::string> pendingPanels;
    std::mutex resourcesMutex;
    std::map<std::string, GRIM::MMO::ResourceDescriptor> pendingResources;
    std::shared_ptr<GRIM::UI::ProviderPanelLease> panelLease =
        std::make_shared<GRIM::UI::ProviderPanelLease>();
    std::mutex panelRequestsMutex;
    std::unordered_map<uint64_t, GRIM::UI::PanelServiceRequest> panelRequests;
};

// Static storage
static std::unordered_map<uint64_t, EventSubscription> s_event_subscriptions;
static uint64_t s_next_event_id = 1;
static std::unordered_map<uint64_t, PluginTimer> s_timers;
static uint64_t s_next_timer_id = 1;
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
    std::string prefix = "[Plugin: " + pluginContext(self).name + "] ";
    std::string full_msg = prefix + message;
    
    switch (level) {
        case GRIM_LOG_DEBUG:    LOG_DEBUG("Plugin", full_msg); break;
        case GRIM_LOG_INFO: LOG_DEBUG("Plugin", full_msg); break;  // Map INFO to DEBUG
    case GRIM_LOG_WARNING:  LOG_ERROR("Plugin", full_msg); break;  // Map WARNING to ERROR
        case GRIM_LOG_ERROR:    LOG_ERROR("Plugin", full_msg); break;
        case GRIM_LOG_CRITICAL: LOG_ERROR("Plugin", full_msg); break;  // Map CRITICAL to ERROR
    }
}

static void api_log_fmt(const GrimPluginAPI* self, GrimLogLevel level, const char* format, ...) {
    char buffer[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    api_log(self, level, buffer);
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
static std::vector<std::string> copyList(const GrimStringList& list) {
    if (list.count > 256 || (list.count && !list.items)) throw std::invalid_argument("Invalid list");
    std::vector<std::string> result;
    for (size_t i = 0; i < list.count; ++i) result.push_back(copyText(list.items[i], true));
    return result;
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
static GRIM::MMO::ToolRegistration copyRegistration(const GrimToolRegistration& source, const PluginContext& owner) {
    using namespace GRIM::MMO;
    const auto& in = source.descriptor;
    if (source.struct_size != sizeof(source) || in.struct_size != sizeof(in) || !source.handler ||
        in.parameter_count > 256 || (in.parameter_count && !in.parameters) ||
        in.is_informational > 1 || in.needs_confirmation > 1)
        throw std::invalid_argument("Invalid tool structure");
    if ((in.required_permissions & owner.permissions) != in.required_permissions)
        throw std::invalid_argument("Tool requires undeclared permissions");
    ToolDescriptor out;
    out.tool_id = copyText(in.tool_id, true);
    out.display_name = copyText(in.display_name, true);
    out.version = copyText(in.version, true);
    out.description = copyText(in.description, true);
    out.short_description = copyText(source.short_description);
    if (!validToolShortDescription(out.short_description)) throw std::invalid_argument("Invalid short description");
    out.usage = copyText(in.usage, true);
    if (source.binding_count > 256 || (source.binding_count && !source.bindings))
        throw std::invalid_argument("Invalid tool bindings");
    for (size_t i = 0; i < source.binding_count; ++i) {
        const auto& binding = source.bindings[i];
        if (!out.bindings.emplace(copyText(binding.placeholder, true), copyText(binding.type, true)).second)
            throw std::invalid_argument("Duplicate tool binding");
    }
    out.category = copyText(in.category, true);
    if (!validToolBindings(out)) throw std::invalid_argument("Invalid tool bindings");
    out.argument_format = copyText(in.argument_format, true);
    out.argument_schema_json = copyText(in.argument_schema_json);
    if (out.argument_format != "raw_text" && out.argument_format != "json")
        throw std::invalid_argument("Unsupported argument format");
    if (!out.argument_schema_json.empty()) {
        const auto schema = nlohmann::json::parse(out.argument_schema_json);
        if (!schema.is_object()) throw std::invalid_argument("Schema must be an object");
    } else if (out.argument_format == "json") throw std::invalid_argument("JSON tools require a schema");
    std::unordered_set<std::string> parameters;
    for (size_t i = 0; i < in.parameter_count; ++i) {
        const auto& p = in.parameters[i];
        if (p.struct_size != sizeof(p) || p.required > 1) throw std::invalid_argument("Invalid parameter");
        ToolParameter value{copyText(p.name, true), copyText(p.type, true), copyText(p.description, true), p.required != 0};
        if (!parameters.insert(value.name).second) throw std::invalid_argument("Duplicate parameter");
        out.parameters.push_back(std::move(value));
    }
    out.aliases = copyList(in.aliases);
    out.keywords = copyList(in.keywords);
    out.capability_tags = copyList(in.capability_tags);
    out.preconditions = copyList(in.preconditions);
    out.permission_bits = in.required_permissions;
    out.is_informational = in.is_informational != 0;
    out.needs_confirmation = in.needs_confirmation != 0;
    out.usage_type = static_cast<ToolUsage>(source.usage_type);
    if (!validToolUsage(out.usage_type)) throw std::invalid_argument("Invalid tool usage type");
    auto gate = owner.gate;
    auto handler = source.handler;
    auto data = source.user_data;
    auto invoke = [gate, handler, data](const std::string& arguments) -> CommandResult {
        ActiveCall active(gate);
        if (!active.entered) return {false, "Plugin is stopping", "ERR_TOOL_UNAVAILABLE"};
        ResultCapture capture;
        GrimToolResultWriter writer{sizeof(GrimToolResultWriter), &capture, completeResult};
        GrimToolCall call{sizeof(GrimToolCall), arguments.c_str(), arguments.size()};
        const auto delivered = handler(&call, &writer, data);
        if (delivered != GRIM_OK || !capture.completed || capture.invalid)
            return {false, "Plugin did not deliver a valid result", "ERR_PLUGIN_RESULT"};
        return std::move(capture.value);
    };
    return {std::move(out), std::move(invoke)};
}
static GrimResult api_register_tools(const GrimPluginAPI* self, const GrimToolRegistration* tools, size_t count) noexcept {
    PluginContext* owner = nullptr;
    try {
        owner = const_cast<PluginContext*>(&pluginContext(self));
        {
            std::lock_guard lock(owner->gate->mutex);
            if (!owner->gate->accepting) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        }
        if (!owner->staging) return GRIM_ERROR_UNSUPPORTED;
        if (!tools || !count || count > 256) throw std::invalid_argument("Invalid batch");
        std::vector<GRIM::MMO::ToolRegistration> copied;
        copied.reserve(count);
        for (size_t i = 0; i < count; ++i)
            copied.push_back(copyRegistration(tools[i], *owner));
        owner->pending.insert(owner->pending.end(), copied.begin(), copied.end());
        return GRIM_OK;
    } catch (...) {
        if (owner && owner->staging) owner->registration_failed = true;
        return GRIM_ERROR_INVALID_PARAM;
    }
}

static GrimResult api_register_ui_panel(const GrimPluginAPI* self, const char* implementation_key) noexcept {
    PluginContext* owner = nullptr;
    try {
        owner = const_cast<PluginContext*>(&pluginContext(self));
        std::lock_guard lock(owner->gate->mutex);
        if (!owner->gate->accepting) return GRIM_ERROR_PLUGIN_NOT_LOADED;
        if (!checkPermission(self, GRIM_PERM_UI)) {
            owner->registration_failed = true;
            return GRIM_ERROR_PERMISSION_DENIED;
        }
        if (!owner->staging) return GRIM_ERROR_UNSUPPORTED;
        const auto key = copyText(implementation_key, true, 256);
        if (key.find_first_of("\r\n") != std::string::npos || owner->pendingPanels.size() >= 64)
            throw std::invalid_argument("Invalid panel implementation declaration");
        for (const auto& existing : owner->pendingPanels)
            if (existing == key) {
                owner->registration_failed = true;
                return GRIM_ERROR_ALREADY_EXISTS;
            }
        owner->pendingPanels.push_back(key);
        return GRIM_OK;
    } catch (...) {
        if (owner && owner->staging) owner->registration_failed = true;
        return GRIM_ERROR_INVALID_PARAM;
    }
}

// Provider-neutral bridge: copied descriptions only; implementation and state stay in the DLL.
static const char* api_get_provider_identity(const GrimPluginAPI* self) noexcept {
    try {
        const auto& owner = pluginContext(self);
        return allocateString(nlohmann::json{{"name", owner.name}, {"instance", owner.instance},
                                            {"path", owner.path}}.dump());
    } catch (...) { return nullptr; }
}
static bool api_is_provider_current(const GrimPluginAPI* self) noexcept {
    try {
        const auto& owner = pluginContext(self);
        ActiveCall active(owner.gate, true);
        if (!active.entered) return false;
        const auto snapshot = GRIM::MMO::ProviderRegistry::instance().snapshot();
        return std::any_of(snapshot.providers.begin(), snapshot.providers.end(), [&](const auto& group) {
            return group.provider.name == owner.name && group.provider.instance == owner.instance;
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
        // Identical validation for staging and live updates.
        GRIM::MMO::ProviderRegistry::validateResources(resourceKind, resources);
        std::lock_guard lock(owner.resourcesMutex);
        if (owner.staging) {
            auto next = owner.pendingResources;
            std::erase_if(next, [&](const auto& entry) { return entry.second.kind == resourceKind; });
            for (const auto& resource : resources)
                if (!next.emplace(resource.resource_id, resource).second)
                    throw std::invalid_argument("Resource ID belongs to another kind");
            if (next.size() > 10000) throw std::invalid_argument("Provider resource limit exceeded");
            owner.pendingResources.swap(next);
            return GRIM_OK;
        }
        return GRIM::MMO::ProviderRegistry::instance().replaceResources(
            {GRIM::MMO::ToolProviderType::Plugin, owner.name, owner.instance, owner.path}, resourceKind, resources)
            ? GRIM_OK : GRIM_ERROR_PLUGIN_NOT_LOADED;
    } catch (...) {
        if (context && context->staging) context->registration_failed = true;
        return GRIM_ERROR_INVALID_PARAM;
    }
}

static GrimResult api_unregister_command(const GrimPluginAPI* self, const char* command_name) {
    if (!command_name || !*command_name) return GRIM_ERROR_INVALID_PARAM;
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto tool = registry.getTool(command_name);
    if (!tool || tool->tool_id != command_name) return GRIM_ERROR_NOT_FOUND;
    const auto& owner = pluginContext(self);
    if (tool->provider.type != GRIM::MMO::ToolProviderType::Plugin || tool->provider.name != owner.name || tool->provider.instance != owner.instance)
        return GRIM_ERROR_PERMISSION_DENIED;
    return registry.unregisterTool(command_name, owner.name, owner.instance) ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
}

static bool api_is_command_registered(const GrimPluginAPI* self, const char* command_name) {
    return command_name && GRIM::MMO::ToolRegistry::instance().isRegistered(command_name);
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
    sub.plugin_name = pluginContext(self).instance;
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
    
    if (it->second.plugin_name != pluginContext(self).instance) {
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
// Memory & Context Implementation
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

static GrimResult api_store_memory(const GrimPluginAPI* self, const char* key, const char* value, const char* category) {
    if (!checkPermission(self, GRIM_PERM_MEMORY)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!key || !value) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    // Store in GRIM's memory system
    // TODO: Implement proper MemoryObject creation
// g_memoryStorage.store(key, value, category ? category : "plugin");
    LOG_DEBUG("PluginAPI", std::string("Memory store not yet implemented for key: ") + key);
    return GRIM_OK;
}

static const char* api_retrieve_memory(const GrimPluginAPI* self, const char* key) {
    if (!checkPermission(self, GRIM_PERM_MEMORY)) {
        return nullptr;
    }
    
    if (!key) return nullptr;
    
    // TODO: Implement proper memory retrieval
    // std::string value = g_memoryStorage.retrieve(key);
    // if (value.empty()) return nullptr;
    // return allocateString(value);
    
    LOG_DEBUG("PluginAPI", std::string("Memory retrieve not yet implemented for key: ") + key);
    return nullptr;
}

static GrimResult api_delete_memory(const GrimPluginAPI* self, const char* key) {
    if (!checkPermission(self, GRIM_PERM_MEMORY)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!key) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    // TODO: Implement proper memory deletion
    // g_memoryStorage.erase(key);
    LOG_DEBUG("PluginAPI", std::string("Memory delete not yet implemented for key: ") + key);
    return GRIM_OK;
}

// ============================================================================
// NLP & AI Implementation
// ============================================================================

static GrimIntent api_classify_intent(const GrimPluginAPI* self, const char* text) {
    (void)text;
    // ABI compatibility only. The application no longer classifies raw user
    // text into host-authored intents.
    return {};
}

static const char* api_resolve_synonym(const GrimPluginAPI* self, const char* word) {
    if (!word) return nullptr;
    
    // Use GRIM's existing synonym system (synonyms.hpp)
    std::string resolved = normalizeWord(word);
    
    // If normalization returns the same word, no synonym was found
    if (resolved == word) {
 LOG_DEBUG("PluginAPI", std::string("No synonym found for: ") + word);
    }
    
    return allocateString(resolved);
}

static const char* api_get_conversation_history(const GrimPluginAPI* self, int limit) {
    // Return JSON array of recent history
    // For now, return empty array
    return allocateString("[]");
}

// ============================================================================
// Output & UI Implementation
// ============================================================================

static void api_send_response(const GrimPluginAPI* self, const char* text) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (text) {
        LOG_DEBUG("Plugin", std::string(text));
        // TODO: Send to actual UI output
    }
}

static void api_send_response_colored(const GrimPluginAPI* self, const char* text, uint32_t rgb_color) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (text) {
        // Extract RGB components
        uint8_t r = (rgb_color >> 16) & 0xFF;
        uint8_t g = (rgb_color >> 8) & 0xFF;
        uint8_t b = rgb_color & 0xFF;
        
        LOG_DEBUG("Plugin", std::string(text));
        // TODO: Send to actual UI output with color
    }
}

static void api_show_notification(const GrimPluginAPI* self, const char* title, const char* message) {
    if (!checkPermission(self, GRIM_PERM_UI)) {
        return;
    }
    
    if (title && message) {
        std::string full_msg = std::string(title) + ": " + message;
        LOG_DEBUG("Plugin", full_msg);
        
        // Show popup UI notification (platform-specific UI)
        #ifdef _WIN32
        notifyPopupActivity();
        showPopup();
        #else
        // On Linux, just log for now (TODO: implement notification daemon integration)
        #endif
    }
}

// ============================================================================
// Voice Implementation
// ============================================================================

static GrimResult api_queue_tts(const GrimPluginAPI* self, const char* text, const GrimVoiceParams* params) {
    if (!checkPermission(self, GRIM_PERM_VOICE)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!text) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    Voice::speak(text, "plugin");
    return GRIM_OK;
}

static GrimResult api_stop_tts(const GrimPluginAPI* self) {
    if (!checkPermission(self, GRIM_PERM_VOICE)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
  
    // TODO: Implement audio stop functionality
    // Voice::stopAudio();
    LOG_DEBUG("PluginAPI", "Audio stop not yet implemented");
    return GRIM_OK;
}

static bool api_is_speaking(const GrimPluginAPI* self) {
    if (!checkPermission(self, GRIM_PERM_VOICE)) {
        return false;
    }
    
    return Voice::isSpeaking();
}

// ============================================================================
// Process Management Implementation
// ============================================================================

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
static const char* api_voice_service(const GrimPluginAPI* self, GrimVoiceOperation operation, const char* arguments_json) noexcept {
    if (operation > GRIM_VOICE_PATHS) return nullptr;
    try {
        const auto request = arguments_json ? nlohmann::json::parse(arguments_json) : nlohmann::json::object();
        if (!request.is_object()) return nullptr;
        const bool cache = operation == GRIM_VOICE_CACHE_SHUTDOWN || operation == GRIM_VOICE_CACHE_INIT || operation == GRIM_VOICE_PRECACHE_COMMON;
        const auto permission = static_cast<GrimPermission>(GRIM_PERM_VOICE | (cache ? GRIM_PERM_FILESYSTEM : 0));
        return runtimeServiceResponse(self, permission, [&] { return GRIM::VoiceCommandServices::execute(operation, request); });
    } catch (...) { return nullptr; }
}
static const char* api_query_perception(const GrimPluginAPI* self, GrimPerceptionQuery query) noexcept {
    if (query != GRIM_PERCEPTION_ANALYZE_SCREEN && query != GRIM_PERCEPTION_READ_TEXT &&
        query != GRIM_PERCEPTION_DETECT_OBJECTS) return nullptr;
    return runtimeServiceResponse(self, GRIM_PERM_SYSTEM, [&] {
        const bool available = GRIM::Perception::isAvailable();
        std::string text;
        if (available) {
            switch (query) {
                case GRIM_PERCEPTION_ANALYZE_SCREEN: text = GRIM::Perception::analyzeScreen(); break;
                case GRIM_PERCEPTION_READ_TEXT: text = GRIM::Perception::readText(); break;
                case GRIM_PERCEPTION_DETECT_OBJECTS: text = GRIM::Perception::detectObjects(); break;
            }
        }
        return nlohmann::json{{"available", available}, {"text", std::move(text)}};
    });
}
static const char* api_schedule_service(const GrimPluginAPI* self, const char* operation,
    const char* arguments, size_t size) noexcept {
    try {
        ActiveCall active(pluginContext(self).gate, true);
        if(!active.entered || !operation || !arguments || size>65536 ||
            !checkPermission(self, static_cast<GrimPermission>(GRIM_PERM_SYSTEM | GRIM_PERM_FILESYSTEM))) return nullptr;
        auto request=nlohmann::json::parse(arguments,arguments+size);
        const auto result=grim::scheduling::schedulingService().execute(operation,request,grim::scheduling::scheduleNow()).dump();
        return allocateString(result);
    } catch (...) {return nullptr;}
}
static GrimScheduleResult api_create_local_calendar(const GrimPluginAPI* self,
    uint32_t version, GrimScheduleProviderV1* out) noexcept {
    try {
        if (!checkPermission(self, GRIM_PERM_SYSTEM)) return GRIM_SCHEDULE_PERMISSION_DENIED;
        return grim::scheduling::LocalCalendar::create(version, out);
    } catch (...) { return GRIM_SCHEDULE_FAILED; }
}
static const char* api_schedule_user_timer(const GrimPluginAPI* self, int32_t seconds, const char* message) noexcept {
    if (seconds <= 0 || !message) return nullptr;
    return runtimeServiceResponse(self, GRIM_PERM_SYSTEM, [&] {
        GRIM::UserTimers::schedule(seconds, message);
        return nlohmann::json(nullptr);
    });
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

static const char* api_get_alias_snapshot(const GrimPluginAPI* self, const char* key) noexcept {
    try {
        (void)pluginContext(self);
        auto entries = nlohmann::json::array();
        std::string info;
        if (key) info = aliases::info(key);
        else for (const auto& [name, target] : aliases::getAll())
            entries.push_back({{"name", name}, {"target", target}});
        return allocateString(nlohmann::json{{"format", "grim-alias-snapshot"}, {"version", 1},
            {"entries", std::move(entries)}, {"info", info}}.dump());
    } catch (...) { return nullptr; }
}

static const char* api_refresh_aliases(const GrimPluginAPI* self) noexcept {
    try {
        (void)pluginContext(self);
        nlohmann::json result{{"format", "grim-alias-refresh"}, {"version", 1},
            {"completed", false}, {"error", ""}};
        if (!checkPermission(self, GRIM_PERM_FILESYSTEM)) {
            result["error"] = "Filesystem permission denied";
        } else {
            try {
                (void)aliases::refreshNow(); // Preserve the old wrapper's ignored status.
                result["completed"] = true;
            } catch (const std::exception& error) { result["error"] = error.what(); }
        }
        return allocateString(result.dump());
    } catch (...) { return nullptr; }
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

static GrimResult api_launch_process(const GrimPluginAPI* self, const char* command, const char* args) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!command) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    #ifdef _WIN32
    // Windows implementation
    std::string cmd_line = command;
    if (args && strlen(args) > 0) {
        cmd_line += " " + std::string(args);
    }
    
    STARTUPINFOA si = {};
    PROCESS_INFORMATION pi = {};
    si.cb = sizeof(si);
    
    if (!CreateProcessA(
        nullptr,
        const_cast<char*>(cmd_line.c_str()),
        nullptr,
        nullptr,
        FALSE,
        0,
        nullptr,
        nullptr,
        &si,
        &pi
    )) {
        LOG_ERROR("PluginAPI", "Failed to launch process: " + cmd_line);
        return GRIM_ERROR_UNKNOWN;
    }
    
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    
    LOG_DEBUG("PluginAPI", "Launched process: " + cmd_line);
    return GRIM_OK;
    
    #else
    // Linux implementation
    pid_t pid = fork();
    
    if (pid < 0) {
        LOG_ERROR("PluginAPI", "Failed to fork process");
        return GRIM_ERROR_UNKNOWN;
    }
    
    if (pid == 0) {
        // Child process
        if (args && strlen(args) > 0) {
            execl("/bin/sh", "sh", "-c", (std::string(command) + " " + args).c_str(), nullptr);
        } else {
            execl("/bin/sh", "sh", "-c", command, nullptr);
        }
        // If exec fails
        _exit(1);
    }
    
    // Parent process
    LOG_DEBUG("PluginAPI", std::string("Launched process: ") + command);
    return GRIM_OK;
    #endif
}

static bool api_is_process_running(const GrimPluginAPI* self, const char* process_name) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return false;
    }
    
    if (!process_name) {
        return false;
    }
    
    #ifdef _WIN32
    // Windows implementation
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return false;
    }
    
    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);
    
    bool found = false;
    if (Process32First(snapshot, &pe32)) {
        do {
            std::string exe_file = pe32.szExeFile;
            std::string search_name = process_name;
            
            std::transform(exe_file.begin(), exe_file.end(), exe_file.begin(), ::tolower);
            std::transform(search_name.begin(), search_name.end(), search_name.begin(), ::tolower);
            
            if (search_name.find(".exe") == std::string::npos) {
                search_name += ".exe";
            }
            
            if (exe_file == search_name) {
                found = true;
                break;
            }
        } while (Process32Next(snapshot, &pe32));
    }
    
    CloseHandle(snapshot);
    return found;
    
    #else
    // Linux implementation - check /proc
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) {
        return false;
    }
    
    bool found = false;
    struct dirent* entry;
    
    while ((entry = readdir(proc_dir)) != nullptr) {
        // Check if directory name is a number (PID)
        if (entry->d_type != DT_DIR) continue;
        
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        
        // Read /proc/[pid]/comm
        std::string comm_path = std::string("/proc/") + entry->d_name + "/comm";
        std::ifstream comm_file(comm_path);
        if (comm_file.is_open()) {
            std::string proc_name;
            std::getline(comm_file, proc_name);
            
            if (proc_name == process_name) {
                found = true;
                break;
            }
        }
    }
    
    closedir(proc_dir);
    return found;
    #endif
}

static GrimResult api_kill_process(const GrimPluginAPI* self, const char* process_name) {
    if (!checkPermission(self, GRIM_PERM_PROCESS)) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    
    if (!process_name) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    #ifdef _WIN32
    // Windows implementation
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return GRIM_ERROR_UNKNOWN;
    }
    
    PROCESSENTRY32 pe32;
    pe32.dwSize = sizeof(PROCESSENTRY32);
    
    bool killed = false;
    if (Process32First(snapshot, &pe32)) {
        do {
            std::string exe_file = pe32.szExeFile;
            std::string search_name = process_name;
            
            std::transform(exe_file.begin(), exe_file.end(), exe_file.begin(), ::tolower);
            std::transform(search_name.begin(), search_name.end(), search_name.begin(), ::tolower);
            
            if (search_name.find(".exe") == std::string::npos) {
                search_name += ".exe";
            }
            
            if (exe_file == search_name) {
                HANDLE hProcess = OpenProcess(PROCESS_TERMINATE, FALSE, pe32.th32ProcessID);
                if (hProcess) {
                    TerminateProcess(hProcess, 0);
                    CloseHandle(hProcess);
                    killed = true;
                    LOG_DEBUG("PluginAPI", "Killed process: " + std::string(process_name));
                }
            }
        } while (Process32Next(snapshot, &pe32));
    }
    
    CloseHandle(snapshot);
    return killed ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
    
    #else
    // Linux implementation - find and kill by name
    DIR* proc_dir = opendir("/proc");
    if (!proc_dir) {
        return GRIM_ERROR_UNKNOWN;
    }
    
    bool killed = false;
    struct dirent* entry;
    
    while ((entry = readdir(proc_dir)) != nullptr) {
        if (entry->d_type != DT_DIR) continue;
        
        int pid = atoi(entry->d_name);
        if (pid <= 0) continue;
        
        // Read /proc/[pid]/comm
        std::string comm_path = std::string("/proc/") + entry->d_name + "/comm";
        std::ifstream comm_file(comm_path);
        if (comm_file.is_open()) {
            std::string proc_name;
            std::getline(comm_file, proc_name);
            
            if (proc_name == process_name) {
                if (kill(pid, SIGTERM) == 0) {
                    killed = true;
                    LOG_DEBUG("PluginAPI", "Killed process: " + std::string(process_name));
                }
            }
        }
    }
    
    closedir(proc_dir);
    return killed ? GRIM_OK : GRIM_ERROR_NOT_FOUND;
    #endif
}

// ============================================================================
// System Information Implementation
// ============================================================================

// ============================================================================
// Timer Implementation (Stub)
// ============================================================================

static GrimTimerHandle api_create_timer(const GrimPluginAPI* self, uint64_t interval_ms, bool repeat, GrimTimerCallback callback, void* user_data) {
    if (!callback) {
        return nullptr;
    }
    
    std::lock_guard lock(s_resourceMutex);
    uint64_t id = s_next_timer_id++;
    
    PluginTimer timer;
    timer.id = id;
    timer.interval_ms = interval_ms;
    timer.repeat = repeat;
    timer.callback = callback;
    timer.user_data = user_data;
    timer.next_fire = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval_ms);
    timer.active = true;
    timer.plugin_name = pluginContext(self).instance;
    timer.gate = pluginContext(self).gate;
    
    s_timers[id] = timer;
    
    return reinterpret_cast<GrimTimerHandle>(id);
}

static GrimResult api_cancel_timer(const GrimPluginAPI* self, GrimTimerHandle handle) {
    if (!handle) {
        return GRIM_ERROR_INVALID_PARAM;
    }
    
    uint64_t id = reinterpret_cast<uint64_t>(handle);
    std::lock_guard lock(s_resourceMutex);
    auto it = s_timers.find(id);
    
    if (it == s_timers.end()) {
        return GRIM_ERROR_NOT_FOUND;
    }
    
    if (it->second.plugin_name != pluginContext(self).instance) {
        return GRIM_ERROR_PERMISSION_DENIED;
    }
    s_timers.erase(it);
    return GRIM_OK;
}

// ============================================================================
// Plugin Lifecycle Implementation
// ============================================================================

static const char* api_get_plugin_name(const GrimPluginAPI* self) {
    return allocateString(pluginContext(self).name);
}

static const char* api_get_plugin_path(const GrimPluginAPI* self) {
    return allocateString(pluginContext(self).path);
}

static uint32_t api_get_permissions(const GrimPluginAPI* self) {
    return pluginContext(self).permissions;
}

static bool api_has_permission(const GrimPluginAPI* self, GrimPermission perm) {
    return checkPermission(self, perm);
}

static void api_request_reload(const GrimPluginAPI* self) {
    LOG_DEBUG("PluginAPI", "Plugin requested reload: " + pluginContext(self).name);
    PluginManager::requestReload(self);
}

// ============================================================================
// Utility Implementation
// ============================================================================

static void* api_allocate_memory(const GrimPluginAPI* self, size_t size) {
    return malloc(size);
}

static void api_free_memory(const GrimPluginAPI* self, void* ptr) {
    free(ptr);
}

static const char* api_string_duplicate(const GrimPluginAPI* self, const char* str) {
    if (!str) return nullptr;
    return allocateString(str);
}

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

static const char* api_input_service(const GrimPluginAPI* self, GrimInputOperation operation,
                                     const char* arguments, size_t size) noexcept {
    try {
        auto& owner = const_cast<PluginContext&>(pluginContext(self));
        ActiveCall active(owner.gate, true);
        if (!active.entered || !checkPermission(self, GRIM_PERM_INPUT) || operation > GRIM_INPUT_CLICK_ON ||
            (operation == GRIM_INPUT_CLICK_ON && !checkPermission(self, GRIM_PERM_SYSTEM)) ||
            size > 65536 || (size && !arguments)) return nullptr;
        const std::string copied = size ? std::string(arguments, size) : std::string{};
        CommandResult result;
        try { result = GRIM::Perception::ExecuteInputCommandService(operation, copied); }
        catch (const std::exception& error) {
            return allocateString(nlohmann::json{{"format", "grim-input-result"}, {"version", 1},
                {"exception", error.what()}}.dump());
        } catch (...) {
            return allocateString(nlohmann::json{{"format", "grim-input-result"}, {"version", 1},
                {"exception", nullptr}}.dump());
        }
        return allocateString(nlohmann::json{{"format", "grim-input-result"}, {"version", 1},
            {"result", {{"success", result.success}, {"message", result.message}, {"error_code", result.errorCode},
                {"category", result.category}, {"voice", result.voice}, {"structured_output_json", result.structuredOutputJson},
                {"color_rgb", (uint32_t(result.color.r) << 16) | (uint32_t(result.color.g) << 8) | result.color.b}}}}.dump());
    } catch (...) { return nullptr; }
}

GrimPluginAPI* createPluginAPI(const std::string& plugin_name, const std::string& plugin_path, uint32_t permissions, bool staging, const GrimPluginAPI* replacing) {
    if (plugin_name.empty()) throw std::invalid_argument("Empty plugin identity");
    for (const auto& [id, context] : s_plugin_contexts) {
        if (context->name == plugin_name && &context->api != replacing)
            throw std::invalid_argument("Duplicate plugin identity: " + plugin_name);
    }
    if (replacing && (pluginContext(replacing).name != plugin_name || !staging))
        throw std::invalid_argument("Invalid replacement identity");
    auto context = std::make_unique<PluginContext>();
    context->instance = plugin_name + "#" + std::to_string(s_next_instance++);
    context->panelLease->owner = context->instance;
    context->panelLease->providerName = plugin_name;
    context->name = plugin_name;
    context->path = plugin_path;
    context->permissions = permissions;
    context->staging = staging;
    context->gate->published = !staging;
    auto& api = context->api;
    api.host_context = context.get();
    
    // Fill in API table
    api.api_version = GRIM_API_VERSION;
    
    // Logging
    api.log = api_log;
    api.log_fmt = api_log_fmt;
    
    // Commands
    api.register_tools = api_register_tools;
    api.unregister_command = api_unregister_command;
    api.is_command_registered = api_is_command_registered;
    api.execute_command = api_execute_command;
    
    // Events
    api.subscribe_event = api_subscribe_event;
    api.unsubscribe_event = api_unsubscribe_event;
    api.emit_event = api_emit_event;
    
    // Memory & Context
    api.get_context = api_get_context;
    api.set_context = api_set_context;
    api.delete_context = api_delete_context;
    api.has_context = api_has_context;
    api.store_memory = api_store_memory;
    api.retrieve_memory = api_retrieve_memory;
    api.delete_memory = api_delete_memory;
    
    // NLP & AI
    api.classify_intent = api_classify_intent;
    api.resolve_synonym = api_resolve_synonym;
    api.get_conversation_history = api_get_conversation_history;
    
    // Output & UI
    api.send_response = api_send_response;
    api.send_response_colored = api_send_response_colored;
    api.show_notification = api_show_notification;
    
    // Voice
    api.queue_tts = api_queue_tts;
    api.stop_tts = api_stop_tts;
    api.is_speaking = api_is_speaking;
    
    // Process Management
    api.launch_process = api_launch_process;
    api.is_process_running = api_is_process_running;
    api.kill_process = api_kill_process;
    
    // System Information
    
    // Timers
    api.create_timer = api_create_timer;
    api.cancel_timer = api_cancel_timer;
    
    // Plugin Lifecycle
    api.get_plugin_name = api_get_plugin_name;
    api.get_plugin_path = api_get_plugin_path;
    api.get_permissions = api_get_permissions;
    api.has_permission = api_has_permission;
    api.request_reload = api_request_reload;
    
    // Utilities
    api.allocate_memory = api_allocate_memory;
    api.free_memory = api_free_memory;
    api.string_duplicate = api_string_duplicate;
    api.string_free = api_string_free;
    api.get_user_error_message = api_get_user_error_message;
    api.get_tool_snapshot = api_get_tool_snapshot;
    api.get_alias_snapshot = api_get_alias_snapshot;
    api.refresh_aliases = api_refresh_aliases;
    api.schedule_user_timer = api_schedule_user_timer;
    api.create_local_calendar = api_create_local_calendar;
    api.schedule_service = api_schedule_service;
    api.stage_manual_memory = api_stage_manual_memory;
    api.search_memory = api_search_memory;
    api.flush_memory_record_if_present = api_flush_memory_record_if_present;
    api.query_perception = api_query_perception;
    api.voice_service = api_voice_service;
    api.ui_panel_service = api_ui_panel_service;
    api.input_service = api_input_service;
    api.register_ui_panel = api_register_ui_panel;
    api.get_provider_identity = api_get_provider_identity;
    api.replace_resources = api_replace_resources;
    api.is_provider_current = api_is_provider_current;
    
    auto* result = &api;
    auto instance = context->instance;
    s_plugin_contexts.emplace(instance, std::move(context));
    return result;
}

// Name-based wrappers are for host callers with exactly one loaded generation.
static GrimPluginAPI* uniquePlugin(const std::string& name) {
    GrimPluginAPI* result = nullptr;
    for (const auto& [id, context] : s_plugin_contexts) if (context->name == name) {
        if (result) throw std::invalid_argument("Ambiguous plugin generation");
        result = &context->api;
    }
    if (!result) throw std::invalid_argument("Missing plugin context");
    return result;
}
bool commitPluginTools(const GrimPluginAPI* api, const GrimPluginAPI* replacing) {
    auto& owner = const_cast<PluginContext&>(pluginContext(api));
    if (owner.registration_failed || !owner.staging) return false;
    auto& providers = GRIM::MMO::ProviderRegistry::instance();
    // Handlers may become reachable as soon as the registry swaps its entries,
    // including from listeners. Late registrations must no longer be staged.
    owner.staging = false;
    bool committed;
    std::vector<GRIM::MMO::ResourceDescriptor> resources;
    for (const auto& [id, resource] : owner.pendingResources) resources.push_back(resource);
    try {
        committed = providers.publish(
            {GRIM::MMO::ToolProviderType::Plugin, owner.name, owner.instance, owner.path},
            owner.pending, replacing ? pluginContext(replacing).instance : std::string{}, resources);
    } catch (...) {
        owner.staging = true;
        throw;
    }
    if (!committed) { owner.staging = true; return false; }
    // Queue retirement before replacement creation. Generation-specific leases
    // prevent a later shutdown of the old DLL from removing its replacement.
    if (replacing) GRIM::UI::retireProviderPanels(pluginContext(replacing).panelLease);
    if (!owner.pendingPanels.empty())
        GRIM::UI::activateProviderPanels(owner.panelLease, std::move(owner.pendingPanels));
    if (replacing) {
        const auto& old = pluginContext(replacing);
        std::lock_guard lock(old.gate->mutex);
        old.gate->accepting = false;
    }
    owner.pending.clear();
    owner.pendingResources.clear();
    owner.staging = false;
    { std::lock_guard lock(owner.gate->mutex); owner.gate->published = true; }
    return true;
}
bool commitPluginTools(const std::string& name) { return commitPluginTools(uniquePlugin(name), nullptr); }
void beginPluginShutdown(const GrimPluginAPI* api) {
    auto& owner = const_cast<PluginContext&>(pluginContext(api));
    {
        std::lock_guard lock(owner.gate->mutex);
        owner.gate->accepting = false;
    }
    GRIM::MMO::ProviderRegistry::instance().retire(owner.name, owner.instance);
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
    const auto instance = pluginContext(api).instance;
    std::lock_guard lock(s_resourceMutex);
    for (auto it = s_event_subscriptions.begin(); it != s_event_subscriptions.end();) {
        if (it->second.plugin_name == instance) it = s_event_subscriptions.erase(it);
        else ++it;
    }
    for (auto it = s_timers.begin(); it != s_timers.end();) {
        if (it->second.plugin_name == instance) it = s_timers.erase(it);
        else ++it;
    }
    s_plugin_contexts.erase(instance);
}
void cleanupPluginContext(const std::string& name) { cleanupPluginContext(uniquePlugin(name)); }

void processTimers() {
    std::vector<PluginTimer> ready;
    const auto now = std::chrono::steady_clock::now();
    {
        std::lock_guard lock(s_resourceMutex);
        for (auto it = s_timers.begin(); it != s_timers.end();) {
            auto& timer = it->second;
            if (now < timer.next_fire) { ++it; continue; }
            ready.push_back(timer);
            if (timer.repeat) {
                timer.next_fire = now + std::chrono::milliseconds(timer.interval_ms);
                ++it;
            } else it = s_timers.erase(it);
        }
    }
    for (const auto& timer : ready) {
        ActiveCall active(timer.gate, true);
        if (active.entered) timer.callback(timer.id, timer.user_data);
    }
}
void emitGlobalEvent(const char* event_name, const void* data, size_t data_size) {
    if (event_name) dispatchEvent(event_name, data, data_size);
}

} // namespace PluginAPI
