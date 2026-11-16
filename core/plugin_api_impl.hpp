#pragma once
#include "plugin_api.hpp"
#include "../MMO/Core/Registries/ProviderRegistry.hpp"
#include <nlohmann/json_fwd.hpp>
#include <string>

// ============================================================================
// Plugin API Implementation Interface
// ============================================================================
// These functions are called by the plugin manager to create and manage
// plugin API instances.
// ============================================================================

namespace PluginAPI {

/**
 * Create a plugin API table for a specific plugin
 * @param provider Provider metadata from the manifest. The host assigns the instance ID.
 * @return Distinct host-owned API table, valid through shutdown until cleanup.
 * Every host call must pass this table as its first argument.
 */
GrimPluginAPI* createPluginAPI(
    GRIM::MMO::ProviderDescriptor provider,
    bool staging = false,
    const GrimPluginAPI* replacing = nullptr
);

std::vector<GRIM::MMO::ToolRegistration> readManifestTools(
    const nlohmann::json& entries, const GrimPluginAPI* api, GrimToolDispatch dispatch);
// Publish the staged provider, tools and initial resources after successful initialization.
bool commitPluginGeneration(const GrimPluginAPI* api, const GrimPluginAPI* replacing = nullptr,
                            const std::vector<GRIM::MMO::ToolRegistration>& tools = {});
bool commitPluginGeneration(const std::string& plugin_name);
// Host lifecycle thread only; never call from inside a plugin callback.
void beginPluginShutdown(const GrimPluginAPI* api);
void beginPluginShutdown(const std::string& plugin_name);
// Remove API context, subscriptions, and timers after shutdown.
void cleanupPluginContext(const GrimPluginAPI* api);
void cleanupPluginContext(const std::string& plugin_name);

/** Dispatch the frame tick to published plugin generations. */
void dispatchSystemTick(double deltaSeconds, double offsetSeconds);

/**
 * Emit an event to all subscribers
 * @param event_name Name of the event
 * @param data Event data pointer
 * @param data_size Size of event data
 */
void emitGlobalEvent(const char* event_name, const void* data, size_t data_size);

} // namespace PluginAPI
