#pragma once
#include "plugin_api.hpp"
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
 * @param plugin_name Name of the plugin
 * @param plugin_path Full path to the plugin DLL
 * @param permissions Permission bitmask (GrimPermission flags)
 * @return Pointer to the API table (static, don't free)
 */
GrimPluginAPI* createPluginAPI(
    const std::string& plugin_name,
    const std::string& plugin_path,
    uint32_t permissions
);

/**
 * Clean up all resources associated with a plugin
 * Called when plugin is unloaded
 * @param plugin_name Name of the plugin to clean up
 */
void cleanupPluginContext(const std::string& plugin_name);

/**
 * Process timer callbacks (call from main loop)
 */
void processTimers();

/**
 * Emit an event to all subscribers
 * @param event_name Name of the event
 * @param data Event data pointer
 * @param data_size Size of event data
 */
void emitGlobalEvent(const char* event_name, const void* data, size_t data_size);

} // namespace PluginAPI
