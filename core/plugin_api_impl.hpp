#pragma once
#include "plugin_api.hpp"
#include "../MMO/Core/Registries/ProviderRegistry.hpp"
#include <nlohmann/json_fwd.hpp>
#include <string>

// Host lifecycle operations for loaded provider instances.
namespace PluginAPI {

struct LoadedProvider;
std::shared_ptr<LoadedProvider> registerInstance(GRIM::MMO::ProviderDescriptor provider);
GrimPluginHandle instanceHandle(const LoadedProvider& provider);
const GrimPluginAPI& hostAPI();
void validateManifestTools(const nlohmann::json& entries, const GRIM::MMO::ProviderDescriptor& provider);
void registerProvider(LoadedProvider& provider, const nlohmann::json& entries, GrimToolDispatch dispatch);
void enableDispatch(LoadedProvider& provider);
void retireProvider(LoadedProvider& provider);
void releaseInstance(LoadedProvider& provider);

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
