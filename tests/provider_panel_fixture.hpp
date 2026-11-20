#pragma once

#include "MMO/Core/Registries/ProviderRegistry.hpp"
#include "ui/ui_panel_service.hpp"

inline GrimPluginHandle prepareProviderPanelFixture(
    GrimPluginAPI& api, const std::shared_ptr<GRIM::UI::ProviderPanelLease>& lease, bool registered = true) {
    static GrimPluginHandle next = 1;
    const auto handle = next++;
    if (registered) GRIM::UI::registerProviderPanelLease(handle, lease);
    api.string_free = [](GrimPluginHandle, const char* value) { delete[] value; };
    return handle;
}
