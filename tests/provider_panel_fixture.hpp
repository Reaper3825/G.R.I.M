#pragma once

#include "MMO/Core/Registries/ProviderRegistry.hpp"
#include "ui/ui_panel_service.hpp"

inline GrimPluginHandle prepareProviderPanelFixture(
    GrimPluginAPI& api, const std::shared_ptr<GRIM::UI::ProviderPanelLease>& lease, bool permitted) {
    static GrimPluginHandle next = 1;
    lease->uiPermitted = permitted;
    const auto handle = next++;
    GRIM::UI::registerProviderPanelLease(handle, lease);
    api.string_free = [](GrimPluginHandle, const char* value) { delete[] value; };
    return handle;
}
