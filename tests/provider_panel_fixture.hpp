#pragma once

#include "ui/ui_panel_service.hpp"
#include <algorithm>

inline void prepareProviderPanelFixture(
    GrimPluginAPI& api, const std::shared_ptr<GRIM::UI::ProviderPanelLease>& lease, bool permitted) {
    lease->uiPermitted = permitted;
    GRIM::UI::registerProviderPanelLease(lease);
    api.host_context = lease.get();
    api.get_provider_identity = [](const GrimPluginAPI* self) -> const char* {
        const auto& owner = *static_cast<const GRIM::UI::ProviderPanelLease*>(self->host_context);
        const auto json = nlohmann::json{{"name", owner.providerName}, {"instance", owner.owner}}.dump();
        auto value = std::make_unique<char[]>(json.size() + 1);
        std::copy_n(json.c_str(), json.size() + 1, value.get());
        return value.release();
    };
    api.string_free = [](const GrimPluginAPI*, const char* value) { delete[] value; };
}
