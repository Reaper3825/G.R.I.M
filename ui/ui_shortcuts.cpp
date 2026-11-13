#include "ui/input/ui_input_router.hpp"
#include "ui_shortcuts.hpp"

#include "../core/input/InputBindings.hpp"
#include "logger.hpp"
#include "ui_root.hpp"

namespace GRIM::UI {
namespace {

void togglePanel(const char* actionId, const char* panelKey)
{
    if (!InputBindings::wasPressed(actionId)) {
        return;
    }

    auto panel = UIRoot::get().getPanelByKey(panelKey);
    if (!panel) {
        LOG_ERROR("UIShortcuts", "Panel shortcut '" + std::string(actionId) +
            "' is unavailable: panel '" + panelKey + "' is not active");
        return;
    }
    UIRoot::get().setVisible(panelKey, !panel->isVisible());
}

} // namespace

void processPanelShortcuts()
{
    const auto& input = GRIM::UI::InputRouter::get();
    if (input.focus || input.popup || input.capture) return;
    togglePanel("toggle_console", "console");
    togglePanel("toggle_settings", "settings");
    togglePanel("toggle_training", "training_control");
    togglePanel("toggle_data_hub", "data_hub");
    togglePanel("toggle_storage", "storage");
    togglePanel("toggle_geospatial", "geospatial");
    togglePanel("toggle_physical_environment", "physical_environment");
    togglePanel("toggle_digital_environment", "digital_environment");
}

} // namespace GRIM::UI