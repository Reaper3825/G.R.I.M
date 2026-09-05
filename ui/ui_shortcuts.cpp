#include "ui_shortcuts.hpp"

#include "../core/input/InputBindings.hpp"
#include "ui_root.hpp"

#include <stdexcept>

namespace GRIM::UI {
namespace {

void togglePanel(const char* actionId, const char* panelName)
{
    if (!InputBindings::wasPressed(actionId)) {
        return;
    }

    auto panel = UIRoot::get().getPanel(panelName);
    if (!panel) {
        throw std::runtime_error(
            "Panel shortcut '" + std::string(actionId) +
            "' references missing panel '" + panelName + "'");
    }
    UIRoot::get().setVisible(panelName, !panel->isVisible());
}

} // namespace

void processPanelShortcuts()
{
    togglePanel("toggle_console", "Console");
    togglePanel("toggle_settings", "Settings");
    togglePanel("toggle_training", "GRIM-text Training Control");
    togglePanel("toggle_data_hub", "DataHub");
    togglePanel("toggle_storage", "Shared Storage");
    togglePanel("toggle_geospatial", "GeoSpatial");
    togglePanel("toggle_physical_environment", "Physical Environment");
    togglePanel("toggle_digital_environment", "Digital Environment");
}

} // namespace GRIM::UI