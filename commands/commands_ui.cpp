#include "commands/commands_ui.hpp"
#include "ui/ui_root.hpp"
#include "logger.hpp"

// Toggle overlay console panel
CommandResult cmdToggleOverlayConsole([[maybe_unused]] const std::string& arg)
{
    auto panel = UIRoot::get().getPanel("Console");
    if (panel)
    {
        bool newState = !panel->isVisible();
        panel->setVisible(newState);
        
        std::string msg = newState ? "Overlay console shown" : "Overlay console hidden";
        LOG_DEBUG("Command", msg);
        
        return {
            true,
            msg,
            "ERR_NONE",
            "routine",
            msg,
            Colors::Green
        };
    }
    
    return {
        false,
        "Overlay console panel not found",
        "ERR_UI_NOT_FOUND",
        "error",
        "Console panel not found",
        Colors::Red
    };
}

// Toggle settings panel
CommandResult cmdToggleSettings([[maybe_unused]] const std::string& arg)
{
    auto panel = UIRoot::get().getPanel("Settings");
    if (!panel)
    {
        // Try alternative name
        panel = UIRoot::get().getPanel("GRIM Settings");
    }
    
    if (panel)
    {
        bool newState = !panel->isVisible();
        panel->setVisible(newState);
        
        std::string msg = newState ? "Settings panel shown" : "Settings panel hidden";
        LOG_DEBUG("Command", msg);
        
        return {
            true,
            msg,
            "ERR_NONE",
            "routine",
            msg,
            Colors::Green
        };
    }
    
    return {
        false,
        "Settings panel not found",
        "ERR_UI_NOT_FOUND",
        "error",
        "Settings panel not found",
        Colors::Red
    };
}
