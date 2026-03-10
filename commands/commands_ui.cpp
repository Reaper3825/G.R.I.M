#include "commands/commands_ui.hpp"
#include "ui/ui_root.hpp"
#include "logger.hpp"
#include "../MMO/UI/UISurfaceRegistry.hpp"
#include <nlohmann/json.hpp>

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

// ================================================================
// MMO UI Surface commands — backed by UISurfaceRegistry
// ================================================================

static GRIM::MMO::SurfaceKind parseSurfaceKind(const std::string& s)
{
    if (s == "overlay_panel") return GRIM::MMO::SurfaceKind::OverlayPanel;
    if (s == "popup")         return GRIM::MMO::SurfaceKind::Popup;
    if (s == "modal")         return GRIM::MMO::SurfaceKind::Modal;
    if (s == "toast")         return GRIM::MMO::SurfaceKind::Toast;
    if (s == "tool_window")   return GRIM::MMO::SurfaceKind::ToolWindow;
    if (s == "inspector")     return GRIM::MMO::SurfaceKind::Inspector;
    throw std::runtime_error("Unknown surface kind: " + s);
}

CommandResult cmdCreateSurface(const std::string& arg)
{
    if (arg.empty()) {
        return {false, "ui.create_surface requires JSON arguments",
                "ERR_MISSING_ARGS", "error", "", Colors::Red};
    }

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(arg);
    } catch (const nlohmann::json::parse_error& e) {
        return {false, std::string("Invalid JSON: ") + e.what(),
                "ERR_INVALID_JSON", "error", "", Colors::Red};
    }

    if (!j.contains("surface_id") || !j.contains("kind") || !j.contains("title")) {
        return {false, "Missing required fields: surface_id, kind, title",
                "ERR_MISSING_FIELDS", "error", "", Colors::Red};
    }

    GRIM::MMO::UISurfaceSpec spec;
    spec.surface_id = j["surface_id"].get<std::string>();
    spec.title      = j["title"].get<std::string>();
    spec.created_by = "ui.create_surface";

    try {
        spec.kind = parseSurfaceKind(j["kind"].get<std::string>());
    } catch (const std::runtime_error& e) {
        return {false, e.what(), "ERR_INVALID_KIND", "error", "", Colors::Red};
    }

    if (j.contains("host_target"))     spec.host_target     = j["host_target"].get<std::string>();
    if (j.contains("auto_dismiss_ms")) spec.auto_dismiss_ms = j["auto_dismiss_ms"].get<int>();

    auto& reg = GRIM::MMO::UISurfaceRegistry::instance();
    std::string err = reg.create(spec);
    if (!err.empty()) {
        return {false, "Failed to create surface: " + err,
                "ERR_SURFACE_CREATE", "error", "", Colors::Red};
    }

    std::string msg = "Created surface '" + spec.surface_id + "' (" + j["kind"].get<std::string>() + ")";
    LOG_DEBUG("UI", msg);
    return {true, msg, "ERR_NONE", "ui", msg, Colors::Green};
}

CommandResult cmdShowSurface(const std::string& arg)
{
    std::string surface_id = arg;

    // Accept JSON or plain string
    if (!arg.empty() && arg.front() == '{') {
        try {
            auto j = nlohmann::json::parse(arg);
            if (j.contains("surface_id"))
                surface_id = j["surface_id"].get<std::string>();
        } catch (...) {}
    }

    if (surface_id.empty()) {
        return {false, "ui.show_surface requires surface_id",
                "ERR_MISSING_ARGS", "error", "", Colors::Red};
    }

    auto& reg = GRIM::MMO::UISurfaceRegistry::instance();
    if (!reg.show(surface_id)) {
        return {false, "Surface '" + surface_id + "' not found",
                "ERR_UI_NOT_FOUND", "error", "", Colors::Red};
    }

    std::string msg = "Surface '" + surface_id + "' shown";
    LOG_DEBUG("UI", msg);
    return {true, msg, "ERR_NONE", "ui", msg, Colors::Green};
}

CommandResult cmdHideSurface(const std::string& arg)
{
    std::string surface_id = arg;

    if (!arg.empty() && arg.front() == '{') {
        try {
            auto j = nlohmann::json::parse(arg);
            if (j.contains("surface_id"))
                surface_id = j["surface_id"].get<std::string>();
        } catch (...) {}
    }

    if (surface_id.empty()) {
        return {false, "ui.hide_surface requires surface_id",
                "ERR_MISSING_ARGS", "error", "", Colors::Red};
    }

    auto& reg = GRIM::MMO::UISurfaceRegistry::instance();
    if (!reg.hide(surface_id)) {
        return {false, "Surface '" + surface_id + "' not found",
                "ERR_UI_NOT_FOUND", "error", "", Colors::Red};
    }

    std::string msg = "Surface '" + surface_id + "' hidden";
    LOG_DEBUG("UI", msg);
    return {true, msg, "ERR_NONE", "ui", msg, Colors::Green};
}

CommandResult cmdDestroySurface(const std::string& arg)
{
    std::string surface_id = arg;

    if (!arg.empty() && arg.front() == '{') {
        try {
            auto j = nlohmann::json::parse(arg);
            if (j.contains("surface_id"))
                surface_id = j["surface_id"].get<std::string>();
        } catch (...) {}
    }

    if (surface_id.empty()) {
        return {false, "ui.destroy_surface requires surface_id",
                "ERR_MISSING_ARGS", "error", "", Colors::Red};
    }

    auto& reg = GRIM::MMO::UISurfaceRegistry::instance();
    if (!reg.destroy(surface_id)) {
        return {false, "Surface '" + surface_id + "' not found",
                "ERR_UI_NOT_FOUND", "error", "", Colors::Red};
    }

    std::string msg = "Surface '" + surface_id + "' destroyed";
    LOG_DEBUG("UI", msg);
    return {true, msg, "ERR_NONE", "ui", msg, Colors::Green};
}
