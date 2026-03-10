// UISurfaceSpec — validated description of a UI surface.
//
// Every UI surface in GRIM (overlay, popup, modal, toast, tool window,
// inspector) is described by a UISurfaceSpec.  The spec is the input
// contract that the registry validates before a surface becomes live.
//
// Surfaces are created via registry-backed UI tools (ui.create_surface
// etc.) that go through ToolRegistry / ActionPolicy / Training Wheels
// like any other tool invocation.
//
// Thread-safe: value type, no mutable shared state.
//======================================================//
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// SurfaceKind — the category of UI host
// =========================================================
enum class SurfaceKind : uint8_t {
    OverlayPanel = 0,   // persistent side / bottom panel
    Popup        = 1,   // ephemeral floating popup
    Modal        = 2,   // blocks input until dismissed
    Toast        = 3,   // auto-dismiss notification
    ToolWindow   = 4,   // resizable tool / inspector window
    Inspector    = 5    // read-only data inspector
};

// =========================================================
// LifetimePolicy — how long the surface stays alive
// =========================================================
enum class LifetimePolicy : uint8_t {
    Persistent  = 0,   // lives until explicitly destroyed
    SessionOnly = 1,   // destroyed when session ends
    AutoDismiss = 2    // dismissed after timeout
};

// =========================================================
// VisibilityState — current display state
// =========================================================
enum class VisibilityState : uint8_t {
    Hidden  = 0,
    Visible = 1
};

// =========================================================
// InputPolicy — how the surface handles input focus
// =========================================================
enum class InputPolicy : uint8_t {
    PassThrough = 0,   // doesn't capture focus
    Focusable   = 1,   // can receive focus but doesn't block
    Exclusive   = 2    // blocks other input (modal)
};

// =========================================================
// WidgetSpec — one UI element within a surface
// =========================================================
struct WidgetSpec {
    std::string widget_id;
    std::string widget_type;     // "label", "button", "text_input", "progress", "list", "image"
    std::string label;           // display text
    std::string data_binding;    // data source key for dynamic content
    std::string action_binding;  // tool invocation on interaction (e.g. "ui.hide_surface")
};

// =========================================================
// LayoutSpec — how widgets are arranged
// =========================================================
struct LayoutSpec {
    std::string direction = "vertical";  // "vertical", "horizontal", "grid"
    int         columns   = 1;           // for grid layout
    int         padding   = 4;           // px
    int         spacing   = 2;           // px between widgets
};

// =========================================================
// UISurfaceSpec — the full surface description
//
// Built by callers (tools, plugins), validated by the Registry
// before the surface becomes live.
// =========================================================
struct UISurfaceSpec {
    std::string        surface_id;
    SurfaceKind        kind           = SurfaceKind::OverlayPanel;
    std::string        title;
    std::string        host_target;       // e.g. "main_window", "secondary_monitor"
    std::string        monitor_target;    // monitor id (empty = primary)

    LayoutSpec         layout;
    std::vector<WidgetSpec> widgets;

    LifetimePolicy     lifetime       = LifetimePolicy::Persistent;
    VisibilityState    visibility     = VisibilityState::Hidden;
    InputPolicy        input_policy   = InputPolicy::Focusable;

    int                auto_dismiss_ms = 0;  // for AutoDismiss lifetime (ms)

    // Source provenance
    std::string        created_by;        // tool_id or plugin_id that created it
};

// =========================================================
// Validation
// =========================================================

// Returns empty string on success, or an error description.
std::string validateSurfaceSpec(const UISurfaceSpec& spec);

} // namespace GRIM::MMO
