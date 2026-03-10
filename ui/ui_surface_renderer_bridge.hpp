// UISurfaceRendererBridge — connects UISurfaceRegistry events to UIRoot.
//
// Listens for SurfaceChangeCallback events and creates/updates/destroys
// DynamicSurfacePanel instances on UIRoot accordingly.
// All UIRoot mutations are routed through postTask for thread safety.
//======================================================//
#pragma once

namespace GRIM::MMO { class UISurfaceRegistry; }

class UISurfaceRendererBridge {
public:
    // Register the bridge callback on UISurfaceRegistry.
    // Call once after both UIRoot::init() and UISurfaceRegistry are live.
    static void install();

private:
    UISurfaceRendererBridge() = delete;
};
