// DynamicSurfacePanel — UIPanel subclass for tool-created UI surfaces.
//
// Wraps a UISurfaceSpec and renders it using OverlayRenderer.
// Created/managed by UISurfaceRendererBridge in response to
// UISurfaceRegistry events.
//======================================================//
#pragma once

#include "ui_panel.hpp"
#include "../MMO/UI/UISurfaceSpec.hpp"

class OverlayRenderer;

class DynamicSurfacePanel : public UIPanel {
public:
    explicit DynamicSurfacePanel(const GRIM::MMO::UISurfaceSpec& spec);

    void updateSpec(const GRIM::MMO::UISurfaceSpec& spec);

    bool drawOverlay(OverlayRenderer& renderer) override;

    const std::string& surfaceId() const { return surface_id_; }

private:
    std::string surface_id_;
    GRIM::MMO::UISurfaceSpec spec_;
};
