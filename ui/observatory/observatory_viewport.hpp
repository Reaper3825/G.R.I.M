#pragma once
#include "observatory_report.hpp"
#include "ui/primitives/ui_3d_viewport.hpp"
#include <optional>

// Owns the native window, render registration, camera and picking state.
class ObservatoryViewport {
public:
    struct Selection { unsigned layer; int tokenId; };
    ObservatoryViewport();
    ~ObservatoryViewport();
    void setReport(std::shared_ptr<const GRIM::Observatory::Report> report, size_t position,
                   unsigned layer, int tokenId);
    void draw(OverlayRenderer& renderer, Vec2 origin, Vec2 size);
    void hide();
    void resetCamera();
    std::optional<Selection> takeSelection();
    std::string status() const;
    UI3DViewportGeometry geometry() const { return viewport_.getGeometry(); }
private:
    struct State;
    std::shared_ptr<State> state_;
    UI3DViewport viewport_;
    static void renderAll(uint32_t);
};
