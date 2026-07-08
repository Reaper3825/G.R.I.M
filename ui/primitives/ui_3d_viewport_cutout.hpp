#pragma once

#include "widget.hpp"

class UI3DViewPortCutOut : public Widget {
public:
    UI3DViewPortCutOut() = default;
    UI3DViewPortCutOut(const Vec2& pos, const Vec2& sz);

    void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) override;
};