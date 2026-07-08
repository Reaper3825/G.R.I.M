#include "ui_3d_viewport_cutout.hpp"

#include "../overlay_renderer.hpp"

UI3DViewPortCutOut::UI3DViewPortCutOut(const Vec2& pos, const Vec2& sz)
{
    setPosition(pos);
    setSize(sz);
}

void UI3DViewPortCutOut::drawOverlay(OverlayRenderer& renderer, const Vec2&)
{
    if (!isVisible())
        return;

    renderer.clearRect(position, size);
}