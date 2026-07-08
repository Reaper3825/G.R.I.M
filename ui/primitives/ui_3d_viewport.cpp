#include "ui_3d_viewport.hpp"

#include "../overlay_renderer.hpp"

#include <cmath>
#include <stdexcept>

UI3DViewport::UI3DViewport(const Vec2& localPos, const Vec2& logicalSize, float scale)
{
    setPosition(localPos);
    setSize(logicalSize);
    setScale(scale);
}

void UI3DViewport::setScale(float scale)
{
    if (scale <= 0.0f)
        throw std::runtime_error("UI3DViewport::setScale requires scale > 0");

    scale_ = scale;
}

void UI3DViewport::setGeometryChangedCallback(GeometryChangedCallback callback)
{
    onGeometryChanged_ = std::move(callback);
}

void UI3DViewport::attachViewport(std::shared_ptr<UI3DViewportAttachment> attachment)
{
    if (!attachment)
        throw std::runtime_error("UI3DViewport::attachViewport requires a valid attachment");

    if (attachment_ && attachment_ != attachment)
        attachment_->onViewportDetached();

    attachment_ = std::move(attachment);

    if (haveGeometry_)
        attachment_->onViewportGeometryChanged(geometry_);
}

void UI3DViewport::detachViewport()
{
    if (!attachment_)
        return;

    attachment_->onViewportDetached();
    attachment_.reset();
}

void UI3DViewport::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos)
{
    syncViewportGeometry(panelPos, true);

    if (!geometry_.visible)
        return;

    renderer.clearRect(geometry_.logicalOrigin, geometry_.logicalSize);
}

bool UI3DViewport::containsScreenPoint(float x, float y) const
{
    if (!geometry_.visible)
        return false;

    return x >= geometry_.logicalOrigin.x
        && x <= geometry_.logicalOrigin.x + geometry_.logicalSize.x
        && y >= geometry_.logicalOrigin.y
        && y <= geometry_.logicalOrigin.y + geometry_.logicalSize.y;
}

void UI3DViewport::syncViewportGeometry(const Vec2& panelPos, bool parentVisible)
{
    UI3DViewportGeometry next = computeGeometry(panelPos);
    next.visible = parentVisible && isVisible();
    publishGeometryIfChanged(next);
}

UI3DViewportGeometry UI3DViewport::computeGeometry(const Vec2& panelPos) const
{
    UI3DViewportGeometry geometry;
    geometry.logicalOrigin = {panelPos.x + position.x, panelPos.y + position.y};
    geometry.logicalSize = size;
    geometry.scale = scale_;
    geometry.visible = isVisible();

    geometry.pixelX = static_cast<int>(std::lround(geometry.logicalOrigin.x * scale_));
    geometry.pixelY = static_cast<int>(std::lround(geometry.logicalOrigin.y * scale_));
    geometry.pixelWidth = static_cast<int>(std::lround(size.x * scale_));
    geometry.pixelHeight = static_cast<int>(std::lround(size.y * scale_));

    if (geometry.pixelWidth < 0 || geometry.pixelHeight < 0)
        throw std::runtime_error("UI3DViewport geometry has negative size");

    return geometry;
}

void UI3DViewport::publishGeometryIfChanged(const UI3DViewportGeometry& geometry)
{
    if (haveGeometry_ && sameGeometry(geometry_, geometry))
        return;

    geometry_ = geometry;
    haveGeometry_ = true;

    if (onGeometryChanged_)
        onGeometryChanged_(geometry_);

    if (attachment_)
        attachment_->onViewportGeometryChanged(geometry_);
}

bool UI3DViewport::sameGeometry(const UI3DViewportGeometry& a, const UI3DViewportGeometry& b)
{
    return a.pixelX == b.pixelX
        && a.pixelY == b.pixelY
        && a.pixelWidth == b.pixelWidth
        && a.pixelHeight == b.pixelHeight
        && a.visible == b.visible
        && a.scale == b.scale;
}