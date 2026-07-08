#include "ui_geospatial_panel.hpp"

#include "overlay_renderer.hpp"
#include "ui_theme.hpp"

#include <utility>

namespace {
    constexpr float kPad = 16.0f;
    constexpr float kButtonH = 28.0f;
    constexpr float kStatusH = 92.0f;
    constexpr float kControlGap = 8.0f;
}

UIGeoSpatialPanel::UIGeoSpatialPanel()
    : UIPanel("GeoSpatial", true)
{
    position = { 260.0f, 220.0f };
    size = { 980.0f, 680.0f };
    setBackground(UITheme::Colors::PanelBg);
    setBorder(UITheme::Colors::DividerLine);

    viewport_ = std::make_shared<UI3DViewport>();

    init_btn_ = std::make_shared<UIButton>(" Init Cesium ", [this]() {
        if (!controller_) { setUiStatus("No GeoSpatial controller attached"); return; }
        controller_->requestInitializeCesium();
    });
    reload_btn_ = std::make_shared<UIButton>(" Reload Tiles ", [this]() {
        if (!controller_) { setUiStatus("No GeoSpatial controller attached"); return; }
        controller_->requestReloadTileset();
    });
    home_btn_ = std::make_shared<UIButton>(" Home ", [this]() {
        if (!controller_) { setUiStatus("No GeoSpatial controller attached"); return; }
        controller_->requestRecenterHome();
    });
    reset_camera_btn_ = std::make_shared<UIButton>(" Reset Camera ", [this]() {
        if (!controller_) { setUiStatus("No GeoSpatial controller attached"); return; }
        controller_->requestResetCamera();
    });
    terrain_btn_ = std::make_shared<UIButton>(" Terrain ", [this]() {
        if (!controller_) { setUiStatus("No GeoSpatial controller attached"); return; }
        controller_->requestToggleTerrain();
    });
    imagery_btn_ = std::make_shared<UIButton>(" Imagery ", [this]() {
        if (!controller_) { setUiStatus("No GeoSpatial controller attached"); return; }
        controller_->requestToggleImagery();
    });

    init_btn_->setSize(110.0f, kButtonH);
    reload_btn_->setSize(115.0f, kButtonH);
    home_btn_->setSize(80.0f, kButtonH);
    reset_camera_btn_->setSize(125.0f, kButtonH);
    terrain_btn_->setSize(90.0f, kButtonH);
    imagery_btn_->setSize(90.0f, kButtonH);
}

void UIGeoSpatialPanel::setController(GRIM::GeoSpatial::GeoSpatialPanelController* controller)
{
    controller_ = controller;
    refreshSnapshot();
}

void UIGeoSpatialPanel::setViewportAttachment(std::shared_ptr<UI3DViewportAttachment> attachment)
{
    if (!viewport_)
        return;

    if (attachment) {
        viewport_->attachViewport(std::move(attachment));
    } else {
        viewport_->detachViewport();
    }
}

void UIGeoSpatialPanel::setVisible(bool visible)
{
    UIPanel::setVisible(visible);

    if (viewport_) {
        if (visible)
            layoutControls();
        viewport_->syncViewportGeometry(position, visible && !isMinimized());
    }
}

void UIGeoSpatialPanel::update(const InputState& input, float dt)
{
    UIPanel::update(input, dt);

    layoutControls();
    refreshSnapshot();

    const bool parentVisible = isVisible() && !isMinimized();
    if (viewport_)
        viewport_->syncViewportGeometry(position, parentVisible);

    if (!parentVisible)
        return;

    if (init_btn_) init_btn_->update(input, dt);
    if (reload_btn_) reload_btn_->update(input, dt);
    if (home_btn_) home_btn_->update(input, dt);
    if (reset_camera_btn_) reset_camera_btn_->update(input, dt);
    if (terrain_btn_) terrain_btn_->update(input, dt);
    if (imagery_btn_) imagery_btn_->update(input, dt);
}

bool UIGeoSpatialPanel::drawOverlay(OverlayRenderer& renderer)
{
    if (!UIPanel::drawOverlay(renderer)) {
        if (viewport_)
            viewport_->syncViewportGeometry(position, false);
        return false;
    }

    layoutControls();

    const float headerY = position.y + titleBarHeight + kPad;
    const float viewportY = headerY + 44.0f;
    const float computedViewportH = size.y - titleBarHeight - kPad * 3.0f - 44.0f - kStatusH;
    const float viewportH = computedViewportH > 80.0f ? computedViewportH : 80.0f;
    const float viewportW = size.x - kPad * 2.0f;

    renderer.drawText({position.x + kPad, headerY},
                      "Cesium Native Viewport Target",
                      UITheme::Colors::TextHeader);
    renderer.drawText({position.x + kPad, headerY + 20.0f},
                      "UI shell only - geospatial state and Cesium ownership attach through controller/viewport hooks.",
                      UITheme::Colors::TextSecondary);

    renderer.drawRoundedBorder({position.x + kPad - 1.0f, viewportY - 1.0f},
                               {viewportW + 2.0f, viewportH + 2.0f},
                               UITheme::Colors::BorderMedium,
                               UITheme::Sizes::SmallRadius);

    if (viewport_)
        viewport_->drawOverlay(renderer, position);

    const float statusY = viewportY + viewportH + kPad;
    renderer.drawRoundedRect({position.x + kPad, statusY},
                             {viewportW, kStatusH},
                             UITheme::Colors::ContentAreaBg,
                             UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder({position.x + kPad, statusY},
                               {viewportW, kStatusH},
                               UITheme::Colors::BorderSubtle,
                               UITheme::Sizes::SmallRadius);

    renderer.drawText({position.x + kPad + 12.0f, statusY + 10.0f},
                      "Status: " + snapshot_.cesium_status,
                      snapshot_.cesium_ready ? UITheme::Colors::Success : UITheme::Colors::Warning);
    renderer.drawText({position.x + kPad + 12.0f, statusY + 30.0f},
                      "Viewport: " + std::string(snapshot_.viewport_ready ? "attached" : "waiting")
                          + " | Tileset: " + snapshot_.active_tileset,
                      UITheme::Colors::TextSecondary);
    renderer.drawText({position.x + kPad + 12.0f, statusY + 50.0f},
                      "Camera: " + snapshot_.camera_mode
                          + " | Terrain: " + std::string(snapshot_.terrain_enabled ? "on" : "off")
                          + " | Imagery: " + std::string(snapshot_.imagery_enabled ? "on" : "off"),
                      UITheme::Colors::TextSecondary);
    renderer.drawText({position.x + kPad + 12.0f, statusY + 70.0f},
                      ui_status_,
                      UITheme::Colors::TextMuted);

    if (init_btn_) init_btn_->drawOverlay(renderer, position);
    if (reload_btn_) reload_btn_->drawOverlay(renderer, position);
    if (home_btn_) home_btn_->drawOverlay(renderer, position);
    if (reset_camera_btn_) reset_camera_btn_->drawOverlay(renderer, position);
    if (terrain_btn_) terrain_btn_->drawOverlay(renderer, position);
    if (imagery_btn_) imagery_btn_->drawOverlay(renderer, position);

    renderer.popClipRect();
    return true;
}

bool UIGeoSpatialPanel::shouldPassThroughAt(float x, float y) const
{
    return viewport_ && viewport_->containsScreenPoint(x, y);
}

void UIGeoSpatialPanel::collectPassThroughRects(std::vector<PanelRect>& rects) const
{
    if (!viewport_)
        return;

    UI3DViewportGeometry geometry = viewport_->getGeometry();
    if (!geometry.visible)
        return;

    rects.push_back({geometry.logicalOrigin, geometry.logicalSize});
}

void UIGeoSpatialPanel::refreshSnapshot()
{
    if (controller_) {
        snapshot_ = controller_->snapshot();
        return;
    }

    snapshot_ = GRIM::GeoSpatial::GeoSpatialPanelSnapshot{};
}

void UIGeoSpatialPanel::layoutControls()
{
    const float buttonY = position.y + titleBarHeight + kPad + 4.0f;
    float x = position.x + size.x - kPad;

    auto placeRight = [&](const std::shared_ptr<UIButton>& button) {
        if (!button)
            return;
        Vec2 sz = button->getSize();
        x -= sz.x;
        button->setPosition(x, buttonY);
        x -= kControlGap;
    };

    placeRight(imagery_btn_);
    placeRight(terrain_btn_);
    placeRight(reset_camera_btn_);
    placeRight(home_btn_);
    placeRight(reload_btn_);
    placeRight(init_btn_);

    if (viewport_) {
        const float headerY = position.y + titleBarHeight + kPad;
        const float viewportY = headerY + 44.0f;
        const float viewportH = size.y - titleBarHeight - kPad * 3.0f - 44.0f - kStatusH;
        viewport_->setPosition(kPad, viewportY - position.y);
        viewport_->setSize(size.x - kPad * 2.0f, viewportH > 80.0f ? viewportH : 80.0f);
    }
}

void UIGeoSpatialPanel::setUiStatus(const std::string& status)
{
    ui_status_ = status;
}