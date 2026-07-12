#include "ui_geospatial_panel.hpp"

#include "core/input_parser.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
    constexpr float kPad = 16.0f;
    constexpr float kButtonH = 28.0f;
    constexpr float kStatusH = 92.0f;
    constexpr float kControlGap = 8.0f;
    constexpr float kSidePanelW = 360.0f;
    constexpr float kFieldH = 26.0f;

    std::string formatCoordinate(double value, int precision)
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(precision) << value;
        return stream.str();
    }

    double parseCoordinate(const std::string& text, const char* field)
    {
        size_t parsedLength = 0;
        const double value = std::stod(text, &parsedLength);
        if (parsedLength != text.size())
            throw std::runtime_error(std::string(field) + " must be a number");
        return value;
    }

    uint32_t previewColor(const std::string& color)
    {
        if (color.size() != 7 || color.front() != '#')
            return UITheme::Colors::Warning;
        try {
            return 0xFF000000u | static_cast<uint32_t>(std::stoul(color.substr(1), nullptr, 16));
        } catch (const std::exception&) {
            return UITheme::Colors::Warning;
        }
    }
}

UIGeoSpatialPanel::UIGeoSpatialPanel()
    : UIPanel("GeoSpatial", true)
{
    position = { 260.0f, 220.0f };
    size = { 1180.0f, 920.0f };
    setBackground(UITheme::Colors::PanelBg);
    setBorder(UITheme::Colors::DividerLine);

    viewport_ = std::make_shared<UI3DViewport>();

    init_btn_ = std::make_shared<UIButton>(" Init Cesium ", [this]() {
        if (!controller_) return;
        controller_->requestInitializeCesium();
    });
    reload_btn_ = std::make_shared<UIButton>(" Reload Tiles ", [this]() {
        if (!controller_) return;
        controller_->requestReloadTileset();
    });
    home_btn_ = std::make_shared<UIButton>(" Home ", [this]() {
        if (!controller_) return;
        controller_->requestRecenterHome();
    });
    reset_camera_btn_ = std::make_shared<UIButton>(" Reset Camera ", [this]() {
        if (!controller_) return;
        controller_->requestResetCamera();
    });

    init_btn_->setSize(110.0f, kButtonH);
    reload_btn_->setSize(115.0f, kButtonH);
    home_btn_->setSize(80.0f, kButtonH);
    reset_camera_btn_->setSize(125.0f, kButtonH);

    lod_override_toggle_ = std::make_shared<UIToggle>("LOD Override", false, [this](bool enabled) {
        if (controller_)
            controller_->requestSetLodOverrideEnabled(enabled);
    });
    lod_level_slider_ = std::make_shared<UISlider>("LOD Target", 0.0f, 20.0f, 6.0f, [this](float level) {
        if (controller_)
            controller_->requestSetLodOverrideLevel(static_cast<int>(std::lround(level)));
    }, 1.0f);

    group_dropdown_ = std::make_shared<UIDropdown>("Group", std::vector<std::string>{"<new group>"}, 0,
        [this](int index, const std::string& item) {
            if (index == 0) clearGroupEditor();
            else selectGroup(item);
        });
    group_id_input_ = std::make_shared<UIInputBox>(&group_id_buffer_);
    group_id_input_->setPlaceholder("unique-group-id");
    group_name_input_ = std::make_shared<UIInputBox>(&group_name_buffer_);
    group_name_input_->setPlaceholder("Group name");
    group_color_input_ = std::make_shared<UIInputBox>(&group_color_buffer_);
    group_color_input_->setPlaceholder("#RRGGBB");

    new_group_btn_ = std::make_shared<UIButton>("New", [this]() { clearGroupEditor(); });
    save_group_btn_ = std::make_shared<UIButton>("Apply", [this]() {
        if (!controller_) return;
        const bool creating = selected_group_id_.empty();
        const size_t previousCount = snapshot_.groups.size();
        controller_->requestUpsertGroup(selected_group_id_, group_name_buffer_, group_color_buffer_);
        refreshSnapshot();
        if (creating && snapshot_.groups.size() == previousCount + 1)
            selectGroup(snapshot_.groups.back().id);
    });
    toggle_group_btn_ = std::make_shared<UIButton>("Hide", [this]() {
        if (controller_ && !selected_group_id_.empty())
            controller_->requestToggleGroupVisibility(selected_group_id_);
    });
    remove_group_btn_ = std::make_shared<UIButton>("Delete + Objects", [this]() {
        if (controller_ && !selected_group_id_.empty())
            controller_->requestRemoveGroup(selected_group_id_);
        clearGroupEditor();
        clearPointEditor();
    });

    point_dropdown_ = std::make_shared<UIDropdown>("Object", std::vector<std::string>{"<new object>"}, 0,
        [this](int index, const std::string& item) {
            if (index == 0) clearPointEditor();
            else if (findPoint(item)) selectPoint(item);
            else if (findArea(item)) selectArea(item);
            else throw std::runtime_error("GeoSpatial object dropdown selected an unknown entity");
        });
    point_group_dropdown_ = std::make_shared<UIDropdown>("In group", std::vector<std::string>{"<create group>"}, 0,
        [](int, const std::string&) {});
    geometry_dropdown_ = std::make_shared<UIDropdown>(
        "Geometry", std::vector<std::string>{"Point", "Cube Area", "Sphere Area"}, 0,
        [this](int index, const std::string&) {
            GRIM::GeoSpatial::GeoSpatialGeometryKind requestedKind;
            if (index == 0)
                requestedKind = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
            else if (index == 1)
                requestedKind = GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea;
            else if (index == 2)
                requestedKind = GRIM::GeoSpatial::GeoSpatialGeometryKind::SphereArea;
            else
                throw std::runtime_error("GeoSpatial geometry dropdown returned an invalid index");

            if (!selected_point_id_.empty() || !selected_area_id_.empty()) {
                const int currentIndex = draft_geometry_kind_ == GRIM::GeoSpatial::GeoSpatialGeometryKind::Point
                    ? 0
                    : (draft_geometry_kind_ == GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea ? 1 : 2);
                geometry_dropdown_->setSelectedIndex(currentIndex);
                editor_status_ = "Create a new object to choose a different geometry type.";
                return;
            }
            draft_geometry_kind_ = requestedKind;
        });
    point_id_input_ = std::make_shared<UIInputBox>(&point_id_buffer_);
    point_id_input_->setPlaceholder("unique-point-id");
    point_name_input_ = std::make_shared<UIInputBox>(&point_name_buffer_);
    point_name_input_->setPlaceholder("Point name");
    point_color_input_ = std::make_shared<UIInputBox>(&point_color_buffer_);
    point_color_input_->setPlaceholder("#RRGGBB");
    longitude_input_ = std::make_shared<UIInputBox>(&longitude_buffer_);
    longitude_input_->setPlaceholder("-180 to 180");
    latitude_input_ = std::make_shared<UIInputBox>(&latitude_buffer_);
    latitude_input_->setPlaceholder("-90 to 90");
    height_input_ = std::make_shared<UIInputBox>(&height_buffer_);
    height_input_->setPlaceholder("meters");
    area_size_input_ = std::make_shared<UIInputBox>(&area_size_buffer_);
    area_size_input_->setPlaceholder("edge / diameter meters");
    area_opacity_input_ = std::make_shared<UIInputBox>(&area_opacity_buffer_);
    area_opacity_input_->setPlaceholder("0.05 to 0.95");

    use_pick_btn_ = std::make_shared<UIButton>("Use Last Pick", [this]() {
        if (!snapshot_.picked_location_valid) {
            editor_status_ = "Click the globe once to pick a WGS84 location.";
            return;
        }
        longitude_buffer_ = formatCoordinate(snapshot_.picked_longitude_degrees, 6);
        latitude_buffer_ = formatCoordinate(snapshot_.picked_latitude_degrees, 6);
        height_buffer_ = formatCoordinate(snapshot_.picked_height_meters, 1);
        longitude_input_->setText(longitude_buffer_);
        latitude_input_->setText(latitude_buffer_);
        height_input_->setText(height_buffer_);
        editor_status_ = "Copied the last globe pick into the geometry draft.";
    });
    new_point_btn_ = std::make_shared<UIButton>("New", [this]() { clearPointEditor(); });
    save_point_btn_ = std::make_shared<UIButton>("Apply", [this]() { savePointDraft(); });
    toggle_point_btn_ = std::make_shared<UIButton>("Hide", [this]() {
        if (controller_ && !selected_area_id_.empty())
            controller_->requestToggleAreaVisibility(selected_area_id_);
        else if (controller_ && !selected_point_id_.empty())
            controller_->requestTogglePointVisibility(selected_point_id_);
    });
    remove_point_btn_ = std::make_shared<UIButton>("Delete", [this]() {
        if (controller_ && !selected_area_id_.empty())
            controller_->requestRemoveArea(selected_area_id_);
        else if (controller_ && !selected_point_id_.empty())
            controller_->requestRemovePoint(selected_point_id_);
        clearPointEditor();
    });
    save_catalog_btn_ = std::make_shared<UIButton>("Save Data", [this]() {
        if (controller_) controller_->requestSavePointCatalog();
    });

    const std::vector<std::shared_ptr<UIButton>> editorButtons = {
        new_group_btn_, save_group_btn_, toggle_group_btn_, remove_group_btn_, use_pick_btn_,
        new_point_btn_, save_point_btn_, toggle_point_btn_, remove_point_btn_, save_catalog_btn_
    };
    for (const std::shared_ptr<UIButton>& button : editorButtons)
        button->setSize(74.0f, kButtonH);
    remove_group_btn_->setSize(118.0f, kButtonH);
    use_pick_btn_->setSize(112.0f, kButtonH);
    save_catalog_btn_->setSize(100.0f, kButtonH);
}

void UIGeoSpatialPanel::setController(GRIM::GeoSpatial::GeoSpatialPanelController* controller)
{
    controller_ = controller;
    refreshSnapshot();
    syncEditors();
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
    syncEditors();

    const bool parentVisible = isVisible() && !isMinimized();
    if (viewport_)
        viewport_->syncViewportGeometry(position, parentVisible);

    if (!parentVisible)
        return;

    if (init_btn_) init_btn_->update(input, dt);
    if (reload_btn_) reload_btn_->update(input, dt);
    if (home_btn_) home_btn_->update(input, dt);
    if (reset_camera_btn_) reset_camera_btn_->update(input, dt);

    updateLayerOutliner(input);

    if (show_layer_outliner_) {
        lod_override_toggle_->update(input, dt);
        lod_level_slider_->update(input, dt);
        return;
    }

    const std::vector<std::shared_ptr<Widget>> editorWidgets = {
        group_dropdown_, group_id_input_, group_name_input_, group_color_input_,
        new_group_btn_, save_group_btn_, toggle_group_btn_, remove_group_btn_,
        point_dropdown_, point_group_dropdown_, geometry_dropdown_, point_id_input_, point_name_input_, point_color_input_,
        longitude_input_, latitude_input_, height_input_, area_size_input_, area_opacity_input_, use_pick_btn_,
        new_point_btn_, save_point_btn_, toggle_point_btn_, remove_point_btn_, save_catalog_btn_
    };
    for (const std::shared_ptr<Widget>& widget : editorWidgets)
        if (widget) widget->update(input, dt);
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
    const float viewportW = size.x - kPad * 3.0f - kSidePanelW;
    const float sideX = position.x + size.x - kPad - kSidePanelW;

    renderer.drawText({position.x + kPad, headerY},
                      "Cesium Native Viewport",
                      UITheme::Colors::TextHeader);
    renderer.drawText({position.x + kPad, headerY + 20.0f},
                      "Click the globe to capture a WGS84 location for the geometry editor.",
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

    renderer.drawRoundedRect({sideX, viewportY},
                             {kSidePanelW, viewportH + kPad + kStatusH},
                             UITheme::Colors::ContentAreaBg,
                             UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder({sideX, viewportY},
                               {kSidePanelW, viewportH + kPad + kStatusH},
                               UITheme::Colors::BorderSubtle,
                               UITheme::Sizes::SmallRadius);
    renderer.drawText({sideX + 12.0f, viewportY + 10.0f}, "Scene Outliner", UITheme::Colors::TextHeader);
    const uint32_t layerTabColor = show_layer_outliner_ ? UITheme::Colors::TextPrimary : UITheme::Colors::TextMuted;
    const uint32_t objectTabColor = show_layer_outliner_ ? UITheme::Colors::TextMuted : UITheme::Colors::TextPrimary;
    renderer.drawText({sideX + 12.0f, viewportY + 34.0f}, "Layers", layerTabColor);
    renderer.drawText({sideX + 92.0f, viewportY + 34.0f}, "Objects", objectTabColor);
    renderer.drawRect({show_layer_outliner_ ? sideX + 12.0f : sideX + 92.0f, viewportY + 53.0f},
                      {52.0f, 2.0f}, UITheme::Colors::Primary);

    if (show_layer_outliner_) {
        lod_override_toggle_->drawOverlay(renderer, position);
        lod_level_slider_->drawOverlay(renderer, position);
        const std::string observedLevel = snapshot_.observed_lod_level >= 0
            ? std::to_string(snapshot_.observed_lod_level)
            : "loading";
        renderer.drawText({sideX + 12.0f, layer_outliner_position_.y - 22.0f},
                          "Observed LOD: " + observedLevel,
                          snapshot_.lod_override_enabled ? UITheme::Colors::Info : UITheme::Colors::TextSecondary);
        drawLayerOutliner(renderer);
    } else {
        renderer.drawText({sideX + 12.0f, viewportY + 66.0f}, "Groups & Geometry", UITheme::Colors::TextHeader);

        auto drawFieldLabel = [&](float y, const char* label) {
            renderer.drawText({sideX + 12.0f, y + 5.0f}, label, UITheme::Colors::TextLabel);
        };
        drawFieldLabel(group_id_input_->getPosition().y, "ID");
        drawFieldLabel(group_name_input_->getPosition().y, "Name");
        drawFieldLabel(group_color_input_->getPosition().y, "Color");
        renderer.drawRoundedRect({sideX + kSidePanelW - 32.0f, group_color_input_->getPosition().y + 4.0f},
                                 {18.0f, 18.0f}, previewColor(group_color_buffer_), 4.0f);
        renderer.drawLine({sideX + 12.0f, point_dropdown_->getPosition().y - 8.0f},
                          {sideX + kSidePanelW - 12.0f, point_dropdown_->getPosition().y - 8.0f},
                          UITheme::Colors::DividerLine);
        drawFieldLabel(point_id_input_->getPosition().y, "ID");
        drawFieldLabel(point_name_input_->getPosition().y, "Name");
        drawFieldLabel(point_color_input_->getPosition().y, "Color");
        drawFieldLabel(longitude_input_->getPosition().y, "Longitude");
        drawFieldLabel(latitude_input_->getPosition().y, "Latitude");
        drawFieldLabel(height_input_->getPosition().y, "Height m");
        drawFieldLabel(area_size_input_->getPosition().y, "Area size m");
        drawFieldLabel(area_opacity_input_->getPosition().y, "Area alpha");
        renderer.drawRoundedRect({sideX + kSidePanelW - 32.0f, point_color_input_->getPosition().y + 4.0f},
                                 {18.0f, 18.0f}, previewColor(point_color_buffer_), 4.0f);

    const std::vector<std::shared_ptr<Widget>> editorWidgets = {
        group_dropdown_, group_id_input_, group_name_input_, group_color_input_,
        new_group_btn_, save_group_btn_, toggle_group_btn_, remove_group_btn_,
        point_dropdown_, point_group_dropdown_, geometry_dropdown_, point_id_input_, point_name_input_, point_color_input_,
        longitude_input_, latitude_input_, height_input_, area_size_input_, area_opacity_input_, use_pick_btn_,
        new_point_btn_, save_point_btn_, toggle_point_btn_, remove_point_btn_, save_catalog_btn_
    };
        for (const std::shared_ptr<Widget>& widget : editorWidgets)
            if (widget) widget->drawOverlay(renderer, position);

        const float catalogStatusY = save_catalog_btn_->getPosition().y + kButtonH + 8.0f;
        renderer.drawText({sideX + 12.0f, catalogStatusY}, snapshot_.point_catalog_status,
                          snapshot_.point_catalog_dirty ? UITheme::Colors::Warning : UITheme::Colors::TextSecondary);
        if (!editor_status_.empty())
            renderer.drawText({sideX + 12.0f, catalogStatusY + 18.0f}, editor_status_, UITheme::Colors::Info);
    }

    if (init_btn_) init_btn_->drawOverlay(renderer, position);
    if (reload_btn_) reload_btn_->drawOverlay(renderer, position);
    if (home_btn_) home_btn_->drawOverlay(renderer, position);
    if (reset_camera_btn_) reset_camera_btn_->drawOverlay(renderer, position);

    if (!show_layer_outliner_) {
        if (group_dropdown_) group_dropdown_->drawExpandedList(renderer, position);
        if (point_group_dropdown_) point_group_dropdown_->drawExpandedList(renderer, position);
        if (geometry_dropdown_) geometry_dropdown_->drawExpandedList(renderer, position);
        if (point_dropdown_) point_dropdown_->drawExpandedList(renderer, position);
    }

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

    placeRight(reset_camera_btn_);
    placeRight(home_btn_);
    placeRight(reload_btn_);
    placeRight(init_btn_);

    if (viewport_) {
        const float headerY = position.y + titleBarHeight + kPad;
        const float viewportY = headerY + 44.0f;
        const float viewportH = size.y - titleBarHeight - kPad * 3.0f - 44.0f - kStatusH;
        viewport_->setPosition(kPad, viewportY - position.y);
        viewport_->setSize(size.x - kPad * 3.0f - kSidePanelW, viewportH > 80.0f ? viewportH : 80.0f);
    }

    const float sideX = position.x + size.x - kPad - kSidePanelW;
    const float fieldX = sideX + 96.0f;
    const float fieldW = kSidePanelW - 110.0f;
    const float viewportY = position.y + titleBarHeight + kPad + 44.0f;
    lod_override_toggle_->setPosition(sideX + 12.0f, viewportY + 62.0f);
    lod_override_toggle_->setSize(kSidePanelW - 24.0f, 40.0f);
    lod_level_slider_->setPosition(sideX + 12.0f, viewportY + 102.0f);
    lod_level_slider_->setSize(kSidePanelW - 24.0f, 48.0f);
    layer_outliner_position_ = {sideX + 12.0f, viewportY + 178.0f};
    layer_outliner_size_ = {kSidePanelW - 24.0f, size.y - (layer_outliner_position_.y - position.y) - kPad};
    float y = viewportY + 92.0f;
    group_dropdown_->setPosition(sideX + 12.0f, y);
    group_dropdown_->setSize(kSidePanelW - 24.0f, 34.0f);
    y += 38.0f;
    for (const std::shared_ptr<UIInputBox>& input : {group_id_input_, group_name_input_, group_color_input_}) {
        input->setPosition(fieldX, y);
        input->setSize(fieldW - (input == group_color_input_ ? 28.0f : 0.0f), kFieldH);
        y += 30.0f;
    }
    new_group_btn_->setPosition(sideX + 12.0f, y);
    save_group_btn_->setPosition(sideX + 92.0f, y);
    toggle_group_btn_->setPosition(sideX + 172.0f, y);
    remove_group_btn_->setPosition(sideX + 252.0f, y);
    y += 44.0f;

    point_dropdown_->setPosition(sideX + 12.0f, y);
    point_dropdown_->setSize(kSidePanelW - 24.0f, 34.0f);
    y += 38.0f;
    point_group_dropdown_->setPosition(sideX + 12.0f, y);
    point_group_dropdown_->setSize(kSidePanelW - 24.0f, 34.0f);
    y += 38.0f;
    geometry_dropdown_->setPosition(sideX + 12.0f, y);
    geometry_dropdown_->setSize(kSidePanelW - 24.0f, 34.0f);
    y += 38.0f;
    for (const std::shared_ptr<UIInputBox>& input : {
             point_id_input_, point_name_input_, point_color_input_, longitude_input_, latitude_input_, height_input_}) {
        input->setPosition(fieldX, y);
        input->setSize(fieldW - (input == point_color_input_ ? 28.0f : 0.0f), kFieldH);
        y += 30.0f;
    }
    for (const std::shared_ptr<UIInputBox>& input : {area_size_input_, area_opacity_input_}) {
        input->setPosition(fieldX, y);
        input->setSize(fieldW, kFieldH);
        y += 30.0f;
    }
    use_pick_btn_->setPosition(sideX + 12.0f, y);
    y += 34.0f;
    new_point_btn_->setPosition(sideX + 12.0f, y);
    save_point_btn_->setPosition(sideX + 92.0f, y);
    toggle_point_btn_->setPosition(sideX + 172.0f, y);
    remove_point_btn_->setPosition(sideX + 252.0f, y);
    y += 38.0f;
    save_catalog_btn_->setPosition(sideX + 12.0f, y);
}

void UIGeoSpatialPanel::updateLayerOutliner(const InputState& input)
{
    const float tabY = layer_outliner_position_.y - 144.0f;
    if (input.mousePressed[0] && input.mousePos.y >= tabY && input.mousePos.y <= tabY + 24.0f) {
        if (input.mousePos.x >= layer_outliner_position_.x &&
            input.mousePos.x <= layer_outliner_position_.x + 68.0f) {
            show_layer_outliner_ = true;
            return;
        }
        if (input.mousePos.x >= layer_outliner_position_.x + 80.0f &&
            input.mousePos.x <= layer_outliner_position_.x + 154.0f) {
            show_layer_outliner_ = false;
            return;
        }
    }

    if (!show_layer_outliner_ || !controller_)
        return;

    constexpr float rowHeight = 58.0f;
    constexpr float checkboxSize = 18.0f;
    const float rowStartY = layer_outliner_position_.y + 24.0f;
    const float checkboxX = layer_outliner_position_.x + 10.0f;
    const float opacityBarX = layer_outliner_position_.x + 38.0f;
    const float opacityBarWidth = layer_outliner_size_.x - 50.0f;

    for (size_t index = 0; index < snapshot_.layers.size(); ++index) {
        const GRIM::GeoSpatial::GeoSpatialLayerSnapshot& layer = snapshot_.layers[index];
        const float rowY = rowStartY + static_cast<float>(index) * rowHeight;
        if (rowY + rowHeight > layer_outliner_position_.y + layer_outliner_size_.y)
            return;

        const float checkboxY = rowY + 7.0f;
        if (input.mousePressed[0] &&
            input.mousePos.x >= checkboxX && input.mousePos.x <= checkboxX + checkboxSize &&
            input.mousePos.y >= checkboxY && input.mousePos.y <= checkboxY + checkboxSize) {
            controller_->requestSetLayerVisibility(layer.id, !layer.visible);
            return;
        }

        if (layer.id == "terrain")
            continue;

        const float opacityBarY = rowY + 36.0f;
        if (input.mouseDown[0] &&
            input.mousePos.x >= opacityBarX && input.mousePos.x <= opacityBarX + opacityBarWidth &&
            input.mousePos.y >= opacityBarY - 4.0f && input.mousePos.y <= opacityBarY + 12.0f) {
            const float opacity = std::clamp((input.mousePos.x - opacityBarX) / opacityBarWidth, 0.0f, 1.0f);
            if (std::abs(opacity - layer.opacity) >= 0.005f)
                controller_->requestSetLayerOpacity(layer.id, opacity);
            return;
        }
    }
}

void UIGeoSpatialPanel::drawLayerOutliner(OverlayRenderer& renderer) const
{
    renderer.drawRoundedRect(layer_outliner_position_, layer_outliner_size_,
                             UITheme::Colors::ScrollboxBg, UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder(layer_outliner_position_, layer_outliner_size_,
                               UITheme::Colors::BorderSubtle, UITheme::Sizes::SmallRadius);
    renderer.drawText({layer_outliner_position_.x + 10.0f, layer_outliner_position_.y + 5.0f},
                      "Visibility", UITheme::Colors::TextMuted);
    renderer.drawText({layer_outliner_position_.x + layer_outliner_size_.x - 64.0f,
                       layer_outliner_position_.y + 5.0f},
                      "Opacity", UITheme::Colors::TextMuted);

    if (snapshot_.layers.empty()) {
        renderer.drawText({layer_outliner_position_.x + 10.0f, layer_outliner_position_.y + 32.0f},
                          "Initialize Cesium to load map layers.", UITheme::Colors::TextMuted);
        return;
    }

    constexpr float rowHeight = 58.0f;
    constexpr float checkboxSize = 18.0f;
    const float rowStartY = layer_outliner_position_.y + 24.0f;
    const float checkboxX = layer_outliner_position_.x + 10.0f;
    const float opacityBarX = layer_outliner_position_.x + 38.0f;
    const float opacityBarWidth = layer_outliner_size_.x - 50.0f;

    for (size_t index = 0; index < snapshot_.layers.size(); ++index) {
        const GRIM::GeoSpatial::GeoSpatialLayerSnapshot& layer = snapshot_.layers[index];
        const float rowY = rowStartY + static_cast<float>(index) * rowHeight;
        if (rowY + rowHeight > layer_outliner_position_.y + layer_outliner_size_.y)
            break;

        renderer.drawRect({layer_outliner_position_.x + 1.0f, rowY},
                          {layer_outliner_size_.x - 2.0f, rowHeight},
                          index % 2 == 0 ? UITheme::Colors::RowEven : UITheme::Colors::RowOdd);

        const Vec2 checkboxPosition{checkboxX, rowY + 7.0f};
        renderer.drawRoundedRect(checkboxPosition, {checkboxSize, checkboxSize},
                                 UITheme::Colors::Background, 3.0f);
        renderer.drawRoundedBorder(checkboxPosition, {checkboxSize, checkboxSize},
                                   layer.visible ? UITheme::Colors::Primary : UITheme::Colors::BorderSubtle, 3.0f);
        if (layer.visible) {
            renderer.drawRoundedRect({checkboxPosition.x + 3.0f, checkboxPosition.y + 3.0f},
                                     {checkboxSize - 6.0f, checkboxSize - 6.0f},
                                     UITheme::Colors::Primary, 2.0f);
        }

        renderer.drawText({layer_outliner_position_.x + 38.0f, rowY + 7.0f}, layer.name,
                          layer.visible ? UITheme::Colors::TextPrimary : UITheme::Colors::TextDisabled);
        const int opacityPercent = static_cast<int>(layer.opacity * 100.0f + 0.5f);
        renderer.drawText({layer_outliner_position_.x + layer_outliner_size_.x - 50.0f, rowY + 7.0f},
                          std::to_string(opacityPercent) + "%", UITheme::Colors::TextSecondary);

        const float opacityBarY = rowY + 36.0f;
        renderer.drawRoundedRect({opacityBarX, opacityBarY}, {opacityBarWidth, 8.0f},
                                 UITheme::Colors::SliderTrack, 4.0f);
        renderer.drawRoundedRect({opacityBarX, opacityBarY}, {opacityBarWidth * layer.opacity, 8.0f},
                                 layer.id == "terrain" ? UITheme::Colors::TextDisabled : UITheme::Colors::Primary, 4.0f);
    }
}

void UIGeoSpatialPanel::syncEditors()
{
    lod_override_toggle_->setState(snapshot_.lod_override_enabled);
    lod_level_slider_->setValue(static_cast<float>(snapshot_.lod_override_level));

    std::vector<std::string> nextGroupOptions{"<new group>"};
    std::vector<std::string> nextPointGroupOptions;
    std::vector<std::string> nextPointOptions{"<new object>"};
    for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups) {
        nextGroupOptions.push_back(group.id);
        nextPointGroupOptions.push_back(group.id);
        for (const GRIM::GeoSpatial::GeoSpatialPointDefinition& point : group.points)
            nextPointOptions.push_back(point.id);
        for (const GRIM::GeoSpatial::GeoSpatialAreaDefinition& area : group.areas)
            nextPointOptions.push_back(area.id);
    }
    if (nextPointGroupOptions.empty())
        nextPointGroupOptions.push_back("<create group>");

    if (group_options_ != nextGroupOptions) {
        group_options_ = nextGroupOptions;
        group_dropdown_->setItems(group_options_);
    }
    if (point_group_options_ != nextPointGroupOptions) {
        point_group_options_ = nextPointGroupOptions;
        point_group_dropdown_->setItems(point_group_options_);
    }
    if (point_options_ != nextPointOptions) {
        point_options_ = nextPointOptions;
        point_dropdown_->setItems(point_options_);
    }

    auto groupOption = std::find(group_options_.begin(), group_options_.end(), selected_group_id_);
    group_dropdown_->setSelectedIndex(groupOption == group_options_.end() ? 0 : static_cast<int>(groupOption - group_options_.begin()));
    auto pointOption = std::find(point_options_.begin(), point_options_.end(), selected_point_id_);
    point_dropdown_->setSelectedIndex(pointOption == point_options_.end() ? 0 : static_cast<int>(pointOption - point_options_.begin()));

    const GRIM::GeoSpatial::GeoSpatialGroupDefinition* group = findGroup(selected_group_id_);
    toggle_group_btn_->setText(group && group->visible ? "Hide" : "Show");
    const GRIM::GeoSpatial::GeoSpatialPointDefinition* point = findPoint(selected_point_id_);
    const GRIM::GeoSpatial::GeoSpatialAreaDefinition* area = findArea(selected_area_id_);
    toggle_point_btn_->setText((point && point->visible) || (area && area->visible) ? "Hide" : "Show");
}

void UIGeoSpatialPanel::selectGroup(const std::string& id)
{
    const GRIM::GeoSpatial::GeoSpatialGroupDefinition* group = findGroup(id);
    if (!group) return;
    selected_group_id_ = group->id;
    group_id_buffer_ = group->id;
    group_name_buffer_ = group->name;
    group_color_buffer_ = group->color;
    group_id_input_->setText(group_id_buffer_);
    group_name_input_->setText(group_name_buffer_);
    group_color_input_->setText(group_color_buffer_);
    auto option = std::find(point_group_options_.begin(), point_group_options_.end(), group->id);
    if (option != point_group_options_.end())
        point_group_dropdown_->setSelectedIndex(static_cast<int>(option - point_group_options_.begin()));
}

void UIGeoSpatialPanel::selectPoint(const std::string& id)
{
    const GRIM::GeoSpatial::GeoSpatialPointDefinition* point = findPoint(id);
    if (!point) return;
    selected_point_id_ = point->id;
    selected_area_id_.clear();
    draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
    geometry_dropdown_->setSelectedIndex(0);
    point_id_buffer_ = point->id;
    point_name_buffer_ = point->name;
    point_color_buffer_ = point->color;
    longitude_buffer_ = formatCoordinate(point->longitude_degrees, 6);
    latitude_buffer_ = formatCoordinate(point->latitude_degrees, 6);
    height_buffer_ = formatCoordinate(point->height_meters, 1);
    point_id_input_->setText(point_id_buffer_);
    point_name_input_->setText(point_name_buffer_);
    point_color_input_->setText(point_color_buffer_);
    longitude_input_->setText(longitude_buffer_);
    latitude_input_->setText(latitude_buffer_);
    height_input_->setText(height_buffer_);
    auto option = std::find(point_group_options_.begin(), point_group_options_.end(), point->group_id);
    if (option != point_group_options_.end())
        point_group_dropdown_->setSelectedIndex(static_cast<int>(option - point_group_options_.begin()));
}

void UIGeoSpatialPanel::selectArea(const std::string& id)
{
    const GRIM::GeoSpatial::GeoSpatialAreaDefinition* area = findArea(id);
    if (!area) return;
    selected_point_id_.clear();
    selected_area_id_ = area->id;
    draft_geometry_kind_ = area->geometry_kind;
    geometry_dropdown_->setSelectedIndex(
        area->geometry_kind == GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea ? 1 : 2);
    point_id_buffer_ = area->id;
    point_name_buffer_ = area->name;
    point_color_buffer_ = area->color;
    longitude_buffer_ = formatCoordinate(area->longitude_degrees, 6);
    latitude_buffer_ = formatCoordinate(area->latitude_degrees, 6);
    height_buffer_ = formatCoordinate(area->height_meters, 1);
    area_size_buffer_ = formatCoordinate(area->size_meters, 1);
    area_opacity_buffer_ = formatCoordinate(area->opacity, 2);
    point_id_input_->setText(point_id_buffer_);
    point_name_input_->setText(point_name_buffer_);
    point_color_input_->setText(point_color_buffer_);
    longitude_input_->setText(longitude_buffer_);
    latitude_input_->setText(latitude_buffer_);
    height_input_->setText(height_buffer_);
    area_size_input_->setText(area_size_buffer_);
    area_opacity_input_->setText(area_opacity_buffer_);
    auto option = std::find(point_group_options_.begin(), point_group_options_.end(), area->group_id);
    if (option != point_group_options_.end())
        point_group_dropdown_->setSelectedIndex(static_cast<int>(option - point_group_options_.begin()));
}

void UIGeoSpatialPanel::clearGroupEditor()
{
    selected_group_id_.clear();
    group_id_buffer_.clear();
    group_name_buffer_.clear();
    group_color_buffer_ = "#4FC3F7";
    group_id_input_->clear();
    group_name_input_->clear();
    group_color_input_->setText(group_color_buffer_);
    group_dropdown_->setSelectedIndex(0);
}

void UIGeoSpatialPanel::clearPointEditor()
{
    selected_point_id_.clear();
    selected_area_id_.clear();
    draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
    point_id_buffer_.clear();
    point_name_buffer_.clear();
    point_color_buffer_ = findGroup(selected_group_id_) ? findGroup(selected_group_id_)->color : "#FFFFFF";
    longitude_buffer_ = "0.000000";
    latitude_buffer_ = "0.000000";
    height_buffer_ = "0.0";
    area_size_buffer_ = "1000.0";
    area_opacity_buffer_ = "0.35";
    point_id_input_->clear();
    point_name_input_->clear();
    point_color_input_->setText(point_color_buffer_);
    longitude_input_->setText(longitude_buffer_);
    latitude_input_->setText(latitude_buffer_);
    height_input_->setText(height_buffer_);
    area_size_input_->setText(area_size_buffer_);
    area_opacity_input_->setText(area_opacity_buffer_);
    geometry_dropdown_->setSelectedIndex(0);
    point_dropdown_->setSelectedIndex(0);
}

void UIGeoSpatialPanel::savePointDraft()
{
    try {
        if (!controller_)
            throw std::runtime_error("GeoSpatial controller is not attached");
        if (snapshot_.groups.empty())
            throw std::runtime_error("Create a group before adding geometry");
        const std::string groupId = point_group_dropdown_->getSelectedItem();
        const double longitude = parseCoordinate(longitude_buffer_, "Longitude");
        const double latitude = parseCoordinate(latitude_buffer_, "Latitude");
        const double height = parseCoordinate(height_buffer_, "Height");
        const bool creating = selected_point_id_.empty() && selected_area_id_.empty();

        if (draft_geometry_kind_ == GRIM::GeoSpatial::GeoSpatialGeometryKind::Point) {
            GRIM::GeoSpatial::GeoSpatialPointDefinition point;
            point.name = point_name_buffer_;
            point.group_id = groupId;
            point.color = point_color_buffer_;
            point.longitude_degrees = longitude;
            point.latitude_degrees = latitude;
            point.height_meters = height;
            controller_->requestUpsertPoint(selected_point_id_, point);
        } else {
            GRIM::GeoSpatial::GeoSpatialAreaDefinition area;
            area.name = point_name_buffer_;
            area.group_id = groupId;
            area.color = point_color_buffer_;
            area.longitude_degrees = longitude;
            area.latitude_degrees = latitude;
            area.height_meters = height;
            area.geometry_kind = draft_geometry_kind_;
            area.size_meters = parseCoordinate(area_size_buffer_, "Area size");
            area.opacity = static_cast<float>(parseCoordinate(area_opacity_buffer_, "Area opacity"));
            controller_->requestUpsertArea(selected_area_id_, area);
        }

        const std::string previousPointId = selected_point_id_;
        const std::string previousAreaId = selected_area_id_;
        refreshSnapshot();
        if (creating) {
            const GRIM::GeoSpatial::GeoSpatialGroupDefinition* targetGroup = findGroup(groupId);
            if (!targetGroup)
                throw std::runtime_error("Geometry group disappeared after save");
            if (draft_geometry_kind_ == GRIM::GeoSpatial::GeoSpatialGeometryKind::Point && !targetGroup->points.empty())
                selectPoint(targetGroup->points.back().id);
            else if (draft_geometry_kind_ != GRIM::GeoSpatial::GeoSpatialGeometryKind::Point && !targetGroup->areas.empty())
                selectArea(targetGroup->areas.back().id);
        } else if (!previousPointId.empty()) {
            selectPoint(previousPointId);
        } else if (!previousAreaId.empty()) {
            selectArea(previousAreaId);
        }
        editor_status_.clear();
    } catch (const std::exception& error) {
        editor_status_ = std::string("Geometry draft: ") + error.what();
    }
}

const GRIM::GeoSpatial::GeoSpatialGroupDefinition* UIGeoSpatialPanel::findGroup(const std::string& id) const
{
    auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
        [&](const GRIM::GeoSpatial::GeoSpatialGroupDefinition& candidate) { return candidate.id == id; });
    return group == snapshot_.groups.end() ? nullptr : &*group;
}

const GRIM::GeoSpatial::GeoSpatialPointDefinition* UIGeoSpatialPanel::findPoint(const std::string& id) const
{
    for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups) {
        auto point = std::find_if(group.points.begin(), group.points.end(),
            [&](const GRIM::GeoSpatial::GeoSpatialPointDefinition& candidate) { return candidate.id == id; });
        if (point != group.points.end())
            return &*point;
    }
    return nullptr;
}

const GRIM::GeoSpatial::GeoSpatialAreaDefinition* UIGeoSpatialPanel::findArea(const std::string& id) const
{
    for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups) {
        auto area = std::find_if(group.areas.begin(), group.areas.end(),
            [&](const GRIM::GeoSpatial::GeoSpatialAreaDefinition& candidate) { return candidate.id == id; });
        if (area != group.areas.end())
            return &*area;
    }
    return nullptr;
}