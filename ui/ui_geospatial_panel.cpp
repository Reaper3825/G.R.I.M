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
    constexpr float kOutlinerRowH = 24.0f;
    constexpr float kOutlinerH = 150.0f;

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

    group_name_input_ = std::make_shared<UIInputBox>(&group_name_buffer_);
    group_name_input_->setPlaceholder("Group name");
    group_color_input_ = std::make_shared<UIInputBox>(&group_color_buffer_);
    group_color_input_->setPlaceholder("#RRGGBB");

    new_group_btn_ = std::make_shared<UIButton>("Add Group", [this]() {
        clearGroupEditor();
        clearPointEditor();
        editor_status_ = "Enter the new group properties, then choose Create Group.";
    });
    save_group_btn_ = std::make_shared<UIButton>("Create Group", [this]() {
        if (!controller_) return;
        const bool creating = selected_group_id_.empty();
        const size_t previousCount = snapshot_.groups.size();
        controller_->requestUpsertGroup(selected_group_id_, group_name_buffer_, group_color_buffer_);
        refreshSnapshot();
        if (creating && snapshot_.groups.size() == previousCount + 1)
            selectGroup(snapshot_.groups.back().id);
    });
    remove_group_btn_ = std::make_shared<UIButton>("Delete Group", [this]() {
        if (controller_ && !selected_group_id_.empty())
            controller_->requestRemoveGroup(selected_group_id_);
        clearGroupEditor();
        clearPointEditor();
    });

    point_group_dropdown_ = std::make_shared<UIDropdown>(
        "Group", std::vector<std::string>{"No groups available"}, 0,
        [this](int, const std::string&) {
            if (!selected_point_id_.empty() || !selected_area_id_.empty())
                saveGeometryDraft();
        });
    geometry_dropdown_ = std::make_shared<UIDropdown>(
        "Geometry", std::vector<std::string>{"Point", "Cube Area", "Sphere Area"}, 0,
        [this](int index, const std::string&) {
            if (index == 0)
                draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
            else if (index == 1)
                draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea;
            else if (index == 2)
                draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::SphereArea;
            else
                throw std::runtime_error("GeoSpatial geometry dropdown returned an invalid index");
            if (!selected_point_id_.empty() || !selected_area_id_.empty())
                saveGeometryDraft();
        });
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

    auto commitSelectedGeometry = [this](const std::string&) {
        if (!selected_point_id_.empty() || !selected_area_id_.empty())
            saveGeometryDraft();
    };
    for (const std::shared_ptr<UIInputBox>& input : {
             point_name_input_, point_color_input_, longitude_input_, latitude_input_, height_input_,
             area_size_input_, area_opacity_input_}) {
        input->OnTextSubmitted.Bind(commitSelectedGeometry);
    }
    auto commitSelectedGroup = [this](const std::string&) {
        if (controller_ && !selected_group_id_.empty() && selected_point_id_.empty() && selected_area_id_.empty()) {
            controller_->requestUpsertGroup(selected_group_id_, group_name_buffer_, group_color_buffer_);
            refreshSnapshot();
        }
    };
    group_name_input_->OnTextSubmitted.Bind(commitSelectedGroup);
    group_color_input_->OnTextSubmitted.Bind(commitSelectedGroup);

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
        if (!selected_point_id_.empty() || !selected_area_id_.empty())
            saveGeometryDraft();
    });
    new_point_btn_ = std::make_shared<UIButton>("Add Point", [this]() {
        if (selected_group_id_.empty()) {
            editor_status_ = "Select a group in the outliner before adding a point.";
            return;
        }
        clearPointEditor();
        draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
        geometry_dropdown_->setSelectedIndex(0);
        editor_status_ = "The new point will be added to the selected group.";
    });
    new_area_btn_ = std::make_shared<UIButton>("Add Area", [this]() {
        if (selected_group_id_.empty()) {
            editor_status_ = "Select a group in the outliner before adding an area.";
            return;
        }
        clearPointEditor();
        draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea;
        geometry_dropdown_->setSelectedIndex(1);
        editor_status_ = "Configure the cube or sphere area in the selected group.";
    });
    save_point_btn_ = std::make_shared<UIButton>("Create Point", [this]() { saveGeometryDraft(); });
    remove_point_btn_ = std::make_shared<UIButton>("Delete Object", [this]() {
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
        new_group_btn_, save_group_btn_, remove_group_btn_, use_pick_btn_,
        new_point_btn_, new_area_btn_, save_point_btn_, remove_point_btn_, save_catalog_btn_
    };
    for (const std::shared_ptr<UIButton>& button : editorButtons)
        button->setSize(74.0f, kButtonH);
    new_group_btn_->setSize(104.0f, kButtonH);
    save_group_btn_->setSize(126.0f, kButtonH);
    remove_group_btn_->setSize(336.0f, kButtonH);
    use_pick_btn_->setSize(112.0f, kButtonH);
    new_point_btn_->setSize(92.0f, kButtonH);
    new_area_btn_->setSize(92.0f, kButtonH);
    save_point_btn_->setSize(112.0f, kButtonH);
    remove_point_btn_->setSize(112.0f, kButtonH);
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

    updateOutliner(input);

    if (init_btn_) init_btn_->update(input, dt);
    if (reload_btn_) reload_btn_->update(input, dt);
    if (home_btn_) home_btn_->update(input, dt);
    if (reset_camera_btn_) reset_camera_btn_->update(input, dt);

    std::vector<std::shared_ptr<Widget>> editorWidgets = {
        group_name_input_, group_color_input_,
        new_group_btn_, save_group_btn_, remove_group_btn_,
        point_group_dropdown_, geometry_dropdown_, point_name_input_, point_color_input_,
        longitude_input_, latitude_input_, height_input_, area_size_input_, area_opacity_input_,
        use_pick_btn_, new_point_btn_, new_area_btn_,
        remove_point_btn_, save_catalog_btn_
    };
    if (selected_point_id_.empty() && selected_area_id_.empty())
        editorWidgets.push_back(save_point_btn_);
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
                      "Click the globe to capture a WGS84 anchor for points and areas.",
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
    renderer.drawText({sideX + 12.0f, viewportY + 29.0f}, "Layers", layerTabColor);
    renderer.drawText({sideX + 92.0f, viewportY + 29.0f}, "Objects", objectTabColor);
    const float tabUnderlineX = show_layer_outliner_ ? sideX + 12.0f : sideX + 92.0f;
    renderer.drawRect({tabUnderlineX, viewportY + 46.0f}, {52.0f, 2.0f}, UITheme::Colors::Primary);
    drawOutliner(renderer);

    auto drawFieldLabel = [&](float y, const char* label) {
        renderer.drawText({sideX + 12.0f, y + 5.0f}, label, UITheme::Colors::TextLabel);
    };
    renderer.drawText({sideX + 12.0f, group_name_input_->getPosition().y - 20.0f},
                      selected_group_id_.empty() ? "New Group Properties" : "Selected Group Properties",
                      UITheme::Colors::TextHeader);
    drawFieldLabel(group_name_input_->getPosition().y, "Name");
    drawFieldLabel(group_color_input_->getPosition().y, "Color");
    renderer.drawRoundedRect({sideX + kSidePanelW - 32.0f, group_color_input_->getPosition().y + 4.0f},
                             {18.0f, 18.0f}, previewColor(group_color_buffer_), 4.0f);
    renderer.drawText({sideX + 12.0f, point_group_dropdown_->getPosition().y - 20.0f},
                      (selected_point_id_.empty() && selected_area_id_.empty())
                          ? "New Geometry Properties" : "Selected Geometry Properties",
                      UITheme::Colors::TextHeader);
    drawFieldLabel(point_name_input_->getPosition().y, "Name");
    drawFieldLabel(point_color_input_->getPosition().y, "Color");
    drawFieldLabel(longitude_input_->getPosition().y, "Longitude");
    drawFieldLabel(latitude_input_->getPosition().y, "Latitude");
    drawFieldLabel(height_input_->getPosition().y, "Height m");
    drawFieldLabel(area_size_input_->getPosition().y, "Area size m");
    drawFieldLabel(area_opacity_input_->getPosition().y, "Area alpha");
    renderer.drawRoundedRect({sideX + kSidePanelW - 32.0f, point_color_input_->getPosition().y + 4.0f},
                             {18.0f, 18.0f}, previewColor(point_color_buffer_), 4.0f);

    std::vector<std::shared_ptr<Widget>> editorWidgets = {
        group_name_input_, group_color_input_,
        new_group_btn_, save_group_btn_, remove_group_btn_,
        point_group_dropdown_, geometry_dropdown_, point_name_input_, point_color_input_,
        longitude_input_, latitude_input_, height_input_, area_size_input_, area_opacity_input_,
        use_pick_btn_, new_point_btn_, new_area_btn_,
        remove_point_btn_, save_catalog_btn_
    };
    if (selected_point_id_.empty() && selected_area_id_.empty())
        editorWidgets.push_back(save_point_btn_);
    for (const std::shared_ptr<Widget>& widget : editorWidgets)
        if (widget) widget->drawOverlay(renderer, position);
    point_group_dropdown_->drawExpandedList(renderer, position);
    geometry_dropdown_->drawExpandedList(renderer, position);

    const float catalogStatusY = save_catalog_btn_->getPosition().y + kButtonH + 8.0f;
    renderer.drawText({sideX + 12.0f, catalogStatusY}, snapshot_.point_catalog_status,
                      snapshot_.point_catalog_dirty ? UITheme::Colors::Warning : UITheme::Colors::TextSecondary);
    if (!editor_status_.empty())
        renderer.drawText({sideX + 12.0f, catalogStatusY + 18.0f}, editor_status_, UITheme::Colors::Info);

    if (init_btn_) init_btn_->drawOverlay(renderer, position);
    if (reload_btn_) reload_btn_->drawOverlay(renderer, position);
    if (home_btn_) home_btn_->drawOverlay(renderer, position);
    if (reset_camera_btn_) reset_camera_btn_->drawOverlay(renderer, position);

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
    outliner_position_ = {sideX + 12.0f, viewportY + 50.0f};
    outliner_size_ = {kSidePanelW - 24.0f, kOutlinerH};
    float y = outliner_position_.y + outliner_size_.y + 30.0f;
    for (const std::shared_ptr<UIInputBox>& input : {group_name_input_, group_color_input_}) {
        input->setPosition(fieldX, y);
        input->setSize(fieldW - (input == group_color_input_ ? 28.0f : 0.0f), kFieldH);
        y += 30.0f;
    }
    new_group_btn_->setPosition(sideX + 12.0f, y);
    save_group_btn_->setPosition(sideX + 122.0f, y);
    y += 34.0f;
    remove_group_btn_->setPosition(sideX + 12.0f, y);
    y += 44.0f;

    y += 20.0f;
    point_group_dropdown_->setPosition(sideX + 12.0f, y);
    point_group_dropdown_->setSize(kSidePanelW - 24.0f, 34.0f);
    y += 38.0f;
    geometry_dropdown_->setPosition(sideX + 12.0f, y);
    geometry_dropdown_->setSize(kSidePanelW - 24.0f, 34.0f);
    y += 38.0f;
    for (const std::shared_ptr<UIInputBox>& input : {
             point_name_input_, point_color_input_, longitude_input_, latitude_input_, height_input_}) {
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
    new_area_btn_->setPosition(sideX + 110.0f, y);
    y += 34.0f;
    save_point_btn_->setPosition(sideX + 12.0f, y);
    remove_point_btn_->setPosition(sideX + 130.0f, y);
    y += 38.0f;
    save_catalog_btn_->setPosition(sideX + 12.0f, y);
}

void UIGeoSpatialPanel::syncEditors()
{
    std::vector<std::string> nextPointGroupOptions;
    nextPointGroupOptions.reserve(snapshot_.groups.size());
    for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups)
        nextPointGroupOptions.push_back(group.name);
    if (nextPointGroupOptions.empty())
        nextPointGroupOptions.push_back("No groups available");
    if (point_group_options_ != nextPointGroupOptions) {
        point_group_options_ = std::move(nextPointGroupOptions);
        point_group_dropdown_->setItems(point_group_options_);
    }

    const GRIM::GeoSpatial::GeoSpatialGroupDefinition* group = findGroup(selected_group_id_);
    save_group_btn_->setText(group ? "Update Group" : "Create Group");
    remove_group_btn_->setText(group
        ? "Delete Group and Its " + std::to_string(group->points.size() + group->areas.size()) + " Object(s)"
        : "Delete Group (select a group first)");
    const bool areaDraft = draft_geometry_kind_ != GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
    save_point_btn_->setText(areaDraft ? "Create Area" : "Create Point");
    remove_point_btn_->setText(!selected_area_id_.empty() ? "Delete Area" : "Delete Point");
}

void UIGeoSpatialPanel::selectGroup(const std::string& id)
{
    const GRIM::GeoSpatial::GeoSpatialGroupDefinition* group = findGroup(id);
    if (!group) return;
    selected_group_id_ = group->id;
    group_name_buffer_ = group->name;
    group_color_buffer_ = group->color;
    group_name_input_->setText(group_name_buffer_);
    group_color_input_->setText(group_color_buffer_);
}

void UIGeoSpatialPanel::selectPoint(const std::string& id)
{
    const GRIM::GeoSpatial::GeoSpatialPointDefinition* point = findPoint(id);
    if (!point) return;
    selected_group_id_ = point->group_id;
    selectGroup(point->group_id);
    selected_point_id_ = point->id;
    selected_area_id_.clear();
    draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
    geometry_dropdown_->setSelectedIndex(0);
    point_name_buffer_ = point->name;
    point_color_buffer_ = point->color;
    longitude_buffer_ = formatCoordinate(point->longitude_degrees, 6);
    latitude_buffer_ = formatCoordinate(point->latitude_degrees, 6);
    height_buffer_ = formatCoordinate(point->height_meters, 1);
    point_name_input_->setText(point_name_buffer_);
    point_color_input_->setText(point_color_buffer_);
    longitude_input_->setText(longitude_buffer_);
    latitude_input_->setText(latitude_buffer_);
    height_input_->setText(height_buffer_);
    auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
        [&](const GRIM::GeoSpatial::GeoSpatialGroupDefinition& candidate) { return candidate.id == point->group_id; });
    if (group != snapshot_.groups.end())
        point_group_dropdown_->setSelectedIndex(static_cast<int>(group - snapshot_.groups.begin()));
}

void UIGeoSpatialPanel::selectArea(const std::string& id)
{
    const GRIM::GeoSpatial::GeoSpatialAreaDefinition* area = findArea(id);
    if (!area) return;
    selected_group_id_ = area->group_id;
    selectGroup(area->group_id);
    selected_point_id_.clear();
    selected_area_id_ = area->id;
    draft_geometry_kind_ = area->geometry_kind;
    geometry_dropdown_->setSelectedIndex(
        area->geometry_kind == GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea ? 1 : 2);
    point_name_buffer_ = area->name;
    point_color_buffer_ = area->color;
    longitude_buffer_ = formatCoordinate(area->longitude_degrees, 6);
    latitude_buffer_ = formatCoordinate(area->latitude_degrees, 6);
    height_buffer_ = formatCoordinate(area->height_meters, 1);
    area_size_buffer_ = formatCoordinate(area->size_meters, 1);
    area_opacity_buffer_ = formatCoordinate(area->opacity, 2);
    point_name_input_->setText(point_name_buffer_);
    point_color_input_->setText(point_color_buffer_);
    longitude_input_->setText(longitude_buffer_);
    latitude_input_->setText(latitude_buffer_);
    height_input_->setText(height_buffer_);
    area_size_input_->setText(area_size_buffer_);
    area_opacity_input_->setText(area_opacity_buffer_);
    auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
        [&](const GRIM::GeoSpatial::GeoSpatialGroupDefinition& candidate) { return candidate.id == area->group_id; });
    if (group != snapshot_.groups.end())
        point_group_dropdown_->setSelectedIndex(static_cast<int>(group - snapshot_.groups.begin()));
}

void UIGeoSpatialPanel::clearGroupEditor()
{
    selected_group_id_.clear();
    group_name_buffer_.clear();
    group_color_buffer_ = "#4FC3F7";
    group_name_input_->clear();
    group_color_input_->setText(group_color_buffer_);
}

void UIGeoSpatialPanel::clearPointEditor()
{
    selected_point_id_.clear();
    selected_area_id_.clear();
    draft_geometry_kind_ = GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
    geometry_dropdown_->setSelectedIndex(0);
    point_name_buffer_.clear();
    point_color_buffer_ = findGroup(selected_group_id_) ? findGroup(selected_group_id_)->color : "#FFFFFF";
    longitude_buffer_ = "0.000000";
    latitude_buffer_ = "0.000000";
    height_buffer_ = "0.0";
    area_size_buffer_ = "1000.0";
    area_opacity_buffer_ = "0.35";
    point_name_input_->clear();
    point_color_input_->setText(point_color_buffer_);
    longitude_input_->setText(longitude_buffer_);
    latitude_input_->setText(latitude_buffer_);
    height_input_->setText(height_buffer_);
    area_size_input_->setText(area_size_buffer_);
    area_opacity_input_->setText(area_opacity_buffer_);
    auto group = std::find_if(snapshot_.groups.begin(), snapshot_.groups.end(),
        [&](const GRIM::GeoSpatial::GeoSpatialGroupDefinition& candidate) { return candidate.id == selected_group_id_; });
    if (group != snapshot_.groups.end())
        point_group_dropdown_->setSelectedIndex(static_cast<int>(group - snapshot_.groups.begin()));
}

void UIGeoSpatialPanel::saveGeometryDraft()
{
    try {
        if (!controller_)
            throw std::runtime_error("GeoSpatial controller is not attached");
        if (snapshot_.groups.empty())
            throw std::runtime_error("Create a group before adding geometry");
        const int groupIndex = point_group_dropdown_->getSelectedIndex();
        if (groupIndex < 0 || groupIndex >= static_cast<int>(snapshot_.groups.size()))
            throw std::runtime_error("Choose a group for the geometry");
        const std::string groupId = snapshot_.groups[static_cast<size_t>(groupIndex)].id;
        const double longitude = parseCoordinate(longitude_buffer_, "Longitude");
        const double latitude = parseCoordinate(latitude_buffer_, "Latitude");
        const double height = parseCoordinate(height_buffer_, "Height");
        const bool creating = selected_point_id_.empty() && selected_area_id_.empty();
        size_t previousCount = 0;
        for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups)
            previousCount += group.points.size() + group.areas.size();

        if (draft_geometry_kind_ == GRIM::GeoSpatial::GeoSpatialGeometryKind::Point) {
            if (!selected_area_id_.empty())
                throw std::runtime_error("Area geometry cannot be changed into a point; create a new point");
            GRIM::GeoSpatial::GeoSpatialPointDefinition point;
            point.name = point_name_buffer_;
            point.group_id = groupId;
            point.color = point_color_buffer_;
            point.longitude_degrees = longitude;
            point.latitude_degrees = latitude;
            point.height_meters = height;
            controller_->requestUpsertPoint(selected_point_id_, point);
        } else {
            if (!selected_point_id_.empty())
                throw std::runtime_error("Point geometry cannot be changed into an area; create a new area");
            GRIM::GeoSpatial::GeoSpatialAreaDefinition area;
            area.name = point_name_buffer_;
            area.group_id = groupId;
            area.color = point_color_buffer_;
            area.longitude_degrees = longitude;
            area.latitude_degrees = latitude;
            area.height_meters = height;
            area.geometry_kind = draft_geometry_kind_;
            area.size_meters = parseCoordinate(area_size_buffer_, "Area size");
            area.opacity = static_cast<float>(parseCoordinate(area_opacity_buffer_, "Area alpha"));
            controller_->requestUpsertArea(selected_area_id_, area);
        }
        refreshSnapshot();
        if (creating) {
            size_t nextCount = 0;
            for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups)
                nextCount += group.points.size() + group.areas.size();
            if (nextCount == previousCount + 1) {
                const GRIM::GeoSpatial::GeoSpatialGroupDefinition& targetGroup =
                    snapshot_.groups[static_cast<size_t>(groupIndex)];
                if (draft_geometry_kind_ == GRIM::GeoSpatial::GeoSpatialGeometryKind::Point && !targetGroup.points.empty())
                    selectPoint(targetGroup.points.back().id);
                else if (draft_geometry_kind_ != GRIM::GeoSpatial::GeoSpatialGeometryKind::Point && !targetGroup.areas.empty())
                    selectArea(targetGroup.areas.back().id);
            }
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

void UIGeoSpatialPanel::updateOutliner(const InputState& input)
{
    const bool clicked = input.mousePressed[0];
    const bool dragging = input.mouseDown[0];
    if (!clicked && !dragging)
        return;

    const float tabTop = outliner_position_.y - 24.0f;
    if (clicked && input.mousePos.y >= tabTop && input.mousePos.y < outliner_position_.y &&
        input.mousePos.x >= outliner_position_.x && input.mousePos.x < outliner_position_.x + 160.0f) {
        show_layer_outliner_ = input.mousePos.x < outliner_position_.x + 80.0f;
        return;
    }

    if (input.mousePos.x < outliner_position_.x ||
        input.mousePos.x > outliner_position_.x + outliner_size_.x ||
        input.mousePos.y < outliner_position_.y || input.mousePos.y > outliner_position_.y + outliner_size_.y) {
        return;
    }

    float rowY = outliner_position_.y;
    if (show_layer_outliner_) {
        const float checkboxX = outliner_position_.x + 8.0f;
        const float opacityX = outliner_position_.x + outliner_size_.x - 92.0f;
        constexpr float opacityWidth = 54.0f;
        constexpr float checkboxSize = 16.0f;
        for (const GRIM::GeoSpatial::GeoSpatialLayerSnapshot& layer : snapshot_.layers) {
            if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
                return;
            if (input.mousePos.y >= rowY && input.mousePos.y < rowY + kOutlinerRowH) {
                if (clicked && input.mousePos.x >= checkboxX && input.mousePos.x <= checkboxX + checkboxSize) {
                    if (controller_)
                        controller_->requestSetLayerVisibility(layer.id, !layer.visible);
                    return;
                }
                if (dragging && layer.id != "terrain" &&
                    input.mousePos.x >= opacityX && input.mousePos.x <= opacityX + opacityWidth) {
                    const float opacity = std::clamp((input.mousePos.x - opacityX) / opacityWidth, 0.0f, 1.0f);
                    if (controller_)
                        controller_->requestSetLayerOpacity(layer.id, opacity);
                    return;
                }
            }
            rowY += kOutlinerRowH;
        }
        return;
    }

    if (!clicked)
        return;

    const float eyeX = outliner_position_.x + outliner_size_.x - 30.0f;
    for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups) {
        if (input.mousePos.y >= rowY && input.mousePos.y < rowY + kOutlinerRowH) {
            if (input.mousePos.x >= eyeX) {
                if (controller_) controller_->requestToggleGroupVisibility(group.id);
            } else {
                selectGroup(group.id);
                clearPointEditor();
            }
            return;
        }
        rowY += kOutlinerRowH;
        for (const GRIM::GeoSpatial::GeoSpatialPointDefinition& point : group.points) {
            if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
                return;
            if (input.mousePos.y >= rowY && input.mousePos.y < rowY + kOutlinerRowH) {
                if (input.mousePos.x >= eyeX) {
                    if (controller_) controller_->requestTogglePointVisibility(point.id);
                } else {
                    selectPoint(point.id);
                }
                return;
            }
            rowY += kOutlinerRowH;
        }
        for (const GRIM::GeoSpatial::GeoSpatialAreaDefinition& area : group.areas) {
            if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
                return;
            if (input.mousePos.y >= rowY && input.mousePos.y < rowY + kOutlinerRowH) {
                if (input.mousePos.x >= eyeX) {
                    if (controller_) controller_->requestToggleAreaVisibility(area.id);
                } else {
                    selectArea(area.id);
                }
                return;
            }
            rowY += kOutlinerRowH;
        }
    }
}

void UIGeoSpatialPanel::drawOutliner(OverlayRenderer& renderer) const
{
    renderer.drawRoundedRect(outliner_position_, outliner_size_, UITheme::Colors::ScrollboxBg, UITheme::Sizes::SmallRadius);
    renderer.drawRoundedBorder(outliner_position_, outliner_size_, UITheme::Colors::BorderSubtle, UITheme::Sizes::SmallRadius);

    float rowY = outliner_position_.y;
    const float eyeX = outliner_position_.x + outliner_size_.x - 25.0f;
    if (show_layer_outliner_) {
        if (snapshot_.layers.empty()) {
            renderer.drawText({outliner_position_.x + 10.0f, rowY + 5.0f},
                              "Initialize Cesium to load layers.", UITheme::Colors::TextMuted);
            return;
        }

        constexpr float checkboxSize = 16.0f;
        const float opacityX = outliner_position_.x + outliner_size_.x - 92.0f;
        constexpr float opacityWidth = 54.0f;
        for (size_t index = 0; index < snapshot_.layers.size(); ++index) {
            if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
                break;
            const GRIM::GeoSpatial::GeoSpatialLayerSnapshot& layer = snapshot_.layers[index];
            renderer.drawRect({outliner_position_.x + 1.0f, rowY}, {outliner_size_.x - 2.0f, kOutlinerRowH},
                              index % 2 == 0 ? UITheme::Colors::RowEven : UITheme::Colors::RowOdd);
            const Vec2 checkboxPosition{outliner_position_.x + 8.0f, rowY + 4.0f};
            renderer.drawRoundedRect(checkboxPosition, {checkboxSize, checkboxSize},
                                     UITheme::Colors::Background, 3.0f);
            renderer.drawRoundedBorder(checkboxPosition, {checkboxSize, checkboxSize},
                                       layer.visible ? UITheme::Colors::Primary : UITheme::Colors::BorderSubtle, 3.0f);
            if (layer.visible) {
                renderer.drawRoundedRect({checkboxPosition.x + 3.0f, checkboxPosition.y + 3.0f},
                                         {checkboxSize - 6.0f, checkboxSize - 6.0f},
                                         UITheme::Colors::Primary, 2.0f);
            }
            renderer.drawText({outliner_position_.x + 34.0f, rowY + 4.0f}, layer.name,
                              layer.visible ? UITheme::Colors::TextPrimary : UITheme::Colors::TextDisabled);
            if (layer.id != "terrain") {
                renderer.drawRoundedRect({opacityX, rowY + 9.0f}, {opacityWidth, 6.0f},
                                         UITheme::Colors::SliderTrack, 3.0f);
                renderer.drawRoundedRect({opacityX, rowY + 9.0f}, {opacityWidth * layer.opacity, 6.0f},
                                         layer.visible ? UITheme::Colors::Primary : UITheme::Colors::TextDisabled, 3.0f);
                renderer.drawText({opacityX + opacityWidth + 5.0f, rowY + 4.0f},
                                  std::to_string(static_cast<int>(std::round(layer.opacity * 100.0f))) + "%",
                                  UITheme::Colors::TextMuted);
            }
            rowY += kOutlinerRowH;
        }
        return;
    }

    if (snapshot_.groups.empty()) {
        renderer.drawText({outliner_position_.x + 10.0f, rowY + 5.0f},
                          "No groups. Choose Add Group below.", UITheme::Colors::TextMuted);
        return;
    }

    for (const GRIM::GeoSpatial::GeoSpatialGroupDefinition& group : snapshot_.groups) {
        if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
            break;
        const bool groupSelected = selected_point_id_.empty() && selected_area_id_.empty() && selected_group_id_ == group.id;
        renderer.drawRect({outliner_position_.x + 1.0f, rowY}, {outliner_size_.x - 2.0f, kOutlinerRowH},
                          groupSelected ? UITheme::Colors::RowSelected : UITheme::Colors::RowEven);
        renderer.drawText({outliner_position_.x + 8.0f, rowY + 4.0f}, ICON_FA_FOLDER_OPEN, previewColor(group.color));
        renderer.drawText({outliner_position_.x + 31.0f, rowY + 4.0f}, group.name, UITheme::Colors::TextPrimary);
        renderer.drawText({eyeX, rowY + 4.0f}, group.visible ? ICON_FA_EYE : ICON_FA_EYE_SLASH,
                          group.visible ? UITheme::Colors::TextPrimary : UITheme::Colors::TextDisabled);
        rowY += kOutlinerRowH;

        for (const GRIM::GeoSpatial::GeoSpatialPointDefinition& point : group.points) {
            if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
                return;
            const bool pointSelected = selected_point_id_ == point.id;
            renderer.drawRect({outliner_position_.x + 1.0f, rowY}, {outliner_size_.x - 2.0f, kOutlinerRowH},
                              pointSelected ? UITheme::Colors::RowSelected : UITheme::Colors::RowOdd);
            renderer.drawText({outliner_position_.x + 27.0f, rowY + 4.0f}, ICON_FA_CIRCLE, previewColor(point.color));
            renderer.drawText({outliner_position_.x + 50.0f, rowY + 4.0f}, point.name, UITheme::Colors::TextSecondary);
            renderer.drawText({eyeX, rowY + 4.0f}, point.visible ? ICON_FA_EYE : ICON_FA_EYE_SLASH,
                              point.visible ? UITheme::Colors::TextPrimary : UITheme::Colors::TextDisabled);
            rowY += kOutlinerRowH;
        }
        for (const GRIM::GeoSpatial::GeoSpatialAreaDefinition& area : group.areas) {
            if (rowY + kOutlinerRowH > outliner_position_.y + outliner_size_.y)
                return;
            const bool areaSelected = selected_area_id_ == area.id;
            renderer.drawRect({outliner_position_.x + 1.0f, rowY}, {outliner_size_.x - 2.0f, kOutlinerRowH},
                              areaSelected ? UITheme::Colors::RowSelected : UITheme::Colors::RowOdd);
            const char* icon = area.geometry_kind == GRIM::GeoSpatial::GeoSpatialGeometryKind::CubeArea
                ? ICON_FA_CUBE : ICON_FA_CIRCLE;
            renderer.drawText({outliner_position_.x + 27.0f, rowY + 4.0f}, icon, previewColor(area.color));
            renderer.drawText({outliner_position_.x + 50.0f, rowY + 4.0f}, area.name, UITheme::Colors::TextSecondary);
            renderer.drawText({eyeX, rowY + 4.0f}, area.visible ? ICON_FA_EYE : ICON_FA_EYE_SLASH,
                              area.visible ? UITheme::Colors::TextPrimary : UITheme::Colors::TextDisabled);
            rowY += kOutlinerRowH;
        }
    }
}