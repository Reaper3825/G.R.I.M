#pragma once

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_3d_viewport.hpp"
#include "primitives/ui_dropdown.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_slider.hpp"
#include "primitives/ui_toggle.hpp"
#include "../geospatial_panel_controller.hpp"

#include <memory>
#include <string>

class OverlayRenderer;
struct InputState;

class UIGeoSpatialPanel : public UIPanel {
public:
    UIGeoSpatialPanel();

    void setController(GRIM::GeoSpatial::GeoSpatialPanelController* controller);
    void setViewportAttachment(std::shared_ptr<UI3DViewportAttachment> attachment);
    void setVisible(bool visible) override;

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;
    bool shouldPassThroughAt(float x, float y) const override;
    void collectPassThroughRects(std::vector<PanelRect>& rects) const override;

private:
    void refreshSnapshot();
    void layoutControls();
    void syncEditors();
    void selectGroup(const std::string& id);
    void selectPoint(const std::string& id);
    void selectArea(const std::string& id);
    void clearGroupEditor();
    void clearPointEditor();
    void savePointDraft();
    void updateLayerOutliner(const InputState& input);
    void drawLayerOutliner(OverlayRenderer& renderer) const;
    const GRIM::GeoSpatial::GeoSpatialGroupDefinition* findGroup(const std::string& id) const;
    const GRIM::GeoSpatial::GeoSpatialPointDefinition* findPoint(const std::string& id) const;
    const GRIM::GeoSpatial::GeoSpatialAreaDefinition* findArea(const std::string& id) const;

    GRIM::GeoSpatial::GeoSpatialPanelController* controller_ = nullptr;
    GRIM::GeoSpatial::GeoSpatialPanelSnapshot snapshot_{};

    std::shared_ptr<UI3DViewport> viewport_;
    std::shared_ptr<UIButton> init_btn_;
    std::shared_ptr<UIButton> reload_btn_;
    std::shared_ptr<UIButton> home_btn_;
    std::shared_ptr<UIButton> reset_camera_btn_;
    std::shared_ptr<UIToggle> lod_override_toggle_;
    std::shared_ptr<UISlider> lod_level_slider_;

    std::shared_ptr<UIDropdown> group_dropdown_;
    std::shared_ptr<UIInputBox> group_id_input_;
    std::shared_ptr<UIInputBox> group_name_input_;
    std::shared_ptr<UIInputBox> group_color_input_;
    std::shared_ptr<UIButton> new_group_btn_;
    std::shared_ptr<UIButton> save_group_btn_;
    std::shared_ptr<UIButton> toggle_group_btn_;
    std::shared_ptr<UIButton> remove_group_btn_;

    std::shared_ptr<UIDropdown> point_dropdown_;
    std::shared_ptr<UIDropdown> point_group_dropdown_;
    std::shared_ptr<UIDropdown> geometry_dropdown_;
    std::shared_ptr<UIInputBox> point_id_input_;
    std::shared_ptr<UIInputBox> point_name_input_;
    std::shared_ptr<UIInputBox> point_color_input_;
    std::shared_ptr<UIInputBox> longitude_input_;
    std::shared_ptr<UIInputBox> latitude_input_;
    std::shared_ptr<UIInputBox> height_input_;
    std::shared_ptr<UIInputBox> area_size_input_;
    std::shared_ptr<UIInputBox> area_opacity_input_;
    std::shared_ptr<UIButton> use_pick_btn_;
    std::shared_ptr<UIButton> new_point_btn_;
    std::shared_ptr<UIButton> save_point_btn_;
    std::shared_ptr<UIButton> toggle_point_btn_;
    std::shared_ptr<UIButton> remove_point_btn_;
    std::shared_ptr<UIButton> save_catalog_btn_;

    std::string selected_group_id_;
    std::string selected_point_id_;
    std::string selected_area_id_;
    std::string group_id_buffer_;
    std::string group_name_buffer_;
    std::string group_color_buffer_ = "#4FC3F7";
    std::string point_id_buffer_;
    std::string point_name_buffer_;
    std::string point_color_buffer_ = "#FFFFFF";
    std::string longitude_buffer_ = "0.000000";
    std::string latitude_buffer_ = "0.000000";
    std::string height_buffer_ = "0.0";
    std::string area_size_buffer_ = "1000.0";
    std::string area_opacity_buffer_ = "0.35";
    GRIM::GeoSpatial::GeoSpatialGeometryKind draft_geometry_kind_ =
        GRIM::GeoSpatial::GeoSpatialGeometryKind::Point;
    std::string editor_status_;
    std::vector<std::string> group_options_;
    std::vector<std::string> point_group_options_;
    std::vector<std::string> point_options_;
    Vec2 layer_outliner_position_{};
    Vec2 layer_outliner_size_{};
    bool show_layer_outliner_ = true;
};