#pragma once

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_3d_viewport.hpp"
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
    void setUiStatus(const std::string& status);

    GRIM::GeoSpatial::GeoSpatialPanelController* controller_ = nullptr;
    GRIM::GeoSpatial::GeoSpatialPanelSnapshot snapshot_{};
    std::string ui_status_ = "Waiting for Cesium native viewport attachment";

    std::shared_ptr<UI3DViewport> viewport_;
    std::shared_ptr<UIButton> init_btn_;
    std::shared_ptr<UIButton> reload_btn_;
    std::shared_ptr<UIButton> home_btn_;
    std::shared_ptr<UIButton> reset_camera_btn_;
    std::shared_ptr<UIButton> terrain_btn_;
    std::shared_ptr<UIButton> imagery_btn_;
};