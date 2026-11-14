#pragma once
#include "ui_panel_service.hpp"
#include "primitives/ui_panel.hpp"
#include "primitives/ui_scrollbox.hpp"
#include <map>

namespace GRIM::UI {
class ProviderDefinedPanel final : public UIPanel {
public:
    ProviderDefinedPanel(nlohmann::json definition, std::shared_ptr<ProviderPanelLease> owner);
    void applyDefinition(nlohmann::json definition);
    void setVisible(bool visible) override;
    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;
private:
    void layout();
    void emit(nlohmann::json event);
    void interact(const std::string& id);
    nlohmann::json definition_;
    std::shared_ptr<ProviderPanelLease> owner_;
    std::shared_ptr<UIScrollBox> scroll_;
    std::map<std::string, std::shared_ptr<Widget>> widgets_;
    Vec2 reportedSize_{-1, -1};
};
}
