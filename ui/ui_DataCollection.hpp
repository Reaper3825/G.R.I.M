#pragma once
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <optional>
#include <atomic>
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"

#include "DataCollection/data_collection_manager.hpp"

class UIDataCollectionPanel : public UIPanel {
public:
    UIDataCollectionPanel();
    ~UIDataCollectionPanel() override;
    std::shared_ptr<UIButton> DataCollectionRunButton;

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;

};