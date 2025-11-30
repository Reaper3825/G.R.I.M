#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_slider.hpp"
#include "ui_progress_bar.hpp"
#include "ui_layout_box.hpp"
#include "ui_inputbox.hpp"
#include "ui_graph.hpp"
#include "ui_training_config.hpp"
#include "../control/training_controller.hpp"
#include "DataCollection/data_collection_manager.hpp"
#include "../hardware/resource_values.hpp"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <optional>
#include <atomic>

class OverlayRenderer;
struct InputState;

struct ModelPreset {
    std::string Name;
    std::string Description;
    std::string ModelID;
    std::string ModelPath;
    std::string VocabularyPath;
    std::vector<std::string> AdditionalFiles;
    
    
};


class UIModelPanel : public UIPanel {
public:
    UIModelPanel();

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;

    // Set the current model preset
    void refreshModelPresets();
    void setActiveModelPreset(const std::string& modelName);






}
