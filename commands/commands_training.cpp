#include "commands_training.hpp"
#include "logger.hpp"
#include "ui/ui_training_panel.hpp"
#include "ui/ui_root.hpp"
#include <memory>

namespace {
void openModelConfig() {
    UIRoot::get().postTask([] {
        auto panel = std::dynamic_pointer_cast<UITrainingPanel>(
            UIRoot::get().getPanelByKey("training_control"));
        if (!panel) {
            LOG_ERROR("Training", "Training panel is unavailable; UI provider may be unloaded");
            return;
        }
        panel->setView(TrainingPanelTab::ModelConfig);
        UIRoot::get().setVisible("training_control", true);
    });
}
}

namespace GRIM {
namespace Training {

void startTraining() {
    LOG_DEBUG("Training", "Local training is disabled; opening model config creator");
    
    openModelConfig();
}

void stopTraining() {
    LOG_DEBUG("Training", "Local training controls are not available in the HPC workflow");
    
    openModelConfig();
}

} // namespace Training
} // namespace GRIM
