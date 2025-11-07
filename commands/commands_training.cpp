#include "commands_training.hpp"
#include "logger.hpp"
#include "ui/ui_training_panel.hpp"
#include <memory>

// External reference to the training panel instance
extern std::shared_ptr<UITrainingPanel> g_trainingPanel;

namespace GRIM {
namespace Training {

void startTraining() {
    LOG_DEBUG("Training", "start_training command invoked");
    
    if (!g_trainingPanel) {
        LOG_ERROR("Training", "Training panel not initialized");
        return;
    }
    
    // Trigger training start through the panel
    g_trainingPanel->startTrainingSession();
}

void stopTraining() {
    LOG_DEBUG("Training", "stop_training command invoked");
    
    if (!g_trainingPanel) {
        LOG_ERROR("Training", "Training panel not initialized");
        return;
    }
    
    // Trigger training stop through the panel
    g_trainingPanel->stopTrainingSession();
}

} // namespace Training
} // namespace GRIM
