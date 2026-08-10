#include "commands_training.hpp"
#include "logger.hpp"
#include "ui/ui_training_panel.hpp"
#include <memory>

// External reference to the training panel instance
extern std::shared_ptr<UITrainingPanel> g_trainingPanel;

namespace GRIM {
namespace Training {

void startTraining() {
    LOG_DEBUG("Training", "Local training is disabled; opening model config creator");
    
    if (!g_trainingPanel) {
        LOG_ERROR("Training", "Training panel not initialized");
        return;
    }
    
    g_trainingPanel->setView(TrainingPanelTab::ModelConfig);
    g_trainingPanel->setVisible(true);
}

void stopTraining() {
    LOG_DEBUG("Training", "Local training controls are not available in the HPC workflow");
    
    if (!g_trainingPanel) {
        LOG_ERROR("Training", "Training panel not initialized");
        return;
    }
    
    g_trainingPanel->setView(TrainingPanelTab::ModelConfig);
    g_trainingPanel->setVisible(true);
}

} // namespace Training
} // namespace GRIM
