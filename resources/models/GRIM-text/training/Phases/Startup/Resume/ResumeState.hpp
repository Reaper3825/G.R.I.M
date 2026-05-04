#pragma once

#include <cstdint>
#include <string>

namespace GRIMText::Training {

struct TrainingContext;

struct ResumeState {
    bool resumed = false;
    std::string loaded_checkpoint_path;
    std::string optimizer_sidecar_path;

    int optimizer_step = 0;
    int global_step = 0;
    float best_val_loss = 0.0f;
    int epochs_completed = 0;
    int accumulation_position = 0;
};

ResumeState captureResumeState(const TrainingContext& ctx);
void ResumeStateReady(TrainingContext& ctx);

} // namespace GRIMText::Training

