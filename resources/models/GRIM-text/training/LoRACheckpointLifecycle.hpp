#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace GRIMText::Training {

struct TrainingContext;

struct LoRACheckpointLifecycleState {
    bool active = false;
    std::string adapter_id;
    std::optional<std::uint64_t> parent_adapter_revision;
    std::uint64_t next_output_adapter_revision = 0;
    std::string base_checkpoint_identity;
    std::array<std::uint8_t, 32> base_checkpoint_sha256{};
    std::string training_config_canonical;
    std::array<std::uint8_t, 32> training_config_sha256{};
};

void initializeLoRACheckpointLifecycle(TrainingContext& ctx);

std::string saveLoRATrainingCheckpointAtBoundary(
    TrainingContext& ctx,
    const std::string& boundary,
    int epochs_completed);

} // namespace GRIMText::Training