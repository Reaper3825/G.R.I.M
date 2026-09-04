#pragma once

#include "LoRAMatrixClass.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace GRIM::Config {
struct AiConfigSnapshot;
}

namespace GRIM::Checkpoint {

using LoRAMatrixClass = GRIM::LoRAMatrixClass;

enum class LoRAMatrixOrientation : std::uint8_t {
    TRANSPOSED_WEIGHT,
    DIRECT_WEIGHT
};

struct LoRAHostTensorState {
    std::vector<std::uint32_t> shape;
    std::vector<float> values;
};

struct LoRATargetTrainingState {
    std::string target_identity;
    std::uint32_t layer_index = 0;
    LoRAMatrixClass matrix_class = LoRAMatrixClass::QKV;
    std::vector<std::uint32_t> base_shape;
    LoRAMatrixOrientation orientation = LoRAMatrixOrientation::TRANSPOSED_WEIGHT;
    std::uint32_t rank = 0;
    float alpha = 0.0f;
    LoRAHostTensorState A;
    LoRAHostTensorState B;
    std::optional<LoRAHostTensorState> A_gradient;
    std::optional<LoRAHostTensorState> B_gradient;
    LoRAHostTensorState A_first_moment;
    LoRAHostTensorState A_second_moment;
    LoRAHostTensorState B_first_moment;
    LoRAHostTensorState B_second_moment;
};

struct LoRAArchitectureMetadata {
    std::uint32_t d_model = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t num_kv_heads = 0;
    std::uint32_t qkv_dim = 0;
    std::uint32_t d_ff = 0;
};

struct LoRAOptimizerTrainingState {
    std::string family;
    std::string canonical_config;
    std::uint64_t optimizer_step = 0;
    float learning_rate_lora = 0.0f;
    std::vector<std::uint8_t> lr_scheduler_state;
    std::vector<std::uint8_t> soft_restart_state;
};

struct LoRATrainingProgressState {
    std::uint64_t global_step = 0;
    std::uint64_t epochs_completed = 0;
    std::uint64_t batch_cursor = 0;
    std::uint32_t accumulation_cursor = 0;
    float best_validation_loss = 0.0f;
    std::vector<std::uint64_t> data_order;
};

struct LoRARngTrainingState {
    std::uint64_t base_seed = 0;
    std::uint64_t data_seed = 0;
    std::uint64_t init_seed = 0;
    std::uint64_t cuda_seed = 0;
    std::string data_rng_state;
    std::vector<std::uint8_t> cuda_rng_state;
};

struct LoRATrainingCheckpointSnapshot {
    std::filesystem::path source_path;
    std::string checkpoint_id;
    std::uint64_t creation_timestamp_ms = 0;
    std::string adapter_id;
    std::optional<std::uint64_t> parent_adapter_revision;
    std::uint64_t output_adapter_revision = 0;
    std::string base_checkpoint_identity;
    std::array<std::uint8_t, 32> base_checkpoint_sha256{};
    LoRAArchitectureMetadata architecture;
    std::string training_config_canonical;
    std::array<std::uint8_t, 32> training_config_sha256{};
    std::vector<LoRATargetTrainingState> targets;
    LoRAOptimizerTrainingState optimizer;
    LoRATrainingProgressState progress;
    LoRARngTrainingState rng;
};

// The selected model.grimcfg owns the directory identity. A bare checkpoint
// name resolves under <selected-model>/lora_checkpoints; paths and traversal
// components are rejected.
std::filesystem::path resolveLoRATrainingCheckpointPath(
    const Config::AiConfigSnapshot& config,
    const std::string& checkpoint_name);

// Strictly reads and validates a complete resumable LoRA training checkpoint.
// expected_base_checkpoint_identity and expected_base_checkpoint_sha256 must
// describe the model checkpoint already loaded by startup. The config digest
// is authored by the future LoRA grouping/config serialization boundary. This
// function does not mutate GPU tensors or TrainingContext and is intentionally
// not wired yet.
LoRATrainingCheckpointSnapshot loadLoRATrainingCheckpoint(
    const Config::AiConfigSnapshot& config,
    const std::string& checkpoint_name,
    const std::string& expected_base_checkpoint_identity,
    const std::array<std::uint8_t, 32>& expected_base_checkpoint_sha256,
    const std::array<std::uint8_t, 32>& expected_training_config_sha256);

} // namespace GRIM::Checkpoint