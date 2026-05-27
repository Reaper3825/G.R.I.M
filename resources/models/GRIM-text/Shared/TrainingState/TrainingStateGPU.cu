//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "TrainingState_GPU.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"  

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

//======================================================//
//  Weight tensor accessors
//  Session 6: Embedding accessors DELETED — weights now owned by EmbeddingLayer (Pattern B).
//  Access via LanguageModel::getEmbeddingLayer()->tokenWeights().
//======================================================//

TrainingState::TrainingState() = default;
TrainingState::~TrainingState() = default;

void TrainingState::allocateStepDeviceWorkspaces(
    const HyperParameters::TrainingStateWorkspaceHP& workspace_hp,
    cudaStream_t stream)
{
    if (initialized) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: TrainingState is already initialized");
    }
    StreamController::fatalIfDefaultStream(stream,
                                           "TrainingState::allocateStepDeviceWorkspaces");
    if (workspace_hp.batch_size <= 0) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: batch_size must be > 0");
    }
    if (workspace_hp.max_tokens_per_batch <= 0) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: max_tokens_per_batch must be > 0");
    }
    const std::size_t token_capacity =
        static_cast<std::size_t>(workspace_hp.max_tokens_per_batch);

    const auto max_tokens = static_cast<int>(token_capacity);
    const auto max_batch_size = workspace_hp.batch_size;

    cached_targets_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, 1),
        false,
        stream,
        "cached_targets");

    cached_token_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_ids");
    std::cout << "✓ Allocated token IDs cache (Tensor API) [" << token_capacity
              << "]" << std::endl;

    cached_token_numeric_values = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_numeric_values");
    std::cout << "✓ Allocated token numeric values cache (Tensor API)" << std::endl;

    if (workspace_hp.mtp_enabled) {
        if (workspace_hp.mtp_k <= 0) {
            throw std::runtime_error(
                "TrainingState::allocateStepDeviceWorkspaces: mtp_enabled=true but mtp_k <= 0");
        }
        cached_mtp_shifted_targets_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(workspace_hp.mtp_k, max_tokens),
            false,
            stream,
            "cached_mtp_shifted_targets");
        std::cout << "✓ Allocated MTP shifted target cache [" << workspace_hp.mtp_k
                  << " x " << token_capacity << "]" << std::endl;
    }

    cached_token_atom_mask = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_atom_mask");
    std::cout << "✓ Allocated token atom mask cache (Tensor API)" << std::endl;

    cached_token_atom_flags = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_atom_flags");
    std::cout << "✓ Allocated token atom flags cache (Tensor API)" << std::endl;

    cached_token_to_slot_map = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_to_slot_map");
    std::cout << "✓ Allocated token-to-slot map cache (Tensor API)" << std::endl;

    read_gate_accum_tensor = Tensor::zeros({2}, stream, "read_gate_accum");
    std::cout << "✓ Allocated read-gate telemetry accumulator [sum,count]" << std::endl;

    const auto atom_side_channel_mib =
        token_capacity * (sizeof(float) + sizeof(std::uint8_t) + sizeof(std::uint32_t)) / 1024 / 1024;
    std::cout << "✓ Allocated numeric values + atom mask + atom flags buffers ("
              << atom_side_channel_mib << " MB)" << std::endl;

    sequence_weights_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_batch_size, 1),
        false,
        stream,
        "sequence_weights");
    sequence_weight_capacity = max_batch_size;
    sequence_weight_count = 0;
}

} // namespace GRIM

#endif  // USE_CUDA