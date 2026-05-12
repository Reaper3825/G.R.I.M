//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

// Include grim_language_model_cuda.hpp for GPUGrimEncoder, FlashAttentionBF16Scratch, etc.
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "TrainingState_GPU.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"  

#include <cstdint>
#include <iostream>
#include <limits>
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
    const HyperParameters::TrainingStateRuntimeCacheHP& hp,
    cudaStream_t stream)
{
    if (initialized) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: TrainingState is already initialized");
    }
    StreamController::fatalIfDefaultStream(stream,
                                           "TrainingState::allocateStepDeviceWorkspaces");
    if (hp.d_model <= 0) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: d_model must be > 0");
    }
    if (hp.vocab_size <= 0) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: vocab_size must be > 0");
    }
    if (hp.token_capacity == 0 ||
        hp.token_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: token_capacity is outside TensorShape int capacity");
    }
    if (hp.logit_token_capacity == 0 ||
        hp.logit_token_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: logit_token_capacity is outside TensorShape int capacity");
    }
    if (hp.batch_rows == 0 ||
        hp.batch_rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "TrainingState::allocateStepDeviceWorkspaces: batch_rows is outside TensorShape int capacity");
    }

    const auto max_tokens = static_cast<int>(hp.token_capacity);
    const auto logit_token_capacity = static_cast<int>(hp.logit_token_capacity);
    const auto max_batch_size = static_cast<int>(hp.batch_rows);
    constexpr int kTextFeatureDim = Batching::BatchPayload::kTextFeatureDim;

    cached_encoder_output = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, hp.d_model),
        false,
        stream,
        "cached_encoder_output");

    cached_logits_tensor = Tensor::empty(
        TensorContract::TensorShape::make_LOGITS(logit_token_capacity, hp.vocab_size),
        false,
        stream,
        "cached_logits");
    std::cout << "✓ Allocated cached_logits [" << hp.logit_token_capacity << " x "
              << hp.vocab_size << "] LOGITS layout" << std::endl;

    cached_targets_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(logit_token_capacity, 1),
        false,
        stream,
        "cached_targets");

    cached_token_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_ids");
    std::cout << "✓ Allocated token IDs cache (Tensor API) [" << hp.token_capacity
              << "]" << std::endl;

    cached_seq_lengths_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_batch_size),
        false,
        stream,
        "cached_seq_lengths");
    std::cout << "✓ Allocated sequence-length cache (Tensor API) [" << hp.batch_rows
              << "]" << std::endl;

    cached_token_numeric_values = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_numeric_values");
    std::cout << "✓ Allocated token numeric values cache (Tensor API)" << std::endl;

    cached_token_atom_mask = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "cached_token_atom_mask");
    std::cout << "✓ Allocated token atom mask cache (Tensor API)" << std::endl;

    cached_token_text_features = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, kTextFeatureDim),
        false,
        stream,
        "cached_token_text_features");
    std::cout << "✓ Allocated token text features cache (Tensor API)" << std::endl;

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

    const auto atom_side_channel_mib =
        hp.token_capacity * (sizeof(float) + sizeof(std::uint8_t) + sizeof(std::uint32_t) +
                             kTextFeatureDim * sizeof(std::uint16_t)) / 1024 / 1024;
    std::cout << "✓ Allocated atom mask + text feature + atom flags buffers ("
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