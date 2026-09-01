#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
#endif

#include <cstdint>
#include <vector>

namespace GRIM {
struct OptimizerState;
struct EncodingLayerParameterTensors;
struct FeedForwardParameterTensors;
struct Tensor;
namespace Config {
struct AiConfigSnapshot;
}
namespace HyperParameters {
struct EmbeddingLayerConstructionHP;
struct EncoderLayerConstructionHP;
struct ExecutionBlockConstructionHP;
struct LMHeadLayerConstructionHP;
struct AtomInsertionBoundaryProjectionHP;
struct NumberEncoderConstructionHP;
struct SlotSeedEncoderConstructionHP;
}
}

namespace GRIMText::Training::Startup {
struct GpuModelState;
}

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

// Phase-1 startup tensor registration boundary.
//
// Ownership contract:
// - TrainingContext::parameter_registry owns migrated writable parameter bundles
//   and the durable ParameterGroup inventory declared in ParameterRegistry.hpp.
// - This module discovers trainable tensors, writes non-owning ParameterGroup
//   metadata, and binds externally owned optimizer moment tensors.
// - OptimizerState owns Adam/RAdam moment tensor storage; ParameterGroup entries
//   only borrow those tensors after bindOptimizerState().
struct OutputUnigramPriorView {
    const float* log_bias = nullptr;
    std::size_t size = 0;
    std::uint32_t vocab_size = 0;
    std::uint32_t seen_tokens = 0;
    std::uint64_t total_targets = 0;
};

void initializeFeedForwardParameterTensors(
    std::vector<GRIM::FeedForwardParameterTensors>& feed_forward_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeEncodingLayerParameterTensors(
    std::vector<GRIM::EncodingLayerParameterTensors>& encoding_layer_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeEmbeddingParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::EmbeddingLayerConstructionHP& embedding_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream,
    bool requires_grad);

void initializeLmHeadParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::LMHeadLayerConstructionHP& lm_head_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream,
    GRIM::Tensor* tied_embedding_weights,
    const OutputUnigramPriorView* output_unigram_prior);

void initializeAtomInsertionBoundaryParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::AtomInsertionBoundaryProjectionHP& atom_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeExecutionBlockParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeNumberEncoderParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::NumberEncoderConstructionHP& number_encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeSelectorParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    bool selector_enabled,
    int d_model,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeLocalAtomRetrievalParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    int d_model,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void initializeSlotSeedEncoderParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream);

void buildParameterGroups(const GRIM::Config::AiConfigSnapshot& config,
                          GRIMText::Training::Startup::GpuModelState& gpu_model_state,
                          ::ParameterRegistry::StartupParameterRegistry& parameter_registry);
void bindOptimizerState(::ParameterRegistry::StartupParameterRegistry& parameter_registry,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream);

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup::ModelRegistration
