//======================================================//
//  InitInferenceState.cu
//  Lightweight inference-only state initialization
//  
//  Minimal GPU buffer allocation for forward-pass only
//  Implemented as LanguageModel::initInferenceState()
//  
//  Author: GRIM Development Team
//  Date: December 9, 2025
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../Embedding/Embedding_GPU.hpp"
#include "../Encoding/Encoding_GPU.hpp"
#include "../ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

namespace GRIM {

#ifdef USE_CUDA

void LanguageModel::initInferenceState() {
    // RULE 20: double-init is a caller-order bug, not a recoverable condition.
    if (training_state_.initialized) {
        throw std::runtime_error(
            "[InitInferenceState] FATAL: training_state_.initialized is already true. "
            "Caller invoked initInferenceState() twice (or after initTrainingState). "
            "This is a call-order bug.");
    }
    
    const auto& model_cfg = config_;
    std::cout << "[InitInferenceState] Initializing INFERENCE-ONLY state..." << std::endl;
    std::cout << "  → Skipping gradient buffers" << std::endl;
    std::cout << "  → Skipping optimizer state" << std::endl;
    std::cout << "  → Minimal activation caches only" << std::endl;

    if (!training_state_.stream_ctrl.isInitialized()) {
        throw std::runtime_error(
            "[InitInferenceState] StreamController not initialized. "
            "Caller must initialize stream_ctrl before initInferenceState().");
    }
    if (training_state_.cublas_handle.get() == nullptr) {
        throw std::runtime_error(
            "[InitInferenceState] cuBLAS handle not initialized. "
            "Caller must call initCuBLASHandle() before initInferenceState().");
    }
    if (!isPBMInitialized()) {
        throw std::runtime_error(
            "[InitInferenceState] PBM not initialized. "
            "Caller must call initPBM() before initInferenceState().");
    }
    if (!gpu_encoder_) {
        throw std::runtime_error(
            "[InitInferenceState] GPU encoder not initialized. "
            "Caller must call initGPU(weight_init_seed) before initInferenceState().");
    }
    if (!embedding_layer_ || !embedding_layer_->weightsReady()) {
        throw std::runtime_error(
            "[InitInferenceState] EmbeddingLayer not assembled by initGPU().");
    }
    if (!lm_head_layer_ || !lm_head_layer_->weightsReady()) {
        throw std::runtime_error(
            "[InitInferenceState] LMHeadLayer not assembled by initGPU().");
    }

    const auto execution_hp = HyperParameters::executionBlockConstructionHP(model_cfg);
    if (execution_hp.enabled && !execution_block_layer_) {
        throw std::runtime_error(
            "[InitInferenceState] ExecutionBlockLayer not assembled by initGPU() while execution_block is enabled.");
    }
    const auto selector_hp = HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
    if (selector_hp.enabled && !decode_time_slot_selector_layer_) {
        throw std::runtime_error(
            "[InitInferenceState] DecodeTimeSlotSelectorLayer not assembled by initGPU() while selector is enabled.");
    }
    const auto mtp_hp = HyperParameters::mtpConstructionHP(model_cfg);
    if (mtp_hp.enabled && static_cast<int>(mtp_heads_.size()) != mtp_hp.k) {
        throw std::runtime_error(
            "[InitInferenceState] MTP heads were not assembled by initGPU() while mtp is enabled.");
    }

    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(primary_stream, "InitInferenceState primary stream");
    cublasSetStream(training_state_.cublas_handle.get(), primary_stream);
    std::cout << "  ✓ Using pre-initialized StreamController and cuBLAS handle" << std::endl;
    
    // 1. Allocate runtime workspaces only.
    // Inference startup allocates maximum capacity from the authored
    // LanguageModelConfig. Prompt/decode geometry still flows through
    // BatchPayload on each request.
    HyperParameters::validateRootConfigDocument(
        model_cfg, "LanguageModel::initInferenceState");
    const auto workspace_hp = HyperParameters::trainingStateWorkspaceHP(model_cfg);
    const size_t max_batch_size = static_cast<size_t>(workspace_hp.batch_size);
    const size_t max_seq_len_cache = static_cast<size_t>(model_cfg.max_cached_seq_len);
    const size_t max_tokens = static_cast<size_t>(workspace_hp.max_tokens_per_batch);
    
    std::cout << "  ℹ Allocating activation caches: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache 
              << ", total_tokens=" << max_tokens << std::endl;

    training_state_.allocateStepDeviceWorkspaces(workspace_hp, primary_stream);
    
    // DELETED: cached_embeddings_tensor - not used in inference (encoder output computed on-the-fly)
    // DELETED: encoder_layer_caches - intermediate tensor caching moved to AutogradIntermediates
    
    // DELETED: FA bf16 buffers — FlashAttentionLayer::ensureScratch() self-manages.
    // Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    
    generation_state_.resetSession();
    std::cout << "  ✓ Reset Phase2 inference session state" << std::endl;
    
    // NOTE: encoder_workspace DELETED (Rule 20/26) — autograd forward creates its own Tensors.

    // 4. Verify ScratchBlock reasoning layer ownership (if enabled)
    if (model_cfg.use_scratch_block) {
        if (!scratch_block_layer_ || !scratch_block_layer_->isEnabled()) {
            throw std::runtime_error(
                "[InitInferenceState] config.use_scratch_block=true but ScratchBlockLayer was not assembled by initGPU()");
        }
        std::cout << "  ✓ ScratchBlock reasoning layer already assembled by initGPU() (d_model="
                  << model_cfg.d_model << ", atom_dim=" << model_cfg.scratch_block_atom_embedding_dim
                  << ", max_atoms=" << model_cfg.scratch_block_max_atoms << ")" << std::endl;
    } else if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        throw std::runtime_error(
            "[InitInferenceState] ScratchBlockLayer exists and is enabled while config.use_scratch_block=false");
    }
    
    training_state_.initialized = true;
    std::cout << "[InitInferenceState] ✓ Inference state initialized successfully" << std::endl;
    std::cout << "  Memory allocated for: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache
              << ", tokens=" << max_tokens << std::endl;
    
    // Compute actual allocated activation buffer sizes (weights excluded — loaded separately)
    const size_t token_cache_bytes = max_tokens * sizeof(float) * 3;  // token_ids + numeric_values + numeric_mask (all float Tensors)
    const size_t encoder_out_bytes = max_tokens * model_cfg.d_model * sizeof(float);
    const size_t logits_bytes = max_tokens * model_cfg.vocab_size * sizeof(float);
    const size_t workspace_bytes = static_cast<size_t>(model_cfg.d_ff) * max_seq_len_cache * 4 * sizeof(float);
    
    const size_t total_bytes = token_cache_bytes + encoder_out_bytes + logits_bytes + workspace_bytes;
    
    std::cout << "  📊 Total GPU activation memory: ~" << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "      (excludes model weights loaded from checkpoint)" << std::endl;
     
    // Ensure all initialization is complete before returning
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error("[InitInferenceState] cudaDeviceSynchronize failed: " +
                                 std::string(cudaGetErrorString(sync_err)));
    }
    
    std::cout << "[InitInferenceState] ✓ GPU synchronization complete" << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM
