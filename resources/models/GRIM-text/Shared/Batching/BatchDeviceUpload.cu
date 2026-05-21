#include "BatchPayload.hpp"
#include "BatchDeviceBindings.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../VerboseLogging.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace GRIM {

// =============================================================================
// uploadBatchToDevice
//
// Performs the H2D copies for a single BatchPayload into TrainingState's
// reusable Category-3 device cache buffers and returns a BatchDeviceBindings
// naming the resulting device pointers. Synchronizes before returning so
// callers may consume the bindings immediately.
//
// This is the single explicit payload-upload sync slice for train/eval and
// inference prefill. It belongs to Shared/Batching because it translates the
// host-side BatchPayload contract into the per-step BatchDeviceBindings device
// view. Startup/Model is intentionally not involved: startup assembles durable
// layers, while this runs once per runtime payload.
// =============================================================================
Batching::BatchDeviceBindings LanguageModel::uploadBatchToDevice(
    const Batching::BatchPayload& payload)
{
    // Re-validate (cheap) so any corruption between buildBatchPayload and the
    // upload site fails loud here instead of inside a kernel.
    payload.validate("uploadBatchToDevice");
    if (!payload.ownsHostInputData()) {
        throw std::runtime_error(
            std::string("uploadBatchToDevice: ") + payload.modeName() +
            " payload has no host input arrays to upload");
    }

    const auto& cfg = config_;
    if (payload.isTraining() && cfg.execution_mode == HyperParameters::ModelExecutionMode::INFERENCE) {
        throw std::runtime_error(
            "uploadBatchToDevice: training BatchPayload cannot be uploaded by an inference-mode LanguageModel");
    }

    if (!training_state_.initialized) {
        if (cfg.execution_mode == HyperParameters::ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
        if (!training_state_.initialized) {
            throw std::runtime_error("uploadBatchToDevice: state initialization completed but flag still false");
        }
    }

    const size_t total_tokens = static_cast<size_t>(payload.total_tokens);

    const auto& token_ids_shape = training_state_.cached_token_ids_tensor.shape.require("uploadBatchToDevice cached_token_ids_tensor");
    if (!token_ids_shape.is_2d_layout()) {
        throw std::runtime_error("uploadBatchToDevice: cached_token_ids_tensor must be a 2D token-id buffer");
    }
    const auto token_dims = token_ids_shape.as_2d();
    const size_t token_limit = static_cast<size_t>(token_dims.cols);
    if (total_tokens > token_limit) {
        throw std::runtime_error(
            "uploadBatchToDevice: total_tokens=" + std::to_string(total_tokens) +
            " exceeds token upload capacity=" + std::to_string(token_limit));
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    int* cached_token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
    if (!cached_token_ids_ptr) {
        throw std::runtime_error("uploadBatchToDevice: cached_token_ids_tensor.data is NULL");
    }
    int* cached_targets_ptr = nullptr;
    if (payload.hasTrainingTargets()) {
        cached_targets_ptr = reinterpret_cast<int*>(training_state_.cached_targets_tensor.data);
        if (!cached_targets_ptr) {
            throw std::runtime_error("uploadBatchToDevice: cached_targets_tensor.data is NULL for training payload");
        }
    }
    int* cached_mtp_shifted_targets_ptr = nullptr;
    if (!payload.mtp_shifted_targets.empty()) {
        const auto mtp_hp = HyperParameters::mtpFeatureHP(cfg);
        if (!mtp_hp.enabled) {
            throw std::runtime_error("uploadBatchToDevice: payload has MTP shifted targets but model config has mtp_enabled=false");
        }
        if (static_cast<int>(payload.mtp_shifted_targets.size()) != mtp_hp.k) {
            throw std::runtime_error(
                "uploadBatchToDevice: payload.mtp_shifted_targets.size()=" +
                std::to_string(payload.mtp_shifted_targets.size()) +
                " != config.mtp_k=" + std::to_string(mtp_hp.k));
        }
        cached_mtp_shifted_targets_ptr = reinterpret_cast<int*>(training_state_.cached_mtp_shifted_targets_tensor.data);
        if (!cached_mtp_shifted_targets_ptr) {
            throw std::runtime_error("uploadBatchToDevice: cached_mtp_shifted_targets_tensor.data is NULL for MTP payload");
        }
    }
    float* cached_numeric_values_ptr = training_state_.cached_token_numeric_values.data;
    if (!cached_numeric_values_ptr) {
        throw std::runtime_error("uploadBatchToDevice: cached_token_numeric_values.data is NULL");
    }
    float* cached_atom_mask_ptr = training_state_.cached_token_atom_mask.data;
    if (!cached_atom_mask_ptr) {
        throw std::runtime_error("uploadBatchToDevice: cached_token_atom_mask.data is NULL");
    }
    int32_t* cached_slot_map_ptr = reinterpret_cast<int32_t*>(training_state_.cached_token_to_slot_map.data);
    if (!cached_slot_map_ptr) {
        throw std::runtime_error("uploadBatchToDevice: cached_token_to_slot_map.data is NULL — "
            "slot map is unconditionally allocated in InitTrainingState");
    }

    const size_t input_ids_bytes   = payload.inputIdBytes();
    const size_t target_ids_bytes  = payload.targetIdBytes();
    const size_t numeric_val_bytes = payload.numericValueBytes();
    const size_t atom_mask_bytes   = payload.atomMaskBytes();
    const size_t atom_flag_bytes   = payload.atomFlagBytes();
    if (!training_state_.cached_token_atom_flags.data && !payload.atom_flags.empty()) {
        throw std::runtime_error("uploadBatchToDevice: cached_token_atom_flags.data is NULL but payload.atom_flags is populated");
    }
    const size_t slot_map_bytes  = payload.slotMapBytes();

    auto copy_start = std::chrono::high_resolution_clock::now();

    // Round 1: input_ids + target_ids. Targets arrive pre-masked from
    // buildBatchPayload Phase 4b; payload.lm_valid_tokens already accounts for
    // the post-masking LM-supervised count.
    CUDA_CHECK(cudaMemcpyAsync(cached_token_ids_ptr, payload.input_ids.data(),
        input_ids_bytes, cudaMemcpyHostToDevice, stream));
    if (payload.hasTrainingTargets()) {
        CUDA_CHECK(cudaMemcpyAsync(cached_targets_ptr, payload.target_ids.data(),
            target_ids_bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Round 2: numeric_values + atom_mask.
    CUDA_CHECK(cudaMemcpyAsync(cached_numeric_values_ptr, payload.numeric_values.data(),
        numeric_val_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(cached_atom_mask_ptr), payload.atom_mask.data(),
        atom_mask_bytes, cudaMemcpyHostToDevice, stream));

    // Round 3: atom_flags.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (training_state_.cached_token_atom_flags.data) {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint32_t*>(training_state_.cached_token_atom_flags.data), payload.atom_flags.data(),
            atom_flag_bytes, cudaMemcpyHostToDevice, stream));
    }

    // Round 4: token_to_slot_map.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(cached_slot_map_ptr, payload.token_to_slot_map.data(),
        slot_map_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Round 5: MTP shifted targets. These are Phase1-authored payload arrays;
    // upload owns the H2D copy so MTP loss consumes BatchDeviceBindings instead
    // of allocating per-head target buffers inside loss assembly.
    if (cached_mtp_shifted_targets_ptr) {
        const size_t mtp_head_bytes = static_cast<size_t>(payload.total_tokens) * sizeof(int);
        for (int k = 0; k < static_cast<int>(payload.mtp_shifted_targets.size()); ++k) {
            CUDA_CHECK(cudaMemcpyAsync(
                cached_mtp_shifted_targets_ptr + static_cast<size_t>(k) * payload.total_tokens,
                payload.mtp_shifted_targets[k].data(),
                mtp_head_bytes,
                cudaMemcpyHostToDevice,
                stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    if constexpr (VerboseLogging::ENABLE_VOCAB_TIMING_LOGS) {
        fprintf(stderr, "[VOCAB_TIMING] uploadBatchToDevice complete: %.2f ms\n", copy_ms);
    }

    // The bindings struct returned below is the canonical reader-facing device
    // view for this step. Fixed-shape training geometry is HyperParameters-owned
    // and must match the realized payload here; TrainingState must not mirror
    // per-step semantics as a hidden global mailbox.

    Batching::BatchDeviceBindings bindings;
    bindings.d_input_ids        = cached_token_ids_ptr;
    bindings.d_target_ids       = cached_targets_ptr;
    bindings.d_numeric_values   = cached_numeric_values_ptr;
    bindings.d_atom_mask        = reinterpret_cast<uint8_t*>(cached_atom_mask_ptr);
    bindings.d_atom_flags       = training_state_.cached_token_atom_flags.data
        ? reinterpret_cast<uint32_t*>(training_state_.cached_token_atom_flags.data)
        : nullptr;
    bindings.d_token_to_slot_map = cached_slot_map_ptr;
    bindings.d_mtp_shifted_targets = cached_mtp_shifted_targets_ptr;

    if (payload.isTraining()) {
        if (cfg.max_cached_batch <= 0) {
            throw std::runtime_error(
                "uploadBatchToDevice: training payload requires config.max_cached_batch > 0");
        }
        if (cfg.max_seq_len <= 0) {
            throw std::runtime_error(
                "uploadBatchToDevice: training payload requires config.max_seq_len > 0");
        }
        if (payload.batch_size != cfg.max_cached_batch || payload.max_seq_len != cfg.max_seq_len) {
            throw std::runtime_error(
                "uploadBatchToDevice: fixed-shape training payload geometry ("
                + std::to_string(payload.batch_size) + "x" + std::to_string(payload.max_seq_len)
                + ") does not match HyperParameters-owned model geometry ("
                + std::to_string(cfg.max_cached_batch) + "x" + std::to_string(cfg.max_seq_len)
                + ")");
        }
    }
    return bindings;
}

}  // namespace GRIM
