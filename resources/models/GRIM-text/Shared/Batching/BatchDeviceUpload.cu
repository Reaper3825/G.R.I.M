#include "BatchPayload.hpp"
#include "BatchDeviceBindings.hpp"
#include "BatchDeviceStorage.hpp"
#include "BatchDeviceUpload.hpp"

#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../VerboseLogging.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace GRIM {
namespace Batching {

namespace {

inline void checkCudaResult(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("BatchDeviceUpload CUDA failure at ") + file + ":" + std::to_string(line) +
            " for " + expr + ": " + cudaGetErrorString(result));
    }
}

const BatchDeviceStorage& requireDeviceStorage(
    const BatchPayload& payload,
    const char* caller)
{
    if (!payload.device_storage) {
        throw std::runtime_error(
            std::string(caller) + ": BatchPayload.device_storage is NULL - "
            "caller must attach explicit batch device storage before upload");
    }
    return *payload.device_storage;
}

}  // namespace

#define BATCH_UPLOAD_CUDA_CHECK(expr) ::GRIM::Batching::checkCudaResult((expr), #expr, __FILE__, __LINE__)

// =============================================================================
// uploadBatchToDevice
//
// Performs the H2D copies for a single BatchPayload into the payload-attached
// BatchDeviceStorage owner and returns a BatchDeviceBindings
// naming the resulting device pointers. Synchronizes before returning so
// callers may consume the bindings immediately.
//
// This is the single explicit payload-upload sync slice for train/eval and
// inference prefill. It belongs to Shared/Batching because it translates the
// host-side BatchPayload contract into the per-step BatchDeviceBindings device
// view. Startup/Model is intentionally not involved: startup assembles durable
// layers, while this runs once per runtime payload.
// =============================================================================
BatchDeviceBindings uploadBatchToDevice(
    const Config::AiConfigSnapshot& config,
    BatchPayload& payload,
    cudaStream_t stream)
{
    // Re-validate (cheap) so any corruption between buildBatchPayload and the
    // upload site fails loud here instead of inside a kernel.
    payload.validate("uploadBatchToDevice");
    if (!payload.ownsHostInputData()) {
        throw std::runtime_error(
            std::string("uploadBatchToDevice: ") + payload.modeName() +
            " payload has no host input arrays to upload");
    }

    const int cfg_batch_size = HyperParameters::snapshotTrainingConfigField<int>(config, "batch_size");
    const int cfg_max_cached_seq_len = HyperParameters::snapshotTrainingConfigField<int>(config, "max_cached_seq_len");
    const int cfg_max_tokens_per_batch = HyperParameters::snapshotTrainingConfigField<int>(config, "max_tokens_per_batch");
    if (payload.isTraining() && HyperParameters::snapshotExecutionMode(config) == HyperParameters::ModelExecutionMode::INFERENCE) {
        throw std::runtime_error(
            "uploadBatchToDevice: training BatchPayload cannot be uploaded by an inference-mode LanguageModel");
    }

    if (!stream) {
        throw std::runtime_error(
            "uploadBatchToDevice: stream is NULL - caller must pass the active upload stream");
    }

    const size_t total_tokens = static_cast<size_t>(payload.total_tokens);
    const BatchDeviceStorage& storage = requireDeviceStorage(payload, "uploadBatchToDevice");

    if (payload.isTraining()) {
        if (payload.batch_size != cfg_batch_size) {
            throw std::runtime_error(
                "uploadBatchToDevice: training payload.batch_size=" +
                std::to_string(payload.batch_size) +
            " != model batch_size=" + std::to_string(cfg_batch_size));
        }
        if (payload.max_seq_len != cfg_max_cached_seq_len) {
            throw std::runtime_error(
                "uploadBatchToDevice: training payload.max_seq_len=" +
                std::to_string(payload.max_seq_len) +
                " != model max_cached_seq_len=" + std::to_string(cfg_max_cached_seq_len));
        }
        if (payload.total_tokens != cfg_max_tokens_per_batch) {
            throw std::runtime_error(
                "uploadBatchToDevice: training payload.total_tokens=" +
                std::to_string(payload.total_tokens) +
                " != model max_tokens_per_batch=" + std::to_string(cfg_max_tokens_per_batch));
        }
    } else {
        if (payload.batch_size > cfg_batch_size) {
            throw std::runtime_error(
                "uploadBatchToDevice: payload.batch_size=" +
                std::to_string(payload.batch_size) +
            " exceeds model batch_size=" + std::to_string(cfg_batch_size));
        }
        if (payload.max_seq_len > cfg_max_cached_seq_len) {
            throw std::runtime_error(
                "uploadBatchToDevice: payload.max_seq_len=" +
                std::to_string(payload.max_seq_len) +
                " exceeds model max_cached_seq_len=" + std::to_string(cfg_max_cached_seq_len));
        }
        if (payload.total_tokens > cfg_max_tokens_per_batch) {
            throw std::runtime_error(
                "uploadBatchToDevice: payload.total_tokens=" +
                std::to_string(payload.total_tokens) +
                " exceeds model max_tokens_per_batch=" + std::to_string(cfg_max_tokens_per_batch));
        }
    }

    const auto& token_ids_shape = storage.input_ids_tensor.shape.require("uploadBatchToDevice input_ids_tensor");
    if (!token_ids_shape.is_2d_layout()) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.input_ids_tensor must be a 2D token-id buffer");
    }
    const size_t token_capacity = static_cast<size_t>(token_ids_shape.as_2d().cols);
    if (total_tokens > token_capacity) {
        throw std::runtime_error(
            "uploadBatchToDevice: total_tokens=" + std::to_string(total_tokens) +
            " exceeds token-id buffer capacity=" + std::to_string(token_capacity));
    }

    if (payload.hasTrainingTargets()) {
        const auto& targets_shape = storage.target_ids_tensor.shape.require("uploadBatchToDevice target_ids_tensor");
        if (!targets_shape.is_2d_layout()) {
            throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.target_ids_tensor must be a 2D target upload buffer");
        }
        const size_t target_capacity = static_cast<size_t>(targets_shape.as_2d().rows);
        if (total_tokens > target_capacity) {
            throw std::runtime_error(
                "uploadBatchToDevice: total_tokens=" + std::to_string(total_tokens) +
                " exceeds target buffer capacity=" + std::to_string(target_capacity));
        }
    }

    int* cached_token_ids_ptr = reinterpret_cast<int*>(storage.input_ids_tensor.data);
    if (!cached_token_ids_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.input_ids_tensor.data is NULL");
    }
    int* cached_targets_ptr = nullptr;
    if (payload.hasTrainingTargets()) {
        cached_targets_ptr = reinterpret_cast<int*>(storage.target_ids_tensor.data);
        if (!cached_targets_ptr) {
            throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.target_ids_tensor.data is NULL for training payload");
        }
    }
    float* cached_numeric_values_ptr = storage.numeric_values_tensor.data;
    if (!cached_numeric_values_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.numeric_values_tensor.data is NULL");
    }
    float* cached_atom_mask_ptr = storage.atom_mask_tensor.data;
    if (!cached_atom_mask_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.atom_mask_tensor.data is NULL");
    }
    int32_t* cached_slot_map_ptr = reinterpret_cast<int32_t*>(storage.token_to_slot_index_map_tensor.data);
    if (!cached_slot_map_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.token_to_slot_index_map_tensor.data is NULL");
    }
    uint32_t* cached_atom_entry_ids_ptr =
        reinterpret_cast<uint32_t*>(storage.atom_entry_ids_tensor.data);
    if (!cached_atom_entry_ids_ptr) {
        throw std::runtime_error(
            "uploadBatchToDevice: BatchDeviceStorage.atom_entry_ids_tensor.data is NULL");
    }
    int* cached_atom_positions_ptr = reinterpret_cast<int*>(storage.atom_positions_tensor.data);
    if (!cached_atom_positions_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.atom_positions_tensor.data is NULL");
    }
    int* cached_atom_types_ptr = reinterpret_cast<int*>(storage.atom_types_tensor.data);
    if (!cached_atom_types_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.atom_types_tensor.data is NULL");
    }

    const size_t input_ids_bytes   = payload.inputIdBytes();
    const size_t target_ids_bytes  = payload.targetIdBytes();
    const size_t numeric_val_bytes = payload.numericValueBytes();
    const size_t atom_mask_bytes   = payload.atomMaskBytes();
    const size_t atom_flag_bytes   = payload.atomFlagBytes();
    if (!storage.atom_flags_tensor.data && !payload.atom_flags.empty()) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.atom_flags_tensor.data is NULL but payload.atom_flags is populated");
    }
    const size_t slot_map_bytes  = payload.slotMapBytes();
    const size_t atom_position_bytes = payload.atomPositionBytes();
    const size_t atom_type_bytes = payload.atomTypeBytes();

    auto copy_start = std::chrono::high_resolution_clock::now();

    // Round 1: input_ids + target_ids. Targets arrive pre-masked from
    // buildBatchPayload Phase 4b; payload.lm_valid_tokens already accounts for
    // the post-masking LM-supervised count.
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(cached_token_ids_ptr, payload.input_ids.data(),
        input_ids_bytes, cudaMemcpyHostToDevice, stream));
    if (payload.hasTrainingTargets()) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(cached_targets_ptr, payload.target_ids.data(),
            target_ids_bytes, cudaMemcpyHostToDevice, stream));
    }
    BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Round 2: numeric_values + atom_mask.
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(cached_numeric_values_ptr, payload.numeric_values.data(),
        numeric_val_bytes, cudaMemcpyHostToDevice, stream));
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(cached_atom_mask_ptr), payload.atom_mask.data(),
        atom_mask_bytes, cudaMemcpyHostToDevice, stream));

    // Round 3: atom_flags.
    BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (storage.atom_flags_tensor.data) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint32_t*>(storage.atom_flags_tensor.data), payload.atom_flags.data(),
            atom_flag_bytes, cudaMemcpyHostToDevice, stream));
    }
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
        cached_atom_entry_ids_ptr,
        payload.atom_entry_ids.data(),
        payload.atom_entry_ids.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice,
        stream));

    // Round 4: token_to_slot_index_map.
    BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(cached_slot_map_ptr, payload.token_to_slot_index_map.data(),
        slot_map_bytes, cudaMemcpyHostToDevice, stream));
    BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Round 5: compact authored atom positions/types.
    if (payload.authoredAtomCount() > payload.total_tokens) {
        throw std::runtime_error(
            "uploadBatchToDevice: payload.authoredAtomCount()=" +
            std::to_string(payload.authoredAtomCount()) +
            " exceeds payload.total_tokens=" + std::to_string(payload.total_tokens));
    }
    if (atom_position_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_atom_positions_ptr,
            payload.atom_positions.data(),
            atom_position_bytes,
            cudaMemcpyHostToDevice,
            stream));
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_atom_types_ptr,
            payload.atom_types.data(),
            atom_type_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }
    BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Round 6: NumberEncoder digit-place channels (compact, atom-aligned).
    int* cached_atom_digit_values_ptr = nullptr;
    int* cached_atom_digit_pow10_ptr = nullptr;
    float* cached_atom_digit_mask_ptr = nullptr;
    float* cached_atom_digit_slot_features_ptr = nullptr;
    float* cached_atom_global_features_ptr = nullptr;
    if (payload.number_encoder_digit_slots > 0) {
        if (storage.number_encoder_digit_slots_capacity != payload.number_encoder_digit_slots) {
            throw std::runtime_error(
                "uploadBatchToDevice: payload.number_encoder_digit_slots=" +
                std::to_string(payload.number_encoder_digit_slots) +
                " != BatchDeviceStorage.number_encoder_digit_slots_capacity=" +
                std::to_string(storage.number_encoder_digit_slots_capacity));
        }
        cached_atom_digit_values_ptr = reinterpret_cast<int*>(storage.atom_digit_values_tensor.data);
        cached_atom_digit_pow10_ptr = reinterpret_cast<int*>(storage.atom_digit_pow10_index_tensor.data);
        cached_atom_digit_mask_ptr = storage.atom_digit_mask_tensor.data;
        cached_atom_digit_slot_features_ptr = storage.atom_digit_slot_features_tensor.data;
        cached_atom_global_features_ptr = storage.atom_global_features_tensor.data;
        if (!cached_atom_digit_values_ptr || !cached_atom_digit_pow10_ptr ||
            !cached_atom_digit_mask_ptr || !cached_atom_digit_slot_features_ptr ||
            !cached_atom_global_features_ptr) {
            throw std::runtime_error(
                "uploadBatchToDevice: NumberEncoder storage tensors are NULL while "
                "payload.number_encoder_digit_slots > 0 — createBatchDeviceStorage must "
                "allocate them when number_encoder_enabled=true");
        }
        if (!payload.atom_digit_values.empty()) {
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_atom_digit_values_ptr, payload.atom_digit_values.data(),
                payload.atom_digit_values.size() * sizeof(int),
                cudaMemcpyHostToDevice, stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_atom_digit_pow10_ptr, payload.atom_digit_pow10_index.data(),
                payload.atom_digit_pow10_index.size() * sizeof(int),
                cudaMemcpyHostToDevice, stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_atom_digit_mask_ptr, payload.atom_digit_mask.data(),
                payload.atom_digit_mask.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_atom_digit_slot_features_ptr, payload.atom_digit_slot_features.data(),
                payload.atom_digit_slot_features.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_atom_global_features_ptr, payload.atom_global_features.data(),
                payload.atom_global_features.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream));
        }
        BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Candidate atom-entry pool (arg/option selector). row_atom_offset is uploaded
    // whenever the selector-owned pool storage exists, even for batches
    // with zero atoms (empty windows); values/types only when the pool is non-empty.
    float* cached_pool_values_ptr = nullptr;
    double* cached_pool_float_values_ptr = nullptr;
    int64_t* cached_pool_int_values_ptr = nullptr;
    uint8_t* cached_pool_kinds_ptr = nullptr;
    int*   cached_pool_types_ptr = nullptr;
    int*   cached_row_atom_offset_ptr = nullptr;
    int*   cached_arg_select_targets_ptr = nullptr;
    if (!payload.row_atom_offset.empty()) {
        if (!storage.pool_numeric_values_tensor.data ||
            !storage.row_atom_offset_tensor.data ||
            !storage.arg_select_targets_tensor.data) {
            throw std::runtime_error(
                "uploadBatchToDevice: selector candidate metadata is present but "
                "selector-owned device storage is unavailable");
        }
        cached_pool_values_ptr = storage.pool_numeric_values_tensor.data;
        cached_pool_float_values_ptr =
            reinterpret_cast<double*>(storage.pool_numeric_float_values_tensor.data);
        cached_pool_int_values_ptr =
            reinterpret_cast<int64_t*>(storage.pool_numeric_int_values_tensor.data);
        cached_pool_kinds_ptr =
            reinterpret_cast<uint8_t*>(storage.pool_numeric_kinds_tensor.data);
        cached_pool_types_ptr = reinterpret_cast<int*>(storage.pool_atom_types_tensor.data);
        if (!cached_pool_float_values_ptr || !cached_pool_int_values_ptr ||
            !cached_pool_kinds_ptr || !cached_pool_types_ptr) {
            throw std::runtime_error(
                "uploadBatchToDevice: exact atom-entry pool storage is incomplete");
        }
        cached_row_atom_offset_ptr = reinterpret_cast<int*>(storage.row_atom_offset_tensor.data);
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_row_atom_offset_ptr, payload.row_atom_offset.data(),
            payload.row_atom_offset.size() * sizeof(int),
            cudaMemcpyHostToDevice, stream));
        if (storage.arg_select_targets_tensor.data && !payload.arg_select_targets.empty()) {
            cached_arg_select_targets_ptr = reinterpret_cast<int*>(storage.arg_select_targets_tensor.data);
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_arg_select_targets_ptr, payload.arg_select_targets.data(),
                payload.arg_select_targets.size() * sizeof(int),
                cudaMemcpyHostToDevice, stream));
        }
        if (payload.num_pool_atoms > 0) {
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_pool_values_ptr, payload.pool_numeric_values.data(),
                payload.pool_numeric_values.size() * sizeof(float),
                cudaMemcpyHostToDevice, stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_pool_float_values_ptr,
                payload.pool_numeric_float_values.data(),
                payload.pool_numeric_float_values.size() * sizeof(double),
                cudaMemcpyHostToDevice,
                stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_pool_int_values_ptr,
                payload.pool_numeric_int_values.data(),
                payload.pool_numeric_int_values.size() * sizeof(int64_t),
                cudaMemcpyHostToDevice,
                stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_pool_kinds_ptr,
                payload.pool_numeric_kinds.data(),
                payload.pool_numeric_kinds.size() * sizeof(uint8_t),
                cudaMemcpyHostToDevice,
                stream));
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_pool_types_ptr, payload.pool_atom_types.data(),
                payload.pool_atom_types.size() * sizeof(int),
                cudaMemcpyHostToDevice, stream));
        }
        BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto copy_end = std::chrono::high_resolution_clock::now();
    auto copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    if constexpr (VerboseLogging::ENABLE_VOCAB_TIMING_LOGS) {
        fprintf(stderr, "[VOCAB_TIMING] uploadBatchToDevice complete: %.2f ms\n", copy_ms);
    }

    // The bindings struct returned below is the canonical reader-facing device
    // view for this step. Batch geometry and valid-token counts stay on the
    // Phase1-authored BatchPayload; durable buffer ownership stays on the
    // explicit payload-attached BatchDeviceStorage owner.

    Batching::BatchDeviceBindings bindings;
    bindings.d_input_ids        = cached_token_ids_ptr;
    bindings.d_target_ids       = cached_targets_ptr;
    bindings.d_numeric_values   = cached_numeric_values_ptr;
    bindings.d_atom_mask        = reinterpret_cast<uint8_t*>(cached_atom_mask_ptr);
    bindings.d_atom_flags       = storage.atom_flags_tensor.data
        ? reinterpret_cast<uint32_t*>(storage.atom_flags_tensor.data)
        : nullptr;
    bindings.d_atom_entry_ids   = cached_atom_entry_ids_ptr;
    bindings.d_token_to_slot_index_map = cached_slot_map_ptr;
    bindings.d_atom_positions   = cached_atom_positions_ptr;
    bindings.d_atom_types       = cached_atom_types_ptr;
    bindings.d_atom_digit_values        = cached_atom_digit_values_ptr;
    bindings.d_atom_digit_pow10_index   = cached_atom_digit_pow10_ptr;
    bindings.d_atom_digit_mask          = cached_atom_digit_mask_ptr;
    bindings.d_atom_digit_slot_features = cached_atom_digit_slot_features_ptr;
    bindings.d_atom_global_features     = cached_atom_global_features_ptr;
    bindings.d_pool_numeric_values = (payload.num_pool_atoms > 0) ? cached_pool_values_ptr : nullptr;
    bindings.d_pool_numeric_float_values =
        (payload.num_pool_atoms > 0) ? cached_pool_float_values_ptr : nullptr;
    bindings.d_pool_numeric_int_values =
        (payload.num_pool_atoms > 0) ? cached_pool_int_values_ptr : nullptr;
    bindings.d_pool_numeric_kinds =
        (payload.num_pool_atoms > 0) ? cached_pool_kinds_ptr : nullptr;
    bindings.d_pool_atom_types     = (payload.num_pool_atoms > 0) ? cached_pool_types_ptr : nullptr;
    bindings.d_row_atom_offset     = cached_row_atom_offset_ptr;
    bindings.num_pool_atoms        = payload.num_pool_atoms;
    bindings.d_arg_select_targets       = cached_arg_select_targets_ptr;
    return bindings;
}

std::shared_ptr<BatchDeviceStorage> createBatchDeviceStorage(
    const Config::AiConfigSnapshot& config,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error("createBatchDeviceStorage: stream is NULL");
    }

    const auto workspace_hp = HyperParameters::trainingStateWorkspaceHP(config);
    if (workspace_hp.batch_size <= 0) {
        throw std::runtime_error("createBatchDeviceStorage: batch_size must be > 0");
    }
    if (workspace_hp.max_tokens_per_batch <= 0) {
        throw std::runtime_error("createBatchDeviceStorage: max_tokens_per_batch must be > 0");
    }

    const std::size_t token_capacity =
        static_cast<std::size_t>(workspace_hp.max_tokens_per_batch);
    const int max_tokens = static_cast<int>(token_capacity);

    auto storage = std::make_shared<BatchDeviceStorage>();
    storage->batch_size_capacity = workspace_hp.batch_size;
    storage->max_seq_len_capacity = HyperParameters::snapshotTrainingConfigField<int>(config, "max_cached_seq_len");
    storage->max_tokens_capacity = max_tokens;

    storage->target_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, 1),
        false,
        stream,
        "batch_target_ids");
    storage->input_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_input_ids");
    storage->numeric_values_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_numeric_values");
    storage->atom_mask_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_atom_mask");
    storage->atom_flags_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_atom_flags");
    storage->atom_entry_ids_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_atom_entry_ids");
    storage->token_to_slot_index_map_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_token_to_slot_index_map");
    storage->atom_positions_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_atom_positions");
    storage->atom_types_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_atom_types");

    const auto number_encoder_hp = HyperParameters::numberEncoderConstructionHP(config);
    if (number_encoder_hp.enabled) {
        if (number_encoder_hp.max_digit_slots <= 0) {
            throw std::runtime_error(
                "createBatchDeviceStorage: number_encoder_enabled=true but max_digit_slots <= 0");
        }
        const int digit_slots = number_encoder_hp.max_digit_slots;
        const int slot_capacity = max_tokens * digit_slots;
        storage->number_encoder_digit_slots_capacity = digit_slots;
        storage->atom_digit_values_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, slot_capacity),
            false,
            stream,
            "batch_atom_digit_values");
        storage->atom_digit_pow10_index_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, slot_capacity),
            false,
            stream,
            "batch_atom_digit_pow10_index");
        storage->atom_digit_mask_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, slot_capacity),
            false,
            stream,
            "batch_atom_digit_mask");
        storage->atom_digit_slot_features_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, slot_capacity * BatchPayload::kNumberSlotFeatureDim),
            false,
            stream,
            "batch_atom_digit_slot_features");
        storage->atom_global_features_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens * BatchPayload::kNumberGlobalFeatureDim),
            false,
            stream,
            "batch_atom_global_features");
    }

    const bool selector_enabled =
        HyperParameters::snapshotTrainingConfigField<bool>(
            config, "selector_enabled");
    if (selector_enabled) {
        // Candidate atom-entry pool (arg/option selector). Pool capacity is
        // max_tokens (every token could be an atom); row_atom_offset is batch+1.
        storage->pool_numeric_values_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_pool_numeric_values");
        storage->pool_numeric_float_values_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens * 2),
            false,
            stream,
            "batch_pool_numeric_float_values");
        storage->pool_numeric_int_values_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens * 2),
            false,
            stream,
            "batch_pool_numeric_int_values");
        storage->pool_numeric_kinds_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_pool_numeric_kinds");
        storage->pool_atom_types_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_pool_atom_types");
        storage->row_atom_offset_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, storage->batch_size_capacity + 1),
            false,
            stream,
            "batch_row_atom_offset");
        storage->arg_select_targets_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false, stream, "batch_arg_select_targets");
    }

    return storage;
}

void attachBatchDeviceStorage(
    BatchPayload& payload,
    std::shared_ptr<BatchDeviceStorage> storage,
    const char* caller)
{
    if (!storage) {
        throw std::runtime_error(std::string(caller) + ": storage is NULL");
    }
    payload.device_storage = std::move(storage);
}

}  // namespace Batching
}  // namespace GRIM

#undef BATCH_UPLOAD_CUDA_CHECK
