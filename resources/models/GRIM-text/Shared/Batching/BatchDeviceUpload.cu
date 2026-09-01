#include "BatchPayload.hpp"
#include "BatchDeviceBindings.hpp"
#include "BatchDeviceStorage.hpp"
#include "BatchDeviceUpload.hpp"

#include "../AtomInsertion/AtomInsertionDecisionLayout.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../VerboseLogging.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
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
    const bool local_atom_retrieval_enabled =
        HyperParameters::snapshotTrainingConfigField<bool>(
            config, "local_atom_retrieval_enabled");
    if (payload.local_atom_retrieval_enabled !=
        local_atom_retrieval_enabled) {
        throw std::runtime_error(
            "uploadBatchToDevice: payload local-atom retrieval feature does "
            "not match the compiled model semantic");
    }
    if (payload.isTraining() && HyperParameters::snapshotExecutionMode(config) == HyperParameters::ModelExecutionMode::INFERENCE) {
        throw std::runtime_error(
            "uploadBatchToDevice: training BatchPayload cannot be uploaded in inference mode");
    }

    if (!stream) {
        throw std::runtime_error(
            "uploadBatchToDevice: stream is NULL - caller must pass the active upload stream");
    }

    const size_t total_tokens = static_cast<size_t>(payload.total_tokens);
    const BatchDeviceStorage& storage = requireDeviceStorage(payload, "uploadBatchToDevice");
    if (payload.batch_size > storage.batch_size_capacity) {
        throw std::runtime_error(
            "uploadBatchToDevice: payload batch size exceeds sequence-length capacity");
    }

    const size_t atom_gap_rows = payload.EnableAtomIdentification
        ? static_cast<size_t>(payload.atomInsertionGapRowCount())
        : 0;
    if (atom_gap_rows >
        static_cast<size_t>(storage.max_atom_insertion_gap_rows_capacity)) {
        throw std::runtime_error(
            "uploadBatchToDevice: atom insertion gap rows exceed device capacity");
    }

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

    const auto& sequence_lengths_shape = storage.sequence_lengths_tensor.shape.require(
        "uploadBatchToDevice sequence_lengths_tensor");
    if (!sequence_lengths_shape.is_2d_layout() ||
        sequence_lengths_shape.as_2d().rows != 1 ||
        sequence_lengths_shape.as_2d().cols < payload.batch_size) {
        throw std::runtime_error(
            "uploadBatchToDevice: BatchDeviceStorage.sequence_lengths_tensor "
            "must have shape [1, batch_capacity]");
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
    int* cached_sequence_lengths_ptr =
        reinterpret_cast<int*>(storage.sequence_lengths_tensor.data);
    if (!cached_sequence_lengths_ptr) {
        throw std::runtime_error(
            "uploadBatchToDevice: BatchDeviceStorage.sequence_lengths_tensor.data is NULL");
    }
    int* cached_targets_ptr = nullptr;
    if (payload.hasTrainingTargets()) {
        cached_targets_ptr = reinterpret_cast<int*>(storage.target_ids_tensor.data);
        if (!cached_targets_ptr) {
            throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.target_ids_tensor.data is NULL for training payload");
        }
    }
    uint8_t* cached_atom_gap_targets_ptr = nullptr;
    uint8_t* cached_atom_gap_mask_ptr = nullptr;
    if (payload.EnableAtomIdentification) {
        cached_atom_gap_mask_ptr = reinterpret_cast<uint8_t*>(
            storage.atom_insertion_valid_gap_mask_tensor.data);
        if (!cached_atom_gap_mask_ptr) {
            throw std::runtime_error(
                "uploadBatchToDevice: atom insertion gap-mask storage is NULL");
        }
        if (payload.isTraining()) {
            cached_atom_gap_targets_ptr = reinterpret_cast<uint8_t*>(
                storage.atom_insertion_gap_targets_tensor.data);
            if (!cached_atom_gap_targets_ptr) {
                throw std::runtime_error(
                    "uploadBatchToDevice: atom insertion target storage is NULL");
            }
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
    uint32_t* cached_local_atom_indices_ptr = nullptr;
    if (local_atom_retrieval_enabled) {
        cached_local_atom_indices_ptr = reinterpret_cast<uint32_t*>(
            storage.token_local_atom_indices_tensor.data);
        if (!cached_local_atom_indices_ptr) {
            throw std::runtime_error(
                "uploadBatchToDevice: local atom index storage is NULL while "
                "the compiled retrieval feature is enabled");
        }
    }
    int* cached_atom_positions_ptr = reinterpret_cast<int*>(storage.atom_positions_tensor.data);
    if (!cached_atom_positions_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.atom_positions_tensor.data is NULL");
    }
    int* cached_atom_types_ptr = reinterpret_cast<int*>(storage.atom_types_tensor.data);
    if (!cached_atom_types_ptr) {
        throw std::runtime_error("uploadBatchToDevice: BatchDeviceStorage.atom_types_tensor.data is NULL");
    }
    int* cached_local_query_positions_ptr = reinterpret_cast<int*>(
        storage.local_atom_query_positions_tensor.data);
    int* cached_local_query_types_ptr = reinterpret_cast<int*>(
        storage.local_atom_query_types_tensor.data);
    int* cached_local_query_targets_ptr = reinterpret_cast<int*>(
        storage.local_atom_query_targets_tensor.data);
    int* cached_local_row_type_offsets_ptr = reinterpret_cast<int*>(
        storage.local_atom_row_type_candidate_offsets_tensor.data);
    int* cached_local_candidate_first_close_ptr = reinterpret_cast<int*>(
        storage.local_atom_candidate_first_close_positions_tensor.data);
    int* cached_local_candidate_content_offsets_ptr = reinterpret_cast<int*>(
        storage.local_atom_candidate_content_offsets_tensor.data);
    int* cached_local_candidate_content_positions_ptr = reinterpret_cast<int*>(
        storage.local_atom_candidate_content_positions_tensor.data);
    if (local_atom_retrieval_enabled) {
        if (!cached_local_query_positions_ptr || !cached_local_query_types_ptr ||
            !cached_local_query_targets_ptr || !cached_local_row_type_offsets_ptr ||
            !cached_local_candidate_first_close_ptr ||
            !cached_local_candidate_content_offsets_ptr ||
            !cached_local_candidate_content_positions_ptr) {
            throw std::runtime_error(
                "uploadBatchToDevice: local atom retrieval metadata storage is incomplete");
        }
    } else if (cached_local_query_positions_ptr || cached_local_query_types_ptr ||
               cached_local_query_targets_ptr || cached_local_row_type_offsets_ptr ||
               cached_local_candidate_first_close_ptr ||
               cached_local_candidate_content_offsets_ptr ||
               cached_local_candidate_content_positions_ptr) {
        throw std::runtime_error(
            "uploadBatchToDevice: local atom retrieval storage exists while "
            "the compiled feature is disabled");
    }
    if (payload.localAtomQueryCount() > storage.max_tokens_capacity ||
        payload.localAtomCandidateCount() > storage.max_tokens_capacity ||
        payload.localAtomContentPositionCount() > storage.max_tokens_capacity) {
        throw std::runtime_error(
            "uploadBatchToDevice: local atom compact metadata exceeds token capacity");
    }
    const std::size_t local_row_type_capacity =
        static_cast<std::size_t>(storage.batch_size_capacity) *
            GRIM::Tokenizer::kAtomTypeCount +
        1;
    if (payload.local_atom_row_type_candidate_offsets.size() >
        local_row_type_capacity) {
        throw std::runtime_error(
            "uploadBatchToDevice: local atom row/type offsets exceed device capacity");
    }

    const size_t input_ids_bytes   = payload.inputIdBytes();
    const size_t sequence_length_bytes = payload.sequenceLengthBytes();
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
    const size_t local_atom_index_bytes = local_atom_retrieval_enabled
        ? payload.localAtomIndexBytes()
        : 0;
    const size_t local_query_position_bytes = payload.localAtomQueryPositionBytes();
    const size_t local_query_type_bytes = payload.localAtomQueryTypeBytes();
    const size_t local_query_target_bytes = payload.localAtomQueryTargetBytes();
    const size_t local_row_type_offset_bytes = payload.localAtomRowTypeOffsetBytes();
    const size_t local_candidate_first_close_bytes =
        payload.localAtomCandidateFirstCloseBytes();
    const size_t local_candidate_content_offset_bytes =
        payload.localAtomCandidateContentOffsetBytes();
    const size_t local_candidate_content_position_bytes =
        payload.localAtomCandidateContentPositionBytes();

    auto copy_start = std::chrono::high_resolution_clock::now();

    // Round 1: input IDs, row lengths, and the active objective's supervision. LM targets
    // arrive pre-masked from
    // buildBatchPayload Phase 4; payload.lm_valid_tokens already accounts for
    // the post-masking LM-supervised count. Atom gap labels share this same
    // authoritative H2D synchronization boundary.
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(cached_token_ids_ptr, payload.input_ids.data(),
        input_ids_bytes, cudaMemcpyHostToDevice, stream));
    BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
        cached_sequence_lengths_ptr,
        payload.seq_lengths.data(),
        sequence_length_bytes,
        cudaMemcpyHostToDevice,
        stream));
    if (payload.hasTrainingTargets()) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(cached_targets_ptr, payload.target_ids.data(),
            target_ids_bytes, cudaMemcpyHostToDevice, stream));
    }
    if (payload.EnableAtomIdentification) {
        if (payload.isTraining()) {
            BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
                cached_atom_gap_targets_ptr,
                payload.atom_insertion_gap_targets.data(),
                payload.atomInsertionGapTargetBytes(),
                cudaMemcpyHostToDevice,
                stream));
        }
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_atom_gap_mask_ptr,
            payload.atom_insertion_valid_gap_mask.data(),
            payload.atomInsertionGapMaskBytes(),
            cudaMemcpyHostToDevice,
            stream));
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
    if (local_atom_index_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_atom_indices_ptr,
            payload.token_local_atom_indices.data(),
            local_atom_index_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }

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
    if (local_query_position_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_query_positions_ptr,
            payload.local_atom_query_positions.data(),
            local_query_position_bytes,
            cudaMemcpyHostToDevice,
            stream));
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_query_types_ptr,
            payload.local_atom_query_types.data(),
            local_query_type_bytes,
            cudaMemcpyHostToDevice,
            stream));
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_query_targets_ptr,
            payload.local_atom_query_targets.data(),
            local_query_target_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }
    if (local_row_type_offset_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_row_type_offsets_ptr,
            payload.local_atom_row_type_candidate_offsets.data(),
            local_row_type_offset_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }
    if (local_candidate_first_close_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_candidate_first_close_ptr,
            payload.local_atom_candidate_first_close_positions.data(),
            local_candidate_first_close_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }
    if (local_candidate_content_offset_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_candidate_content_offsets_ptr,
            payload.local_atom_candidate_content_offsets.data(),
            local_candidate_content_offset_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }
    if (local_candidate_content_position_bytes > 0) {
        BATCH_UPLOAD_CUDA_CHECK(cudaMemcpyAsync(
            cached_local_candidate_content_positions_ptr,
            payload.local_atom_candidate_content_positions.data(),
            local_candidate_content_position_bytes,
            cudaMemcpyHostToDevice,
            stream));
    }
    BATCH_UPLOAD_CUDA_CHECK(cudaStreamSynchronize(stream));

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
    bindings.d_sequence_lengths = cached_sequence_lengths_ptr;
    bindings.d_target_ids       = cached_targets_ptr;
    bindings.d_atom_insertion_gap_targets = cached_atom_gap_targets_ptr;
    bindings.d_atom_insertion_valid_gap_mask = cached_atom_gap_mask_ptr;
    bindings.d_numeric_values   = cached_numeric_values_ptr;
    bindings.d_atom_mask        = reinterpret_cast<uint8_t*>(cached_atom_mask_ptr);
    bindings.d_atom_flags       = storage.atom_flags_tensor.data
        ? reinterpret_cast<uint32_t*>(storage.atom_flags_tensor.data)
        : nullptr;
    bindings.d_atom_entry_ids   = cached_atom_entry_ids_ptr;
    bindings.d_token_local_atom_indices = cached_local_atom_indices_ptr;
    bindings.d_token_to_slot_index_map = cached_slot_map_ptr;
    bindings.d_atom_positions   = cached_atom_positions_ptr;
    bindings.d_atom_types       = cached_atom_types_ptr;
    bindings.d_local_atom_query_positions = cached_local_query_positions_ptr;
    bindings.d_local_atom_query_types = cached_local_query_types_ptr;
    bindings.d_local_atom_query_targets = cached_local_query_targets_ptr;
    bindings.d_local_atom_row_type_candidate_offsets =
        cached_local_row_type_offsets_ptr;
    bindings.d_local_atom_candidate_first_close_positions =
        cached_local_candidate_first_close_ptr;
    bindings.d_local_atom_candidate_content_offsets =
        cached_local_candidate_content_offsets_ptr;
    bindings.d_local_atom_candidate_content_positions =
        cached_local_candidate_content_positions_ptr;
    bindings.local_atom_query_count = payload.localAtomQueryCount();
    bindings.local_atom_candidate_count = payload.localAtomCandidateCount();
    bindings.local_atom_content_position_count =
        payload.localAtomContentPositionCount();
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
    const bool local_atom_retrieval_enabled =
        HyperParameters::snapshotTrainingConfigField<bool>(
            config, "local_atom_retrieval_enabled");
    if (workspace_hp.batch_size <= 0) {
        throw std::runtime_error("createBatchDeviceStorage: batch_size must be > 0");
    }
    if (workspace_hp.max_tokens_per_batch <= 0) {
        throw std::runtime_error("createBatchDeviceStorage: max_tokens_per_batch must be > 0");
    }

    const std::size_t token_capacity =
        static_cast<std::size_t>(workspace_hp.max_tokens_per_batch);
    const int max_tokens = static_cast<int>(token_capacity);
    const int max_cached_seq_len =
        HyperParameters::snapshotTrainingConfigField<int>(
            config, "max_cached_seq_len");
    if (max_cached_seq_len < 2) {
        throw std::runtime_error(
            "createBatchDeviceStorage: max_cached_seq_len must be at least 2");
    }
    if (workspace_hp.batch_size >
        std::numeric_limits<int>::max() / (max_cached_seq_len - 1)) {
        throw std::runtime_error(
            "createBatchDeviceStorage: atom gap capacity overflows int");
    }
    const int max_atom_gap_rows = workspace_hp.batch_size *
        (max_cached_seq_len - 1);
    if (max_atom_gap_rows > std::numeric_limits<int>::max() /
            AtomInsertion::kAtomDecisionClassCount) {
        throw std::runtime_error(
            "createBatchDeviceStorage: atom target byte capacity overflows Tensor shape");
    }
    static_assert(
        sizeof(float) == 4,
        "raw atom upload capacity assumes four-byte Tensor storage elements");
    const int atom_target_bytes =
        max_atom_gap_rows * AtomInsertion::kAtomDecisionClassCount;
    const int atom_target_storage_elements =
        atom_target_bytes / static_cast<int>(sizeof(float)) +
        (atom_target_bytes % static_cast<int>(sizeof(float)) != 0 ? 1 : 0);
    const int atom_gap_mask_storage_elements =
        max_atom_gap_rows / static_cast<int>(sizeof(float)) +
        (max_atom_gap_rows % static_cast<int>(sizeof(float)) != 0 ? 1 : 0);

    auto storage = std::make_shared<BatchDeviceStorage>();
    storage->batch_size_capacity = workspace_hp.batch_size;
    storage->max_seq_len_capacity = max_cached_seq_len;
    storage->max_tokens_capacity = max_tokens;
    storage->max_atom_insertion_gap_rows_capacity = max_atom_gap_rows;

    storage->target_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, 1),
        false,
        stream,
        "batch_target_ids");
    static_assert(sizeof(int) == sizeof(float),
                  "sequence-length upload capacity assumes four-byte int storage");
    storage->sequence_lengths_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, workspace_hp.batch_size),
        false,
        stream,
        "batch_sequence_lengths");
    storage->input_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, max_tokens),
        false,
        stream,
        "batch_input_ids");
    // Tensor is the canonical owner even though these two buffers are exposed
    // as uint8 through BatchDeviceBindings. Float-backed capacities round each
    // raw byte channel up independently to a complete storage element.
    storage->atom_insertion_gap_targets_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, atom_target_storage_elements),
        false,
        stream,
        "batch_atom_insertion_gap_targets");
    storage->atom_insertion_valid_gap_mask_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(
            1,
            atom_gap_mask_storage_elements),
        false,
        stream,
        "batch_atom_insertion_valid_gap_mask");
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
    if (local_atom_retrieval_enabled) {
        storage->token_local_atom_indices_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_token_local_atom_indices");
        storage->local_atom_query_positions_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_local_atom_query_positions");
        storage->local_atom_query_types_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_local_atom_query_types");
        storage->local_atom_query_targets_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_local_atom_query_targets");
        if (workspace_hp.batch_size >
            (std::numeric_limits<int>::max() - 1) /
                GRIM::Tokenizer::kAtomTypeCount) {
            throw std::runtime_error(
                "createBatchDeviceStorage: local atom row/type offset capacity overflows int");
        }
        const int row_type_offset_capacity =
            workspace_hp.batch_size * GRIM::Tokenizer::kAtomTypeCount + 1;
        storage->local_atom_row_type_candidate_offsets_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, row_type_offset_capacity),
            false,
            stream,
            "batch_local_atom_row_type_candidate_offsets");
        storage->local_atom_candidate_first_close_positions_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_local_atom_candidate_first_close_positions");
        storage->local_atom_candidate_content_offsets_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens + 1),
            false,
            stream,
            "batch_local_atom_candidate_content_offsets");
        storage->local_atom_candidate_content_positions_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, max_tokens),
            false,
            stream,
            "batch_local_atom_candidate_content_positions");
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
