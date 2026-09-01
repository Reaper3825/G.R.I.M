//======================================================//
//  BatchDeviceStorage.hpp
//
//  Explicit owner of reusable device buffers for the
//  BatchPayload upload boundary. BatchDeviceBindings is
//  only the borrowed step-local pointer view.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <memory>

namespace GRIM {
namespace Batching {

struct BatchPayload;

struct BatchDeviceStorage {
    Tensor input_ids_tensor;
    // Per-row valid token extents [batch_size]. The attention kernels consume
    // these as read-only key bounds for padded non-causal batches.
    Tensor sequence_lengths_tensor;
    Tensor target_ids_tensor;
    // Raw uint8 atom-identification channels. Tensor owns the byte capacity;
    // BatchDeviceBindings exposes typed borrowed pointers for the active step.
    Tensor atom_insertion_gap_targets_tensor;
    Tensor atom_insertion_valid_gap_mask_tensor;
    Tensor numeric_values_tensor;
    Tensor atom_mask_tensor;
    Tensor atom_flags_tensor;
    Tensor atom_entry_ids_tensor;
    // Retrieval-only storage is absent when the compiled model feature is off.
    Tensor token_local_atom_indices_tensor;
    Tensor token_to_slot_index_map_tensor;
    Tensor atom_positions_tensor;
    Tensor atom_types_tensor;
    // Compact retrieval-only metadata buffers. These share this root batch
    // owner and are never allocated by the retrieval forward/loss code.
    Tensor local_atom_query_positions_tensor;
    Tensor local_atom_query_types_tensor;
    Tensor local_atom_query_targets_tensor;
    Tensor local_atom_row_type_candidate_offsets_tensor;
    Tensor local_atom_candidate_first_close_positions_tensor;
    Tensor local_atom_candidate_content_offsets_tensor;
    Tensor local_atom_candidate_content_positions_tensor;

    // Candidate atom-entry pool (arg/option selector). Allocated independently
    // when selector_enabled=true; pool capacity is max_tokens (every token could
    // be an atom), row_atom_offset capacity is batch_size + 1.
    Tensor pool_numeric_values_tensor;      // float [1, max_tokens]
    Tensor pool_numeric_float_values_tensor; // double [max_tokens], stored in 2x float bytes
    Tensor pool_numeric_int_values_tensor;  // int64 [max_tokens], stored in 2x float bytes
    Tensor pool_numeric_kinds_tensor;       // uint8 [max_tokens]
    Tensor pool_atom_types_tensor;          // int32 [1, max_tokens]
    Tensor row_atom_offset_tensor;          // int32 [1, batch_size + 1]
    Tensor arg_select_targets_tensor;       // int32 [1, max_tokens] selector supervision

    int batch_size_capacity = 0;
    int max_seq_len_capacity = 0;
    int max_tokens_capacity = 0;
    int max_atom_insertion_gap_rows_capacity = 0;
};

std::shared_ptr<BatchDeviceStorage> createBatchDeviceStorage(
    const Config::AiConfigSnapshot& config,
    cudaStream_t stream);

void attachBatchDeviceStorage(
    BatchPayload& payload,
    std::shared_ptr<BatchDeviceStorage> storage,
    const char* caller);

}  // namespace Batching
}  // namespace GRIM

#endif  // USE_CUDA
