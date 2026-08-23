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
    Tensor target_ids_tensor;
    Tensor numeric_values_tensor;
    Tensor atom_mask_tensor;
    Tensor atom_flags_tensor;
    Tensor atom_entry_ids_tensor;
    Tensor token_to_slot_index_map_tensor;
    Tensor atom_positions_tensor;
    Tensor atom_types_tensor;

    // NumberEncoder digit-place upload caches (Category 3 workspace; contents
    // are valid only for the active upload boundary). Allocated only when
    // number_encoder_enabled=true; geometry capacity is max_tokens * digit_slots.
    Tensor atom_digit_values_tensor;        // int32 [1, max_tokens * digit_slots]
    Tensor atom_digit_pow10_index_tensor;   // int32 [1, max_tokens * digit_slots]
    Tensor atom_digit_mask_tensor;          // float [1, max_tokens * digit_slots]
    Tensor atom_digit_slot_features_tensor; // float [1, max_tokens * digit_slots * kNumberSlotFeatureDim]
    Tensor atom_global_features_tensor;     // float [1, max_tokens * kNumberGlobalFeatureDim]

    // Numeric auxiliary target mirrors populated at the batch upload boundary
    // and exposed through BatchDeviceBindings.
    Tensor number_aux_target_digits_tensor;     // int32 [1, max_tokens * digit_slots]
    Tensor number_aux_target_pow10_index_tensor; // int32 [1, max_tokens * digit_slots]
    Tensor number_aux_target_digit_mask_tensor;  // uint8 [max_tokens * digit_slots]
    Tensor number_aux_target_valid_tensor;       // uint8 [max_tokens]
    Tensor number_aux_target_sign_negative_tensor; // uint8 [max_tokens]

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
    int number_encoder_digit_slots_capacity = 0;
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
