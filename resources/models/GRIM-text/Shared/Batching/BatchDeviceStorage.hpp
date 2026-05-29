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
    Tensor token_to_slot_map_tensor;
    Tensor mtp_shifted_targets_tensor;

    int batch_size_capacity = 0;
    int max_seq_len_capacity = 0;
    int max_tokens_capacity = 0;
    int mtp_k_capacity = 0;
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
