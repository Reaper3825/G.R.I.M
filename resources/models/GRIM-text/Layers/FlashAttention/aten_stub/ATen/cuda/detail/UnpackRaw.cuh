// Minimal ATen stub for standalone GRIM build (no PyTorch).
// Provides PhiloxCudaState and at::cuda::philox::unpack for flash-attention.
// PyTorch source: aten/src/ATen/cuda/detail/PhiloxCudaStateRaw.cuh, UnpackRaw.cuh

#pragma once
#define GRIM_PHILOX_CUDA_STATE_DEFINED

#include <cstdint>
#include <tuple>

namespace at {

struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(uint64_t seed, uint64_t offset) : seed(seed), offset(offset) {}

  uint64_t seed = 0;
  uint64_t offset = 0;
};

}  // namespace at

namespace at::cuda::philox {

__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState arg) {
  return std::make_tuple(arg.seed, arg.offset);
}

}  // namespace at::cuda::philox
