#pragma once

// Compatibility stub for flash-attention v2.5.9 (hardware_info.h added in later upstream).
// Provides device query helpers. Include cuda_runtime.h before this in host code.

#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#include <tuple>
#endif

#define CHECK_CUDA(call)                                                        \
  do {                                                                         \
    cudaError_t status_ = call;                                                 \
    if (status_ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(status_));                                     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

inline int get_current_device() {
  int device;
  CHECK_CUDA(cudaGetDevice(&device));
  return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
  int capability_major, capability_minor;
  CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
  return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  return multiprocessor_count;
}
