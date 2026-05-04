#pragma once

#include <cstddef>

#include <cuda_runtime_api.h>

#include "../LogRecorder/LogRecorder.hpp"

namespace GRIM::TeacherLogits {

struct Buffer {
    float* device = nullptr;
    std::size_t capacity = 0;  // number of floats allocated

    Buffer() = default;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
};

// Ensure device buffer can hold batch_size * seq_len * vocab floats.
bool ensureCapacity(Buffer& buf, std::size_t tokens, int vocab, cudaStream_t stream = nullptr);

// Copy logits from an existing device tensor into the teacher buffer.
bool copyFromDevice(Buffer& buf,
                    const float* src,
                    std::size_t tokens,
                    int vocab,
                    cudaStream_t stream = nullptr);

// Copy logits from host into the teacher buffer (expects contiguous logits).
bool copyFromHost(Buffer& buf,
                  const float* src_host,
                  std::size_t tokens,
                  int vocab,
                  cudaStream_t stream = nullptr);

// Release any allocated device memory.
void release(Buffer& buf);

inline float* data(Buffer& buf) { return buf.device; }
inline const float* data(const Buffer& buf) { return buf.device; }

}  // namespace GRIM::TeacherLogits
