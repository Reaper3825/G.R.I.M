#pragma once

#include <cstddef>

namespace GRIM {

// Softmax log-sum-exp buffer for FlashAttention (float32, dense, no padding).
struct SoftmaxLseBuffer {
    float* data = nullptr;
    std::size_t bytes = 0;
};

// Returns the byte size for a contiguous [batch, num_heads, seqlen] FP32 buffer.
std::size_t softmaxLseBytes(int batch, int num_heads, int seqlen);

// Bind a pre-allocated buffer to the requested shape.
// Returns false if storage is null or too small.
bool bindSoftmaxLse(SoftmaxLseBuffer& buffer,
                    float* storage,
                    std::size_t storage_bytes,
                    int batch,
                    int num_heads,
                    int seqlen);

}  // namespace GRIM
