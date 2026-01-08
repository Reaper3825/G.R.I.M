#include "softmax_lse.hpp"

#include <cstddef>
#include <cstdio>

namespace GRIM {

std::size_t softmaxLseBytes(int batch, int num_heads, int seqlen) {
    if (batch <= 0 || num_heads <= 0 || seqlen <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(batch) *
           static_cast<std::size_t>(num_heads) *
           static_cast<std::size_t>(seqlen) *
           sizeof(float);
}

bool bindSoftmaxLse(SoftmaxLseBuffer& buffer,
                    float* storage,
                    std::size_t storage_bytes,
                    int batch,
                    int num_heads,
                    int seqlen) {
    buffer = {};
    const std::size_t required = softmaxLseBytes(batch, num_heads, seqlen);
    if (required == 0) {
        return false;
    }
    if (!storage) {
        fprintf(stderr, "[SoftmaxLSE] storage is null (required bytes=%zu).\n", required);
        return false;
    }
    if (storage_bytes < required) {
        fprintf(stderr,
                "[SoftmaxLSE] storage too small (required=%zu bytes, provided=%zu bytes).\n",
                required,
                storage_bytes);
        return false;
    }
    buffer.data = storage;
    buffer.bytes = required;
    return true;
}

}  // namespace GRIM
