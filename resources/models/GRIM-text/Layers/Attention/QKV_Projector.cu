//======================================================//
//  QKV_Projector.cu
//  Tensor reshape utility for attention output
//
//  History: Originally contained launchQkvProjection, launchGQAProjection,
//  launchQKVReshapeToBHSD, launchReshapeToBHSD — all superseded by
//  autograd system (autograd::matmul + autograd::split_and_reshape_qkv).
//  Cleaned Feb 2026 (Rule 26: delete dead code).
//
//  Only launchReshapeFromBHSD remains — called by TensorContract_GPU.cu
//  (autograd::reshape_bhsd_to_flat).
//======================================================//

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "QKV_Projector.hpp"
#include "../../Shared/TensorConversion/TensorConversion.hpp"

namespace GRIM {

void launchReshapeFromBHSD(const float* src,
                           float* dst,
                           int batch_size,
                           int seq_len,
                           int num_heads,
                           int head_dim,
                           cudaStream_t stream) {
    // Rule 20: Fail loud on null pointers
    if (!src) throw std::runtime_error("[ReshapeFromBHSD] src is NULL");
    if (!dst) throw std::runtime_error("[ReshapeFromBHSD] dst is NULL");

    // Rule 20: Fail loud on invalid dimensions
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("[ReshapeFromBHSD] Invalid dimensions batch=" +
                std::to_string(batch_size) + " seq=" + std::to_string(seq_len) +
                " heads=" + std::to_string(num_heads) + " head_dim=" + std::to_string(head_dim));
    }

    // Delegate to TensorConversion: [batch, heads, seq, head_dim] -> [batch, seq, model]
    TensorConversion::convert_BHSD_to_BSM(src, dst, batch_size, num_heads, seq_len, head_dim, stream);
}

} // namespace GRIM
