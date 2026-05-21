//======================================================//
//  MTP_GPU.cu
//  Multi-Token Prediction (MTP) CUDA primitives
//
//  Shared kernels only. Autograd loss orchestration lives in
//  training/Autograd/AutogradMtpAuxiliaryLoss.cu.
//======================================================//

#include "MTP_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace MTP {

// MTP accuracy: per-token argmax(logits[t]) == targets[t], count valid (targets[t] != -1)
__global__ void kernelMTPAccuracy(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    int total_tokens,
    int vocab_size,
    int* __restrict__ d_correct,
    int* __restrict__ d_valid
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    int my_correct = 0, my_valid = 0;
    // Masking: target == -1 means masked/padding — must match CrossEntropyNLL exactly
    if (t < total_tokens && targets[t] != -1) {
        my_valid = 1;
        const float* row = logits + t * static_cast<size_t>(vocab_size);
        int best = 0;
        float best_val = row[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > best_val) {
                best_val = row[v];
                best = v;
            }
        }
        if (best == targets[t]) my_correct = 1;
    }
    atomicAdd(d_valid, my_valid);
    atomicAdd(d_correct, my_correct);
}

void launchMTPAccuracyKernel(
    const float* logits,
    const int* targets,
    const Batching::BatchPayload& payload,
    int* d_correct,
    int* d_valid,
    cudaStream_t stream
) {
    payload.validate("launchMTPAccuracyKernel");
    if (!stream) {
        throw std::runtime_error("launchMTPAccuracyKernel: stream is NULL — caller MUST provide valid CUDA stream");
    }
    if (!logits) throw std::runtime_error("launchMTPAccuracyKernel: logits is NULL");
    if (!targets) throw std::runtime_error("launchMTPAccuracyKernel: targets is NULL");
    if (!d_correct) throw std::runtime_error("launchMTPAccuracyKernel: d_correct is NULL");
    if (!d_valid) throw std::runtime_error("launchMTPAccuracyKernel: d_valid is NULL");
    const int block = 256;
    const int grid = (payload.total_tokens + block - 1) / block;
    kernelMTPAccuracy<<<grid, block, 0, stream>>>(
        logits,
        targets,
        payload.total_tokens,
        payload.vocab_size,
        d_correct,
        d_valid);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launchMTPAccuracyKernel: kernel launch failed: ") + cudaGetErrorString(err));
    }
}

}  // namespace MTP
}  // namespace GRIM
