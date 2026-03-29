#include "execution_block_data_stream_GPU.hpp"

#ifdef USE_CUDA

namespace GRIM {

using namespace ExecutionBlockInternal;

__global__ void kernelEntropy(
    float* __restrict__ out,
    const float* __restrict__ probs,
    int N
) {
    if (threadIdx.x != 0) return;
    float ent = 0.0f;
    for (int i = 0; i < N; ++i) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    out[0] = ent;
}

__global__ void kernelAccumScalar(
    float* __restrict__ out,
    const float* __restrict__ in
) {
    if (threadIdx.x == 0)
        out[0] += in[0];
}

__global__ void kernelScaleNegAvg(
    float* __restrict__ out,
    const float* __restrict__ in,
    float weight,
    int count
) {
    if (threadIdx.x == 0)
        out[0] = -weight * (in[0] / fmaxf(static_cast<float>(count), 1.0f));
}

Tensor ExecutionBlockLayer::computeEntropyLoss(
    const std::vector<ExecutionBlockStepOutput>& steps,
    float weight,
    cudaStream_t stream) const
{
    if (steps.empty() || weight <= 0.0f) {
        return Tensor::zeros({1, 1}, stream, "exec_entropy_zero");
    }

    auto accum = Tensor::zeros({1, 1}, stream, "exec_entropy_accum");
    auto tmp   = Tensor::zeros({1, 1}, stream, "exec_entropy_tmp");
    int count = 0;

    for (const auto& s : steps) {
        auto accum_ent = [&](const Tensor& probs, int n) {
            if (!probs.data || n <= 0) return;
            kernelEntropy<<<1, 1, 0, stream>>>(tmp.data, probs.data, n);
            kernelAccumScalar<<<1, 1, 0, stream>>>(accum.data, tmp.data);
            count++;
        };
        if (s.p_arg1.data) accum_ent(s.p_arg1, s.p_arg1.shape.flat.cols);
        if (s.p_arg2.data) accum_ent(s.p_arg2, s.p_arg2.shape.flat.cols);
        if (s.p_op.data) accum_ent(s.p_op, s.p_op.shape.flat.cols);
        if (s.p_write.data) accum_ent(s.p_write, s.p_write.shape.flat.cols);
    }

    auto result = Tensor::zeros({1, 1}, stream, "exec_entropy_loss");
    kernelScaleNegAvg<<<1, 1, 0, stream>>>(result.data, accum.data, weight, count);
    return result;
}

}  // namespace GRIM

#endif  // USE_CUDA
