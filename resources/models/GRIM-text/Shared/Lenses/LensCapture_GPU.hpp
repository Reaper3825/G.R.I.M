#pragma once
#ifdef USE_CUDA
#include "LensMetadata.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"

namespace GRIM::Lenses {
// Scoped to one executeModelForward invocation. Reuses detached parameters and
// one optional gated weight matrix; publishes only completed owned host data.
class LensCaptureSession {
public:
    LensCaptureSession(const LensCaptureRequest& request,
        const Batching::BatchPayload& payload,
        const NamedConceptSpanDefinitions& definitions, int layer_count,
        int cache_prefix_length, bool effective_causal, bool dropout_enabled,
        std::uint64_t invocation_id,
        const HyperParameters::LMHeadLayerConstructionHP& hp,
        const LMHeadParameterTensors& parameters,
        cudaStream_t stream, cublasHandle_t handle);
    void captureLayer(int layer, const Tensor& hidden,
                      const Tensor* actual_logits = nullptr);
    void publish(Forward::ModelForwardOutputs& outputs);
private:
    LensCaptureRequest request_;
    const Batching::BatchPayload& payload_;
    HyperParameters::LMHeadLayerConstructionHP hp_;
    LMHeadParameterTensors parameters_;
    Tensor effective_weights_;
    cudaStream_t stream_;
    cublasHandle_t handle_;
    std::vector<LensCaptureMetadata> positions_;
    std::shared_ptr<LensCaptureResult> result_;
    int next_layer_ = 0;
};
std::uint64_t nextForwardInvocationId();
}
#endif
