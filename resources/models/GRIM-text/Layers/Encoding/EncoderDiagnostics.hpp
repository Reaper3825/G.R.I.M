#pragma once
//======================================================//
//  EncoderDiagnostics.hpp
//  Rule 21 diagnostics for encoder-layer residual output
//======================================================//

#include <cuda_runtime_api.h>

#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM::EncoderDiagnostics {

struct LayerResidualDiagnosticRequest {
    const Tensor& input;
    const Tensor& attention_raw;
    const Tensor& attention_branch;
    const Tensor& residual1;
    const Tensor& ffn_raw;
    const Tensor& ffn_branch;
    const Tensor& output;
    const Tensor* layer_scale1;
    const Tensor* layer_scale2;
    const HyperParameters::EncoderLayerConstructionHP& hp;
    const Batching::BatchPayload& payload;
    cudaStream_t stream;
    int layer_idx;
    bool emitLayerResidualDiag;
};

void emitLayerResidualDiagnostic(const LayerResidualDiagnosticRequest& request);

} // namespace GRIM::EncoderDiagnostics