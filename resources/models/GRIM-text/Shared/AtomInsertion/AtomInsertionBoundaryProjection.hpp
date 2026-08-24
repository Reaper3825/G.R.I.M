//======================================================//
//  AtomInsertionBoundaryProjection.hpp
//  Primitive-composed adjacent-state boundary projection
//======================================================//

#pragma once

#include "../Batching/BatchPayload.hpp"
#include "../Forward/ModelForwardOutputs.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GRIM::AtomInsertion {

// Constructs one representation for every adjacent row inside each fixed
// BatchPayload rectangle:
//
//   left  = select_fixed_group_rows(H, offset=0, rows=S-1)
//   right = select_fixed_group_rows(H, offset=1, rows=S-1)
//   gaps  = left*W_left + right*W_right + bias
//
// With rows [BOS, byte0, ..., byteN-1, EOS, PAD...], the first N+1 rows in
// each output group are the real byte gaps. Padded gap rows remain present and
// are excluded by the BatchPayload-authored device gap mask at loss time.
//
// Every intermediate is retained on ModelForwardOutputs. Backward is therefore
// the graph assembled from select_fixed_group_rows, matmul, add, and
// broadcast_add primitive GradFns; there is no atom-specific backward node and
// no private boundary-projection memory owner.
void forwardAtomInsertionBoundaryProjection(
    const HyperParameters::AtomInsertionBoundaryProjectionHP& hp,
    const AtomInsertionBoundaryParameterTensors& parameters,
    const Tensor& contextual_states,
    const Batching::BatchPayload& payload,
    bool EnableAtomIdentification,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs);

} // namespace GRIM::AtomInsertion
