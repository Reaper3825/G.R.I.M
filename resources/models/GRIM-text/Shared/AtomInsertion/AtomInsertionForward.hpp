//======================================================//
//  AtomInsertionForward.hpp
//  Downstream forward entry for the atom-identification model
//======================================================//

#pragma once

#include "AtomInsertionBoundaryProjection.hpp"

#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../Forward/ModelForwardOutputs.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GRIM::AtomInsertion {

// Downstream entry intended to be called from executeModelForward after its
// encoder has produced full-context, non-causal byte states:
//
//   contextual byte states [B*S, D]
//       -> adjacent-state boundary projection [B*(S-1), D]
//       -> existing LM head [B*(S-1), V]
//       -> ModelForwardOutputs::logits_tensor
//
// This function does not run an encoder and cannot infer whether attention was
// causal. executeModelForward validates the compiled non-causal encoder
// semantic before calling it. Boundary configuration is supplied by the
// AtomInsertionBoundaryProjectionHP config-system grouping; the LM-head
// grouping independently carries the task mode needed by forwardLmHead().
void forwardAtomInsertion(
    const HyperParameters::AtomInsertionBoundaryProjectionHP& boundary_hp,
    const AtomInsertionBoundaryParameterTensors& boundary_parameters,
    const HyperParameters::LMHeadLayerConstructionHP& lm_head_hp,
    const LMHeadParameterTensors& lm_head_parameters,
    const Tensor& contextual_states,
    const Batching::BatchPayload& payload,
    bool EnableAtomIdentification,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs);

} // namespace GRIM::AtomInsertion
