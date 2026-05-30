#pragma once
//======================================================//
//  DecodeTimeSlotSelector — decode-time selector ops and
//  selector-op surface over the durable trainable tensor
//  owner declared in Startup/Model/ParameterRegistry.
//
//  Returns an ordered score vector over { NULL } ∪ L
//  where L is the live slot candidate set.
//
//  Required baseline tensors:
//    W_q_select       — query projection (d_model → d_selector)
//    W_k_select       — key projection (d_slot_features → d_selector)
//    null_key_select  — learnable NULL key vector (d_selector)
//    null_logit_bias  — learnable scalar NULL bias
//
//  Startup/GpuModelState is the durable owner of these
//  trainable tensors via the adjacent registry boundary.
//  Callers invoke the selector ops directly against that
//  owner and pass the grouped selector config explicitly
//  at the call boundary.
//
//  Pattern B (self-managed autograd): forward op uses
//  autograd::matmul, autograd::add, autograd::concat.
//  Backward is automatic via the grad_fn chain.
//======================================================//

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct cublasContext;
using cublasHandle_t = cublasContext*;
#endif

#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIM {

void validateDecodeTimeSlotSelectorConfig(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp);

DecodeTimeSlotSelector createDecodeTimeSlotSelector(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    uint64_t seed,
    cudaStream_t init_stream);

// Forward: compute scores over { NULL } ∪ L using autograd primitives.
//
// h_t:             [1, d_model] hidden state (non-owning view OK)
// slot_features:   [num_live, d_slot_features] fixed slot features (non-owning view OK)
// num_live_slots:  number of live candidates in L
// stream:          CUDA stream for async execution
//
// Returns Forward::SelectorForwardResult with autograd-tracked scores tensor.
// Caller MUST keep the result alive until backward() completes.
Forward::SelectorForwardResult forwardDecodeTimeSlotSelector(
    const DecodeTimeSlotSelector& selector,
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const Tensor& h_t,
    const Tensor& slot_features,
    int num_live_slots,
    cudaStream_t stream,
    cublasHandle_t cublas_handle);

} // namespace GRIM
