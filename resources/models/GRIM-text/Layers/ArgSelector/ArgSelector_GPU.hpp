#pragma once
//======================================================//
//  ArgSelector_GPU.hpp
//  Arg/option selector scoring head (execution-INDEPENDENT).
//
//  Given the encoder hidden states and the candidate atom-entry "menu" (keys
//  encoded from numeric meaning), produce a selection-logit matrix
//  [total_tokens, num_pool_atoms]: for token t and candidate entry e,
//      score(t, e) = <W_q · h_t, key_e> * softmax_scale
//  with candidates OUTSIDE token t's row window masked to -inf so each token only
//  selects among its own row's entries (prompt + already-generated). This is the
//  "selector owns selection" head from docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md;
//  candidates/features are data-derived (NumberEncoder keys), never execution
//  state (GRIM/Docs/DeletedCode.md).
//
//  Autograd: encoder_output and W_q carry gradient (the trunk + the selector
//  query projection train). `keys` are detached (the NumberEncoder trains from
//  the input-side fusion). Scores are produced with autograd::matmul so backward
//  reuses the tested matmul GradFns; the row-window mask is an additive constant.
//======================================================//

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace ArgSelector {

// Build a row-window additive mask M[total_tokens, num_pool_atoms]: 0.0 where
// candidate e is in token t's row window [row_atom_offset[row], row_atom_offset[row+1]),
// and a large negative sentinel elsewhere. `d_row_atom_offset` is device int[batch+1].
void launchRowWindowMask(
    float* d_mask,                 // [total_tokens * num_pool_atoms], pre-zeroed
    const int* d_row_atom_offset,  // device [batch_size + 1]
    int total_tokens,
    int max_seq_len,
    int batch_size,
    int num_pool_atoms,
    cudaStream_t stream);

// Selection-logit head. Returns [total_tokens, num_pool_atoms] (grad-tracked
// through encoder_output and W_q). softmax_scale is typically 1/sqrt(d_model).
Tensor argSelectorForward(
    const Tensor& encoder_output,  // [total_tokens, d_model]
    const Tensor& W_q,             // [d_model, d_model] query projection
    const Tensor& keys,            // [num_pool_atoms, d_model] detached candidate keys
    const Batching::BatchPayload& payload,
    const int* d_row_atom_offset,  // device [batch_size + 1]
    int num_pool_atoms,
    float softmax_scale,
    cudaStream_t stream);

}  // namespace ArgSelector
}  // namespace GRIM

#endif  // USE_CUDA
