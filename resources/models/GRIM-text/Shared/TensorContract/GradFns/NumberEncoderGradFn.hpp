#pragma once
//======================================================//
//  NumberEncoderGradFn.hpp
//  Single-owner header for the NumberEncoder autograd node.
//
//  NumberEncoder is the numeric-meaning input path of the selector
//  architecture (docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md):
//
//  Forward, per authored numeric atom a at token position p:
//    slot_i      = digit_emb[digit_i]
//                + pow10_emb[pow10_bucket_i]
//                + W_c2 @ tanh(W_c1 @ slot_feat_i + b_c1)        (real slots only)
//    pooled_a    = mean over real digit slots of slot_i           (mask-weighted)
//    global_a    = W_g2 @ tanh(W_g1 @ global_feat_a + b_g1)
//    out[p, :]   = pooled_a + global_a
//    out[q, :]   = 0 for every non-atom token position q
//
//  The digit/pow10 indices, per-slot features, per-slot mask, and global
//  features are CURRENT-token arg_number metadata materialized by
//  BatchPayload and uploaded through BatchDeviceBindings. Padding digit
//  slots have mask == 0 and contribute exactly zero in forward and backward.
//
//  Backward writes gradients ONLY into the eight registered NumberEncoder
//  parameter grad buffers, always with atomicAdd (Rule 20 GradFn
//  accumulation contract). The data channels are not trainable: no gradient
//  flows to AtomTable metadata, and no upstream tensor chain exists because
//  every input besides the parameters is authored batch data.
//
//  Owns:
//    - struct NumberEncoderForwardParams (caller-facing param view bundle)
//    - struct NumberEncoderGradFn        (backward node)
//    - autograd::number_encode(...)      (forward op; defined in .cu)
//======================================================//

#include "../TensorContract_GPU.hpp"  // GradFn, Tensor, TensorShape
#include "../../HyperParameters/HyperparameterGroupings.hpp"

#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

// Non-owning view bundle over the eight registry-owned NumberEncoder
// parameter tensors (or detached per-call copies when the caller runs with
// connect_parameter_graph=false). The caller keeps the referenced tensors
// alive for the duration of the forward call window.
struct NumberEncoderForwardParams {
    const Tensor* digit_emb = nullptr;   // [10, d_model]
    const Tensor* pow10_emb = nullptr;   // [pow10_buckets, d_model]
    const Tensor* W_c1 = nullptr;        // [kNumberSlotFeatureDim, d_hidden]
    const Tensor* b_c1 = nullptr;        // [1, d_hidden]
    const Tensor* W_c2 = nullptr;        // [d_hidden, d_model]
    const Tensor* W_g1 = nullptr;        // [kNumberGlobalFeatureDim, d_hidden]
    const Tensor* b_g1 = nullptr;        // [1, d_hidden]
    const Tensor* W_g2 = nullptr;        // [d_hidden, d_model]
};

/**
 * NumberEncoderGradFn — backward node for the NumberEncoder forward op.
 *
 * STORES STABLE DATA, NOT Tensor* (ISSUE #48 pattern):
 *   - per-parameter grad pointers into the registered leaf grad buffers
 *   - W_c2 / W_g2 weight VALUE pointers (registry tensors are durable
 *     Category 2 state and outlive the tape window)
 *   - owned saved activations (post-tanh hidden states) needed for the
 *     tanh backward
 *
 * Batch-aware: digit indices, slot features, masks, and global features are
 * re-read from the backward-time BatchDeviceBindings, exactly like
 * EmbeddingGradFn re-reads d_input_ids.
 */
struct NumberEncoderGradFn : public GradFn {
    // Captured parameter grad destinations (NULL when that param does not
    // require grad). All backward writes are atomicAdd accumulation.
    float* digit_emb_grad = nullptr;
    float* pow10_emb_grad = nullptr;
    float* W_c1_grad = nullptr;
    float* b_c1_grad = nullptr;
    float* W_c2_grad = nullptr;
    float* W_g1_grad = nullptr;
    float* b_g1_grad = nullptr;
    float* W_g2_grad = nullptr;

    // Captured weight VALUES needed by the tanh-MLP backward equations.
    const float* W_c2_data = nullptr;
    const float* W_g2_data = nullptr;

    // Owned saved forward activations (Category 1, freed by release_saved()).
    Tensor slot_hidden;    // [num_atoms * digit_slots, d_hidden] post-tanh
    Tensor global_hidden;  // [num_atoms, d_hidden] post-tanh

    // Captured geometry.
    int num_atoms = 0;
    int digit_slots = 0;
    int d_hidden = 0;
    int d_model = 0;
    int pow10_buckets = 0;

    NumberEncoderGradFn();
    ~NumberEncoderGradFn() override;

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

/**
 * ArgSelectorKeysGradFn — backward node for encodeAtomEntryPoolKeys.
 *
 * Propagates the arg/option-selector loss from the candidate KEY tensor back
 * into the eight NumberEncoder parameters, so the selection objective TEACHES
 * the numeric-meaning encoder which candidate entries must be distinguishable.
 * This gradient SUMS (atomicAdd) with the input-side number_encode gradient
 * into the same registry leaf grad buffers — the NumberEncoder learns from both
 * the LM fusion path and the selection path.
 *
 * Structurally identical to NumberEncoderGradFn, except:
 *   - grad_output is the dense per-entry key grad [num_pool_atoms, d_model]
 *     (one row per candidate entry, NOT scattered to token positions);
 *   - the digit/slot/global channels are re-read from the candidate-POOL
 *     bindings (BatchDeviceBindings::d_pool_*), not the per-token d_atom_*;
 *   - the shared NumberEncoder backward kernels run in DENSE mode
 *     (atom_positions = nullptr ⇒ grad row index == entry index).
 */
struct ArgSelectorKeysGradFn : public GradFn {
    float* digit_emb_grad = nullptr;
    float* pow10_emb_grad = nullptr;
    float* W_c1_grad = nullptr;
    float* b_c1_grad = nullptr;
    float* W_c2_grad = nullptr;
    float* W_g1_grad = nullptr;
    float* b_g1_grad = nullptr;
    float* W_g2_grad = nullptr;

    const float* W_c2_data = nullptr;
    const float* W_g2_data = nullptr;

    Tensor slot_hidden;    // [num_atoms * digit_slots, d_hidden] post-tanh
    Tensor global_hidden;  // [num_atoms, d_hidden] post-tanh

    int num_atoms = 0;     // E = num_pool_atoms captured at forward
    int digit_slots = 0;
    int d_hidden = 0;
    int d_model = 0;
    int pow10_buckets = 0;

    ArgSelectorKeysGradFn();
    ~ArgSelectorKeysGradFn() override;

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override;
    void release_saved() override;
};

/**
 * NumberEncoder forward op.
 *
 * Consumes the authored digit-place channels for the active payload
 * (payload.number_encoder_digit_slots, bindings.d_atom_digit_*) and the
 * eight NumberEncoder parameters; produces a dense [total_tokens, d_model]
 * tensor that is zero everywhere except authored numeric-atom positions.
 *
 * Fail-loud requirements (Rule 20):
 *   - hp.enabled must be true and geometry must match the payload channels
 *   - every device binding for the digit channels must be non-NULL
 *   - all eight parameter tensors must have data and consistent shapes
 *
 * Attaches a NumberEncoderGradFn when any parameter requires grad.
 */
Tensor number_encode(const NumberEncoderForwardParams& params,
                     const HyperParameters::NumberEncoderConstructionHP& hp,
                     const Batching::BatchPayload& payload,
                     const Batching::BatchDeviceBindings& bindings,
                     cudaStream_t stream);

/**
 * encodeAtomEntryPoolKeys — dense per-entry selector key encoding.
 *
 * Encodes every candidate atom-entry in the batch pool into a [num_pool_atoms,
 * d_model] key tensor using the SAME NumberEncoder weights/math as number_encode,
 * so a number's selector key matches its input representation. Feature channels
 * come from the candidate-pool bindings (BatchDeviceBindings::d_pool_digit_*).
 *
 * GRADIENT: when any NumberEncoder parameter requires grad (training), the result
 * is CONNECTED via an ArgSelectorKeysGradFn, so the selection loss flows through
 * the keys into the NumberEncoder parameters — the selector teaches the encoder
 * which entries to keep distinguishable. Pass DETACHED parameter views (inference)
 * to get a forward-only, requires_grad=false key tensor instead.
 */
Tensor encodeAtomEntryPoolKeys(
    const NumberEncoderForwardParams& params,
    const HyperParameters::NumberEncoderConstructionHP& hp,
    const int* d_pool_digit_values,
    const int* d_pool_digit_pow10_index,
    const float* d_pool_digit_mask,
    const float* d_pool_digit_slot_features,
    const float* d_pool_global_features,
    int num_pool_atoms,
    cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM
