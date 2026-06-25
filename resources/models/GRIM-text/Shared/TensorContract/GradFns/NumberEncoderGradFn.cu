//======================================================//
//  NumberEncoderGradFn.cu
//  Implementation of the NumberEncoder autograd node.
//
//  Owns:
//    - kernel_number_encoder_slot_hidden     (forward, per real digit slot)
//    - kernel_number_encoder_global_hidden   (forward, per atom)
//    - kernel_number_encoder_output          (forward, pooled scatter)
//    - kernel_number_encoder_backward_emb    (backward, digit/pow10 embeddings)
//    - kernel_number_encoder_backward_slot   (backward, W_c2/W_c1/b_c1)
//    - kernel_number_encoder_backward_global (backward, W_g2/W_g1/b_g1)
//    - NumberEncoderGradFn methods
//    - autograd::number_encode(...)
//
//  Rule 20 GradFn discipline: every parameter-gradient write below is an
//  atomicAdd accumulation into a registered leaf grad buffer. Forward
//  saves only the post-tanh hidden activations; digit indices, features,
//  and masks are re-read from backward-time BatchDeviceBindings.
//======================================================//

#include "NumberEncoderGradFn.hpp"
#include "../TensorContract_GPU.hpp"
#include "../../Batching/BatchPayload.hpp"
#include "../../Batching/BatchDeviceBindings.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

// ─── Forward declarations: defined in TensorContract_GPU.cu at global scope ───
void setCurrentGradFnOp(const char* op_name, void* gradfn_ptr);
void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace {

constexpr int NUMBER_ENCODER_BLOCK_SIZE = 256;
constexpr int kSlotFeatDim = GRIM::Batching::BatchPayload::kNumberSlotFeatureDim;
constexpr int kGlobalFeatDim = GRIM::Batching::BatchPayload::kNumberGlobalFeatureDim;

// Forward: per-(atom, slot) contribution-MLP hidden state.
//   slot_hidden[a*S+i, h] = tanh(b_c1[h] + sum_f W_c1[f, h] * slot_feat[a*S+i, f])
// Padding slots (mask == 0) stay exactly zero.
__global__ void kernel_number_encoder_slot_hidden(
    const float* __restrict__ slot_features,  // [A*S, F]
    const float* __restrict__ slot_mask,      // [A*S]
    const float* __restrict__ W_c1,           // [F, H]
    const float* __restrict__ b_c1,           // [H]
    float* __restrict__ slot_hidden,          // [A*S, H]
    int total_slots,
    int d_hidden)
{
    const int slot = blockIdx.x;
    if (slot >= total_slots) return;

    float* hidden_row = slot_hidden + static_cast<size_t>(slot) * d_hidden;
    if (slot_mask[slot] == 0.0f) {
        for (int h = threadIdx.x; h < d_hidden; h += blockDim.x) {
            hidden_row[h] = 0.0f;
        }
        return;
    }

    const float* feat = slot_features + static_cast<size_t>(slot) * kSlotFeatDim;
    for (int h = threadIdx.x; h < d_hidden; h += blockDim.x) {
        float acc = b_c1 ? b_c1[h] : 0.0f;  // bias optional (gated by use_bias)
        #pragma unroll
        for (int f = 0; f < kSlotFeatDim; ++f) {
            acc += W_c1[f * d_hidden + h] * feat[f];
        }
        hidden_row[h] = tanhf(acc);
    }
}

// Forward: per-atom global mantissa/exponent MLP hidden state.
//   global_hidden[a, h] = tanh(b_g1[h] + sum_f W_g1[f, h] * global_feat[a, f])
__global__ void kernel_number_encoder_global_hidden(
    const float* __restrict__ global_features,  // [A, G]
    const float* __restrict__ W_g1,             // [G, H]
    const float* __restrict__ b_g1,             // [H]
    float* __restrict__ global_hidden,          // [A, H]
    int num_atoms,
    int d_hidden)
{
    const int atom = blockIdx.x;
    if (atom >= num_atoms) return;

    const float* feat = global_features + static_cast<size_t>(atom) * kGlobalFeatDim;
    float* hidden_row = global_hidden + static_cast<size_t>(atom) * d_hidden;
    for (int h = threadIdx.x; h < d_hidden; h += blockDim.x) {
        float acc = b_g1 ? b_g1[h] : 0.0f;  // bias optional (gated by use_bias)
        #pragma unroll
        for (int f = 0; f < kGlobalFeatDim; ++f) {
            acc += W_g1[f * d_hidden + h] * feat[f];
        }
        hidden_row[h] = tanhf(acc);
    }
}

// Forward: pooled digit-place contribution + global head, scattered into the
// dense output at this atom's token position. Output buffer arrives zeroed,
// so non-atom rows stay exactly zero.
__global__ void kernel_number_encoder_output(
    const int* __restrict__ atom_positions,     // [A]
    const int* __restrict__ digit_values,       // [A*S]
    const int* __restrict__ pow10_index,        // [A*S]
    const float* __restrict__ slot_mask,        // [A*S]
    const float* __restrict__ slot_hidden,      // [A*S, H]
    const float* __restrict__ global_hidden,    // [A, H]
    const float* __restrict__ digit_emb,        // [10, M]
    const float* __restrict__ pow10_emb,        // [P, M]
    const float* __restrict__ W_c2,             // [H, M]
    const float* __restrict__ W_g2,             // [H, M]
    float* __restrict__ output,                 // [total_tokens, M] pre-zeroed
    int num_atoms,
    int digit_slots,
    int d_hidden,
    int d_model,
    int pow10_buckets,
    int total_tokens)
{
    const int atom = blockIdx.x;
    if (atom >= num_atoms) return;

    const int token_pos = atom_positions[atom];
    if (token_pos < 0 || token_pos >= total_tokens) {
        printf("FATAL: OOB atom token_pos=%d (total_tokens=%d) at atom=%d in kernel_number_encoder_output\n",
               token_pos, total_tokens, atom);
        __trap();
    }

    // Real-slot count for the mask-weighted mean. Payload build guarantees
    // every numeric atom has at least one real digit slot.
    int real_slots = 0;
    for (int i = 0; i < digit_slots; ++i) {
        if (slot_mask[atom * digit_slots + i] != 0.0f) ++real_slots;
    }
    if (real_slots <= 0) {
        printf("FATAL: numeric atom=%d has zero real digit slots in kernel_number_encoder_output\n", atom);
        __trap();
    }
    const float inv_n = 1.0f / static_cast<float>(real_slots);

    float* out_row = output + static_cast<size_t>(token_pos) * d_model;
    const float* g_hidden = global_hidden + static_cast<size_t>(atom) * d_hidden;

    for (int m = threadIdx.x; m < d_model; m += blockDim.x) {
        float pooled = 0.0f;
        for (int i = 0; i < digit_slots; ++i) {
            const int slot = atom * digit_slots + i;
            if (slot_mask[slot] == 0.0f) continue;

            const int dv = digit_values[slot];
            const int pi = pow10_index[slot];
            if (dv < 0 || dv > 9 || pi < 0 || pi >= pow10_buckets) {
                printf("FATAL: OOB digit channel (digit=%d pow10_idx=%d buckets=%d) at atom=%d slot=%d in kernel_number_encoder_output\n",
                       dv, pi, pow10_buckets, atom, i);
                __trap();
            }

            float slot_val = digit_emb[dv * d_model + m] + pow10_emb[pi * d_model + m];
            const float* s_hidden = slot_hidden + static_cast<size_t>(slot) * d_hidden;
            for (int h = 0; h < d_hidden; ++h) {
                slot_val += W_c2[h * d_model + m] * s_hidden[h];
            }
            pooled += slot_val;
        }
        pooled *= inv_n;

        float global_val = 0.0f;
        for (int h = 0; h < d_hidden; ++h) {
            global_val += W_g2[h * d_model + m] * g_hidden[h];
        }

        out_row[m] = pooled + global_val;
    }
}

// Forward (selector keys): dense per-ENTRY numeric-meaning encoding for the
// arg/option selector candidate pool. Identical math to kernel_number_encoder_output
// but writes keys[e, :] densely (one row per pool entry) instead of scattering to a
// token position, and does NOT trap on zero real slots (non-numeric pool entries
// have all-zero digit masks and simply receive the global-only contribution).
__global__ void kernel_number_encoder_keys_dense(
    const int* __restrict__ digit_values,       // [E*S]
    const int* __restrict__ pow10_index,        // [E*S]
    const float* __restrict__ slot_mask,        // [E*S]
    const float* __restrict__ slot_hidden,      // [E*S, H]
    const float* __restrict__ global_hidden,    // [E, H]
    const float* __restrict__ digit_emb,        // [10, M]
    const float* __restrict__ pow10_emb,        // [P, M]
    const float* __restrict__ W_c2,             // [H, M]
    const float* __restrict__ W_g2,             // [H, M]
    float* __restrict__ keys,                   // [E, M] pre-zeroed
    int num_entries,
    int digit_slots,
    int d_hidden,
    int d_model,
    int pow10_buckets)
{
    const int e = blockIdx.x;
    if (e >= num_entries) return;

    int real_slots = 0;
    for (int i = 0; i < digit_slots; ++i) {
        if (slot_mask[e * digit_slots + i] != 0.0f) ++real_slots;
    }
    const float inv_n = real_slots > 0 ? 1.0f / static_cast<float>(real_slots) : 0.0f;

    float* out_row = keys + static_cast<size_t>(e) * d_model;
    const float* g_hidden = global_hidden + static_cast<size_t>(e) * d_hidden;

    for (int m = threadIdx.x; m < d_model; m += blockDim.x) {
        float pooled = 0.0f;
        for (int i = 0; i < digit_slots; ++i) {
            const int slot = e * digit_slots + i;
            if (slot_mask[slot] == 0.0f) continue;
            const int dv = digit_values[slot];
            const int pi = pow10_index[slot];
            if (dv < 0 || dv > 9 || pi < 0 || pi >= pow10_buckets) {
                printf("FATAL: OOB digit channel (digit=%d pow10_idx=%d buckets=%d) at entry=%d slot=%d in kernel_number_encoder_keys_dense\n",
                       dv, pi, pow10_buckets, e, i);
                __trap();
            }
            float slot_val = digit_emb[dv * d_model + m] + pow10_emb[pi * d_model + m];
            const float* s_hidden = slot_hidden + static_cast<size_t>(slot) * d_hidden;
            for (int h = 0; h < d_hidden; ++h) {
                slot_val += W_c2[h * d_model + m] * s_hidden[h];
            }
            pooled += slot_val;
        }
        pooled *= inv_n;

        float global_val = 0.0f;
        for (int h = 0; h < d_hidden; ++h) {
            global_val += W_g2[h * d_model + m] * g_hidden[h];
        }
        out_row[m] = pooled + global_val;
    }
}

// Backward: digit / pow10 embedding gradients.
//   grad_digit_emb[dv, m] += grad_out[pos, m] / n_a   (real slots only)
//   grad_pow10_emb[pi, m] += grad_out[pos, m] / n_a
__global__ void kernel_number_encoder_backward_emb(
    const float* __restrict__ grad_output,    // [total_tokens, M]
    const int* __restrict__ atom_positions,   // [A]; null ⇒ dense (token_pos = atom)
    const int* __restrict__ digit_values,     // [A*S]
    const int* __restrict__ pow10_index,      // [A*S]
    const float* __restrict__ slot_mask,      // [A*S]
    float* __restrict__ digit_emb_grad,       // [10, M] nullable
    float* __restrict__ pow10_emb_grad,       // [P, M] nullable
    int num_atoms,
    int digit_slots,
    int d_model,
    int pow10_buckets)
{
    const int slot = blockIdx.x;
    if (slot >= num_atoms * digit_slots) return;
    if (slot_mask[slot] == 0.0f) return;

    const int atom = slot / digit_slots;
    int real_slots = 0;
    for (int i = 0; i < digit_slots; ++i) {
        if (slot_mask[atom * digit_slots + i] != 0.0f) ++real_slots;
    }
    const float inv_n = 1.0f / static_cast<float>(real_slots);

    const int token_pos = atom_positions ? atom_positions[atom] : atom;
    const float* grad_row = grad_output + static_cast<size_t>(token_pos) * d_model;

    const int dv = digit_values[slot];
    const int pi = pow10_index[slot];
    if (dv < 0 || dv > 9 || pi < 0 || pi >= pow10_buckets) {
        printf("FATAL: OOB digit channel (digit=%d pow10_idx=%d buckets=%d) at slot=%d in kernel_number_encoder_backward_emb\n",
               dv, pi, pow10_buckets, slot);
        __trap();
    }

    for (int m = threadIdx.x; m < d_model; m += blockDim.x) {
        const float g = grad_row[m] * inv_n;
        if (digit_emb_grad) atomicAdd(&digit_emb_grad[dv * d_model + m], g);
        if (pow10_emb_grad) atomicAdd(&pow10_emb_grad[pi * d_model + m], g);
    }
}

// Backward: contribution-MLP gradients per real digit slot.
//   gs[m]            = grad_out[pos, m] / n_a
//   grad_W_c2[h, m] += slot_hidden[h] * gs[m]
//   dhid[h]          = (1 - slot_hidden[h]^2) * sum_m W_c2[h, m] * gs[m]
//   grad_b_c1[h]    += dhid[h]
//   grad_W_c1[f, h] += slot_feat[f] * dhid[h]
__global__ void kernel_number_encoder_backward_slot(
    const float* __restrict__ grad_output,    // [total_tokens, M]
    const int* __restrict__ atom_positions,   // [A]; null ⇒ dense (token_pos = atom)
    const float* __restrict__ slot_mask,      // [A*S]
    const float* __restrict__ slot_features,  // [A*S, F]
    const float* __restrict__ slot_hidden,    // [A*S, H]
    const float* __restrict__ W_c2,           // [H, M]
    float* __restrict__ W_c2_grad,            // [H, M] nullable
    float* __restrict__ W_c1_grad,            // [F, H] nullable
    float* __restrict__ b_c1_grad,            // [H] nullable
    int num_atoms,
    int digit_slots,
    int d_hidden,
    int d_model)
{
    const int slot = blockIdx.x;
    if (slot >= num_atoms * digit_slots) return;
    if (slot_mask[slot] == 0.0f) return;

    const int atom = slot / digit_slots;
    int real_slots = 0;
    for (int i = 0; i < digit_slots; ++i) {
        if (slot_mask[atom * digit_slots + i] != 0.0f) ++real_slots;
    }
    const float inv_n = 1.0f / static_cast<float>(real_slots);

    const int token_pos = atom_positions ? atom_positions[atom] : atom;
    const float* grad_row = grad_output + static_cast<size_t>(token_pos) * d_model;
    const float* s_hidden = slot_hidden + static_cast<size_t>(slot) * d_hidden;
    const float* feat = slot_features + static_cast<size_t>(slot) * kSlotFeatDim;

    for (int h = threadIdx.x; h < d_hidden; h += blockDim.x) {
        const float sh = s_hidden[h];
        float w_dot_g = 0.0f;
        for (int m = 0; m < d_model; ++m) {
            const float gs = grad_row[m] * inv_n;
            if (W_c2_grad) atomicAdd(&W_c2_grad[h * d_model + m], sh * gs);
            w_dot_g += W_c2[h * d_model + m] * gs;
        }
        const float dhid = (1.0f - sh * sh) * w_dot_g;
        if (b_c1_grad) atomicAdd(&b_c1_grad[h], dhid);
        if (W_c1_grad) {
            #pragma unroll
            for (int f = 0; f < kSlotFeatDim; ++f) {
                atomicAdd(&W_c1_grad[f * d_hidden + h], feat[f] * dhid);
            }
        }
    }
}

// Backward: global-MLP gradients per atom (no pooling normalization — the
// global head adds directly into the atom's output row).
__global__ void kernel_number_encoder_backward_global(
    const float* __restrict__ grad_output,      // [total_tokens, M]
    const int* __restrict__ atom_positions,     // [A]; null ⇒ dense (token_pos = atom)
    const float* __restrict__ global_features,  // [A, G]
    const float* __restrict__ global_hidden,    // [A, H]
    const float* __restrict__ W_g2,             // [H, M]
    float* __restrict__ W_g2_grad,              // [H, M] nullable
    float* __restrict__ W_g1_grad,              // [G, H] nullable
    float* __restrict__ b_g1_grad,              // [H] nullable
    int num_atoms,
    int d_hidden,
    int d_model)
{
    const int atom = blockIdx.x;
    if (atom >= num_atoms) return;

    const int token_pos = atom_positions ? atom_positions[atom] : atom;
    const float* grad_row = grad_output + static_cast<size_t>(token_pos) * d_model;
    const float* g_hidden = global_hidden + static_cast<size_t>(atom) * d_hidden;
    const float* feat = global_features + static_cast<size_t>(atom) * kGlobalFeatDim;

    for (int h = threadIdx.x; h < d_hidden; h += blockDim.x) {
        const float gh = g_hidden[h];
        float w_dot_g = 0.0f;
        for (int m = 0; m < d_model; ++m) {
            const float g = grad_row[m];
            if (W_g2_grad) atomicAdd(&W_g2_grad[h * d_model + m], gh * g);
            w_dot_g += W_g2[h * d_model + m] * g;
        }
        const float dhid = (1.0f - gh * gh) * w_dot_g;
        if (b_g1_grad) atomicAdd(&b_g1_grad[h], dhid);
        if (W_g1_grad) {
            #pragma unroll
            for (int f = 0; f < kGlobalFeatDim; ++f) {
                atomicAdd(&W_g1_grad[f * d_hidden + h], feat[f] * dhid);
            }
        }
    }
}

}  // anonymous namespace

namespace GRIM {
namespace autograd {

// ═══════════════════════════════════════════════════════════════════════════
// NumberEncoderGradFn — method bodies
// ═══════════════════════════════════════════════════════════════════════════

NumberEncoderGradFn::NumberEncoderGradFn() {
    op_name = "number_encode";
}

NumberEncoderGradFn::~NumberEncoderGradFn() {
    release_saved();
}

void NumberEncoderGradFn::apply_impl(const Tensor& grad_output,
                                     cudaStream_t stream,
                                     const Batching::BatchPayload* backward_payload,
                                     const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("number_encode", this);

    if (applied) {
        return;
    }
    applied = true;

    if (!backward_payload) {
        throw std::runtime_error("NumberEncoderGradFn::apply: backward_payload is NULL - orchestration MUST pass the active BatchPayload into backward()");
    }
    if (!backward_bindings) {
        throw std::runtime_error("NumberEncoderGradFn::apply: backward_bindings is NULL - orchestration MUST pass the active BatchDeviceBindings into backward()");
    }
    if (!grad_output.data) {
        throw std::runtime_error("NumberEncoderGradFn::apply: grad_output.data is NULL");
    }
    if (backward_payload->number_encoder_digit_slots != digit_slots) {
        throw std::runtime_error("NumberEncoderGradFn::apply: backward payload digit_slots=" +
                                 std::to_string(backward_payload->number_encoder_digit_slots) +
                                 " != captured digit_slots=" + std::to_string(digit_slots));
    }
    if (backward_payload->authoredAtomCount() != num_atoms) {
        throw std::runtime_error("NumberEncoderGradFn::apply: backward payload atom count=" +
                                 std::to_string(backward_payload->authoredAtomCount()) +
                                 " != captured num_atoms=" + std::to_string(num_atoms));
    }
    const size_t expected_count = static_cast<size_t>(backward_payload->total_tokens) *
                                  static_cast<size_t>(d_model);
    if (grad_output.numel() != expected_count) {
        throw std::runtime_error("NumberEncoderGradFn::apply: grad_output element count mismatch. expected=" +
                                 std::to_string(expected_count) + " got=" +
                                 std::to_string(grad_output.numel()));
    }
    if (!backward_bindings->d_atom_positions ||
        !backward_bindings->d_atom_digit_values ||
        !backward_bindings->d_atom_digit_pow10_index ||
        !backward_bindings->d_atom_digit_mask ||
        !backward_bindings->d_atom_digit_slot_features ||
        !backward_bindings->d_atom_global_features) {
        throw std::runtime_error("NumberEncoderGradFn::apply: NumberEncoder device channels are NULL on backward bindings");
    }
    if (!slot_hidden.data || !global_hidden.data) {
        throw std::runtime_error("NumberEncoderGradFn::apply: saved hidden activations were released before backward");
    }
    if (!W_c2_data || !W_g2_data) {
        throw std::runtime_error("NumberEncoderGradFn::apply: captured W_c2/W_g2 weight pointers are NULL");
    }

    const int total_slots = num_atoms * digit_slots;

    if (digit_emb_grad || pow10_emb_grad) {
        kernel_number_encoder_backward_emb<<<total_slots, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
            grad_output.data,
            backward_bindings->d_atom_positions,
            backward_bindings->d_atom_digit_values,
            backward_bindings->d_atom_digit_pow10_index,
            backward_bindings->d_atom_digit_mask,
            digit_emb_grad,
            pow10_emb_grad,
            num_atoms,
            digit_slots,
            d_model,
            pow10_buckets);
        trackKernelLaunch("kernel_number_encoder_backward_emb", stream);
    }

    if (W_c2_grad || W_c1_grad || b_c1_grad) {
        kernel_number_encoder_backward_slot<<<total_slots, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
            grad_output.data,
            backward_bindings->d_atom_positions,
            backward_bindings->d_atom_digit_mask,
            backward_bindings->d_atom_digit_slot_features,
            slot_hidden.data,
            W_c2_data,
            W_c2_grad,
            W_c1_grad,
            b_c1_grad,
            num_atoms,
            digit_slots,
            d_hidden,
            d_model);
        trackKernelLaunch("kernel_number_encoder_backward_slot", stream);
    }

    if (W_g2_grad || W_g1_grad || b_g1_grad) {
        kernel_number_encoder_backward_global<<<num_atoms, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
            grad_output.data,
            backward_bindings->d_atom_positions,
            backward_bindings->d_atom_global_features,
            global_hidden.data,
            W_g2_data,
            W_g2_grad,
            W_g1_grad,
            b_g1_grad,
            num_atoms,
            d_hidden,
            d_model);
        trackKernelLaunch("kernel_number_encoder_backward_global", stream);
    }

    // No upstream chain: every non-parameter input is authored batch data
    // (AtomTable metadata stays immutable during the training step), and the
    // parameter tensors are registered leaves.
}

void NumberEncoderGradFn::release_saved() {
    if (released_) return;
    GradFn::release_saved();
    slot_hidden = Tensor();
    global_hidden = Tensor();
    digit_emb_grad = nullptr;
    pow10_emb_grad = nullptr;
    W_c1_grad = nullptr;
    b_c1_grad = nullptr;
    W_c2_grad = nullptr;
    W_g1_grad = nullptr;
    b_g1_grad = nullptr;
    W_g2_grad = nullptr;
    W_c2_data = nullptr;
    W_g2_data = nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// ArgSelectorKeysGradFn — method bodies
//
// Reuses the NumberEncoder backward kernels in DENSE mode (atom_positions =
// nullptr) over the candidate-POOL channels, so the arg/option-selection loss
// trains the eight NumberEncoder parameters. Grads accumulate (atomicAdd) into
// the SAME registry leaf grad buffers as the input-side number_encode backward.
// ═══════════════════════════════════════════════════════════════════════════

ArgSelectorKeysGradFn::ArgSelectorKeysGradFn() {
    op_name = "arg_selector_keys";
}

ArgSelectorKeysGradFn::~ArgSelectorKeysGradFn() {
    release_saved();
}

void ArgSelectorKeysGradFn::apply_impl(const Tensor& grad_output,
                                       cudaStream_t stream,
                                       const Batching::BatchPayload* /*backward_payload*/,
                                       const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("arg_selector_keys", this);

    if (applied) {
        return;
    }
    applied = true;

    if (!backward_bindings) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: backward_bindings is NULL - orchestration MUST pass the active BatchDeviceBindings into backward()");
    }
    if (!grad_output.data) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: grad_output.data is NULL");
    }
    if (num_atoms <= 0) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: captured num_atoms <= 0");
    }
    if (backward_bindings->num_pool_atoms != num_atoms) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: backward bindings num_pool_atoms=" +
                                 std::to_string(backward_bindings->num_pool_atoms) +
                                 " != captured num_atoms=" + std::to_string(num_atoms));
    }
    const size_t expected_count = static_cast<size_t>(num_atoms) * static_cast<size_t>(d_model);
    if (grad_output.numel() != expected_count) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: grad_output element count mismatch. expected=" +
                                 std::to_string(expected_count) + " got=" +
                                 std::to_string(grad_output.numel()));
    }
    if (!backward_bindings->d_pool_digit_values ||
        !backward_bindings->d_pool_digit_pow10_index ||
        !backward_bindings->d_pool_digit_mask ||
        !backward_bindings->d_pool_digit_slot_features ||
        !backward_bindings->d_pool_global_features) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: candidate-pool device channels are NULL on backward bindings");
    }
    if (!slot_hidden.data || !global_hidden.data) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: saved hidden activations were released before backward");
    }
    if (!W_c2_data || !W_g2_data) {
        throw std::runtime_error("ArgSelectorKeysGradFn::apply: captured W_c2/W_g2 weight pointers are NULL");
    }

    const int total_slots = num_atoms * digit_slots;

    // DENSE mode: atom_positions = nullptr ⇒ each backward kernel indexes
    // grad_output by entry directly (grad_output is [num_pool_atoms, d_model]).
    if (digit_emb_grad || pow10_emb_grad) {
        kernel_number_encoder_backward_emb<<<total_slots, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
            grad_output.data,
            nullptr,
            backward_bindings->d_pool_digit_values,
            backward_bindings->d_pool_digit_pow10_index,
            backward_bindings->d_pool_digit_mask,
            digit_emb_grad,
            pow10_emb_grad,
            num_atoms,
            digit_slots,
            d_model,
            pow10_buckets);
        trackKernelLaunch("kernel_number_encoder_backward_emb(pool)", stream);
    }

    if (W_c2_grad || W_c1_grad || b_c1_grad) {
        kernel_number_encoder_backward_slot<<<total_slots, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
            grad_output.data,
            nullptr,
            backward_bindings->d_pool_digit_mask,
            backward_bindings->d_pool_digit_slot_features,
            slot_hidden.data,
            W_c2_data,
            W_c2_grad,
            W_c1_grad,
            b_c1_grad,
            num_atoms,
            digit_slots,
            d_hidden,
            d_model);
        trackKernelLaunch("kernel_number_encoder_backward_slot(pool)", stream);
    }

    if (W_g2_grad || W_g1_grad || b_g1_grad) {
        kernel_number_encoder_backward_global<<<num_atoms, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
            grad_output.data,
            nullptr,
            backward_bindings->d_pool_global_features,
            global_hidden.data,
            W_g2_data,
            W_g2_grad,
            W_g1_grad,
            b_g1_grad,
            num_atoms,
            d_hidden,
            d_model);
        trackKernelLaunch("kernel_number_encoder_backward_global(pool)", stream);
    }

    // No upstream chain: the pool feature channels are immutable batch data and
    // the parameters are registered leaves (grads accumulated above).
}

void ArgSelectorKeysGradFn::release_saved() {
    if (released_) return;
    GradFn::release_saved();
    slot_hidden = Tensor();
    global_hidden = Tensor();
    digit_emb_grad = nullptr;
    pow10_emb_grad = nullptr;
    W_c1_grad = nullptr;
    b_c1_grad = nullptr;
    W_c2_grad = nullptr;
    W_g1_grad = nullptr;
    b_g1_grad = nullptr;
    W_g2_grad = nullptr;
    W_c2_data = nullptr;
    W_g2_data = nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// autograd::number_encode — forward op
// ═══════════════════════════════════════════════════════════════════════════

namespace {

const Tensor& requireParam(const Tensor* t, int expected_rows, int expected_cols, const char* name) {
    if (!t) {
        throw std::runtime_error(std::string("autograd::number_encode: parameter view '") + name + "' is NULL");
    }
    if (!t->data) {
        throw std::runtime_error(std::string("autograd::number_encode: parameter '") + name + "'.data is NULL");
    }
    if (!t->shape.is_2d_layout()) {
        throw std::runtime_error(std::string("autograd::number_encode: parameter '") + name + "' is not 2D");
    }
    const auto dims = t->shape.as_2d();
    if (dims.rows != expected_rows || dims.cols != expected_cols) {
        throw std::runtime_error(std::string("autograd::number_encode: parameter '") + name + "' shape=[" +
                                 std::to_string(dims.rows) + ", " + std::to_string(dims.cols) +
                                 "] != expected [" + std::to_string(expected_rows) + ", " +
                                 std::to_string(expected_cols) + "]");
    }
    return *t;
}

// Optional parameter (e.g. the use_bias-gated hidden biases). When the tensor is
// absent/unallocated, returns the provided default-empty tensor so callers can keep
// using a reference: its .data is NULL (kernels treat that as a zero bias) and its
// requires_grad is false (no grad captured). Shape is still validated when present.
const Tensor& optionalParam(const Tensor* t, const Tensor& absent,
                            int expected_rows, int expected_cols, const char* name) {
    if (!t || !t->data) {
        return absent;
    }
    return requireParam(t, expected_rows, expected_cols, name);
}

}  // namespace

Tensor number_encode(const NumberEncoderForwardParams& params,
                     const HyperParameters::NumberEncoderConstructionHP& hp,
                     const Batching::BatchPayload& payload,
                     const Batching::BatchDeviceBindings& bindings,
                     cudaStream_t stream) {
    payload.validate("autograd::number_encode");
    if (!hp.enabled) {
        throw std::runtime_error("autograd::number_encode: called while hp.enabled=false");
    }
    if (!stream) {
        throw std::runtime_error("autograd::number_encode: stream is NULL");
    }
    if (hp.pow10_buckets != 2 * hp.max_abs_pow10 + 1 || hp.pow10_buckets <= 0) {
        throw std::runtime_error("autograd::number_encode: hp.pow10_buckets=" +
                                 std::to_string(hp.pow10_buckets) +
                                 " inconsistent with max_abs_pow10=" +
                                 std::to_string(hp.max_abs_pow10));
    }
    if (payload.number_encoder_digit_slots != hp.max_digit_slots) {
        throw std::runtime_error("autograd::number_encode: payload digit_slots=" +
                                 std::to_string(payload.number_encoder_digit_slots) +
                                 " != hp.max_digit_slots=" + std::to_string(hp.max_digit_slots) +
                                 " — payload build and model config disagree");
    }
    const int num_atoms = payload.authoredAtomCount();
    if (num_atoms <= 0) {
        throw std::runtime_error("autograd::number_encode: payload has no authored atoms; caller must gate on authoredAtomCount() > 0");
    }
    if (!bindings.d_atom_positions ||
        !bindings.d_atom_digit_values ||
        !bindings.d_atom_digit_pow10_index ||
        !bindings.d_atom_digit_mask ||
        !bindings.d_atom_digit_slot_features ||
        !bindings.d_atom_global_features) {
        throw std::runtime_error("autograd::number_encode: NumberEncoder device channels are NULL on bindings — uploadBatchToDevice must populate them when number_encoder is enabled");
    }

    const int d_model = hp.d_model;
    const int d_hidden = hp.d_hidden;
    const int digit_slots = hp.max_digit_slots;
    const int total_slots = num_atoms * digit_slots;

    const Tensor empty_bias;  // default-empty fallback for use_bias-gated biases
    const Tensor& digit_emb = requireParam(params.digit_emb, 10, d_model, "digit_emb");
    const Tensor& pow10_emb = requireParam(params.pow10_emb, hp.pow10_buckets, d_model, "pow10_emb");
    const Tensor& W_c1 = requireParam(params.W_c1, kSlotFeatDim, d_hidden, "W_c1");
    const Tensor& b_c1 = optionalParam(params.b_c1, empty_bias, 1, d_hidden, "b_c1");
    const Tensor& W_c2 = requireParam(params.W_c2, d_hidden, d_model, "W_c2");
    const Tensor& W_g1 = requireParam(params.W_g1, kGlobalFeatDim, d_hidden, "W_g1");
    const Tensor& b_g1 = optionalParam(params.b_g1, empty_bias, 1, d_hidden, "b_g1");
    const Tensor& W_g2 = requireParam(params.W_g2, d_hidden, d_model, "W_g2");

    const bool any_requires_grad =
        digit_emb.requires_grad || pow10_emb.requires_grad ||
        W_c1.requires_grad || b_c1.requires_grad || W_c2.requires_grad ||
        W_g1.requires_grad || b_g1.requires_grad || W_g2.requires_grad;

    // Saved activations: owned by the GradFn when training, locals otherwise.
    Tensor slot_hidden = Tensor::empty(
        ::TensorContract::TensorShape::make_BSM(total_slots, d_hidden),
        false, stream, "number_encoder_slot_hidden");
    Tensor global_hidden = Tensor::empty(
        ::TensorContract::TensorShape::make_BSM(num_atoms, d_hidden),
        false, stream, "number_encoder_global_hidden");

    kernel_number_encoder_slot_hidden<<<total_slots, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
        bindings.d_atom_digit_slot_features,
        bindings.d_atom_digit_mask,
        W_c1.data,
        b_c1.data,
        slot_hidden.data,
        total_slots,
        d_hidden);
    trackKernelLaunch("kernel_number_encoder_slot_hidden", stream);

    kernel_number_encoder_global_hidden<<<num_atoms, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
        bindings.d_atom_global_features,
        W_g1.data,
        b_g1.data,
        global_hidden.data,
        num_atoms,
        d_hidden);
    trackKernelLaunch("kernel_number_encoder_global_hidden", stream);

    auto output_shape = ::TensorContract::TensorShape::make_BSM(payload.total_tokens, d_model);
    Tensor result = Tensor::zeros(output_shape, any_requires_grad, stream, "number_encode_result");

    kernel_number_encoder_output<<<num_atoms, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
        bindings.d_atom_positions,
        bindings.d_atom_digit_values,
        bindings.d_atom_digit_pow10_index,
        bindings.d_atom_digit_mask,
        slot_hidden.data,
        global_hidden.data,
        digit_emb.data,
        pow10_emb.data,
        W_c2.data,
        W_g2.data,
        result.data,
        num_atoms,
        digit_slots,
        d_hidden,
        d_model,
        hp.pow10_buckets,
        payload.total_tokens);
    trackKernelLaunch("kernel_number_encoder_output", stream);

    if (any_requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<NumberEncoderGradFn>();
        grad_fn->num_atoms = num_atoms;
        grad_fn->digit_slots = digit_slots;
        grad_fn->d_hidden = d_hidden;
        grad_fn->d_model = d_model;
        grad_fn->pow10_buckets = hp.pow10_buckets;
        grad_fn->slot_hidden = std::move(slot_hidden);
        grad_fn->global_hidden = std::move(global_hidden);
        grad_fn->W_c2_data = W_c2.data;
        grad_fn->W_g2_data = W_g2.data;

        auto captureGrad = [](const Tensor& t) -> float* {
            if (!t.requires_grad) return nullptr;
            auto& mutable_t = const_cast<Tensor&>(t);
            mutable_t.ensure_grad();
            return mutable_t.grad_data();
        };
        grad_fn->digit_emb_grad = captureGrad(digit_emb);
        grad_fn->pow10_emb_grad = captureGrad(pow10_emb);
        grad_fn->W_c1_grad = captureGrad(W_c1);
        grad_fn->b_c1_grad = captureGrad(b_c1);
        grad_fn->W_c2_grad = captureGrad(W_c2);
        grad_fn->W_g1_grad = captureGrad(W_g1);
        grad_fn->b_g1_grad = captureGrad(b_g1);
        grad_fn->W_g2_grad = captureGrad(W_g2);

        result.grad_fn = grad_fn;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// encodeAtomEntryPoolKeys — dense per-entry selector keys (forward-only/detached)
//
// Encodes every candidate atom-entry in the batch pool into a d_model key vector
// using the SAME NumberEncoder weights and math as the input-side fusion, so a
// number's selector key matches its input representation. Keys are DETACHED: the
// NumberEncoder trains from the input-side path; the selector consumes keys as a
// fixed (per-step) candidate representation and learns to query them via W_q.
// ═══════════════════════════════════════════════════════════════════════════
Tensor encodeAtomEntryPoolKeys(
    const NumberEncoderForwardParams& params,
    const HyperParameters::NumberEncoderConstructionHP& hp,
    const int* d_pool_digit_values,
    const int* d_pool_digit_pow10_index,
    const float* d_pool_digit_mask,
    const float* d_pool_digit_slot_features,
    const float* d_pool_global_features,
    int num_pool_atoms,
    cudaStream_t stream) {
    if (!hp.enabled) {
        throw std::runtime_error("encodeAtomEntryPoolKeys: called while hp.enabled=false");
    }
    if (num_pool_atoms <= 0) {
        throw std::runtime_error("encodeAtomEntryPoolKeys: num_pool_atoms must be > 0");
    }
    if (!stream) {
        throw std::runtime_error("encodeAtomEntryPoolKeys: stream is NULL");
    }
    if (!d_pool_digit_values || !d_pool_digit_pow10_index || !d_pool_digit_mask ||
        !d_pool_digit_slot_features || !d_pool_global_features) {
        throw std::runtime_error("encodeAtomEntryPoolKeys: candidate-pool feature channels are NULL "
                                 "(uploadBatchToDevice must populate them when the selector is enabled)");
    }
    if (hp.pow10_buckets != 2 * hp.max_abs_pow10 + 1 || hp.pow10_buckets <= 0) {
        throw std::runtime_error("encodeAtomEntryPoolKeys: hp.pow10_buckets inconsistent with max_abs_pow10");
    }

    const int d_model = hp.d_model;
    const int d_hidden = hp.d_hidden;
    const int digit_slots = hp.max_digit_slots;
    const int total_slots = num_pool_atoms * digit_slots;

    const Tensor empty_bias;  // default-empty fallback for use_bias-gated biases
    const Tensor& digit_emb = requireParam(params.digit_emb, 10, d_model, "digit_emb");
    const Tensor& pow10_emb = requireParam(params.pow10_emb, hp.pow10_buckets, d_model, "pow10_emb");
    const Tensor& W_c1 = requireParam(params.W_c1, kSlotFeatDim, d_hidden, "W_c1");
    const Tensor& b_c1 = optionalParam(params.b_c1, empty_bias, 1, d_hidden, "b_c1");
    const Tensor& W_c2 = requireParam(params.W_c2, d_hidden, d_model, "W_c2");
    const Tensor& W_g1 = requireParam(params.W_g1, kGlobalFeatDim, d_hidden, "W_g1");
    const Tensor& b_g1 = optionalParam(params.b_g1, empty_bias, 1, d_hidden, "b_g1");
    const Tensor& W_g2 = requireParam(params.W_g2, d_hidden, d_model, "W_g2");

    const bool any_requires_grad =
        digit_emb.requires_grad || pow10_emb.requires_grad ||
        W_c1.requires_grad || b_c1.requires_grad || W_c2.requires_grad ||
        W_g1.requires_grad || b_g1.requires_grad || W_g2.requires_grad;

    // Saved activations: owned by the GradFn when training, locals otherwise.
    Tensor slot_hidden = Tensor::empty(
        ::TensorContract::TensorShape::make_BSM(total_slots, d_hidden),
        false, stream, "selector_pool_slot_hidden");
    Tensor global_hidden = Tensor::empty(
        ::TensorContract::TensorShape::make_BSM(num_pool_atoms, d_hidden),
        false, stream, "selector_pool_global_hidden");
    // Connected when any NumberEncoder parameter requires grad (training): the
    // selection loss flows through these keys into the encoder. Detached
    // (requires_grad=false) when the caller passes detached params (inference).
    Tensor keys = Tensor::zeros(
        ::TensorContract::TensorShape::make_BSM(num_pool_atoms, d_model),
        any_requires_grad, stream, "selector_pool_keys");

    kernel_number_encoder_slot_hidden<<<total_slots, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
        d_pool_digit_slot_features, d_pool_digit_mask, W_c1.data, b_c1.data,
        slot_hidden.data, total_slots, d_hidden);
    trackKernelLaunch("kernel_number_encoder_slot_hidden(pool)", stream);

    kernel_number_encoder_global_hidden<<<num_pool_atoms, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
        d_pool_global_features, W_g1.data, b_g1.data, global_hidden.data,
        num_pool_atoms, d_hidden);
    trackKernelLaunch("kernel_number_encoder_global_hidden(pool)", stream);

    kernel_number_encoder_keys_dense<<<num_pool_atoms, NUMBER_ENCODER_BLOCK_SIZE, 0, stream>>>(
        d_pool_digit_values, d_pool_digit_pow10_index, d_pool_digit_mask,
        slot_hidden.data, global_hidden.data, digit_emb.data, pow10_emb.data,
        W_c2.data, W_g2.data, keys.data,
        num_pool_atoms, digit_slots, d_hidden, d_model, hp.pow10_buckets);
    trackKernelLaunch("kernel_number_encoder_keys_dense", stream);

    if (any_requires_grad) {
        keys.is_leaf = false;
        auto grad_fn = std::make_shared<ArgSelectorKeysGradFn>();
        grad_fn->num_atoms = num_pool_atoms;
        grad_fn->digit_slots = digit_slots;
        grad_fn->d_hidden = d_hidden;
        grad_fn->d_model = d_model;
        grad_fn->pow10_buckets = hp.pow10_buckets;
        grad_fn->slot_hidden = std::move(slot_hidden);
        grad_fn->global_hidden = std::move(global_hidden);
        grad_fn->W_c2_data = W_c2.data;
        grad_fn->W_g2_data = W_g2.data;

        auto captureGrad = [](const Tensor& t) -> float* {
            if (!t.requires_grad) return nullptr;
            auto& mutable_t = const_cast<Tensor&>(t);
            mutable_t.ensure_grad();
            return mutable_t.grad_data();
        };
        grad_fn->digit_emb_grad = captureGrad(digit_emb);
        grad_fn->pow10_emb_grad = captureGrad(pow10_emb);
        grad_fn->W_c1_grad = captureGrad(W_c1);
        grad_fn->b_c1_grad = captureGrad(b_c1);
        grad_fn->W_c2_grad = captureGrad(W_c2);
        grad_fn->W_g1_grad = captureGrad(W_g1);
        grad_fn->b_g1_grad = captureGrad(b_g1);
        grad_fn->W_g2_grad = captureGrad(W_g2);

        keys.grad_fn = grad_fn;
    }

    return keys;
}

}  // namespace autograd
}  // namespace GRIM
