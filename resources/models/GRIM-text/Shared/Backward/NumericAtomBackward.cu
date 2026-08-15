//======================================================//
//  NumericAtomBackward.cu
//  Reverse-time GRU derivative for NumericAtom.
//======================================================//

#include "NumericAtomBackward.hpp"

#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace GRIM {
namespace autograd {

namespace {

constexpr int kBlockSize = 256;

__global__ void kernelNumericAtomBackward(
    const float* __restrict__ upstream_gradient,
    const float* __restrict__ digit_embedding,
    const float* __restrict__ pow10_embedding,
    const float* __restrict__ Wz,
    const float* __restrict__ Uz,
    const float* __restrict__ Wr,
    const float* __restrict__ Ur,
    const float* __restrict__ Wh,
    const float* __restrict__ Uh,
    const float* __restrict__ stop_classifier,
    const float* __restrict__ digit_logits,
    const float* __restrict__ pow10_logits,
    const float* __restrict__ stop_logits,
    const float* __restrict__ final_states,
    const float* __restrict__ step_states,
    const float* __restrict__ update_gates,
    const float* __restrict__ reset_gates,
    const float* __restrict__ candidates,
    const int* __restrict__ atom_positions,
    const int* __restrict__ atom_types,
    const uint8_t* __restrict__ atom_valid,
    const int* __restrict__ target_digits,
    const int* __restrict__ target_pow10,
    const uint8_t* __restrict__ target_digit_mask,
    float* hidden_gradient,
    float* digit_embedding_gradient,
    float* pow10_embedding_gradient,
    float* Wz_gradient,
    float* Uz_gradient,
    float* Wr_gradient,
    float* Ur_gradient,
    float* Wh_gradient,
    float* Uh_gradient,
    float* stop_classifier_gradient,
    int atom_count,
    int digit_slots,
    int digit_classes,
    int pow10_buckets,
    int d_model,
    int int_atom_type,
    int float_atom_type,
    float normalization) {
    const int atom = blockIdx.x;
    if (atom >= atom_count) return;
    const int atom_type = atom_types[atom];
    if (atom_type != int_atom_type && atom_type != float_atom_type) return;
    if (atom_valid[atom] == 0) return;

    extern __shared__ float workspace[];
    float* grad_next = workspace;
    float* grad_state = grad_next + d_model;
    float* grad_az = grad_state + d_model;
    float* grad_ar = grad_az + d_model;
    float* grad_ah = grad_ar + d_model;
    float* softmax_stats = grad_ah + d_model;

    for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
        grad_next[feature] = 0.0f;
    }
    __syncthreads();

    const int opening_row = atom_positions[atom];
    const size_t target_base = static_cast<size_t>(atom) * digit_slots;
    int step_count = 0;
    while (step_count < digit_slots &&
           target_digit_mask[target_base + step_count] != 0) {
        ++step_count;
    }

    // The post-digit state owns the positive typed-CLOSE target. Seed reverse
    // recurrence with that classifier gradient before walking digit steps.
    const int stop_row = atom * (digit_slots + 1) + step_count;
    const float stop_gradient =
        ((1.0f / (1.0f + __expf(-stop_logits[stop_row]))) - 1.0f) *
        upstream_gradient[0] * normalization;
    const float* final_state =
        final_states + static_cast<size_t>(atom) * d_model;
    for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
        grad_next[feature] += stop_gradient * stop_classifier[feature];
        if (stop_classifier_gradient) {
            atomicAdd(&stop_classifier_gradient[feature],
                      stop_gradient * final_state[feature]);
        }
    }
    __syncthreads();

    for (int step = step_count - 1; step >= 0; --step) {
        const size_t target_index = target_base + step;
        const size_t activation_base = target_index * d_model;
        const float* state = step_states + activation_base;
        const float* z = update_gates + activation_base;
        const float* r = reset_gates + activation_base;
        const float* candidate = candidates + activation_base;

        for (int output = threadIdx.x; output < d_model; output += blockDim.x) {
            const float g_next = grad_next[output];
            const float g_candidate = g_next * z[output];
            const float g_update = g_next * (candidate[output] - state[output]);
            grad_ah[output] =
                g_candidate * (1.0f - candidate[output] * candidate[output]);
            grad_az[output] = g_update * z[output] * (1.0f - z[output]);
        }
        __syncthreads();

        for (int input = threadIdx.x; input < d_model; input += blockDim.x) {
            float grad_reset_state = 0.0f;
            for (int output = 0; output < d_model; ++output) {
                grad_reset_state +=
                    Uh[static_cast<size_t>(input) * d_model + output] *
                    grad_ah[output];
            }
            const float grad_reset = grad_reset_state * state[input];
            grad_ar[input] = grad_reset * r[input] * (1.0f - r[input]);
            grad_state[input] =
                grad_next[input] * (1.0f - z[input]) +
                grad_reset_state * r[input];
        }
        __syncthreads();

        for (int input = threadIdx.x; input < d_model; input += blockDim.x) {
            float recurrent_gradient = 0.0f;
            for (int output = 0; output < d_model; ++output) {
                recurrent_gradient +=
                    Uz[static_cast<size_t>(input) * d_model + output] *
                        grad_az[output] +
                    Ur[static_cast<size_t>(input) * d_model + output] *
                        grad_ar[output];
            }
            grad_state[input] += recurrent_gradient;
        }

        const int digit = target_digits[target_index];
        const int pow10 = target_pow10[target_index];
        const size_t matrix_count = static_cast<size_t>(d_model) * d_model;
        for (size_t index = threadIdx.x; index < matrix_count; index += blockDim.x) {
            const int input = static_cast<int>(index / d_model);
            const int output = static_cast<int>(index % d_model);
            const float state_value = state[input];
            if (Uz_gradient) atomicAdd(&Uz_gradient[index], state_value * grad_az[output]);
            if (Ur_gradient) atomicAdd(&Ur_gradient[index], state_value * grad_ar[output]);
            if (Uh_gradient) {
                atomicAdd(&Uh_gradient[index],
                          r[input] * state_value * grad_ah[output]);
            }
        }

        const size_t input_matrix_count =
            static_cast<size_t>(2 * d_model) * d_model;
        for (size_t index = threadIdx.x;
             index < input_matrix_count;
             index += blockDim.x) {
            const int input = static_cast<int>(index / d_model);
            const int output = static_cast<int>(index % d_model);
            const int feature = input < d_model ? input : input - d_model;
            const float input_value = input < d_model
                ? digit_embedding[static_cast<size_t>(digit) * d_model + feature]
                : pow10_embedding[static_cast<size_t>(pow10) * d_model + feature];
            if (Wz_gradient) atomicAdd(&Wz_gradient[index], input_value * grad_az[output]);
            if (Wr_gradient) atomicAdd(&Wr_gradient[index], input_value * grad_ar[output]);
            if (Wh_gradient) atomicAdd(&Wh_gradient[index], input_value * grad_ah[output]);
        }

        for (int input = threadIdx.x; input < 2 * d_model; input += blockDim.x) {
            float input_gradient = 0.0f;
            for (int output = 0; output < d_model; ++output) {
                input_gradient +=
                    Wz[static_cast<size_t>(input) * d_model + output] * grad_az[output] +
                    Wr[static_cast<size_t>(input) * d_model + output] * grad_ar[output] +
                    Wh[static_cast<size_t>(input) * d_model + output] * grad_ah[output];
            }
            if (input < d_model) {
                if (digit_embedding_gradient) {
                    atomicAdd(
                        &digit_embedding_gradient[
                            static_cast<size_t>(digit) * d_model + input],
                        input_gradient);
                }
            } else if (pow10_embedding_gradient) {
                const int feature = input - d_model;
                atomicAdd(
                    &pow10_embedding_gradient[
                        static_cast<size_t>(pow10) * d_model + feature],
                    input_gradient);
            }
        }
        __syncthreads();

        const int decoder_row = atom * (digit_slots + 1) + step;
        if (threadIdx.x == 0) {
            const float* digit_row =
                digit_logits + static_cast<size_t>(decoder_row) * digit_classes;
            const float* pow10_row =
                pow10_logits + static_cast<size_t>(decoder_row) * pow10_buckets;
            float digit_max = digit_row[0];
            for (int cls = 1; cls < digit_classes; ++cls) {
                digit_max = fmaxf(digit_max, digit_row[cls]);
            }
            float digit_sum = 0.0f;
            for (int cls = 0; cls < digit_classes; ++cls) {
                digit_sum += __expf(digit_row[cls] - digit_max);
            }
            float pow10_max = pow10_row[0];
            for (int cls = 1; cls < pow10_buckets; ++cls) {
                pow10_max = fmaxf(pow10_max, pow10_row[cls]);
            }
            float pow10_sum = 0.0f;
            for (int cls = 0; cls < pow10_buckets; ++cls) {
                pow10_sum += __expf(pow10_row[cls] - pow10_max);
            }
            softmax_stats[0] = digit_max;
            softmax_stats[1] = 1.0f / fmaxf(digit_sum, 1e-20f);
            softmax_stats[2] = pow10_max;
            softmax_stats[3] = 1.0f / fmaxf(pow10_sum, 1e-20f);
        }
        __syncthreads();

        const float scale = upstream_gradient[0] * normalization;
        const float* digit_row =
            digit_logits + static_cast<size_t>(decoder_row) * digit_classes;
        const float* pow10_row =
            pow10_logits + static_cast<size_t>(decoder_row) * pow10_buckets;
        const float digit_stop_gradient =
            (1.0f / (1.0f + __expf(-stop_logits[decoder_row]))) * scale;

        for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
            float classifier_state_gradient =
                digit_stop_gradient * stop_classifier[feature];
            if (stop_classifier_gradient) {
                atomicAdd(&stop_classifier_gradient[feature],
                          digit_stop_gradient * state[feature]);
            }
            for (int cls = 0; cls < digit_classes; ++cls) {
                const float probability =
                    __expf(digit_row[cls] - softmax_stats[0]) * softmax_stats[1];
                const float logit_gradient =
                    (probability - (cls == digit ? 1.0f : 0.0f)) * scale;
                classifier_state_gradient +=
                    logit_gradient *
                    digit_embedding[static_cast<size_t>(cls) * d_model + feature];
                if (digit_embedding_gradient) {
                    atomicAdd(
                        &digit_embedding_gradient[
                            static_cast<size_t>(cls) * d_model + feature],
                        logit_gradient * state[feature]);
                }
            }
            for (int cls = 0; cls < pow10_buckets; ++cls) {
                const float probability =
                    __expf(pow10_row[cls] - softmax_stats[2]) * softmax_stats[3];
                const float logit_gradient =
                    (probability - (cls == pow10 ? 1.0f : 0.0f)) * scale;
                classifier_state_gradient +=
                    logit_gradient *
                    pow10_embedding[static_cast<size_t>(cls) * d_model + feature];
                if (pow10_embedding_gradient) {
                    atomicAdd(
                        &pow10_embedding_gradient[
                            static_cast<size_t>(cls) * d_model + feature],
                        logit_gradient * state[feature]);
                }
            }
            grad_state[feature] += classifier_state_gradient;
        }
        __syncthreads();

        for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
            grad_next[feature] = grad_state[feature];
        }
        __syncthreads();
    }

    if (hidden_gradient) {
        for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
            atomicAdd(
                &hidden_gradient[
                    static_cast<size_t>(opening_row) * d_model + feature],
                grad_next[feature]);
        }
    }
}

struct NumericAtomBackwardFn final : public GradFn {
    std::shared_ptr<Tensor> hidden_gradient;
    std::shared_ptr<Tensor> digit_embedding_gradient;
    std::shared_ptr<Tensor> pow10_embedding_gradient;
    std::shared_ptr<Tensor> Wz_gradient;
    std::shared_ptr<Tensor> Uz_gradient;
    std::shared_ptr<Tensor> Wr_gradient;
    std::shared_ptr<Tensor> Ur_gradient;
    std::shared_ptr<Tensor> Wh_gradient;
    std::shared_ptr<Tensor> Uh_gradient;
    std::shared_ptr<Tensor> stop_classifier_gradient;

    const float* digit_embedding = nullptr;
    const float* pow10_embedding = nullptr;
    const float* Wz = nullptr;
    const float* Uz = nullptr;
    const float* Wr = nullptr;
    const float* Ur = nullptr;
    const float* Wh = nullptr;
    const float* Uh = nullptr;
    const float* stop_classifier = nullptr;
    const float* digit_logits = nullptr;
    const float* pow10_logits = nullptr;
    const float* stop_logits = nullptr;
    const float* final_states = nullptr;
    const float* step_states = nullptr;
    const float* update_gates = nullptr;
    const float* reset_gates = nullptr;
    const float* candidates = nullptr;

    int atom_count = 0;
    int digit_slots = 0;
    int digit_classes = 0;
    int pow10_buckets = 0;
    int hidden_rows = 0;
    int d_model = 0;

    NumericAtomBackwardFn() { op_name = "numeric_atom_gru"; }
    ~NumericAtomBackwardFn() override { release_saved(); }

    void capture_inputs(
        Tensor& shared_hidden_state,
        NumberEncoderParameterTensors& parameters,
        const Forward::NumericAtomForwardOutputs& outputs,
        cudaStream_t stream) {
        digit_embedding = parameters.digit_emb.data;
        pow10_embedding = parameters.pow10_emb.data;
        Wz = parameters.numeric_atom_Wz.data;
        Uz = parameters.numeric_atom_Uz.data;
        Wr = parameters.numeric_atom_Wr.data;
        Ur = parameters.numeric_atom_Ur.data;
        Wh = parameters.numeric_atom_Wh.data;
        Uh = parameters.numeric_atom_Uh.data;
        stop_classifier = parameters.numeric_atom_stop_classifier.data;
        digit_logits = outputs.digit_logits.data;
        pow10_logits = outputs.pow10_logits.data;
        stop_logits = outputs.stop_logits.data;
        final_states = outputs.final_states.data;
        step_states = outputs.step_states.data;
        update_gates = outputs.update_gates.data;
        reset_gates = outputs.reset_gates.data;
        candidates = outputs.candidates.data;

        if (shared_hidden_state.requires_grad) {
            hidden_gradient = capture_input_gradient(
                shared_hidden_state, stream, "NumericAtomBackward.hidden");
        }
        if (parameters.digit_emb.requires_grad) {
            digit_embedding_gradient = capture_input_gradient(
                parameters.digit_emb, stream, "NumericAtomBackward.digit_embedding");
        }
        if (parameters.pow10_emb.requires_grad) {
            pow10_embedding_gradient = capture_input_gradient(
                parameters.pow10_emb, stream, "NumericAtomBackward.pow10_embedding");
        }
        if (parameters.numeric_atom_Wz.requires_grad) {
            Wz_gradient = capture_input_gradient(
                parameters.numeric_atom_Wz, stream, "NumericAtomBackward.Wz");
        }
        if (parameters.numeric_atom_Uz.requires_grad) {
            Uz_gradient = capture_input_gradient(
                parameters.numeric_atom_Uz, stream, "NumericAtomBackward.Uz");
        }
        if (parameters.numeric_atom_Wr.requires_grad) {
            Wr_gradient = capture_input_gradient(
                parameters.numeric_atom_Wr, stream, "NumericAtomBackward.Wr");
        }
        if (parameters.numeric_atom_Ur.requires_grad) {
            Ur_gradient = capture_input_gradient(
                parameters.numeric_atom_Ur, stream, "NumericAtomBackward.Ur");
        }
        if (parameters.numeric_atom_Wh.requires_grad) {
            Wh_gradient = capture_input_gradient(
                parameters.numeric_atom_Wh, stream, "NumericAtomBackward.Wh");
        }
        if (parameters.numeric_atom_Uh.requires_grad) {
            Uh_gradient = capture_input_gradient(
                parameters.numeric_atom_Uh, stream, "NumericAtomBackward.Uh");
        }
        if (parameters.numeric_atom_stop_classifier.requires_grad) {
            stop_classifier_gradient = capture_input_gradient(
                parameters.numeric_atom_stop_classifier,
                stream,
                "NumericAtomBackward.stop_classifier");
        }
    }

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* payload,
        const Batching::BatchDeviceBindings* bindings) override {
        if (applied) return;
        applied = true;
        if (!payload || !bindings) {
            throw std::runtime_error(
                "NumericAtomBackward: payload and device bindings are required");
        }
        if (!grad_output.data || grad_output.numel() != 1) {
            throw std::runtime_error(
                "NumericAtomBackward: upstream gradient must be scalar");
        }
        payload->validate("NumericAtomBackward");
        if (payload->total_tokens != hidden_rows ||
            static_cast<int>(payload->atom_positions.size()) != atom_count ||
            payload->number_aux_target_digit_slots != digit_slots ||
            2 * payload->number_aux_target_max_abs_pow10 + 1 != pow10_buckets) {
            throw std::runtime_error(
                "NumericAtomBackward: payload geometry disagrees with captured forward state");
        }
        if (!digit_embedding || !pow10_embedding || !Wz || !Uz || !Wr || !Ur ||
            !Wh || !Uh || !stop_classifier || !digit_logits || !pow10_logits ||
            !stop_logits || !final_states || !step_states ||
            !update_gates || !reset_gates || !candidates) {
            throw std::runtime_error(
                "NumericAtomBackward: borrowed forward state is incomplete");
        }
        if (!bindings->d_atom_positions || !bindings->d_atom_types ||
            !bindings->d_number_aux_target_valid ||
            !bindings->d_number_aux_target_digits ||
            !bindings->d_number_aux_target_pow10_index ||
            !bindings->d_number_aux_target_digit_mask) {
            throw std::runtime_error(
                "NumericAtomBackward: numeric target device bindings are incomplete");
        }

        int valid_digit_steps = 0;
        for (int atom = 0; atom < atom_count; ++atom) {
            if (payload->number_aux_target_valid[static_cast<size_t>(atom)] == 0) continue;
            valid_digit_steps += static_cast<int>(
                payload->number_aux_target_digit_count[static_cast<size_t>(atom)]);
        }
        const int valid_stop_steps = payload->number_aux_target_valid_count;
        if (valid_digit_steps <= 0 || valid_stop_steps <= 0) {
            throw std::runtime_error(
                "NumericAtomBackward: digit and stop supervision are both required");
        }

        const float normalization =
            1.0f / static_cast<float>(3 * valid_digit_steps + valid_stop_steps);
        const size_t shared_bytes =
            (static_cast<size_t>(5) * d_model + 4) * sizeof(float);
        kernelNumericAtomBackward<<<atom_count, kBlockSize, shared_bytes, stream>>>(
            grad_output.data,
            digit_embedding,
            pow10_embedding,
            Wz,
            Uz,
            Wr,
            Ur,
            Wh,
            Uh,
            stop_classifier,
            digit_logits,
            pow10_logits,
            stop_logits,
            final_states,
            step_states,
            update_gates,
            reset_gates,
            candidates,
            bindings->d_atom_positions,
            bindings->d_atom_types,
            bindings->d_number_aux_target_valid,
            bindings->d_number_aux_target_digits,
            bindings->d_number_aux_target_pow10_index,
            bindings->d_number_aux_target_digit_mask,
            hidden_gradient ? hidden_gradient->data : nullptr,
            digit_embedding_gradient ? digit_embedding_gradient->data : nullptr,
            pow10_embedding_gradient ? pow10_embedding_gradient->data : nullptr,
            Wz_gradient ? Wz_gradient->data : nullptr,
            Uz_gradient ? Uz_gradient->data : nullptr,
            Wr_gradient ? Wr_gradient->data : nullptr,
            Ur_gradient ? Ur_gradient->data : nullptr,
            Wh_gradient ? Wh_gradient->data : nullptr,
            Uh_gradient ? Uh_gradient->data : nullptr,
            stop_classifier_gradient ? stop_classifier_gradient->data : nullptr,
            atom_count,
            digit_slots,
            digit_classes,
            pow10_buckets,
            d_model,
            static_cast<int>(Tokenizer::AtomType::ATOM_INT),
            static_cast<int>(Tokenizer::AtomType::ATOM_FLOAT),
            normalization);
        trackKernelLaunch("kernelNumericAtomBackward", stream);

        auto propagate = [&](const std::shared_ptr<Tensor>& gradient, const char* name) {
            if (gradient) {
                propagate_input_gradient(gradient, stream, payload, bindings, name);
            }
        };
        propagate(hidden_gradient, "NumericAtomBackward.hidden");
        propagate(digit_embedding_gradient, "NumericAtomBackward.digit_embedding");
        propagate(pow10_embedding_gradient, "NumericAtomBackward.pow10_embedding");
        propagate(Wz_gradient, "NumericAtomBackward.Wz");
        propagate(Uz_gradient, "NumericAtomBackward.Uz");
        propagate(Wr_gradient, "NumericAtomBackward.Wr");
        propagate(Ur_gradient, "NumericAtomBackward.Ur");
        propagate(Wh_gradient, "NumericAtomBackward.Wh");
        propagate(Uh_gradient, "NumericAtomBackward.Uh");
        propagate(stop_classifier_gradient, "NumericAtomBackward.stop_classifier");
    }

    void release_saved() override {
        GradFn::release_saved();
        hidden_gradient.reset();
        digit_embedding_gradient.reset();
        pow10_embedding_gradient.reset();
        Wz_gradient.reset();
        Uz_gradient.reset();
        Wr_gradient.reset();
        Ur_gradient.reset();
        Wh_gradient.reset();
        Uh_gradient.reset();
        stop_classifier_gradient.reset();
        digit_embedding = nullptr;
        pow10_embedding = nullptr;
        Wz = nullptr;
        Uz = nullptr;
        Wr = nullptr;
        Ur = nullptr;
        Wh = nullptr;
        Uh = nullptr;
        stop_classifier = nullptr;
        digit_logits = nullptr;
        pow10_logits = nullptr;
        stop_logits = nullptr;
        final_states = nullptr;
        step_states = nullptr;
        update_gates = nullptr;
        reset_gates = nullptr;
        candidates = nullptr;
    }
};

void requireShape(
    const Tensor& tensor,
    int rows,
    int columns,
    const char* name) {
    tensor.require(name);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(name) + " must be 2D");
    }
    const auto shape = tensor.shape.as_2d();
    if (shape.rows != rows || shape.cols != columns) {
        throw std::runtime_error(std::string(name) + " has invalid geometry");
    }
}

}  // namespace

void attachNumericAtomBackward(
    Tensor& numeric_atom_loss,
    Tensor& shared_hidden_state,
    NumberEncoderParameterTensors& parameters,
    const Forward::NumericAtomForwardOutputs& outputs,
    cudaStream_t stream) {
    numeric_atom_loss.require("attachNumericAtomBackward.loss");
    shared_hidden_state.require("attachNumericAtomBackward.hidden");
    if (!stream) {
        throw std::runtime_error("attachNumericAtomBackward: stream is NULL");
    }
    if (numeric_atom_loss.numel() != 1 || numeric_atom_loss.grad_fn ||
        numeric_atom_loss.requires_grad) {
        throw std::runtime_error(
            "attachNumericAtomBackward: loss must be a detached scalar");
    }
    if (!outputs.populated() || outputs.atom_count <= 0) {
        throw std::runtime_error(
            "attachNumericAtomBackward: recurrent forward outputs are incomplete");
    }
    if (!shared_hidden_state.shape.is_2d_layout()) {
        throw std::runtime_error(
            "attachNumericAtomBackward: shared hidden state must be 2D");
    }

    const auto hidden_shape = shared_hidden_state.shape.as_2d();
    const int d_model = hidden_shape.cols;
    outputs.step_states.require("attachNumericAtomBackward.step_states");
    if (!outputs.step_states.shape.is_2d_layout()) {
        throw std::runtime_error(
            "attachNumericAtomBackward: step states must be 2D");
    }
    const auto step_state_shape = outputs.step_states.shape.as_2d();
    if (outputs.decoder_step_capacity <= 0 ||
        step_state_shape.rows !=
            outputs.atom_count * outputs.decoder_step_capacity) {
        throw std::runtime_error(
            "attachNumericAtomBackward: step-state rows are not decoder aligned");
    }
    const int digit_slots = outputs.decoder_step_capacity;
    requireShape(parameters.digit_emb, outputs.digit_classes, d_model,
                 "attachNumericAtomBackward.digit_emb");
    requireShape(parameters.pow10_emb, outputs.pow10_buckets, d_model,
                 "attachNumericAtomBackward.pow10_emb");
    requireShape(parameters.numeric_atom_Wz, 2 * d_model, d_model,
                 "attachNumericAtomBackward.Wz");
    requireShape(parameters.numeric_atom_Uz, d_model, d_model,
                 "attachNumericAtomBackward.Uz");
    requireShape(parameters.numeric_atom_Wr, 2 * d_model, d_model,
                 "attachNumericAtomBackward.Wr");
    requireShape(parameters.numeric_atom_Ur, d_model, d_model,
                 "attachNumericAtomBackward.Ur");
    requireShape(parameters.numeric_atom_Wh, 2 * d_model, d_model,
                 "attachNumericAtomBackward.Wh");
    requireShape(parameters.numeric_atom_Uh, d_model, d_model,
                 "attachNumericAtomBackward.Uh");
    requireShape(parameters.numeric_atom_stop_classifier, 1, d_model,
                 "attachNumericAtomBackward.stop_classifier");
    requireShape(outputs.digit_logits, outputs.decoder_row_count, outputs.digit_classes,
                 "attachNumericAtomBackward.digit_logits");
    requireShape(outputs.pow10_logits, outputs.decoder_row_count, outputs.pow10_buckets,
                 "attachNumericAtomBackward.pow10_logits");
    requireShape(outputs.stop_logits, outputs.decoder_row_count, 1,
                 "attachNumericAtomBackward.stop_logits");
    requireShape(outputs.final_states, outputs.atom_count, d_model,
                 "attachNumericAtomBackward.final_states");
    requireShape(outputs.step_states, outputs.atom_count * digit_slots, d_model,
                 "attachNumericAtomBackward.step_states");
    requireShape(outputs.update_gates, outputs.atom_count * digit_slots, d_model,
                 "attachNumericAtomBackward.update_gates");
    requireShape(outputs.reset_gates, outputs.atom_count * digit_slots, d_model,
                 "attachNumericAtomBackward.reset_gates");
    requireShape(outputs.candidates, outputs.atom_count * digit_slots, d_model,
                 "attachNumericAtomBackward.candidates");
        if (digit_slots <= 0 || outputs.decoder_row_count !=
            outputs.atom_count * (digit_slots + 1)) {
        throw std::runtime_error(
            "attachNumericAtomBackward: recurrent geometry is inconsistent");
    }
    if (!shared_hidden_state.requires_grad || !parameters.digit_emb.requires_grad ||
        !parameters.pow10_emb.requires_grad || !parameters.numeric_atom_Wz.requires_grad ||
        !parameters.numeric_atom_Uz.requires_grad || !parameters.numeric_atom_Wr.requires_grad ||
        !parameters.numeric_atom_Ur.requires_grad || !parameters.numeric_atom_Wh.requires_grad ||
        !parameters.numeric_atom_Uh.requires_grad ||
        !parameters.numeric_atom_stop_classifier.requires_grad) {
        throw std::runtime_error(
            "attachNumericAtomBackward: all recurrent training inputs must require gradients");
    }

    auto grad_fn = std::make_shared<NumericAtomBackwardFn>();
    grad_fn->atom_count = outputs.atom_count;
    grad_fn->digit_slots = digit_slots;
    grad_fn->digit_classes = outputs.digit_classes;
    grad_fn->pow10_buckets = outputs.pow10_buckets;
    grad_fn->hidden_rows = hidden_shape.rows;
    grad_fn->d_model = d_model;
    grad_fn->capture_inputs(shared_hidden_state, parameters, outputs, stream);

    numeric_atom_loss.requires_grad = true;
    numeric_atom_loss.is_leaf = false;
    numeric_atom_loss.grad_fn = std::move(grad_fn);
}

}  // namespace autograd
}  // namespace GRIM
