//======================================================//
//  NumericAtomForward.cu
//  Teacher-forced recurrent numeric decoder.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "NumericAtomForward.hpp"

#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Forward {

namespace {

constexpr int kBlockSize = 256;

__device__ float numericAtomSigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

// One block owns one authored atom. Numeric atoms advance sequentially through
// their digit steps; separate atoms execute independently in parallel.
__global__ void kernelNumericAtomForward(
    const float* __restrict__ hidden,
    const float* __restrict__ digit_embedding,
    const float* __restrict__ pow10_embedding,
    const float* __restrict__ Wz,
    const float* __restrict__ Uz,
    const float* __restrict__ Wr,
    const float* __restrict__ Ur,
    const float* __restrict__ Wh,
    const float* __restrict__ Uh,
    const int* __restrict__ atom_positions,
    const int* __restrict__ atom_types,
    const int* __restrict__ target_digits,
    const int* __restrict__ target_pow10,
    const uint8_t* __restrict__ target_digit_mask,
    const int* __restrict__ row_atom_index,
    const uint8_t* __restrict__ row_mask,
    const int* __restrict__ row_step_index,
    float* __restrict__ digit_logits,
    float* __restrict__ pow10_logits,
    float* __restrict__ final_states,
    float* __restrict__ step_states,
    float* __restrict__ saved_update_gates,
    float* __restrict__ saved_reset_gates,
    float* __restrict__ saved_candidates,
    int atom_count,
    int total_rows,
    int digit_slots,
    int digit_classes,
    int pow10_buckets,
    int d_model,
    int int_atom_type,
    int float_atom_type) {
    const int atom = blockIdx.x;
    if (atom >= atom_count) return;

    const int atom_type = atom_types[atom];
    if (atom_type != int_atom_type && atom_type != float_atom_type) return;

    extern __shared__ float workspace[];
    float* state = workspace;
    float* update_gate = state + d_model;
    float* reset_gate = update_gate + d_model;
    float* candidate = reset_gate + d_model;

    const int opening_row = atom_positions[atom];
    for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
        state[feature] = hidden[static_cast<size_t>(opening_row) * d_model + feature];
    }
    __syncthreads();

    const size_t target_base = static_cast<size_t>(atom) * digit_slots;
    for (int step = 0; step < digit_slots; ++step) {
        const size_t target_index = target_base + step;
        if (target_digit_mask[target_index] == 0) break;

        const int row = opening_row + step;
        if (row < 0 || row >= total_rows ||
            row_atom_index[row] != atom || row_step_index[row] != step) {
            break;
        }

        if (row_mask[row] != 0) {
            for (int cls = threadIdx.x; cls < digit_classes; cls += blockDim.x) {
                float logit = 0.0f;
                for (int feature = 0; feature < d_model; ++feature) {
                    logit += state[feature] *
                        digit_embedding[static_cast<size_t>(cls) * d_model + feature];
                }
                digit_logits[static_cast<size_t>(row) * digit_classes + cls] = logit;
            }
            for (int cls = threadIdx.x; cls < pow10_buckets; cls += blockDim.x) {
                float logit = 0.0f;
                for (int feature = 0; feature < d_model; ++feature) {
                    logit += state[feature] *
                        pow10_embedding[static_cast<size_t>(cls) * d_model + feature];
                }
                pow10_logits[static_cast<size_t>(row) * pow10_buckets + cls] = logit;
            }
        }

        const int digit = target_digits[target_index];
        const int pow10 = target_pow10[target_index];
        const size_t activation_base = target_index * d_model;
        for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
            step_states[activation_base + feature] = state[feature];
        }
        for (int output = threadIdx.x; output < d_model; output += blockDim.x) {
            float z_value = 0.0f;
            float r_value = 0.0f;
            for (int input = 0; input < d_model; ++input) {
                const float digit_value =
                    digit_embedding[static_cast<size_t>(digit) * d_model + input];
                const float pow10_value =
                    pow10_embedding[static_cast<size_t>(pow10) * d_model + input];
                z_value += digit_value * Wz[static_cast<size_t>(input) * d_model + output];
                z_value += pow10_value * Wz[static_cast<size_t>(d_model + input) * d_model + output];
                z_value += state[input] * Uz[static_cast<size_t>(input) * d_model + output];
                r_value += digit_value * Wr[static_cast<size_t>(input) * d_model + output];
                r_value += pow10_value * Wr[static_cast<size_t>(d_model + input) * d_model + output];
                r_value += state[input] * Ur[static_cast<size_t>(input) * d_model + output];
            }
            update_gate[output] = numericAtomSigmoid(z_value);
            reset_gate[output] = numericAtomSigmoid(r_value);
            saved_update_gates[activation_base + output] = update_gate[output];
            saved_reset_gates[activation_base + output] = reset_gate[output];
        }
        __syncthreads();

        for (int output = threadIdx.x; output < d_model; output += blockDim.x) {
            float candidate_value = 0.0f;
            for (int input = 0; input < d_model; ++input) {
                const float digit_value =
                    digit_embedding[static_cast<size_t>(digit) * d_model + input];
                const float pow10_value =
                    pow10_embedding[static_cast<size_t>(pow10) * d_model + input];
                candidate_value +=
                    digit_value * Wh[static_cast<size_t>(input) * d_model + output];
                candidate_value +=
                    pow10_value * Wh[static_cast<size_t>(d_model + input) * d_model + output];
                candidate_value += reset_gate[input] * state[input] *
                    Uh[static_cast<size_t>(input) * d_model + output];
            }
            candidate[output] = tanhf(candidate_value);
            saved_candidates[activation_base + output] = candidate[output];
        }
        __syncthreads();

        for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
            const float z = update_gate[feature];
            state[feature] =
                (1.0f - z) * state[feature] + z * candidate[feature];
        }
        __syncthreads();
    }

    for (int feature = threadIdx.x; feature < d_model; feature += blockDim.x) {
        final_states[static_cast<size_t>(atom) * d_model + feature] = state[feature];
    }
}

void requireMatrixShape(
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
        throw std::runtime_error(
            std::string(name) + " shape=[" + std::to_string(shape.rows) + "," +
            std::to_string(shape.cols) + "] expected=[" + std::to_string(rows) +
            "," + std::to_string(columns) + "]");
    }
}

}  // namespace

NumericAtomForwardOutputs NumericAtomForward(
    const Tensor& shared_hidden_state,
    const NumberEncoderParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream) {
    if (!stream) {
        throw std::runtime_error("NumericAtomForward: stream is NULL");
    }
    payload.validate("NumericAtomForward");
    shared_hidden_state.require("NumericAtomForward.shared_hidden_state");
    if (!shared_hidden_state.shape.is_2d_layout()) {
        throw std::runtime_error("NumericAtomForward: shared hidden state must be 2D");
    }

    const auto hidden_shape = shared_hidden_state.shape.as_2d();
    const int d_model = hidden_shape.cols;
    const int digit_slots = payload.number_aux_target_digit_slots;
    const int pow10_buckets = 2 * payload.number_aux_target_max_abs_pow10 + 1;
    const int atom_count = static_cast<int>(payload.atom_positions.size());
    if (hidden_shape.rows != payload.total_tokens || d_model <= 0 || digit_slots <= 0) {
        throw std::runtime_error(
            "NumericAtomForward: hidden state and payload geometry disagree");
    }

    requireMatrixShape(parameters.digit_emb, 10, d_model,
                       "NumericAtomForward.parameters.digit_emb");
    requireMatrixShape(parameters.pow10_emb, pow10_buckets, d_model,
                       "NumericAtomForward.parameters.pow10_emb");
    requireMatrixShape(parameters.numeric_atom_Wz, 2 * d_model, d_model,
                       "NumericAtomForward.parameters.numeric_atom_Wz");
    requireMatrixShape(parameters.numeric_atom_Uz, d_model, d_model,
                       "NumericAtomForward.parameters.numeric_atom_Uz");
    requireMatrixShape(parameters.numeric_atom_Wr, 2 * d_model, d_model,
                       "NumericAtomForward.parameters.numeric_atom_Wr");
    requireMatrixShape(parameters.numeric_atom_Ur, d_model, d_model,
                       "NumericAtomForward.parameters.numeric_atom_Ur");
    requireMatrixShape(parameters.numeric_atom_Wh, 2 * d_model, d_model,
                       "NumericAtomForward.parameters.numeric_atom_Wh");
    requireMatrixShape(parameters.numeric_atom_Uh, d_model, d_model,
                       "NumericAtomForward.parameters.numeric_atom_Uh");

    if (!bindings.d_atom_positions || !bindings.d_atom_types ||
        !bindings.d_number_aux_target_digits ||
        !bindings.d_number_aux_target_pow10_index ||
        !bindings.d_number_aux_target_digit_mask ||
        !bindings.d_number_aux_target_atom_index ||
        !bindings.d_number_aux_target_row_mask ||
        !bindings.d_number_aux_target_step_index) {
        throw std::runtime_error(
            "NumericAtomForward: numeric decoder device bindings are incomplete");
    }

    // Fail before launching if the host-authored causal routing cannot carry
    // every teacher-forced digit step for a numeric atom.
    for (int atom = 0; atom < atom_count; ++atom) {
        const auto atom_type = static_cast<Tokenizer::AtomType>(
            payload.atom_types[static_cast<size_t>(atom)]);
        if (!Tokenizer::isNumericAtom(atom_type)) continue;

        const int opening_row = payload.atom_positions[static_cast<size_t>(atom)];
        const int digit_count = static_cast<int>(
            payload.number_aux_target_digit_count[static_cast<size_t>(atom)]);
        for (int step = 0; step < digit_count; ++step) {
            const int row = opening_row + step;
            if (row < 0 || row >= payload.total_tokens ||
                payload.number_aux_target_atom_index[static_cast<size_t>(row)] != atom ||
                payload.number_aux_target_step_index[static_cast<size_t>(row)] != step) {
                throw std::runtime_error(
                    "NumericAtomForward: causal row routing does not cover every numeric decoder step");
            }
        }
    }

    NumericAtomForwardOutputs outputs;
    outputs.evaluated = true;
    outputs.row_count = hidden_shape.rows;
    outputs.atom_count = atom_count;
    outputs.decoder_step_capacity = digit_slots;
    outputs.digit_classes = 10;
    outputs.pow10_buckets = pow10_buckets;
    outputs.digit_logits = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(outputs.row_count, outputs.digit_classes),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.digit_logits");
    outputs.pow10_logits = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(outputs.row_count, outputs.pow10_buckets),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.pow10_logits");
    if (atom_count == 0) {
        return outputs;
    }
    outputs.final_states = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(atom_count, d_model),
        /*requires_grad=*/false,
        stream,
        "numeric_atom.final_states");
    const int activation_rows = atom_count * digit_slots;
    const auto activation_shape =
        TensorContract::TensorShape::make_BSM(activation_rows, d_model);
    outputs.step_states = Tensor::zeros(
        activation_shape, false, stream, "numeric_atom.step_states");
    outputs.update_gates = Tensor::zeros(
        activation_shape, false, stream, "numeric_atom.update_gates");
    outputs.reset_gates = Tensor::zeros(
        activation_shape, false, stream, "numeric_atom.reset_gates");
    outputs.candidates = Tensor::zeros(
        activation_shape, false, stream, "numeric_atom.candidates");

    const size_t shared_bytes = static_cast<size_t>(4) * d_model * sizeof(float);
    kernelNumericAtomForward<<<atom_count, kBlockSize, shared_bytes, stream>>>(
        shared_hidden_state.data,
        parameters.digit_emb.data,
        parameters.pow10_emb.data,
        parameters.numeric_atom_Wz.data,
        parameters.numeric_atom_Uz.data,
        parameters.numeric_atom_Wr.data,
        parameters.numeric_atom_Ur.data,
        parameters.numeric_atom_Wh.data,
        parameters.numeric_atom_Uh.data,
        bindings.d_atom_positions,
        bindings.d_atom_types,
        bindings.d_number_aux_target_digits,
        bindings.d_number_aux_target_pow10_index,
        bindings.d_number_aux_target_digit_mask,
        bindings.d_number_aux_target_atom_index,
        bindings.d_number_aux_target_row_mask,
        bindings.d_number_aux_target_step_index,
        outputs.digit_logits.data,
        outputs.pow10_logits.data,
        outputs.final_states.data,
        outputs.step_states.data,
        outputs.update_gates.data,
        outputs.reset_gates.data,
        outputs.candidates.data,
        atom_count,
        outputs.row_count,
        digit_slots,
        outputs.digit_classes,
        outputs.pow10_buckets,
        d_model,
        static_cast<int>(Tokenizer::AtomType::ATOM_INT),
        static_cast<int>(Tokenizer::AtomType::ATOM_FLOAT));
    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("NumericAtomForward: recurrent kernel launch failed: ") +
            cudaGetErrorString(launch_error));
    }

    return outputs;
}

}  // namespace Forward
}  // namespace GRIM
