//======================================================//
//  AtomInsertionLoss.cu
//  Typed delimiter loss assembled from autograd primitives
//======================================================//

#include "AtomInsertionLoss.hpp"

#include "../../UnigramByte/TokenLayout.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr int kBlockSize = 256;

void checkCudaLaunch(const char* caller, const char* kernel_name) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + kernel_name + " launch failed: " +
            cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float stableSoftplus(float value) {
    return fmaxf(value, 0.0f) + log1pf(expf(-fabsf(value)));
}

__global__ void kernelMaskedBinaryCrossEntropyWithLogitsForward(
    const float* __restrict__ logits,
    const uint8_t* __restrict__ targets,
    const uint8_t* __restrict__ valid_gap_mask,
    float* __restrict__ loss,
    float* __restrict__ saved_probabilities,
    int gap_rows,
    int labels_per_gap,
    float positive_label_weight,
    float negative_label_weight,
    float inverse_normalization) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int label_count = gap_rows * labels_per_gap;
    if (index >= label_count) return;

    const float logit = logits[index];
    if (saved_probabilities) {
        saved_probabilities[index] = 1.0f / (1.0f + expf(-logit));
    }

    const int gap = index / labels_per_gap;
    if (valid_gap_mask[gap] == 0) return;

    const uint8_t target = targets[index];
    if (target > 1) {
        asm volatile("trap;");
        return;
    }

    const float weighted_loss = target != 0
        ? positive_label_weight * stableSoftplus(-logit)
        : negative_label_weight * stableSoftplus(logit);
    atomicAdd(loss, weighted_loss * inverse_normalization);
}

__global__ void kernelMaskedBinaryCrossEntropyWithLogitsBackward(
    const float* __restrict__ grad_output,
    const float* __restrict__ saved_probabilities,
    const uint8_t* __restrict__ targets,
    const uint8_t* __restrict__ valid_gap_mask,
    float* __restrict__ grad_logits,
    int gap_rows,
    int labels_per_gap,
    float positive_label_weight,
    float negative_label_weight,
    float inverse_normalization) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int label_count = gap_rows * labels_per_gap;
    if (index >= label_count) return;

    const int gap = index / labels_per_gap;
    if (valid_gap_mask[gap] == 0) return;

    const uint8_t target = targets[index];
    if (target > 1) {
        asm volatile("trap;");
        return;
    }

    const float probability = saved_probabilities[index];
    const float label_weight = target != 0
        ? positive_label_weight
        : negative_label_weight;
    grad_logits[index] += grad_output[0] * label_weight *
        (probability - static_cast<float>(target)) * inverse_normalization;
}

} // namespace

namespace GRIM::autograd {

namespace {

// Primitive autograd operation: masked, weighted BCE with logits. This node
// represents one mathematical tensor operation; it is not an atom-insertion
// backward controller. Like the existing text/selector losses, it owns the
// transformed forward values required by backward. Input-gradient storage and
// fan-in are supplied by the ordinary GradFn/AutogradEngine contract.
struct MaskedBinaryCrossEntropyWithLogitsGradFn final : GradFn {
    std::shared_ptr<Tensor> logits_gradient;
    Tensor saved_probabilities;
    int gap_rows = 0;
    int labels_per_gap = 0;
    float positive_label_weight = 1.0f;
    float negative_label_weight = 1.0f;
    float inverse_normalization = 0.0f;

    MaskedBinaryCrossEntropyWithLogitsGradFn() {
        op_name = "masked_binary_cross_entropy_with_logits";
    }

    ~MaskedBinaryCrossEntropyWithLogitsGradFn() override {
        release_saved();
    }

    void capture_input(Tensor& logits, cudaStream_t stream) {
        if (logits.requires_grad) {
            logits_gradient = capture_input_gradient(
                logits,
                stream,
                "MaskedBinaryCrossEntropyWithLogitsGradFn::capture_input");
        }
    }

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override {
        setCurrentGradFnOp("masked_binary_cross_entropy_with_logits", this);
        if (applied) return;
        applied = true;

        if (!backward_payload || !backward_bindings) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: scheduler "
                "must provide BatchPayload and BatchDeviceBindings");
        }
        backward_payload->validate(
            "MaskedBinaryCrossEntropyWithLogitsGradFn::apply");
        if (!backward_payload->EnableAtomIdentification) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: "
                "EnableAtomIdentification is false");
        }
        if (backward_payload->atomInsertionGapRowCount() != gap_rows) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: backward "
                "payload gap geometry differs from forward");
        }
        if (!backward_bindings->d_atom_insertion_gap_targets ||
            !backward_bindings->d_atom_insertion_valid_gap_mask) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: atom "
                "supervision was not uploaded for this step");
        }
        if (!saved_probabilities.data || !logits_gradient) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: saved "
                "probabilities or captured input gradient is missing");
        }
        if (!saved_probabilities.shape.is_2d_layout()) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: saved "
                "probabilities must be 2D");
        }
        const auto probability_shape = saved_probabilities.shape.as_2d();
        if (probability_shape.rows != gap_rows ||
            probability_shape.cols != labels_per_gap) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: saved "
                "probability geometry differs from forward");
        }
        if (!grad_output.data || !grad_output.shape.is_2d_layout()) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: invalid "
                "scalar upstream gradient");
        }
        const auto grad_shape = grad_output.shape.as_2d();
        if (grad_shape.rows != 1 || grad_shape.cols != 1) {
            throw std::runtime_error(
                "MaskedBinaryCrossEntropyWithLogitsGradFn::apply: upstream "
                "gradient must have shape [1,1]");
        }

        const int label_count = gap_rows * labels_per_gap;
        const int blocks = 1 + (label_count - 1) / kBlockSize;
        kernelMaskedBinaryCrossEntropyWithLogitsBackward<<<
            blocks, kBlockSize, 0, stream>>>(
            grad_output.data,
            saved_probabilities.data,
            backward_bindings->d_atom_insertion_gap_targets,
            backward_bindings->d_atom_insertion_valid_gap_mask,
            logits_gradient->data,
            gap_rows,
            labels_per_gap,
            positive_label_weight,
            negative_label_weight,
            inverse_normalization);
        checkCudaLaunch(
            "MaskedBinaryCrossEntropyWithLogitsGradFn::apply",
            "kernelMaskedBinaryCrossEntropyWithLogitsBackward");

        propagate_input_gradient(
            logits_gradient,
            stream,
            backward_payload,
            backward_bindings,
            "MaskedBinaryCrossEntropyWithLogitsGradFn::apply");
    }

    void release_saved() override {
        if (released_) return;
        GradFn::release_saved();
        logits_gradient.reset();
        saved_probabilities = Tensor();
    }
};

Tensor maskedBinaryCrossEntropyWithLogits(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const AtomInsertion::AtomInsertionLossConfig& config,
    float normalization_weight,
    cudaStream_t stream) {
    constexpr const char* caller = "maskedBinaryCrossEntropyWithLogits";
    if (!stream) {
        throw std::runtime_error(
            "maskedBinaryCrossEntropyWithLogits: stream is NULL");
    }
    if (!logits.data || !logits.shape.is_2d_layout()) {
        throw std::runtime_error(
            "maskedBinaryCrossEntropyWithLogits: logits must be a valid 2D tensor");
    }
    const auto shape = logits.shape.as_2d();
    const int gap_rows = payload.atomInsertionGapRowCount();
    if (gap_rows >
        std::numeric_limits<int>::max() / Tokenizer::ATOM_VOCAB_SIZE) {
        throw std::runtime_error(
            "maskedBinaryCrossEntropyWithLogits: label count overflows int");
    }
    if (shape.rows != gap_rows ||
        shape.cols != Tokenizer::ATOM_VOCAB_SIZE) {
        throw std::runtime_error(
            "maskedBinaryCrossEntropyWithLogits: expected logits shape [" +
            std::to_string(gap_rows) + "," +
            std::to_string(Tokenizer::ATOM_VOCAB_SIZE) + "] got [" +
            std::to_string(shape.rows) + "," +
            std::to_string(shape.cols) + "]");
    }
    if (!bindings.d_atom_insertion_gap_targets ||
        !bindings.d_atom_insertion_valid_gap_mask) {
        throw std::runtime_error(
            "maskedBinaryCrossEntropyWithLogits: atom supervision was not "
            "uploaded for this step");
    }
    if (!(normalization_weight > 0.0f) ||
        !std::isfinite(normalization_weight)) {
        throw std::runtime_error(
            "maskedBinaryCrossEntropyWithLogits: invalid normalization weight");
    }

    Tensor result = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),
        logits.requires_grad,
        stream,
        "masked_bce_with_logits_result");

    Tensor saved_probabilities;
    if (logits.requires_grad) {
        saved_probabilities = Tensor::empty(
            logits.shape,
            false,
            stream,
            "masked_bce_saved_probabilities");
    }

    const int label_count = gap_rows * Tokenizer::ATOM_VOCAB_SIZE;
    const int blocks = 1 + (label_count - 1) / kBlockSize;
    const float inverse_normalization = 1.0f / normalization_weight;
    kernelMaskedBinaryCrossEntropyWithLogitsForward<<<
        blocks, kBlockSize, 0, stream>>>(
        logits.data,
        bindings.d_atom_insertion_gap_targets,
        bindings.d_atom_insertion_valid_gap_mask,
        result.data,
        saved_probabilities.data,
        gap_rows,
        Tokenizer::ATOM_VOCAB_SIZE,
        config.positive_label_weight,
        config.negative_label_weight,
        inverse_normalization);
    checkCudaLaunch(caller, "kernelMaskedBinaryCrossEntropyWithLogitsForward");

    if (logits.requires_grad) {
        result.is_leaf = false;
        auto grad_fn =
            std::make_shared<MaskedBinaryCrossEntropyWithLogitsGradFn>();
        grad_fn->capture_input(logits, stream);
        grad_fn->saved_probabilities = std::move(saved_probabilities);
        grad_fn->gap_rows = gap_rows;
        grad_fn->labels_per_gap = Tokenizer::ATOM_VOCAB_SIZE;
        grad_fn->positive_label_weight = config.positive_label_weight;
        grad_fn->negative_label_weight = config.negative_label_weight;
        grad_fn->inverse_normalization = inverse_normalization;
        result.grad_fn = std::move(grad_fn);
    }

    return result;
}

} // namespace

} // namespace GRIM::autograd

namespace GRIM::AtomInsertion {

void AtomInsertionLossConfig::validate(const char* caller) const {
    const std::string prefix = caller && caller[0] != '\0'
        ? std::string(caller)
        : std::string("AtomInsertionLossConfig::validate");
    if (!(positive_label_weight > 0.0f) ||
        !std::isfinite(positive_label_weight)) {
        throw std::runtime_error(
            prefix + ": positive_label_weight must be finite and positive");
    }
    if (!(negative_label_weight > 0.0f) ||
        !std::isfinite(negative_label_weight)) {
        throw std::runtime_error(
            prefix + ": negative_label_weight must be finite and positive");
    }
}

Tensor atomInsertionLoss(
    Tensor& full_gap_vocab_logits,
    Forward::ModelForwardOutputs& forward_outputs,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool EnableAtomIdentification,
    const AtomInsertionLossConfig& config,
    AtomInsertionLossStats* out_stats,
    cudaStream_t stream) {
    constexpr const char* caller = "atomInsertionLoss";
    if (!EnableAtomIdentification || !payload.EnableAtomIdentification) {
        throw std::runtime_error(
            "atomInsertionLoss: EnableAtomIdentification must be true on both "
            "the entry point and BatchPayload");
    }
    if (!payload.isTraining()) {
        throw std::runtime_error(
            "atomInsertionLoss: loss requires a training BatchPayload");
    }
    config.validate(caller);
    payload.validate(caller);
    if (!full_gap_vocab_logits.data ||
        !full_gap_vocab_logits.shape.is_2d_layout()) {
        throw std::runtime_error(
            "atomInsertionLoss: full_gap_vocab_logits must be a valid 2D tensor");
    }
    const auto full_shape = full_gap_vocab_logits.shape.as_2d();
    const int gap_rows = payload.atomInsertionGapRowCount();
    if (full_shape.rows != gap_rows || full_shape.cols != payload.vocab_size) {
        throw std::runtime_error(
            "atomInsertionLoss: full logits shape does not match payload gap "
            "rows and vocabulary size");
    }

    const int positive_count = payload.atom_insertion_positive_label_count;
    const int total_valid_labels =
        payload.atom_insertion_valid_gap_count * Tokenizer::ATOM_VOCAB_SIZE;
    const int negative_count = total_valid_labels - positive_count;
    if (positive_count < 0 || negative_count < 0) {
        throw std::runtime_error(
            "atomInsertionLoss: invalid authored positive/negative label counts");
    }
    const float normalization_weight =
        static_cast<float>(positive_count) * config.positive_label_weight +
        static_cast<float>(negative_count) * config.negative_label_weight;
    if (!(normalization_weight > 0.0f) ||
        !std::isfinite(normalization_weight)) {
        throw std::runtime_error(
            "atomInsertionLoss: normalization weight must be finite and positive");
    }

    // The slice's GradFn connects backward to the full-vocabulary head and lets
    // the scheduler accumulate any other users. The BCE primitive owns its
    // saved sigmoid probabilities; it does not borrow this slice's value buffer.
    forward_outputs.atom_insertion_delimiter_logits = autograd::slice_columns(
        full_gap_vocab_logits,
        Tokenizer::ATOM_TOKEN_OFFSET,
        Tokenizer::ATOM_VOCAB_SIZE,
        stream);

    if (out_stats) {
        out_stats->valid_gap_count = payload.atom_insertion_valid_gap_count;
        out_stats->positive_label_count = positive_count;
        out_stats->negative_label_count = negative_count;
        out_stats->normalization_weight = normalization_weight;
    }

    return autograd::maskedBinaryCrossEntropyWithLogits(
        forward_outputs.atom_insertion_delimiter_logits,
        payload,
        bindings,
        config,
        normalization_weight,
        stream);
}

} // namespace GRIM::AtomInsertion
