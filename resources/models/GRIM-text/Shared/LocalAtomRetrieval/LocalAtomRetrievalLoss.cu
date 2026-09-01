//======================================================//
//  LocalAtomRetrievalLoss.cu
//  Standalone causal selector cross-entropy.
//
//  Deliberately not wired into ModelForward or the training hot path.
//======================================================//

#include "LocalAtomRetrievalLoss.hpp"

#include "../UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace GRIM::LocalAtomRetrieval {

namespace {

constexpr int kQueryBlockSize = 256;

int blocksForQueries(int query_count, const char* caller) {
    if (query_count <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": query_count must be positive");
    }
    return 1 + (query_count - 1) / kQueryBlockSize;
}

void checkCuda(cudaError_t status, const char* caller) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + cudaGetErrorString(status));
    }
}

int expectedCandidateClassCount(
    const Batching::BatchPayload& payload,
    const char* caller) {
    const auto& offsets = payload.local_atom_row_type_candidate_offsets;
    const std::size_t expected_offset_count =
        static_cast<std::size_t>(payload.batch_size) *
            Tokenizer::kAtomTypeCount +
        1;
    if (offsets.size() != expected_offset_count || offsets.empty()) {
        throw std::runtime_error(
            std::string(caller) +
            ": payload row/type candidate offsets have invalid geometry");
    }

    int maximum_bank_width = 0;
    for (std::size_t bank = 0; bank + 1 < offsets.size(); ++bank) {
        const int width = offsets[bank + 1] - offsets[bank];
        if (width < 0) {
            throw std::runtime_error(
                std::string(caller) +
                ": payload row/type candidate offsets are not monotonic");
        }
        maximum_bank_width = std::max(maximum_bank_width, width);
    }
    if (maximum_bank_width == std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            std::string(caller) + ": candidate class count overflows int");
    }
    return maximum_bank_width + 1;
}

void requireBindings(
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const char* caller) {
    if (bindings.local_atom_query_count != payload.localAtomQueryCount() ||
        bindings.local_atom_candidate_count !=
            payload.localAtomCandidateCount()) {
        throw std::runtime_error(
            std::string(caller) +
            ": local atom binding counts disagree with BatchPayload");
    }
    if (!bindings.d_local_atom_query_positions ||
        !bindings.d_local_atom_query_types ||
        !bindings.d_local_atom_query_targets ||
        !bindings.d_local_atom_row_type_candidate_offsets ||
        !bindings.d_local_atom_candidate_first_close_positions) {
        throw std::runtime_error(
            std::string(caller) +
            ": local atom BatchDeviceBindings are incomplete");
    }
}

__device__ bool isCausalCandidateClass(
    int query,
    int candidate_class,
    const int* query_positions,
    const int* query_types,
    const int* row_type_candidate_offsets,
    const int* candidate_first_close_positions,
    int candidate_count,
    int type_count,
    int sequence_length) {
    if (candidate_class == Batching::kLocalAtomNoReferenceTarget) {
        return true;
    }

    const int query_position = query_positions[query];
    const int type = query_types[query];
    if (query_position < 0 || type < 0 || type >= type_count) {
        return false;
    }
    const int row = query_position / sequence_length;
    const int bank = row * type_count + type;
    const int candidate_begin = row_type_candidate_offsets[bank];
    const int candidate_end = row_type_candidate_offsets[bank + 1];
    const int candidate = candidate_begin + candidate_class - 1;
    return candidate >= candidate_begin && candidate < candidate_end &&
           candidate >= 0 && candidate < candidate_count &&
           candidate_first_close_positions[candidate] < query_position;
}

// One thread owns one query row. This intentionally favors a transparent
// reference equation over a tuned block reduction.
__global__ void kernelLocalAtomRetrievalLossForward(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    const int* __restrict__ query_positions,
    const int* __restrict__ query_types,
    const int* __restrict__ row_type_candidate_offsets,
    const int* __restrict__ candidate_first_close_positions,
    float* __restrict__ mean_loss,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int candidate_class_count) {
    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= query_count) {
        return;
    }

    const float* row =
        logits + static_cast<std::size_t>(query) * candidate_class_count;
    float maximum = row[Batching::kLocalAtomNoReferenceTarget];
    for (int candidate_class = 1;
         candidate_class < candidate_class_count;
         ++candidate_class) {
        if (isCausalCandidateClass(
                query,
                candidate_class,
                query_positions,
                query_types,
                row_type_candidate_offsets,
                candidate_first_close_positions,
                candidate_count,
                type_count,
                sequence_length)) {
            maximum = fmaxf(maximum, row[candidate_class]);
        }
    }

    float exponential_sum = 0.0f;
    for (int candidate_class = 0;
         candidate_class < candidate_class_count;
         ++candidate_class) {
        if (isCausalCandidateClass(
                query,
                candidate_class,
                query_positions,
                query_types,
                row_type_candidate_offsets,
                candidate_first_close_positions,
                candidate_count,
                type_count,
                sequence_length)) {
            exponential_sum += expf(row[candidate_class] - maximum);
        }
    }

    const int target = targets[query];
    if (target < 0 || target >= candidate_class_count ||
        !isCausalCandidateClass(
            query,
            target,
            query_positions,
            query_types,
            row_type_candidate_offsets,
            candidate_first_close_positions,
            candidate_count,
            type_count,
            sequence_length)) {
        // BatchPayload::validate() prevents this. Preserve fail-visible output
        // if a corrupted device binding ever disagrees with its host payload.
        atomicAdd(mean_loss, CUDART_INF_F);
        return;
    }

    const float query_loss =
        logf(fmaxf(exponential_sum, 1.0e-20f)) + maximum - row[target];
    atomicAdd(mean_loss, query_loss / static_cast<float>(query_count));
}

__global__ void kernelLocalAtomRetrievalLossBackward(
    const float* __restrict__ logits,
    const float* __restrict__ grad_loss,
    const int* __restrict__ targets,
    const int* __restrict__ query_positions,
    const int* __restrict__ query_types,
    const int* __restrict__ row_type_candidate_offsets,
    const int* __restrict__ candidate_first_close_positions,
    float* __restrict__ grad_logits,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int candidate_class_count) {
    const int query = blockIdx.x * blockDim.x + threadIdx.x;
    if (query >= query_count) {
        return;
    }

    const float* row =
        logits + static_cast<std::size_t>(query) * candidate_class_count;
    float maximum = row[Batching::kLocalAtomNoReferenceTarget];
    for (int candidate_class = 1;
         candidate_class < candidate_class_count;
         ++candidate_class) {
        if (isCausalCandidateClass(
                query,
                candidate_class,
                query_positions,
                query_types,
                row_type_candidate_offsets,
                candidate_first_close_positions,
                candidate_count,
                type_count,
                sequence_length)) {
            maximum = fmaxf(maximum, row[candidate_class]);
        }
    }

    float exponential_sum = 0.0f;
    for (int candidate_class = 0;
         candidate_class < candidate_class_count;
         ++candidate_class) {
        if (isCausalCandidateClass(
                query,
                candidate_class,
                query_positions,
                query_types,
                row_type_candidate_offsets,
                candidate_first_close_positions,
                candidate_count,
                type_count,
                sequence_length)) {
            exponential_sum += expf(row[candidate_class] - maximum);
        }
    }

    const float inverse_sum =
        exponential_sum > 0.0f ? 1.0f / exponential_sum : 0.0f;
    const float scale = grad_loss[0] / static_cast<float>(query_count);
    const int target = targets[query];
    float* gradient_row =
        grad_logits + static_cast<std::size_t>(query) * candidate_class_count;

    for (int candidate_class = 0;
         candidate_class < candidate_class_count;
         ++candidate_class) {
        if (!isCausalCandidateClass(
                query,
                candidate_class,
                query_positions,
                query_types,
                row_type_candidate_offsets,
                candidate_first_close_positions,
                candidate_count,
                type_count,
                sequence_length)) {
            continue;
        }
        const float probability =
            expf(row[candidate_class] - maximum) * inverse_sum;
        const float target_indicator =
            candidate_class == target ? 1.0f : 0.0f;
        atomicAdd(
            gradient_row + candidate_class,
            scale * (probability - target_indicator));
    }
}

struct LocalAtomRetrievalLossGradFn final : public GradFn {
    const Tensor* logits_ = nullptr;
    std::shared_ptr<Tensor> logits_gradient_;
    int query_count_ = 0;
    int candidate_count_ = 0;
    int type_count_ = 0;
    int sequence_length_ = 0;
    int candidate_class_count_ = 0;

    LocalAtomRetrievalLossGradFn() {
        op_name = "LocalAtomRetrievalLoss";
    }

    void captureInput(
        Tensor& logits,
        const Batching::BatchPayload& payload,
        int candidate_class_count,
        cudaStream_t stream) {
        logits_ = &logits;
        query_count_ = payload.localAtomQueryCount();
        candidate_count_ = payload.localAtomCandidateCount();
        type_count_ = Tokenizer::kAtomTypeCount;
        sequence_length_ = payload.max_seq_len;
        candidate_class_count_ = candidate_class_count;
        logits_gradient_ = capture_input_gradient(
            logits,
            stream,
            "LocalAtomRetrievalLossGradFn::captureInput");
    }

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override {
        setCurrentGradFnOp("LocalAtomRetrievalLossBackwards", this);
        if (applied) {
            return;
        }
        applied = true;
        if (!stream || !backward_payload || !backward_bindings) {
            throw std::runtime_error(
                "LocalAtomRetrievalLossGradFn::apply: stream/payload/bindings are required");
        }
        if (!logits_ || !logits_gradient_) {
            throw std::runtime_error(
                "LocalAtomRetrievalLossGradFn::apply: logits handles are unavailable");
        }
        grad_output.require("LocalAtomRetrievalLossGradFn::apply grad_output");
        if (grad_output.numel() != 1) {
            throw std::runtime_error(
                "LocalAtomRetrievalLossGradFn::apply: grad_output must be scalar");
        }

        backward_payload->validate("LocalAtomRetrievalLossGradFn::apply");
        requireBindings(
            *backward_payload,
            *backward_bindings,
            "LocalAtomRetrievalLossGradFn::apply");
        if (backward_payload->localAtomQueryCount() != query_count_ ||
            backward_payload->localAtomCandidateCount() != candidate_count_ ||
            backward_payload->max_seq_len != sequence_length_ ||
            expectedCandidateClassCount(
                *backward_payload,
                "LocalAtomRetrievalLossGradFn::apply") !=
                candidate_class_count_) {
            throw std::runtime_error(
                "LocalAtomRetrievalLossGradFn::apply: backward geometry differs from forward");
        }

        logits_->require("LocalAtomRetrievalLossGradFn::apply logits");
        logits_gradient_->require(
            "LocalAtomRetrievalLossGradFn::apply logits_gradient");
        kernelLocalAtomRetrievalLossBackward<<<
            blocksForQueries(query_count_, "LocalAtomRetrievalLossGradFn::apply"),
            kQueryBlockSize,
            0,
            stream>>>(
            logits_->data,
            grad_output.data,
            backward_bindings->d_local_atom_query_targets,
            backward_bindings->d_local_atom_query_positions,
            backward_bindings->d_local_atom_query_types,
            backward_bindings->d_local_atom_row_type_candidate_offsets,
            backward_bindings->d_local_atom_candidate_first_close_positions,
            logits_gradient_->data,
            query_count_,
            candidate_count_,
            type_count_,
            sequence_length_,
            candidate_class_count_);
        checkCuda(
            cudaGetLastError(),
            "LocalAtomRetrievalLossGradFn::apply kernel");

        propagate_input_gradient(
            logits_gradient_,
            stream,
            backward_payload,
            backward_bindings,
            "LocalAtomRetrievalLossGradFn::apply logits");
    }

    void release_saved() override {
        GradFn::release_saved();
        logits_gradient_.reset();
        logits_ = nullptr;
    }
};

} // namespace

Tensor LocalAtomRetrievalLoss(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream) {
    constexpr const char* caller = "LocalAtomRetrievalLoss";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    payload.validate(caller);

    const int query_count = payload.localAtomQueryCount();
    Tensor loss = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),
        query_count > 0 && logits.requires_grad,
        stream,
        "local_atom_retrieval_loss");
    if (query_count == 0) {
        return loss;
    }

    requireBindings(payload, bindings, caller);
    logits.require(caller);
    if (!logits.shape.is_2d_layout()) {
        throw std::runtime_error(
            std::string(caller) + ": logits must be 2D");
    }
    const auto logits_shape = logits.shape.as_2d();
    const int candidate_class_count =
        expectedCandidateClassCount(payload, caller);
    if (logits_shape.rows != query_count ||
        logits_shape.cols != candidate_class_count) {
        throw std::runtime_error(
            std::string(caller) +
            ": logits must be [localAtomQueryCount, candidate_class_count]");
    }

    kernelLocalAtomRetrievalLossForward<<<
        blocksForQueries(query_count, caller),
        kQueryBlockSize,
        0,
        stream>>>(
        logits.data,
        bindings.d_local_atom_query_targets,
        bindings.d_local_atom_query_positions,
        bindings.d_local_atom_query_types,
        bindings.d_local_atom_row_type_candidate_offsets,
        bindings.d_local_atom_candidate_first_close_positions,
        loss.data,
        query_count,
        payload.localAtomCandidateCount(),
        Tokenizer::kAtomTypeCount,
        payload.max_seq_len,
        candidate_class_count);
    checkCuda(cudaGetLastError(), caller);

    if (logits.requires_grad) {
        loss.is_leaf = false;
        auto grad_fn = std::make_shared<LocalAtomRetrievalLossGradFn>();
        grad_fn->captureInput(logits, payload, candidate_class_count, stream);
        loss.grad_fn = std::move(grad_fn);
    }
    return loss;
}

} // namespace GRIM::LocalAtomRetrieval
