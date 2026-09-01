//======================================================//
//  LocalAtomRetrievalBackwards.cu
//  Explicit backward for scoring pre-encoded local atom banks.
//======================================================//

#include "LocalAtomRetrievalBackwards.hpp"

#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIM::LocalAtomRetrieval {

namespace {

constexpr int kBlockSize = 256;

int blocksFor(std::size_t count, const char* caller) {
    if (count == 0) {
        throw std::runtime_error(std::string(caller) + ": element count is zero");
    }
    const std::size_t blocks =
        1 + (count - 1) / static_cast<std::size_t>(kBlockSize);
    if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(caller) + ": CUDA grid overflows int");
    }
    return static_cast<int>(blocks);
}

void checkCuda(cudaError_t status, const char* caller) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + cudaGetErrorString(status));
    }
}

void requireBindings(
    const Batching::BatchDeviceBindings& bindings,
    const char* caller) {
    if (!bindings.d_local_atom_query_positions ||
        !bindings.d_local_atom_query_types ||
        !bindings.d_local_atom_row_type_candidate_offsets ||
        !bindings.d_local_atom_candidate_first_close_positions) {
        throw std::runtime_error(
            std::string(caller) +
            ": local atom BatchDeviceBindings are incomplete");
    }
}

__global__ void kernelQueryAndNoReferenceBackward(
    const float* __restrict__ grad_logits,
    const float* __restrict__ query_embeddings,
    const float* __restrict__ candidate_embeddings,
    const float* __restrict__ type_no_reference_key,
    const int* __restrict__ query_positions,
    const int* __restrict__ query_types,
    const int* __restrict__ row_type_candidate_offsets,
    const int* __restrict__ candidate_first_close_positions,
    float* __restrict__ grad_query_embeddings,
    float* __restrict__ grad_type_no_reference_key,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int retrieval_dim,
    int class_count,
    float score_scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * retrieval_dim;
    if (index >= count) {
        return;
    }

    const int query = static_cast<int>(index / retrieval_dim);
    const int feature = static_cast<int>(index % retrieval_dim);
    const int query_position = query_positions[query];
    const int type = query_types[query];
    if (query_position < 0 || type < 0 || type >= type_count) {
        return;
    }

    const int row = query_position / sequence_length;
    const int bank = row * type_count + type;
    const int candidate_begin = row_type_candidate_offsets[bank];
    const int candidate_end = row_type_candidate_offsets[bank + 1];
    const float* grad_row =
        grad_logits + static_cast<std::size_t>(query) * class_count;

    float gradient = grad_row[0] * type_no_reference_key[
        static_cast<std::size_t>(type) * retrieval_dim + feature];
    for (int candidate = candidate_begin;
         candidate < candidate_end && candidate < candidate_count;
         ++candidate) {
        if (candidate < 0 ||
            candidate_first_close_positions[candidate] >= query_position) {
            continue;
        }
        const int local_class = candidate - candidate_begin + 1;
        gradient += grad_row[local_class] * candidate_embeddings[
            static_cast<std::size_t>(candidate) * retrieval_dim + feature];
    }
    if (grad_query_embeddings) {
        atomicAdd(grad_query_embeddings + index, score_scale * gradient);
    }
    if (grad_type_no_reference_key) {
        atomicAdd(
            grad_type_no_reference_key +
                static_cast<std::size_t>(type) * retrieval_dim + feature,
            score_scale * grad_row[0] * query_embeddings[index]);
    }
}

__global__ void kernelCandidateBackward(
    const float* __restrict__ grad_logits,
    const float* __restrict__ query_embeddings,
    const int* __restrict__ query_positions,
    const int* __restrict__ query_types,
    const int* __restrict__ row_type_candidate_offsets,
    const int* __restrict__ candidate_first_close_positions,
    float* __restrict__ grad_candidate_embeddings,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int retrieval_dim,
    int class_count,
    float score_scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t row_width =
        static_cast<std::size_t>(class_count) * retrieval_dim;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * row_width;
    if (index >= count) {
        return;
    }

    const int query = static_cast<int>(index / row_width);
    const std::size_t within_query = index % row_width;
    const int local_class = static_cast<int>(within_query / retrieval_dim);
    const int feature = static_cast<int>(within_query % retrieval_dim);
    if (local_class == 0) {
        return;
    }

    const int query_position = query_positions[query];
    const int type = query_types[query];
    if (query_position < 0 || type < 0 || type >= type_count) {
        return;
    }
    const int row = query_position / sequence_length;
    const int bank = row * type_count + type;
    const int candidate_begin = row_type_candidate_offsets[bank];
    const int candidate_end = row_type_candidate_offsets[bank + 1];
    const int candidate = candidate_begin + local_class - 1;
    if (candidate < candidate_begin || candidate >= candidate_end ||
        candidate < 0 || candidate >= candidate_count ||
        candidate_first_close_positions[candidate] >= query_position) {
        return;
    }

    const float gradient = grad_logits[
        static_cast<std::size_t>(query) * class_count + local_class] *
        query_embeddings[
            static_cast<std::size_t>(query) * retrieval_dim + feature];
    atomicAdd(
        grad_candidate_embeddings +
            static_cast<std::size_t>(candidate) * retrieval_dim + feature,
        score_scale * gradient);
}

} // namespace

static void localAtomRetrievalBackwards(
    const Tensor& grad_logits,
    const Tensor& query_embeddings,
    const Tensor& candidate_embeddings,
    const Tensor& type_no_reference_key,
    Tensor* grad_query_embeddings,
    Tensor* grad_candidate_embeddings,
    Tensor* grad_type_no_reference_key,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int retrieval_dim,
    int class_count,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream) {
    constexpr const char* caller = "LocalAtomRetrievalBackwards";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    grad_logits.require(caller);
    query_embeddings.require(caller);
    candidate_embeddings.require(caller);
    type_no_reference_key.require(caller);
    if (!grad_logits.shape.is_2d_layout() ||
        !query_embeddings.shape.is_2d_layout() ||
        !candidate_embeddings.shape.is_2d_layout() ||
        !type_no_reference_key.shape.is_2d_layout()) {
        throw std::runtime_error(
            std::string(caller) + ": all retrieval tensors must be 2D");
    }
    const auto grad_shape = grad_logits.shape.as_2d();
    const auto query_shape = query_embeddings.shape.as_2d();
    const auto candidate_shape = candidate_embeddings.shape.as_2d();
    const auto no_reference_shape = type_no_reference_key.shape.as_2d();
    if (grad_shape.rows != query_count || grad_shape.cols != class_count ||
        query_shape.rows != query_count || query_shape.cols != retrieval_dim ||
        candidate_shape.rows != candidate_count ||
        candidate_shape.cols != retrieval_dim ||
        no_reference_shape.rows != type_count ||
        no_reference_shape.cols != retrieval_dim) {
        throw std::runtime_error(
            std::string(caller) + ": retrieval tensor shape mismatch");
    }
    if (query_count <= 0 || candidate_count <= 0 || type_count <= 0 ||
        sequence_length <= 0 || retrieval_dim <= 0 || class_count <= 1) {
        throw std::runtime_error(
            std::string(caller) + ": invalid saved retrieval geometry");
    }
    auto requireGradientShape = [&](const Tensor* gradient,
                                    int rows,
                                    int cols,
                                    const char* name) {
        if (!gradient) {
            return;
        }
        gradient->require(caller);
        if (!gradient->shape.is_2d_layout()) {
            throw std::runtime_error(
                std::string(caller) + ": " + name + " must be 2D");
        }
        const auto shape = gradient->shape.as_2d();
        if (shape.rows != rows || shape.cols != cols) {
            throw std::runtime_error(
                std::string(caller) + ": " + name + " shape mismatch");
        }
    };
    requireGradientShape(
        grad_query_embeddings, query_count, retrieval_dim, "grad_query_embeddings");
    requireGradientShape(
        grad_candidate_embeddings,
        candidate_count,
        retrieval_dim,
        "grad_candidate_embeddings");
    requireGradientShape(
        grad_type_no_reference_key,
        type_count,
        retrieval_dim,
        "grad_type_no_reference_key");
    requireBindings(bindings, caller);

    const float score_scale =
        1.0f / std::sqrt(static_cast<float>(retrieval_dim));
    if (grad_query_embeddings || grad_type_no_reference_key) {
        const std::size_t query_element_count =
            static_cast<std::size_t>(query_count) * retrieval_dim;
        kernelQueryAndNoReferenceBackward<<<
            blocksFor(query_element_count, caller),
            kBlockSize,
            0,
            stream>>>(
            grad_logits.data,
            query_embeddings.data,
            candidate_embeddings.data,
            type_no_reference_key.data,
            bindings.d_local_atom_query_positions,
            bindings.d_local_atom_query_types,
            bindings.d_local_atom_row_type_candidate_offsets,
            bindings.d_local_atom_candidate_first_close_positions,
            grad_query_embeddings ? grad_query_embeddings->data : nullptr,
            grad_type_no_reference_key
                ? grad_type_no_reference_key->data
                : nullptr,
            query_count,
            candidate_count,
            type_count,
            sequence_length,
            retrieval_dim,
            class_count,
            score_scale);
        checkCuda(cudaGetLastError(), "LocalAtomRetrievalBackwards query");
    }

    if (grad_candidate_embeddings) {
        const std::size_t candidate_work_count =
            static_cast<std::size_t>(query_count) *
            class_count * retrieval_dim;
        kernelCandidateBackward<<<
            blocksFor(candidate_work_count, caller),
            kBlockSize,
            0,
            stream>>>(
            grad_logits.data,
            query_embeddings.data,
            bindings.d_local_atom_query_positions,
            bindings.d_local_atom_query_types,
            bindings.d_local_atom_row_type_candidate_offsets,
            bindings.d_local_atom_candidate_first_close_positions,
            grad_candidate_embeddings->data,
            query_count,
            candidate_count,
            type_count,
            sequence_length,
            retrieval_dim,
            class_count,
            score_scale);
        checkCuda(cudaGetLastError(), "LocalAtomRetrievalBackwards candidates");
    }
}

LocalAtomRetrievalGradFn::LocalAtomRetrievalGradFn() {
    op_name = "LocalAtomRetrievalForward";
}

void LocalAtomRetrievalGradFn::captureInputs(
    Tensor& query_embeddings,
    Tensor& candidate_embeddings,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int retrieval_dim,
    int class_count,
    cudaStream_t stream) {
    query_embeddings_ = &query_embeddings;
    candidate_embeddings_ = &candidate_embeddings;
    parameter_registry_ = &parameter_registry;
    query_count_ = query_count;
    candidate_count_ = candidate_count;
    type_count_ = type_count;
    sequence_length_ = sequence_length;
    retrieval_dim_ = retrieval_dim;
    class_count_ = class_count;

    auto& type_no_reference_key =
        parameter_registry.requireLocalAtomRetrievalParameters(
            "LocalAtomRetrievalGradFn::captureInputs")
            .type_no_reference_key;

    if (query_embeddings.requires_grad) {
        query_embeddings_gradient_ = capture_input_gradient(
            query_embeddings,
            stream,
            "LocalAtomRetrievalGradFn::captureInputs queries");
    }
    if (candidate_embeddings.requires_grad) {
        candidate_embeddings_gradient_ = capture_input_gradient(
            candidate_embeddings,
            stream,
            "LocalAtomRetrievalGradFn::captureInputs candidates");
    }
    if (type_no_reference_key.requires_grad) {
        type_no_reference_key_gradient_ = capture_input_gradient(
            type_no_reference_key,
            stream,
            "LocalAtomRetrievalGradFn::captureInputs no_reference_key");
    }
}

void LocalAtomRetrievalGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings) {
    setCurrentGradFnOp("LocalAtomRetrievalBackwards", this);
    if (applied) {
        return;
    }
    applied = true;
    if (!backward_payload) {
        throw std::runtime_error(
            "LocalAtomRetrievalGradFn::apply: backward_payload is NULL");
    }
    if (!backward_bindings) {
        throw std::runtime_error(
            "LocalAtomRetrievalGradFn::apply: backward_bindings is NULL");
    }
    if (!query_embeddings_ || !candidate_embeddings_ || !parameter_registry_) {
        throw std::runtime_error(
            "LocalAtomRetrievalGradFn::apply: borrowed Tensor/registry handle is NULL");
    }
    backward_payload->validate("LocalAtomRetrievalGradFn::apply");
    if (backward_payload->localAtomQueryCount() != query_count_ ||
        backward_payload->localAtomCandidateCount() != candidate_count_ ||
        backward_payload->max_seq_len != sequence_length_ ||
        backward_bindings->local_atom_query_count != query_count_ ||
        backward_bindings->local_atom_candidate_count != candidate_count_) {
        throw std::runtime_error(
            "LocalAtomRetrievalGradFn::apply: backward batch geometry differs from forward");
    }

    auto& type_no_reference_key =
        parameter_registry_->requireLocalAtomRetrievalParameters(
            "LocalAtomRetrievalGradFn::apply")
            .type_no_reference_key;
    if (type_no_reference_key_gradient_ &&
        type_no_reference_key.grad() != type_no_reference_key_gradient_.get()) {
        throw std::runtime_error(
            "LocalAtomRetrievalGradFn::apply: registry parameter gradient owner changed since forward");
    }
    localAtomRetrievalBackwards(
        grad_output,
        *query_embeddings_,
        *candidate_embeddings_,
        type_no_reference_key,
        query_embeddings_gradient_.get(),
        candidate_embeddings_gradient_.get(),
        type_no_reference_key_gradient_.get(),
        query_count_,
        candidate_count_,
        type_count_,
        sequence_length_,
        retrieval_dim_,
        class_count_,
        *backward_bindings,
        stream);

    if (query_embeddings_gradient_) {
        propagate_input_gradient(
            query_embeddings_gradient_,
            stream,
            backward_payload,
            backward_bindings,
            "LocalAtomRetrievalGradFn::apply queries");
    }
    if (candidate_embeddings_gradient_) {
        propagate_input_gradient(
            candidate_embeddings_gradient_,
            stream,
            backward_payload,
            backward_bindings,
            "LocalAtomRetrievalGradFn::apply candidates");
    }
    if (type_no_reference_key_gradient_) {
        propagate_input_gradient(
            type_no_reference_key_gradient_,
            stream,
            backward_payload,
            backward_bindings,
            "LocalAtomRetrievalGradFn::apply no_reference_key");
    }
}

void LocalAtomRetrievalGradFn::release_saved() {
    GradFn::release_saved();
    query_embeddings_gradient_.reset();
    candidate_embeddings_gradient_.reset();
    type_no_reference_key_gradient_.reset();
    query_embeddings_ = nullptr;
    candidate_embeddings_ = nullptr;
    parameter_registry_ = nullptr;
}

} // namespace GRIM::LocalAtomRetrieval
