//======================================================//
//  LocalAtomRetrievalForward.cu
//  Scoring over pre-encoded sequence-local atom banks.
//======================================================//

#include "LocalAtomRetrievalForward.hpp"
#include "LocalAtomRetrievalBackwards.hpp"

#include "../UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIM::LocalAtomRetrieval {

namespace {

constexpr int kBlockSize = 256;
constexpr float kMaskedLogit = -1.0e9f;

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

TensorContract::Shape2D requireMatrix(
    const Tensor& tensor,
    const char* name,
    const char* caller) {
    tensor.require(caller);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(
            std::string(caller) + ": " + name + " must be 2D");
    }
    return tensor.shape.as_2d();
}

int maximumBankWidth(
    const Batching::BatchPayload& payload,
    const char* caller) {
    const auto& offsets = payload.local_atom_row_type_candidate_offsets;
    const std::size_t expected =
        static_cast<std::size_t>(payload.batch_size) *
            Tokenizer::kAtomTypeCount +
        1;
    if (offsets.size() != expected || offsets.empty()) {
        throw std::runtime_error(
            std::string(caller) +
            ": payload row/type candidate offsets have invalid geometry");
    }

    int maximum = 0;
    for (std::size_t bank = 0; bank + 1 < offsets.size(); ++bank) {
        const int width = offsets[bank + 1] - offsets[bank];
        if (width < 0) {
            throw std::runtime_error(
                std::string(caller) +
                ": payload row/type candidate offsets are not monotonic");
        }
        maximum = std::max(maximum, width);
    }
    return maximum;
}

void requireBindings(
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const char* caller) {
    if (bindings.local_atom_query_count != payload.localAtomQueryCount() ||
        bindings.local_atom_candidate_count !=
            payload.localAtomCandidateCount() ||
        bindings.local_atom_content_position_count !=
            payload.localAtomContentPositionCount()) {
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

__global__ void kernelLocalAtomRetrievalForward(
    const float* __restrict__ query_embeddings,
    const float* __restrict__ candidate_embeddings,
    const float* __restrict__ type_no_reference_key,
    const int* __restrict__ query_positions,
    const int* __restrict__ query_types,
    const int* __restrict__ row_type_candidate_offsets,
    const int* __restrict__ candidate_first_close_positions,
    float* __restrict__ logits,
    int query_count,
    int type_count,
    int sequence_length,
    int candidate_count,
    int retrieval_dim,
    int class_count,
    float score_scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * class_count;
    if (index >= count) {
        return;
    }

    const int query = static_cast<int>(index / class_count);
    const int local_class = static_cast<int>(index % class_count);
    const int query_position = query_positions[query];
    const int type = query_types[query];
    if (query_position < 0 || type < 0 || type >= type_count) {
        logits[index] = kMaskedLogit;
        return;
    }

    const float* query_vector =
        query_embeddings + static_cast<std::size_t>(query) * retrieval_dim;
    const float* candidate_vector = nullptr;
    if (local_class == 0) {
        candidate_vector = type_no_reference_key +
            static_cast<std::size_t>(type) * retrieval_dim;
    } else {
        const int row = query_position / sequence_length;
        const int bank = row * type_count + type;
        const int candidate_begin = row_type_candidate_offsets[bank];
        const int candidate_end = row_type_candidate_offsets[bank + 1];
        const int candidate = candidate_begin + local_class - 1;
        if (candidate < candidate_begin || candidate >= candidate_end ||
            candidate < 0 || candidate >= candidate_count ||
            candidate_first_close_positions[candidate] >= query_position) {
            logits[index] = kMaskedLogit;
            return;
        }
        candidate_vector = candidate_embeddings +
            static_cast<std::size_t>(candidate) * retrieval_dim;
    }

    float dot = 0.0f;
    for (int feature = 0; feature < retrieval_dim; ++feature) {
        dot += query_vector[feature] * candidate_vector[feature];
    }
    logits[index] = dot * score_scale;
}

} // namespace

LocalAtomRetrievalForwardOutputs LocalAtomRetrievalForward(
    Tensor& query_embeddings,
    Tensor& candidate_embeddings,
    const LocalAtomRetrievalParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream) {
    constexpr const char* caller = "LocalAtomRetrievalForward";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    payload.validate(caller);
    requireBindings(payload, bindings, caller);

    LocalAtomRetrievalForwardOutputs outputs;
    outputs.query_count = payload.localAtomQueryCount();
    outputs.candidate_count = payload.localAtomCandidateCount();
    if (outputs.query_count == 0) {
        return outputs;
    }
    if (outputs.candidate_count <= 0 || payload.max_seq_len <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": invalid compact retrieval geometry");
    }

    const auto query_shape =
        requireMatrix(query_embeddings, "query_embeddings", caller);
    const auto candidate_shape =
        requireMatrix(candidate_embeddings, "candidate_embeddings", caller);
    const auto no_reference_shape = requireMatrix(
        parameters.type_no_reference_key,
        "type_no_reference_key",
        caller);

    const int retrieval_dim = query_shape.cols;
    if (retrieval_dim <= 0 || query_shape.rows != outputs.query_count) {
        throw std::runtime_error(
            std::string(caller) + ": query embedding shape mismatch");
    }
    if (candidate_shape.rows != outputs.candidate_count ||
        candidate_shape.cols != retrieval_dim) {
        throw std::runtime_error(
            std::string(caller) +
            ": candidate embeddings must be [candidate_count, retrieval_dim]");
    }
    if (no_reference_shape.rows != Tokenizer::kAtomTypeCount ||
        no_reference_shape.cols != retrieval_dim) {
        throw std::runtime_error(
            std::string(caller) +
            ": no-reference keys must be [kAtomTypeCount, retrieval_dim]");
    }

    const int maximum_bank_width = maximumBankWidth(payload, caller);
    if (maximum_bank_width <= 0 ||
        maximum_bank_width == std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            std::string(caller) + ": invalid maximum candidate-bank width");
    }
    const int class_count = maximum_bank_width + 1;
    const bool requires_grad =
        query_embeddings.requires_grad ||
        candidate_embeddings.requires_grad ||
        parameters.type_no_reference_key.requires_grad;

    outputs.logits = Tensor::empty(
        TensorContract::TensorShape::make_LOGITS(
            outputs.query_count, class_count),
        requires_grad,
        stream,
        "local_atom_retrieval_logits");
    outputs.class_count = class_count;
    outputs.retrieval_dim = retrieval_dim;

    const std::size_t logit_count =
        static_cast<std::size_t>(outputs.query_count) * class_count;
    kernelLocalAtomRetrievalForward<<<
        blocksFor(logit_count, caller), kBlockSize, 0, stream>>>(
        query_embeddings.data,
        candidate_embeddings.data,
        parameters.type_no_reference_key.data,
        bindings.d_local_atom_query_positions,
        bindings.d_local_atom_query_types,
        bindings.d_local_atom_row_type_candidate_offsets,
        bindings.d_local_atom_candidate_first_close_positions,
        outputs.logits.data,
        outputs.query_count,
        Tokenizer::kAtomTypeCount,
        payload.max_seq_len,
        outputs.candidate_count,
        retrieval_dim,
        class_count,
        1.0f / std::sqrt(static_cast<float>(retrieval_dim)));
    checkCuda(cudaGetLastError(), caller);

    if (requires_grad) {
        outputs.logits.is_leaf = false;
        auto grad_fn = std::make_shared<LocalAtomRetrievalGradFn>();
        grad_fn->query_count = outputs.query_count;
        grad_fn->candidate_count = outputs.candidate_count;
        grad_fn->type_count = Tokenizer::kAtomTypeCount;
        grad_fn->sequence_length = payload.max_seq_len;
        grad_fn->retrieval_dim = retrieval_dim;
        grad_fn->class_count = class_count;
        grad_fn->captureInputs(
            query_embeddings,
            candidate_embeddings,
            parameters.type_no_reference_key,
            stream);
        outputs.logits.grad_fn = std::move(grad_fn);
    }

    return outputs;
}

} // namespace GRIM::LocalAtomRetrieval
