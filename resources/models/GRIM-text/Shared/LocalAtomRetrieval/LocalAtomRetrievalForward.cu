//======================================================//
//  LocalAtomRetrievalForward.cu
//  Encoder-derived sequence-local retrieval signals.
//======================================================//

#include "LocalAtomRetrievalForward.hpp"
#include "LocalAtomRetrievalBackwards.hpp"

#include "../UnigramByte/TokenLayout.hpp"
#include "../InferenceState/LocalAtomRetrievalInferenceState.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

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

__device__ float scaledDotProduct(
    const float* left,
    const float* right,
    int retrieval_dim,
    float score_scale) {
    float dot = 0.0f;
    for (int feature = 0; feature < retrieval_dim; ++feature) {
        dot += left[feature] * right[feature];
    }
    return dot * score_scale;
}

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

int candidateSlotCount(
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
    if (maximum_bank_width <= 0 ||
        maximum_bank_width == std::numeric_limits<int>::max()) {
        throw std::runtime_error(
            std::string(caller) + ": invalid maximum candidate-bank width");
    }
    return maximum_bank_width + 1; // Slot zero is NO_REFERENCE.
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
        !bindings.d_local_atom_candidate_first_close_positions ||
        !bindings.d_local_atom_candidate_content_offsets ||
        !bindings.d_local_atom_candidate_content_positions) {
        throw std::runtime_error(
            std::string(caller) +
            ": local atom BatchDeviceBindings are incomplete");
    }
}

__global__ void kernelGatherQueriesForward(
    const float* __restrict__ encoder_output,
    const int* __restrict__ query_positions,
    float* __restrict__ query_embeddings,
    int query_count,
    int retrieval_dim) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * retrieval_dim;
    if (index >= count) {
        return;
    }
    const int query = static_cast<int>(index / retrieval_dim);
    const int feature = static_cast<int>(index % retrieval_dim);
    query_embeddings[index] = encoder_output[
        static_cast<std::size_t>(query_positions[query]) * retrieval_dim +
        feature];
}

__global__ void kernelGatherQueriesBackward(
    const float* __restrict__ grad_queries,
    const int* __restrict__ query_positions,
    float* __restrict__ grad_encoder,
    int query_count,
    int retrieval_dim) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * retrieval_dim;
    if (index >= count) {
        return;
    }
    const int query = static_cast<int>(index / retrieval_dim);
    const int feature = static_cast<int>(index % retrieval_dim);
    atomicAdd(
        grad_encoder +
            static_cast<std::size_t>(query_positions[query]) * retrieval_dim +
            feature,
        grad_queries[index]);
}

__global__ void kernelMeanPoolCandidatesForward(
    const float* __restrict__ encoder_output,
    const int* __restrict__ content_offsets,
    const int* __restrict__ content_positions,
    float* __restrict__ candidate_embeddings,
    int candidate_count,
    int retrieval_dim) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(candidate_count) * retrieval_dim;
    if (index >= count) {
        return;
    }
    const int candidate = static_cast<int>(index / retrieval_dim);
    const int feature = static_cast<int>(index % retrieval_dim);
    const int begin = content_offsets[candidate];
    const int end = content_offsets[candidate + 1];
    float sum = 0.0f;
    for (int content = begin; content < end; ++content) {
        sum += encoder_output[
            static_cast<std::size_t>(content_positions[content]) *
                retrieval_dim +
            feature];
    }
    candidate_embeddings[index] = sum / static_cast<float>(end - begin);
}

__global__ void kernelMeanPoolCandidatesBackward(
    const float* __restrict__ grad_candidates,
    const int* __restrict__ content_offsets,
    const int* __restrict__ content_positions,
    float* __restrict__ grad_encoder,
    int candidate_count,
    int retrieval_dim) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(candidate_count) * retrieval_dim;
    if (index >= count) {
        return;
    }
    const int candidate = static_cast<int>(index / retrieval_dim);
    const int feature = static_cast<int>(index % retrieval_dim);
    const int begin = content_offsets[candidate];
    const int end = content_offsets[candidate + 1];
    const float gradient =
        grad_candidates[index] / static_cast<float>(end - begin);
    for (int content = begin; content < end; ++content) {
        atomicAdd(
            grad_encoder +
                static_cast<std::size_t>(content_positions[content]) *
                    retrieval_dim +
                feature,
            gradient);
    }
}

class QueryGatherGradFn final : public GradFn {
public:
    QueryGatherGradFn(
        Tensor& encoder_output,
        int query_count,
        int total_tokens,
        int retrieval_dim,
        cudaStream_t stream)
        : query_count_(query_count),
          total_tokens_(total_tokens),
          retrieval_dim_(retrieval_dim) {
        op_name = "LocalAtomQueryGather";
        encoder_gradient_ = capture_input_gradient(
            encoder_output, stream, "LocalAtomQueryGather capture");
    }

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override {
        setCurrentGradFnOp("LocalAtomQueryGatherBackward", this);
        if (applied) {
            return;
        }
        applied = true;
        if (!backward_payload || !backward_bindings ||
            !backward_bindings->d_local_atom_query_positions) {
            throw std::runtime_error(
                "LocalAtomQueryGatherBackward: backward batch data is incomplete");
        }
        backward_payload->validate("LocalAtomQueryGatherBackward");
        if (backward_payload->localAtomQueryCount() != query_count_ ||
            backward_payload->total_tokens != total_tokens_ ||
            backward_bindings->local_atom_query_count != query_count_) {
            throw std::runtime_error(
                "LocalAtomQueryGatherBackward: batch geometry differs from forward");
        }
        const auto shape = requireMatrix(
            grad_output, "grad_output", "LocalAtomQueryGatherBackward");
        if (shape.rows != query_count_ || shape.cols != retrieval_dim_) {
            throw std::runtime_error(
                "LocalAtomQueryGatherBackward: gradient shape mismatch");
        }
        const std::size_t count =
            static_cast<std::size_t>(query_count_) * retrieval_dim_;
        kernelGatherQueriesBackward<<<
            blocksFor(count, "LocalAtomQueryGatherBackward"),
            kBlockSize,
            0,
            stream>>>(
            grad_output.data,
            backward_bindings->d_local_atom_query_positions,
            encoder_gradient_->data,
            query_count_,
            retrieval_dim_);
        checkCuda(cudaGetLastError(), "LocalAtomQueryGatherBackward");
        propagate_input_gradient(
            encoder_gradient_, stream, backward_payload, backward_bindings,
            "LocalAtomQueryGatherBackward");
    }

    void release_saved() override {
        GradFn::release_saved();
        encoder_gradient_.reset();
    }

private:
    std::shared_ptr<Tensor> encoder_gradient_;
    int query_count_ = 0;
    int total_tokens_ = 0;
    int retrieval_dim_ = 0;
};

class CandidateMeanPoolGradFn final : public GradFn {
public:
    CandidateMeanPoolGradFn(
        Tensor& encoder_output,
        int candidate_count,
        int content_position_count,
        int total_tokens,
        int retrieval_dim,
        cudaStream_t stream)
        : candidate_count_(candidate_count),
          content_position_count_(content_position_count),
          total_tokens_(total_tokens),
          retrieval_dim_(retrieval_dim) {
        op_name = "LocalAtomCandidateMeanPool";
        encoder_gradient_ = capture_input_gradient(
            encoder_output, stream, "LocalAtomCandidateMeanPool capture");
    }

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override {
        setCurrentGradFnOp("LocalAtomCandidateMeanPoolBackward", this);
        if (applied) {
            return;
        }
        applied = true;
        if (!backward_payload || !backward_bindings ||
            !backward_bindings->d_local_atom_candidate_content_offsets ||
            !backward_bindings->d_local_atom_candidate_content_positions) {
            throw std::runtime_error(
                "LocalAtomCandidateMeanPoolBackward: backward batch data is incomplete");
        }
        backward_payload->validate("LocalAtomCandidateMeanPoolBackward");
        if (backward_payload->localAtomCandidateCount() != candidate_count_ ||
            backward_payload->localAtomContentPositionCount() !=
                content_position_count_ ||
            backward_payload->total_tokens != total_tokens_ ||
            backward_bindings->local_atom_candidate_count != candidate_count_ ||
            backward_bindings->local_atom_content_position_count !=
                content_position_count_) {
            throw std::runtime_error(
                "LocalAtomCandidateMeanPoolBackward: batch geometry differs from forward");
        }
        const auto shape = requireMatrix(
            grad_output, "grad_output", "LocalAtomCandidateMeanPoolBackward");
        if (shape.rows != candidate_count_ || shape.cols != retrieval_dim_) {
            throw std::runtime_error(
                "LocalAtomCandidateMeanPoolBackward: gradient shape mismatch");
        }
        const std::size_t count =
            static_cast<std::size_t>(candidate_count_) * retrieval_dim_;
        kernelMeanPoolCandidatesBackward<<<
            blocksFor(count, "LocalAtomCandidateMeanPoolBackward"),
            kBlockSize,
            0,
            stream>>>(
            grad_output.data,
            backward_bindings->d_local_atom_candidate_content_offsets,
            backward_bindings->d_local_atom_candidate_content_positions,
            encoder_gradient_->data,
            candidate_count_,
            retrieval_dim_);
        checkCuda(cudaGetLastError(), "LocalAtomCandidateMeanPoolBackward");
        propagate_input_gradient(
            encoder_gradient_, stream, backward_payload, backward_bindings,
            "LocalAtomCandidateMeanPoolBackward");
    }

    void release_saved() override {
        GradFn::release_saved();
        encoder_gradient_.reset();
    }

private:
    std::shared_ptr<Tensor> encoder_gradient_;
    int candidate_count_ = 0;
    int content_position_count_ = 0;
    int total_tokens_ = 0;
    int retrieval_dim_ = 0;
};

Tensor gatherQueries(
    Tensor& encoder_output,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool connect_parameter_graph,
    cudaStream_t stream) {
    const auto encoder_shape = encoder_output.shape.as_2d();
    const int query_count = payload.localAtomQueryCount();
    const int retrieval_dim = encoder_shape.cols;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * retrieval_dim;
    Tensor result = Tensor::empty(
        TensorContract::TensorShape::make_BSM(query_count, retrieval_dim),
        connect_parameter_graph,
        stream,
        "local_atom_query_embeddings");
    kernelGatherQueriesForward<<<
        blocksFor(count, "LocalAtomQueryGather"), kBlockSize, 0, stream>>>(
        encoder_output.data,
        bindings.d_local_atom_query_positions,
        result.data,
        query_count,
        retrieval_dim);
    checkCuda(cudaGetLastError(), "LocalAtomQueryGather");
    if (connect_parameter_graph) {
        result.is_leaf = false;
        result.grad_fn = std::make_shared<QueryGatherGradFn>(
            encoder_output,
            query_count,
            payload.total_tokens,
            retrieval_dim,
            stream);
    }
    return result;
}

Tensor meanPoolCandidates(
    Tensor& encoder_output,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool connect_parameter_graph,
    cudaStream_t stream) {
    const auto encoder_shape = encoder_output.shape.as_2d();
    const int candidate_count = payload.localAtomCandidateCount();
    const int retrieval_dim = encoder_shape.cols;
    const std::size_t count =
        static_cast<std::size_t>(candidate_count) * retrieval_dim;
    Tensor result = Tensor::empty(
        TensorContract::TensorShape::make_BSM(candidate_count, retrieval_dim),
        connect_parameter_graph,
        stream,
        "local_atom_candidate_embeddings");
    kernelMeanPoolCandidatesForward<<<
        blocksFor(count, "LocalAtomCandidateMeanPool"),
        kBlockSize,
        0,
        stream>>>(
        encoder_output.data,
        bindings.d_local_atom_candidate_content_offsets,
        bindings.d_local_atom_candidate_content_positions,
        result.data,
        candidate_count,
        retrieval_dim);
    checkCuda(cudaGetLastError(), "LocalAtomCandidateMeanPool");
    if (connect_parameter_graph) {
        result.is_leaf = false;
        result.grad_fn = std::make_shared<CandidateMeanPoolGradFn>(
            encoder_output,
            candidate_count,
            payload.localAtomContentPositionCount(),
            payload.total_tokens,
            retrieval_dim,
            stream);
    }
    return result;
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
    int candidate_slot_count,
    float score_scale) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count =
        static_cast<std::size_t>(query_count) * candidate_slot_count;
    if (index >= count) {
        return;
    }

    const int query = static_cast<int>(index / candidate_slot_count);
    const int candidate_slot =
        static_cast<int>(index % candidate_slot_count);
    const int query_position = query_positions[query];
    const int type = query_types[query];
    if (query_position < 0 || type < 0 || type >= type_count) {
        logits[index] = kMaskedLogit;
        return;
    }

    const float* query_vector =
        query_embeddings + static_cast<std::size_t>(query) * retrieval_dim;
    const float* candidate_vector = nullptr;
    if (candidate_slot == 0) {
        candidate_vector = type_no_reference_key +
            static_cast<std::size_t>(type) * retrieval_dim;
    } else {
        const int row = query_position / sequence_length;
        const int bank = row * type_count + type;
        const int candidate_begin = row_type_candidate_offsets[bank];
        const int candidate_end = row_type_candidate_offsets[bank + 1];
        const int candidate = candidate_begin + candidate_slot - 1;
        if (candidate < candidate_begin || candidate >= candidate_end ||
            candidate < 0 || candidate >= candidate_count ||
            candidate_first_close_positions[candidate] >= query_position) {
            logits[index] = kMaskedLogit;
            return;
        }
        candidate_vector = candidate_embeddings +
            static_cast<std::size_t>(candidate) * retrieval_dim;
    }

    logits[index] = scaledDotProduct(
        query_vector, candidate_vector, retrieval_dim, score_scale);
}

__global__ void kernelLocalAtomRetrievalDecodeForward(
    const float* __restrict__ query_embedding,
    const float* __restrict__ typed_candidate_embeddings,
    const float* __restrict__ type_no_reference_key,
    float* __restrict__ logits,
    int query_type,
    int candidate_count,
    int retrieval_dim,
    float score_scale) {
    const int candidate_slot = blockIdx.x * blockDim.x + threadIdx.x;
    const int candidate_slot_count = candidate_count + 1;
    if (candidate_slot >= candidate_slot_count) {
        return;
    }
    const float* key = candidate_slot == 0
        ? type_no_reference_key +
            static_cast<std::size_t>(query_type) * retrieval_dim
        : typed_candidate_embeddings +
            static_cast<std::size_t>(candidate_slot - 1) * retrieval_dim;
    logits[candidate_slot] = scaledDotProduct(
        query_embedding, key, retrieval_dim, score_scale);
}

} // namespace

void LocalAtomRetrievalForward(
    Tensor& encoder_output,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool connect_parameter_graph,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs) {
    constexpr const char* caller = "LocalAtomRetrievalForward";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    if (forward_outputs.local_atom_query_embeddings.data ||
        forward_outputs.local_atom_candidate_embeddings.data ||
        forward_outputs.local_atom_retrieval_logits.data) {
        throw std::runtime_error(
            std::string(caller) +
            ": ModelForwardOutputs already contains retrieval tensors");
    }
    payload.validate(caller);
    const int query_count = payload.localAtomQueryCount();
    if (query_count == 0) {
        return;
    }
    requireBindings(payload, bindings, caller);
    const int candidate_count = payload.localAtomCandidateCount();
    if (candidate_count <= 0 || payload.max_seq_len <= 0) {
        throw std::runtime_error(
            std::string(caller) + ": invalid compact retrieval geometry");
    }
    const auto encoder_shape =
        requireMatrix(encoder_output, "encoder_output", caller);
    if (encoder_shape.rows != payload.total_tokens ||
        encoder_shape.cols <= 0) {
        throw std::runtime_error(
            std::string(caller) +
            ": encoder output must be [payload.total_tokens, d_model]");
    }
    if (connect_parameter_graph && !encoder_output.requires_grad) {
        throw std::runtime_error(
            std::string(caller) +
            ": connected retrieval requires a graph-connected encoder output");
    }

    auto& parameters =
        parameter_registry.requireLocalAtomRetrievalParameters(caller);
    const auto no_reference_shape = requireMatrix(
        parameters.type_no_reference_key,
        "type_no_reference_key",
        caller);
    const int retrieval_dim = encoder_shape.cols;
    if (no_reference_shape.rows != Tokenizer::kAtomTypeCount ||
        no_reference_shape.cols != retrieval_dim) {
        throw std::runtime_error(
            std::string(caller) +
            ": no-reference keys must be [kAtomTypeCount, d_model]");
    }

    forward_outputs.local_atom_query_embeddings = gatherQueries(
        encoder_output,
        payload,
        bindings,
        connect_parameter_graph,
        stream);
    forward_outputs.local_atom_candidate_embeddings = meanPoolCandidates(
        encoder_output,
        payload,
        bindings,
        connect_parameter_graph,
        stream);

    const int candidate_slot_count = candidateSlotCount(payload, caller);
    const bool requires_grad = connect_parameter_graph &&
        (forward_outputs.local_atom_query_embeddings.requires_grad ||
         forward_outputs.local_atom_candidate_embeddings.requires_grad ||
         parameters.type_no_reference_key.requires_grad);
    Tensor logits = Tensor::empty(
        TensorContract::TensorShape::make_LOGITS(
            query_count, candidate_slot_count),
        requires_grad,
        stream,
        "local_atom_retrieval_logits");
    const std::size_t logit_count =
        static_cast<std::size_t>(query_count) * candidate_slot_count;
    kernelLocalAtomRetrievalForward<<<
        blocksFor(logit_count, caller), kBlockSize, 0, stream>>>(
        forward_outputs.local_atom_query_embeddings.data,
        forward_outputs.local_atom_candidate_embeddings.data,
        parameters.type_no_reference_key.data,
        bindings.d_local_atom_query_positions,
        bindings.d_local_atom_query_types,
        bindings.d_local_atom_row_type_candidate_offsets,
        bindings.d_local_atom_candidate_first_close_positions,
        logits.data,
        query_count,
        Tokenizer::kAtomTypeCount,
        payload.max_seq_len,
        candidate_count,
        retrieval_dim,
        candidate_slot_count,
        1.0f / std::sqrt(static_cast<float>(retrieval_dim)));
    checkCuda(cudaGetLastError(), caller);

    if (requires_grad) {
        logits.is_leaf = false;
        logits.grad_fn = makeLocalAtomRetrievalGradFn(
            forward_outputs.local_atom_query_embeddings,
            forward_outputs.local_atom_candidate_embeddings,
            parameter_registry,
            query_count,
            candidate_count,
            Tokenizer::kAtomTypeCount,
            payload.max_seq_len,
            retrieval_dim,
            candidate_slot_count,
            stream);
    }
    forward_outputs.local_atom_retrieval_logits = std::move(logits);
}

void LocalAtomRetrievalDecodeForward(
    Tensor& current_encoder_output,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const LocalAtomRetrievalInferenceState& inference_state,
    Tokenizer::AtomType query_type,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs) {
    constexpr const char* caller = "LocalAtomRetrievalDecodeForward";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    if (forward_outputs.local_atom_query_embeddings.data ||
        forward_outputs.local_atom_candidate_embeddings.data ||
        forward_outputs.local_atom_retrieval_logits.data) {
        throw std::runtime_error(
            std::string(caller) +
            ": ModelForwardOutputs already contains retrieval tensors");
    }
    const int query_type_index =
        Tokenizer::atomTypeIndexOrThrow(query_type, caller);
    const auto encoder_shape = requireMatrix(
        current_encoder_output, "current_encoder_output", caller);
    if (encoder_shape.rows != 1 || encoder_shape.cols <= 0 ||
        current_encoder_output.requires_grad) {
        throw std::runtime_error(
            std::string(caller) +
            ": decode encoder output must be detached [1, d_model]");
    }
    if (!inference_state.allocated ||
        inference_state.d_model != encoder_shape.cols) {
        throw std::runtime_error(
            std::string(caller) + ": inference retrieval state geometry mismatch");
    }

    auto& parameters =
        parameter_registry.requireLocalAtomRetrievalParameters(caller);
    const auto no_reference_shape = requireMatrix(
        parameters.type_no_reference_key,
        "type_no_reference_key",
        caller);
    const int retrieval_dim = encoder_shape.cols;
    if (no_reference_shape.rows != Tokenizer::kAtomTypeCount ||
        no_reference_shape.cols != retrieval_dim) {
        throw std::runtime_error(
            std::string(caller) + ": no-reference key shape mismatch");
    }

    const int candidate_count = inference_state.candidateCount(query_type);
    const Tensor& candidate_storage =
        inference_state.candidateEmbeddingStorage(query_type);
    const auto candidate_shape = requireMatrix(
        candidate_storage, "typed_candidate_embeddings", caller);
    if (candidate_shape.rows < candidate_count ||
        candidate_shape.cols != retrieval_dim) {
        throw std::runtime_error(
            std::string(caller) + ": typed candidate storage shape mismatch");
    }

    forward_outputs.local_atom_query_embeddings = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, retrieval_dim),
        false,
        stream,
        "local_atom_decode_query_embedding");
    checkCuda(
        cudaMemcpyAsync(
            forward_outputs.local_atom_query_embeddings.data,
            current_encoder_output.data,
            static_cast<std::size_t>(retrieval_dim) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream),
        caller);

    const int candidate_slot_count = candidate_count + 1;
    forward_outputs.local_atom_retrieval_logits = Tensor::empty(
        TensorContract::TensorShape::make_LOGITS(1, candidate_slot_count),
        false,
        stream,
        "local_atom_decode_logits");
    kernelLocalAtomRetrievalDecodeForward<<<
        blocksFor(static_cast<std::size_t>(candidate_slot_count), caller),
        kBlockSize,
        0,
        stream>>>(
        forward_outputs.local_atom_query_embeddings.data,
        candidate_storage.data,
        parameters.type_no_reference_key.data,
        forward_outputs.local_atom_retrieval_logits.data,
        query_type_index,
        candidate_count,
        retrieval_dim,
        1.0f / std::sqrt(static_cast<float>(retrieval_dim)));
    checkCuda(cudaGetLastError(), caller);
}

} // namespace GRIM::LocalAtomRetrieval
