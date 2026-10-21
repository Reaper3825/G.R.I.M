#include "CachedInferenceForward.hpp"

#include "../Batching/BatchDeviceUpload.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../InferenceState/LocalAtomRetrievalInferenceState.hpp"
#include "../LocalAtomRetrieval/LocalAtomRetrievalForward.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIM::Forward {
namespace {
void requireCudaSuccess(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("runCachedInferenceForward: ") +
            operation + ": " + cudaGetErrorString(error));
    }
}

struct CachedForwardScope {
    ModelForwardOutputs& outputs;
    cudaStream_t stream;
    bool synchronized = false;
    ~CachedForwardScope() {
        // On exceptions, finish any queued CPU copies before their destinations
        // are destroyed. The result is declared before this scope for that reason.
        if (!synchronized) (void)cudaStreamSynchronize(stream);
        outputs.clear();
    }
};
} // namespace

CachedInferenceResult runCachedInferenceForward(
    ModelForwardRequest request,
    Batching::BatchPayload& payload,
    std::shared_ptr<Batching::BatchDeviceStorage>& reusable_storage,
    LocalAtomRetrievalInferenceState* retrieval_state,
    CachedInferenceReadback readback) {
    constexpr const char* caller = "runCachedInferenceForward";
    if (!request.config || !request.kv_cache) {
        throw std::invalid_argument(
            "runCachedInferenceForward: config and initialized KV cache are required");
    }
    if (payload.device_storage) {
        reusable_storage = payload.device_storage;
    } else {
        if (!reusable_storage) {
            reusable_storage = Batching::createBatchDeviceStorage(*request.config, request.stream);
        }
        Batching::attachBatchDeviceStorage(payload, reusable_storage, caller);
    }
    const auto bindings = Batching::uploadBatchToDevice(*request.config, payload, request.stream);
    request.payload = &payload;
    request.bindings = &bindings;
    request.graph = ModelForwardGraphPolicy{
        /*connect_parameter_graph=*/false,
        /*enable_dropout=*/false,
        /*emit_local_atom_retrieval=*/
            retrieval_state != nullptr && payload.isInferencePrefill()};

    CachedInferenceResult result;
    ModelForwardRuntimePayload runtime;
    auto outputs = executeModelForward(request, runtime);
    CachedForwardScope cleanup{outputs, request.stream};
    result.lens_snapshot = outputs.lens_snapshot;
    result.lens_capture_result = outputs.lens_capture_result;

    const int q_len = payload.total_tokens;
    if (retrieval_state && payload.isInferencePrefill()) {
        retrieval_state->seedFromPrefill(outputs.local_atom_candidate_embeddings, payload, request.stream);
    } else if (retrieval_state && payload.isInferenceDecode() && q_len == 1 &&
               Tokenizer::isAtomOpenTokenId(payload.input_ids[0])) {
        LocalAtomRetrieval::LocalAtomRetrievalDecodeForward(
            outputs.encoder_output_tensor, *request.parameter_registry, *retrieval_state,
            Tokenizer::tokenIdToAtomType(payload.input_ids[0]), request.stream, outputs);
    }

    if (readback.last_logits) {
        const auto& logits = outputs.logits_tensor;
        const auto matrix = logits.shape.as_2d();
        if (!logits.data || matrix.rows != q_len || matrix.cols != payload.vocab_size) {
            throw std::runtime_error("runCachedInferenceForward: logits shape/data mismatch");
        }
        result.last_logits.resize(static_cast<std::size_t>(matrix.cols));
        requireCudaSuccess(cudaMemcpyAsync(
            result.last_logits.data(),
            logits.data + static_cast<std::size_t>(q_len - 1) * matrix.cols,
            result.last_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, request.stream),
            "last logits copy failed");
    }
    if (readback.encoder_row) {
        const auto& encoder = outputs.encoder_output_tensor;
        const auto matrix = encoder.shape.as_2d();
        const auto model_hp = HyperParameters::modelHP(*request.config);
        if (!encoder.data || matrix.rows != q_len || matrix.cols != model_hp.encoder_d_model) {
            throw std::runtime_error("runCachedInferenceForward: encoder shape/data mismatch");
        }
        result.encoder_row.resize(static_cast<std::size_t>(matrix.cols));
        requireCudaSuccess(cudaMemcpyAsync(
            result.encoder_row.data(),
            encoder.data + static_cast<std::size_t>(q_len - 1) * matrix.cols,
            result.encoder_row.size() * sizeof(float), cudaMemcpyDeviceToHost, request.stream),
            "encoder row copy failed");
    }
    if (readback.retrieval_logits) {
        const auto& logits = outputs.local_atom_retrieval_logits;
        if (!logits.data) {
            throw std::runtime_error("runCachedInferenceForward: requested retrieval logits were not produced");
        }
        const auto matrix = logits.shape.as_2d();
        if (matrix.rows != 1 || matrix.cols <= 0) {
            throw std::runtime_error("runCachedInferenceForward: retrieval logits shape mismatch");
        }
        result.retrieval_logits.resize(static_cast<std::size_t>(matrix.cols));
        requireCudaSuccess(cudaMemcpyAsync(
            result.retrieval_logits.data(), logits.data,
            result.retrieval_logits.size() * sizeof(float), cudaMemcpyDeviceToHost, request.stream),
            "retrieval logits copy failed");
    }
    requireCudaSuccess(cudaStreamSynchronize(request.stream), "stream synchronization failed");
    cleanup.synchronized = true;
    return result;
}

} // namespace GRIM::Forward
