"""Exercise cached-forward orchestration with host CUDA/forward substitutes.

Compiles the production runner body; does not build or execute GPU/model targets.
"""
from pathlib import Path

from test_add_gradfn_host import compile_and_run


def main():
    root = Path(__file__).resolve().parent.parent
    def without_includes(path):
        return '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                         if not line.startswith('#include') and line != '#pragma once')
    header = without_includes(root / 'Shared/Forward/CachedInferenceForward.hpp')
    body = without_includes(root / 'Shared/Forward/CachedInferenceForward.cu')
    source = r'''#include <cassert>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using cudaStream_t = int;
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaMemcpyDeviceToHost = 1;
int copies = 0, syncs = 0, clears = 0, allocations = 0, forwards = 0, decode_scores = 0;
bool fail_encoder = false;
const char* cudaGetErrorString(int) { return "host failure"; }
int cudaStreamSynchronize(int) { ++syncs; return 0; }
int cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, int, int) {
    ++copies; std::memcpy(dst, src, bytes); return 0;
}
namespace GRIM {
namespace Config { struct AiConfigSnapshot {}; }
struct Matrix { int rows, cols; };
struct Shape { Matrix matrix; Matrix as_2d() const { return matrix; } };
struct Tensor { float* data = nullptr; Shape shape{}; };
namespace Batching {
struct BatchDeviceStorage {};
struct BatchDeviceBindings {};
struct BatchPayload {
    bool prefill = true;
    int total_tokens = 2, vocab_size = 3;
    std::vector<int> input_ids{10, 11};
    std::shared_ptr<BatchDeviceStorage> device_storage;
    bool isInferencePrefill() const { return prefill; }
    bool isInferenceDecode() const { return !prefill; }
};
std::shared_ptr<BatchDeviceStorage> createBatchDeviceStorage(const Config::AiConfigSnapshot&, int) {
    ++allocations; return std::make_shared<BatchDeviceStorage>();
}
void attachBatchDeviceStorage(BatchPayload& p, std::shared_ptr<BatchDeviceStorage> s, const char*) {
    p.device_storage = std::move(s);
}
BatchDeviceBindings uploadBatchToDevice(const Config::AiConfigSnapshot&, BatchPayload&, int) { return {}; }
}
namespace HyperParameters {
struct ModelHP { int encoder_d_model = 2; };
ModelHP modelHP(const Config::AiConfigSnapshot&) { return {}; }
}
namespace Tokenizer {
bool isAtomOpenTokenId(int id) { return id == 101; }
int tokenIdToAtomType(int) { return 1; }
}
struct LocalAtomRetrievalInferenceState {
    int seeded = 0;
    void seedFromPrefill(const Tensor& candidates, const Batching::BatchPayload&, int) {
        assert(candidates.data); ++seeded;
    }
};
namespace Forward {
struct ModelForwardOutputs {
    Tensor logits_tensor, encoder_output_tensor, local_atom_retrieval_logits, local_atom_candidate_embeddings;
    std::shared_ptr<int> lens_snapshot, lens_capture_result;
    void clear() {
        ++clears; lens_snapshot.reset(); lens_capture_result.reset();
        logits_tensor.data = encoder_output_tensor.data = local_atom_retrieval_logits.data = nullptr;
    }
};
struct ModelForwardGraphPolicy {
    bool connect_parameter_graph = false;
    bool enable_dropout = false;
    bool emit_local_atom_retrieval = false;
};
struct ModelForwardRequest {
    const Config::AiConfigSnapshot* config = nullptr;
    int* kv_cache = nullptr;
    int* parameter_registry = nullptr;
    int stream = 1;
    const Batching::BatchPayload* payload = nullptr;
    const Batching::BatchDeviceBindings* bindings = nullptr;
    ModelForwardGraphPolicy graph;
};
struct ModelForwardRuntimePayload {};
float logits[]{1, 2, 3, 4, 5, 6}, encoder[]{7, 8, 9, 10}, retrieval[]{0.1f, 0.2f};
ModelForwardOutputs executeModelForward(const ModelForwardRequest& r, ModelForwardRuntimePayload&) {
    assert(r.payload && r.bindings);
    assert(!r.graph.connect_parameter_graph && !r.graph.enable_dropout);
    assert(!r.graph.emit_local_atom_retrieval || r.payload->isInferencePrefill());
    ++forwards; *r.kv_cache += r.payload->total_tokens;
    ModelForwardOutputs out;
    out.logits_tensor = {logits, {{r.payload->total_tokens, 3}}};
    out.encoder_output_tensor = {encoder, {{r.payload->total_tokens, fail_encoder ? 99 : 2}}};
    if (r.graph.emit_local_atom_retrieval) out.local_atom_candidate_embeddings = out.encoder_output_tensor;
    out.lens_snapshot = std::make_shared<int>(42);
    out.lens_capture_result = out.lens_snapshot;
    return out;
}
}
namespace LocalAtomRetrieval {
void LocalAtomRetrievalDecodeForward(Tensor&, int&, const LocalAtomRetrievalInferenceState&, int, int,
                                    Forward::ModelForwardOutputs& out) {
    ++decode_scores;
    out.local_atom_retrieval_logits = {Forward::retrieval, {{1, 2}}};
}
}
}
HEADER
BODY
int main() {
    using namespace GRIM;
    Config::AiConfigSnapshot config;
    int cache = 0, registry = 0;
    Forward::ModelForwardRequest request;
    request.config = &config; request.kv_cache = &cache; request.parameter_registry = &registry;
    // Caller policy is deliberately incompatible; the runner must override it.
    request.graph = {true, true, false};
    Batching::BatchPayload prefill;
    std::shared_ptr<Batching::BatchDeviceStorage> storage;
    LocalAtomRetrievalInferenceState state;
    auto first = Forward::runCachedInferenceForward(request, prefill, storage, &state);
    assert((first.last_logits == std::vector<float>{4, 5, 6}));
    assert(first.encoder_row.empty() && first.retrieval_logits.empty());
    assert(*first.lens_snapshot == 42 && *first.lens_capture_result == 42);
    assert(copies == 1 && clears == 1 && state.seeded == 1 && cache == 2);
    Batching::BatchPayload decode;
    decode.prefill = false; decode.total_tokens = 1; decode.input_ids = {101};
    auto second = Forward::runCachedInferenceForward(request, decode, storage, &state, {true, true, true});
    assert((second.last_logits == std::vector<float>{1, 2, 3}));
    assert((second.encoder_row == std::vector<float>{7, 8}));
    assert((second.retrieval_logits == std::vector<float>{0.1f, 0.2f}));
    assert(decode_scores == 1 && cache == 3 && allocations == 1);
    assert(decode.device_storage == prefill.device_storage);
    decode.input_ids = {12};
    int previous_copies = copies;
    auto skipped = Forward::runCachedInferenceForward(request, decode, storage, &state, {false, false, false});
    assert(skipped.last_logits.empty() && skipped.encoder_row.empty() && skipped.retrieval_logits.empty());
    assert(copies == previous_copies && decode_scores == 1);
    // Explicitly requesting an absent retrieval tensor must fail, even though
    // ordinary decode tokens legitimately do not produce retrieval logits.
    bool missing_rejected = false;
    int missing_clears = clears;
    try { Forward::runCachedInferenceForward(request, decode, storage, &state, {false, false, true}); }
    catch (const std::runtime_error& error) {
        missing_rejected = std::string(error.what()).find("requested retrieval logits") != std::string::npos;
    }
    assert(missing_rejected && clears == missing_clears + 1);
    previous_copies = copies;
    auto inspect = Forward::runCachedInferenceForward(request, prefill, storage, &state, {false});
    assert(inspect.last_logits.empty() && inspect.lens_snapshot);
    assert(copies == previous_copies);
    // Failure after a queued logits copy must still synchronize and clear outputs.
    fail_encoder = true;
    int previous_clears = clears, previous_syncs = syncs;
    bool rejected = false;
    try { Forward::runCachedInferenceForward(request, decode, storage, nullptr, {true, true}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && clears == previous_clears + 1 && syncs == previous_syncs + 1);
    request.kv_cache = nullptr;
    int previous_forwards = forwards;
    rejected = false;
    try { Forward::runCachedInferenceForward(request, decode, storage, nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && forwards == previous_forwards);
}
'''
    source = source.replace('HEADER', header).replace('BODY', body)
    compile_and_run(source)
    print('Cached inference forward host tests passed.')


if __name__ == '__main__':
    main()
