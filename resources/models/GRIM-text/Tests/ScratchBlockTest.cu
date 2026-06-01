//======================================================//
//  ScratchBlockTest.cu
//  Smoke tests for the ScratchBlock runtime shell
//======================================================//

#include "ScratchBlockTest.hpp"

#include "../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <utility>

using namespace GRIM;
using namespace GRIM::Test;

namespace {

HyperParameters::ScratchBlockConstructionHP makeScratchBlockHp(bool enabled) {
    HyperParameters::ScratchBlockConstructionHP hp;
    hp.enabled = enabled;
    hp.d_model = 64;
    hp.max_atoms = 16;
    hp.atom_embedding_dim = 32;
    hp.atom_token_start = HyperParameters::ATOM_TOKEN_START;
    hp.atom_token_end = HyperParameters::ATOM_TOKEN_END;
    hp.atom_scale = 1.0f;
    return hp;
}

Batching::BatchPayload makeInferencePayload(int total_tokens) {
    Batching::BatchPayload payload;
    payload.mode = Batching::BatchPayloadMode::InferencePrefill;
    payload.batch_size = 1;
    payload.max_seq_len = total_tokens;
    payload.total_tokens = total_tokens;
    payload.actual_tokens = total_tokens;
    payload.padding_tokens = 0;
    payload.valid_tokens = 0;
    payload.lm_valid_tokens = 0;
    payload.vocab_size = 512;
    payload.seq_lengths = {total_tokens};
    payload.valid_target_counts = {0};
    payload.input_ids.assign(total_tokens, 0);
    payload.target_ids.assign(total_tokens, -1);
    payload.numeric_values.assign(total_tokens, 0.0f);
    payload.atom_mask.assign(total_tokens, 0);
    payload.atom_flags.assign(total_tokens, 0u);
    payload.token_to_slot_map.assign(total_tokens, -1);
    payload.fits_in_cache = true;
    return payload;
}

bool testLoggingAccessors(std::string& message) {
    ScratchBlockLayer layer(makeScratchBlockHp(false), nullptr);

    SB_ASSERT_FALSE(layer.isLoggingEnabled(), "Logging should start disabled");
    layer.setLoggingEnabled(true);
    SB_ASSERT_TRUE(layer.isLoggingEnabled(), "Logging should be enabled after setLoggingEnabled(true)");

    layer.setGlobalStep(42);
    SB_ASSERT_EQ(layer.globalStep(), 42ULL, "Global step accessor mismatch");
    return true;
}

bool testDisabledConstructorSkipsRuntimeBuffers(std::string& message) {
    ScratchBlockLayer layer(makeScratchBlockHp(false), nullptr);

    SB_ASSERT_TRUE(layer.atomPositionsBuffer() == nullptr, "Disabled ScratchBlock should not allocate atom positions buffer");
    SB_ASSERT_TRUE(layer.numAtomsBuffer() == nullptr, "Disabled ScratchBlock should not allocate num-atoms buffer");
    SB_ASSERT_TRUE(layer.atomEmbeddingsBuffer() == nullptr, "Disabled ScratchBlock should not allocate atom embeddings buffer");
    return true;
}

bool testEnabledConstructorAllocatesRuntimeBuffers(std::string& message) {
    const auto hp = makeScratchBlockHp(true);
    cudaStream_t stream = nullptr;
    SB_ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream), "Failed to create CUDA stream");

    {
        ScratchBlockLayer layer(hp, stream);
        SB_ASSERT_TRUE(layer.atomPositionsBuffer() != nullptr, "Enabled ScratchBlock should allocate atom positions buffer");
        SB_ASSERT_TRUE(layer.numAtomsBuffer() != nullptr, "Enabled ScratchBlock should allocate num-atoms buffer");
        SB_ASSERT_TRUE(layer.atomEmbeddingsBuffer() != nullptr, "Enabled ScratchBlock should allocate atom embeddings buffer");
    }

    SB_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream), "Failed to destroy CUDA stream");
    return true;
}

bool testMoveConstructTransfersRuntimeBuffers(std::string& message) {
    const auto hp = makeScratchBlockHp(true);
    cudaStream_t stream = nullptr;
    SB_ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream), "Failed to create CUDA stream");

    {
        ScratchBlockLayer source(hp, stream);
        int* positions_before = source.atomPositionsBuffer();
        int* num_atoms_before = source.numAtomsBuffer();
        float* embeddings_before = source.atomEmbeddingsBuffer();

        ScratchBlockLayer moved(std::move(source));
        SB_ASSERT_TRUE(moved.atomPositionsBuffer() == positions_before, "Move construction should transfer atom positions buffer");
        SB_ASSERT_TRUE(moved.numAtomsBuffer() == num_atoms_before, "Move construction should transfer num-atoms buffer");
        SB_ASSERT_TRUE(moved.atomEmbeddingsBuffer() == embeddings_before, "Move construction should transfer atom embeddings buffer");
        SB_ASSERT_TRUE(source.atomPositionsBuffer() == nullptr, "Moved-from layer should release atom positions buffer");
        SB_ASSERT_TRUE(source.numAtomsBuffer() == nullptr, "Moved-from layer should release num-atoms buffer");
        SB_ASSERT_TRUE(source.atomEmbeddingsBuffer() == nullptr, "Moved-from layer should release atom embeddings buffer");
    }

    SB_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream), "Failed to destroy CUDA stream");
    return true;
}

bool testRunForwardKernelsRequiresExplicitBindings(std::string& message) {
    const auto hp = makeScratchBlockHp(true);
    const auto payload = makeInferencePayload(4);

    cudaStream_t stream = nullptr;
    SB_ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream), "Failed to create CUDA stream");

    float* d_output = nullptr;
    SB_ASSERT_CUDA_SUCCESS(
        cudaMalloc(&d_output, static_cast<size_t>(payload.total_tokens) * hp.d_model * sizeof(float)),
        "Failed to allocate output buffer");

    bool threw = false;
    try {
        ScratchBlockLayer layer(hp, stream);
        GRIM::ScratchBlockParameterTensors scratch_parameters{};
        const Batching::BatchDeviceBindings bindings{};
        layer.runForwardKernels(d_output, scratch_parameters, hp, payload, bindings, stream, false);
    } catch (const std::runtime_error&) {
        threw = true;
    }

    cudaFree(d_output);
    SB_ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream), "Failed to destroy CUDA stream");
    SB_ASSERT_TRUE(threw, "runForwardKernels should require explicit BatchDeviceBindings device pointers");
    return true;
}

}  // namespace

void registerScratchBlockTests(ScratchBlockTestSuite& suite) {
    suite.addTest("Accessors: Logging Controls", testLoggingAccessors);
    suite.addTest("Runtime Buffers: Disabled Construction", testDisabledConstructorSkipsRuntimeBuffers);
    suite.addTest("Runtime Buffers: Enabled Construction", testEnabledConstructorAllocatesRuntimeBuffers);
    suite.addTest("Runtime Buffers: Move Construction", testMoveConstructTransfersRuntimeBuffers);
    suite.addTest("Boundary: Explicit Device Bindings Required", testRunForwardKernelsRequiresExplicitBindings);
}

int GRIM::Test::runScratchBlockTests() {
    ScratchBlockTestSuite suite;
    registerScratchBlockTests(suite);

    const auto results = suite.runAll();
    int failed = 0;
    for (const auto& result : results) {
        if (!result.passed) {
            failed++;
        }
    }
    return failed;
}

#ifdef SCRATCH_BLOCK_TEST_MAIN
int main() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    return GRIM::Test::runScratchBlockTests() > 0 ? 1 : 0;
}
#endif
