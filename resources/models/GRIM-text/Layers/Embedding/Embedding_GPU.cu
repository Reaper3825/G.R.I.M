/**
 * @file Embedding_GPU.cu
 * @brief EmbeddingRuntime memory management only
 *
 * PRODUCTION PATH: autograd::embedding() in TensorContract_GPU.cu
 * 
 * This file now ONLY contains:
 * - destroyEmbeddingRuntime(): Frees GPU buffers for EmbeddingRuntime struct
 *
 * ═══════════════════════════════════════════════════════════════════════════════════
 * LEGACY CODE DELETED (Issue #92 / Rule 20: No Backwards Compatibility)
 * ═══════════════════════════════════════════════════════════════════════════════════
 * The following were DEAD CODE - production uses autograd::embedding() which has 
 * kernel_embedding_forward/backward in TensorContract_GPU.cu:
 *
 * - EmbeddingLookupKernel: Legacy forward kernel
 * - EmbeddingRMSNormKernel: Legacy fused forward+RMSNorm kernel
 * - EmbeddingBackwardKernel: Legacy backward kernel  
 * - PositionEmbeddingBackwardKernel: Legacy position backward kernel
 * - launchEmbeddingLookup(): Legacy forward launcher
 * - launchEmbeddingBackward(): Legacy backward launcher
 * - launchPositionEmbeddingBackward(): Legacy position backward launcher
 * - validateTokenIds(): Legacy validation helper
 * - ValidateTokenIdsKernel: Legacy validation kernel
 * - EmbeddingLayer class: Legacy stateless wrapper
 * - validateRuntime(): Legacy runtime validation helper
 * - validateForwardArgs(): Legacy args validation helper
 * - embeddingRuntimeForward(): Already deleted earlier
 * - embeddingRuntimeForwardSingle(): Already deleted earlier
 *
 * Test files also deleted:
 * - Tests/embedding_self_test.cu (legacy kernel tests)
 * - Tests/embedding_autograd_test.cu (legacy kernel tests)
 * ═══════════════════════════════════════════════════════════════════════════════════
 */

#include "Embedding_GPU.hpp"
#include <cuda_runtime.h>

namespace GRIM {

//======================================================//
// Runtime Lifecycle - ONLY RETAINED FUNCTIONALITY
//======================================================//

void destroyEmbeddingRuntime(EmbeddingRuntime* runtime) {
    if (!runtime) return;

    // Only free buffers we own - TrainingTensors may own them instead
    if (runtime->owns_token_buffer && runtime->token_buffer) {
        cudaFree(runtime->token_buffer);
    }
    runtime->token_buffer = nullptr;
    
    if (runtime->owns_position_buffer && runtime->position_buffer) {
        cudaFree(runtime->position_buffer);
    }
    runtime->position_buffer = nullptr;
    
    if (runtime->owns_gamma_buffer && runtime->gamma_buffer) {
        cudaFree(runtime->gamma_buffer);
    }
    runtime->gamma_buffer = nullptr;
    
    // Always free these (always owned by runtime)
    if (runtime->single_token_id) { 
        cudaFree(runtime->single_token_id); 
        runtime->single_token_id = nullptr; 
    }
    if (runtime->single_position) { 
        cudaFree(runtime->single_position); 
        runtime->single_position = nullptr; 
    }
    if (runtime->owns_stream && runtime->stream) { 
        cudaStreamDestroy(runtime->stream); 
        runtime->stream = nullptr; 
    }

    delete runtime;
}

} // namespace GRIM
