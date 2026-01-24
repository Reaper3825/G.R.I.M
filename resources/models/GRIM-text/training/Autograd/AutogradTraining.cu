//======================================================//
//  AutogradTraining.cu
//  Implementation of autograd-based training flow
//======================================================//

#include "AutogradTraining.hpp"

// MUST include full definition of GPUGrimEncoder for method calls
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/LayernNorm/RMSNorm_Kernel_GPU.hpp"
#include "../../Layers/grim_layer_gpu.hpp"                    // LayerWorkspace
#include "../../Layers/Encoding/Encoding_GPU.hpp"             // EncodingLayer::useExternalWeights
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"     // ScratchBlockLayer
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"      // NumericHead forward/backward
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../../Shared/VerboseLogging.hpp"

// Logging macros - guarded by VerboseLogging flags for production
#define AG_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[AutogradTraining] INFO: " << msg << std::endl; \
    } \
} while(0)
#define AG_ERROR(msg) do { std::cerr << "[AutogradTraining] ERROR: " << msg << std::endl; } while(0)
#define AG_WARN(msg) do { std::cerr << "[AutogradTraining] WARN: " << msg << std::endl; } while(0)

namespace GRIM {
namespace Autograd {

namespace {

constexpr int kBlockSize = 256;

__global__ void scaleGradientsKernel(float* __restrict__ data, float scale, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

inline void scaleGradBuffer(float* data, size_t n, float scale, cudaStream_t stream) {
    if (!data || n == 0) {
        return;
    }
    const int n_int = static_cast<int>(n);
    const int blocks = (n_int + kBlockSize - 1) / kBlockSize;
    scaleGradientsKernel<<<blocks, kBlockSize, 0, stream>>>(data, scale, n_int);
}

//======================================================//
// Issue #57 FIX: Position embedding support
// Generate position IDs [0,1,2,...,seq_len-1] repeated for each batch element
// and add position embeddings to token embeddings
//======================================================//

__global__ void generatePositionIdsKernel(int* __restrict__ position_ids,
                                          int total_tokens,
                                          int seq_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_tokens) return;
    
    // Position within sequence = idx % seq_len
    // This gives [0,1,2,...,seq_len-1] for each batch element
    position_ids[idx] = idx % seq_len;
}

inline void generatePositionIds(int* position_ids, int total_tokens, int seq_len, cudaStream_t stream) {
    const int blocks = (total_tokens + kBlockSize - 1) / kBlockSize;
    generatePositionIdsKernel<<<blocks, kBlockSize, 0, stream>>>(position_ids, total_tokens, seq_len);
}

//======================================================================
// Broadcast Bias Add Kernel
// Adds a [1, vocab_size] bias to each row of [total_tokens, vocab_size] logits
//======================================================================
__global__ void broadcastBiasAddKernel(
    float* __restrict__ output,           // [total_tokens, vocab_size]
    const float* __restrict__ bias,       // [vocab_size]
    int total_tokens,
    int vocab_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(total_tokens) * vocab_size;
    if (idx >= total) return;
    
    const int vocab_idx = idx % vocab_size;
    output[idx] += bias[vocab_idx];
}

inline void launchBroadcastBiasAdd(
    float* output,
    const float* bias,
    int total_tokens,
    int vocab_size,
    cudaStream_t stream
) {
    const size_t total = static_cast<size_t>(total_tokens) * vocab_size;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    broadcastBiasAddKernel<<<blocks, kBlockSize, 0, stream>>>(output, bias, total_tokens, vocab_size);
}

//======================================================================
// Valid Token Counting Kernel
// Counts number of tokens where mask >= 0.5
//======================================================================
__global__ void countValidTokensKernel(
    const float* __restrict__ valid_mask,
    int* __restrict__ count,
    int num_tokens
) {
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    
    int local_count = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_tokens; i += gridDim.x * blockDim.x) {
        if (valid_mask[i] >= 0.5f) {
            local_count++;
        }
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_count, local_count);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(count, s_count);
    }
}

inline int countValidTokens(const float* valid_mask, int num_tokens, cudaStream_t stream) {
    if (!valid_mask) return num_tokens;  // All valid if no mask
    
    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    
    const int blocks = std::min(256, (num_tokens + kBlockSize - 1) / kBlockSize);
    countValidTokensKernel<<<blocks, kBlockSize, 0, stream>>>(valid_mask, d_count, num_tokens);
    
    int h_count = 0;
    cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_count);
    
    return h_count;
}

//======================================================================
// Gradient Norm Computation Kernels
// Computes L2 norm: sqrt(sum(grad^2))
//======================================================================
__global__ void sumSquaredKernel(
    const float* __restrict__ data,
    double* __restrict__ partial_sum,
    int n
) {
    __shared__ double s_sum;
    if (threadIdx.x == 0) s_sum = 0.0;
    __syncthreads();
    
    double local_sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const double val = static_cast<double>(data[i]);
        local_sum += val * val;
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(partial_sum, s_sum);
    }
}

inline double computeSumSquared(const float* data, size_t n, double* d_sum, cudaStream_t stream) {
    if (!data || n == 0) return 0.0;
    
    const int n_int = static_cast<int>(n);
    const int blocks = std::min(256, (n_int + kBlockSize - 1) / kBlockSize);
    sumSquaredKernel<<<blocks, kBlockSize, 0, stream>>>(data, d_sum, n_int);
    return 0.0;  // Result accumulated in d_sum
}

} // namespace

// NOTE: linkEncoderWeightsToTrainingState was removed.
// Encoder owns its weights internally; optimizer accesses gradients via
// enc->getAttnWqkvGrad(), enc->getFFNW1Grad(), etc.
// See buildParameterGroups() in LanguageModel_Training.cu.

//======================================================================
// Context Initialization
//======================================================================

AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    int batch_size,
    int seq_len,
    float grad_scale,
    uint64_t step
) {
    AutogradContext ctx{};
    ctx.config = config;
    ctx.training_state = training_state;
    ctx.gpu_encoder = gpu_encoder;
    ctx.scratch_block = scratch_block;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.batch_size = batch_size;
    ctx.seq_len = seq_len;
    ctx.grad_scale = grad_scale;
    ctx.step = step;
    ctx.error_layer = -1;
    
    // Pre-allocate encoder layer outputs vector
    if (config) {
        ctx.encoder_layer_outputs.reserve(config->num_layers);
    }
    
    if (!ctx.isValid()) {
        ctx.error_message = "Invalid context: missing required fields";
        AG_ERROR("initAutogradContext: " << ctx.error_message);
    }
    
    return ctx;
}

//======================================================================
// Autograd Forward Pass
// PRODUCTION-READY: Runs entire model with autograd graph intact
//======================================================================

ForwardResult executeAutogradForward(AutogradContext& ctx) {
    ForwardResult result{};
    result.success = false;
    
    if (!ctx.isValid()) {
        result.error_message = "Invalid context";
        AG_ERROR("executeAutogradForward: " << result.error_message);
        return result;
    }
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    
    const int total_tokens = ctx.batch_size * ctx.seq_len;
    result.total_tokens = total_tokens;
    result.vocab_size = cfg->vocab_size;
    
    AG_INFO("Autograd Forward: batch=" << ctx.batch_size << " seq=" << ctx.seq_len 
            << " tokens=" << total_tokens << " vocab=" << cfg->vocab_size);
    
    // Set autograd cuBLAS handle for all matmul operations
    autograd::set_autograd_cublas_handle(ctx.cublas_handle);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 1: Embedding Lookup
    //  Input: token_ids [total_tokens]
    //  Output: embeddings [total_tokens, d_model]
    //  
    //  Uses autograd::embedding() for gradient tracking
    // ═══════════════════════════════════════════════════════════════════════════
    
    // RULE 20: Fail loud - validate required buffers
    if (!ts->cached_token_ids) {
        throw std::runtime_error("AutogradForward: cached_token_ids buffer is NULL");
    }
    
    // Embedding weights tensor (already a Tensor in TrainingState)
    Tensor& emb_weights = ts->embedding_weights;
    if (!emb_weights.data) {
        throw std::runtime_error("AutogradForward: embedding_weights.data is NULL");
    }
    emb_weights.requires_grad = true;
    
    // Ensure embedding weights have correct shape
    if (!emb_weights.shape.is_valid()) {
        emb_weights.shape = TensorContract::TensorShape::make_BSM(cfg->vocab_size, cfg->d_model);
    }
    
    // Use autograd::embedding for proper gradient tracking
    // This performs: output[i] = weight[token_ids[i]] with gradient scatter-add backward
    Tensor emb_output = autograd::embedding(
        emb_weights,
        ts->cached_token_ids,
        total_tokens,
        ctx.stream
    );
    
    // Copy to cached buffer for compatibility with rest of pipeline
    if (ts->cached_embeddings) {
        cudaMemcpyAsync(ts->cached_embeddings, emb_output.data,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Issue #57 FIX: Add position embeddings
    // PyTorch baseline: x = tok_emb->forward(idx) + pos_emb->forward(pos)
    // GRIM was MISSING this step, causing training plateau!
    // ═══════════════════════════════════════════════════════════════════════════
    if (ts->position_embedding_weights.data) {
        ts->position_embedding_weights.requires_grad = true;
        
        // Ensure position embedding weights have correct shape [max_seq_len, d_model]
        if (!ts->position_embedding_weights.shape.is_valid()) {
            ts->position_embedding_weights.shape = TensorContract::TensorShape::make_BSM(
                cfg->max_seq_len, cfg->d_model);
        }
        
        // Allocate temporary buffer for position IDs on device
        int* d_position_ids = nullptr;
        cudaMallocAsync(&d_position_ids, total_tokens * sizeof(int), ctx.stream);
        
        // Generate position IDs: [0,1,2,...,seq_len-1] repeated for each batch
        generatePositionIds(d_position_ids, total_tokens, ctx.seq_len, ctx.stream);
        
        // Look up position embeddings with autograd tracking
        Tensor pos_emb_output = autograd::embedding(
            ts->position_embedding_weights,
            d_position_ids,
            total_tokens,
            ctx.stream
        );
        
        // Free temporary position IDs (embedding lookup already copied them)
        cudaFreeAsync(d_position_ids, ctx.stream);
        
        // Add token embeddings + position embeddings (both tracked by autograd)
        emb_output = autograd::add(emb_output, pos_emb_output, ctx.stream);
        
        // Update cached embeddings with the combined result
        if (ts->cached_embeddings) {
            cudaMemcpyAsync(ts->cached_embeddings, emb_output.data,
                            static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                            cudaMemcpyDeviceToDevice, ctx.stream);
        }
        
        AG_INFO("Step 1b: Position embeddings added (Issue #57 FIX)");
    }
    
    // Store in context for backward
    ctx.embedding_tensor = std::move(emb_output);
    AG_INFO("Step 1: Embedding complete, shape=[" << total_tokens << ", " << cfg->d_model << "]");
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 1.5: ScratchBlock (numeric/code processing)
    //  
    //  Processes numeric atoms (integers, floats, hex, etc.) and injects
    //  learned embeddings into the token representations. This enables
    //  the model to understand numeric patterns and code structures.
    //  
    //  NOTE: ScratchBlock operates IN-PLACE on the embedding buffer.
    //  The backward pass uses cached atom positions and types.
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        AG_INFO("Step 1.5: Running ScratchBlock (numeric/code processing)...");
        
        ScratchBlockForwardArgs sb_args{};
        
        // Input/output are the same (in-place operation on embeddings)
        // Use cached_embeddings since that's where embedding data lives
        sb_args.input = TensorContract::TensorView::make_BSM(
            ts->cached_embeddings, total_tokens, cfg->d_model, "sb_input");
        sb_args.output = TensorContract::TensorView::make_BSM(
            ts->cached_embeddings, total_tokens, cfg->d_model, "sb_output");
        
        sb_args.total_tokens = total_tokens;
        sb_args.seq_len = ctx.seq_len;
        sb_args.token_ids = ts->cached_token_ids;
        
        // Numeric values and mask from DataLoader (passed via context)
        sb_args.token_numeric_values = ctx.token_numeric_values;
        sb_args.token_numeric_mask = ctx.token_numeric_mask;
        
        // GRMT v4: text features for atom injection
        sb_args.token_text_features = ctx.token_text_features;
        sb_args.token_text_mask = ctx.token_text_mask;
        
        sb_args.stream = ctx.stream;
        
        // Cache atom embeddings for backward
        if (ts->cached_scratch_block_embeddings) {
            sb_args.cache_atom_embeddings = TensorContract::TensorView::make_BSM(
                ts->cached_scratch_block_embeddings,
                ctx.scratch_block->config().max_atoms,
                ctx.scratch_block->config().atom_embedding_dim,
                "sb_cache_embeddings");
        }
        sb_args.cache_atom_positions = ts->cached_scratch_block_positions;
        sb_args.cache_atom_types = ts->cached_scratch_block_types;
        sb_args.cache_num_atoms = ts->cached_scratch_block_num_atoms;
        
        // Run ScratchBlock forward
        ctx.scratch_block->forward(sb_args);
        
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("AutogradForward: ScratchBlock forward CUDA error: " + 
                                     std::string(cudaGetErrorString(cuda_err)));
        }
        
        AG_INFO("Step 1.5: ScratchBlock complete");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 2: Encoder Layers (transformer blocks)
    //  
    //  PRODUCTION READY: Each layer uses autograd ops internally.
    //  The returned Tensors are stored in ctx.encoder_layer_outputs to
    //  keep the autograd graph alive for backward propagation.
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (!ctx.gpu_encoder) {
        throw std::runtime_error("AutogradForward: gpu_encoder is NULL - pass encoder in context");
    }
    if (!ts->encoder_workspace) {
        throw std::runtime_error("AutogradForward: encoder_workspace is NULL - TrainingState MUST allocate workspace");
    }
    
    const int num_layers = ctx.gpu_encoder->getNumLayers();
    ctx.encoder_layer_outputs.clear();
    ctx.encoder_layer_outputs.reserve(num_layers);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  Issue #56 FIX: Reserve space for intermediate tensor storage
    //  
    //  CRITICAL: We MUST keep all intermediate tensors alive until backward
    //  completes. Without this, grad_fn objects are destroyed when forward()
    //  returns, causing use-after-free in backward pass.
    // ═══════════════════════════════════════════════════════════════════════════
    ctx.layer_intermediates.layers.clear();
    ctx.layer_intermediates.layers.reserve(num_layers);
    
    // ISSUE #60 FIX: First layer input MUST preserve embedding_tensor's grad_fn!
    // Previous code used from_ptr() which creates a NEW tensor with NO grad_fn,
    // breaking the autograd chain - EmbeddingGradFn was never called!
    // 
    // For first layer: use embedding_tensor directly (preserves grad_fn chain)
    // For subsequent layers: use layer output (which has its own grad_fn chain)
    ctx.embedding_tensor.is_leaf = false;  // Ensure it's not treated as a leaf
    
    AG_INFO("Step 2: Running " << num_layers << " encoder layers with autograd...");
    AG_INFO("  embedding_tensor.grad_fn=" << (void*)ctx.embedding_tensor.grad_fn 
            << " requires_grad=" << ctx.embedding_tensor.requires_grad);
    
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
        if (!enc_layer) {
            throw std::runtime_error("AutogradForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
        }
        
        // Setup workspace
        LayerWorkspace<float> ws{};
        ws.data = ts->encoder_workspace;
        ws.bytes = enc_layer->requiredWorkspaceBytes(total_tokens, ctx.seq_len);
        enc_layer->setWorkspace(ws.data, ws.bytes);
        
        // ═══════════════════════════════════════════════════════════════════════
        //  Issue #56 FIX: Create intermediates storage and pass to forward()
        //  
        //  forward() stores all intermediate tensors in this struct instead of
        //  local variables. This keeps the autograd graph alive until backward.
        // ═══════════════════════════════════════════════════════════════════════
        ctx.layer_intermediates.layers.emplace_back();
        ForwardIntermediates& layer_storage = ctx.layer_intermediates.layers.back();
        
        // ISSUE #60 FIX: First layer uses embedding_tensor directly (with grad_fn intact)
        // Subsequent layers use previous layer output (which has proper grad_fn chain)
        Tensor& layer_input = (layer_idx == 0) 
            ? ctx.embedding_tensor 
            : ctx.encoder_layer_outputs.back();
        
        // Run layer forward with intermediates storage
        Tensor layer_output = enc_layer->forward(layer_input, ctx.seq_len, ctx.stream, layer_storage);
        
        // Store output reference in encoder_layer_outputs (for compatibility)
        // NOTE: The actual output data lives in layer_storage.output
        ctx.encoder_layer_outputs.push_back(std::move(layer_output));
    }
    
    AG_INFO("Step 2: All " << num_layers << " encoder layers complete");
    
    // Final encoder output is the last layer's output
    float* encoder_output = ctx.encoder_layer_outputs.back().data;
    result.encoder_output = encoder_output;
    
    // Also copy to cached buffer for compatibility
    if (ts->cached_encoder_outputs) {
        cudaMemcpyAsync(ts->cached_encoder_outputs, encoder_output,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 3: Final RMSNorm (before LM head)
    //  Uses autograd::rms_norm for gradient tracking
    // ═══════════════════════════════════════════════════════════════════════════
    
    Tensor normalized_output;
    if (ts->final_rms_gamma.data) {
        ts->final_rms_gamma.requires_grad = true;
        
        // Create input tensor from encoder output
        Tensor rms_input = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,
            true
        );
        rms_input.is_leaf = false;
        rms_input.grad_fn = ctx.encoder_layer_outputs.back().grad_fn;
        rms_input.owns_grad_fn = false;  // Borrowed from encoder output
        
        // Apply autograd RMSNorm
        normalized_output = autograd::rms_norm(rms_input, ts->final_rms_gamma, 
                                               cfg->rms_epsilon, ctx.stream);
        
        encoder_output = normalized_output.data;
        AG_INFO("Step 3: Final RMSNorm applied with autograd");
    } else {
        // No final RMSNorm, use encoder output directly
        normalized_output = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false, true
        );
        normalized_output.is_leaf = false;
        normalized_output.grad_fn = ctx.encoder_layer_outputs.back().grad_fn;
        normalized_output.owns_grad_fn = false;  // Borrowed, not owned
    }
    
    // Store for backward
    ctx.encoder_output_tensor = std::move(normalized_output);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 4: LM Head → Logits
    //  Input: encoder_output [total_tokens, d_model]
    //  Weights: lm_head_weights [vocab_size, d_model]
    //  Output: logits [total_tokens, vocab_size]
    //  
    //  Using autograd::matmul for proper gradient tracking through the
    //  entire computation graph (encoder → final_rms → lm_head → loss)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Get LM head weights tensor (already a Tensor in TrainingState)
    Tensor& lm_weights = ts->lm_head_weights;
    lm_weights.requires_grad = true;  // Gradients for weight update
    
    // RULE 20: Fail loud - validate cached_logits buffer
    float* logits_output = ts->cached_logits;
    if (!logits_output) {
        throw std::runtime_error("AutogradTraining: cached_logits buffer is NULL - TrainingState MUST allocate logits buffer");
    }
    
    // Create input tensor referencing the encoder output
      // Link grad_fn to continue the backward chain
    // ISSUE #48 FIX: Store in context so pointer remains valid until backward completes
    ctx.lm_input_tensor = Tensor::from_ptr(
        ctx.encoder_output_tensor.data,
        TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
        false,  // doesn't own data
        true    // requires_grad
    );
    ctx.lm_input_tensor.is_leaf = false;
    ctx.lm_input_tensor.grad_fn = ctx.encoder_output_tensor.grad_fn;  // Link to RMSNorm or last encoder layer
    ctx.lm_input_tensor.owns_grad_fn = false;  // Borrowed, not owned
    
    // RULE 20: Validate shapes
    ctx.lm_input_tensor.shape.require("lm_input");
    lm_weights.shape = TensorContract::TensorShape::make_BSM(cfg->vocab_size, cfg->d_model);
    lm_weights.shape.require("lm_weights");
    
    // Execute autograd matmul: logits = encoder @ lm_weights^T
    // transpose_b=true because lm_weights is [vocab_size, d_model]
    Tensor logits_tensor = autograd::matmul(
        ctx.lm_input_tensor,
        lm_weights,
        ctx.stream,
        nullptr,  // a_cache: autograd will cache internally
        nullptr,  // b_cache: weights persist in memory
        true      // transpose_b=true
    );
    
    AG_INFO("Step 4: LM head matmul complete");
    
    // RULE 20: Validate output shape
    const auto expected_shape = TensorContract::TensorShape::make_LOGITS(total_tokens, cfg->vocab_size);
    if (logits_tensor.shape.total_elements() != expected_shape.total_elements()) {
        throw std::runtime_error("AutogradTraining: logits shape mismatch - got " + 
                                 std::to_string(logits_tensor.shape.total_elements()) + 
                                 " elements, expected " + 
                                 std::to_string(expected_shape.total_elements()));
    }
    
    // Update logits shape to use LOGITS layout (semantic correctness)
    logits_tensor.shape = expected_shape;
    
    // Copy logits data to cached buffer (for compatibility with loss computation)
    cudaMemcpyAsync(logits_output, logits_tensor.data,
                    logits_tensor.shape.total_elements() * sizeof(float),
                    cudaMemcpyDeviceToDevice, ctx.stream);
    
    // Add bias if present (applied to cached buffer)
    if (cfg->use_bias && ts->lm_head_bias.data) {
        launchBroadcastBiasAdd(
            logits_output,
            ts->lm_head_bias.data,
            total_tokens,
            cfg->vocab_size,
            ctx.stream
        );
        AG_INFO("Applied LM head bias (broadcast add)");
    }
    
    // CRITICAL FIX: Don't overwrite data pointer - this causes heap corruption!
    // The matmul output tensor owns its allocated memory (owns_data=true).
    // We copied the data to cached_logits above, so loss computation can use
    // cached_logits directly. The logits_tensor keeps its original data pointer
    // and will properly free it when destroyed.
    //
    // Move the autograd tensor to context (grad_fn is preserved)
    // The context owns the tensor lifetime during the forward-backward cycle
    ctx.logits_tensor = std::move(logits_tensor);
    // NOTE: ctx.logits_tensor.data still points to matmul's allocated buffer (correct!)
    // The cached_logits copy is used by loss computation for compatibility.
    
    // Create a lightweight result that references the context's tensor
    // NOTE: result.logits is just a reference view, ctx owns the actual Tensor
    result.logits.data = ctx.logits_tensor.data;
    result.logits.shape = ctx.logits_tensor.shape;
    result.logits.requires_grad = ctx.logits_tensor.requires_grad;
    result.logits.is_leaf = ctx.logits_tensor.is_leaf;
    // CRITICAL: Do NOT share grad_fn - the context owns it.
    // If we set result.logits.grad_fn = ctx.logits_tensor.grad_fn, then when
    // result.logits is destroyed it would call `delete grad_fn` (double-free!)
    result.logits.grad_fn = nullptr;  // Context owns the grad_fn, not the result
    // Do NOT take ownership (the context owns it)
    result.logits.owns_data = false;
    // Note: owns_grad removed - ownership now managed by shared_ptr<Tensor> grad_
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 5: NumericHead Forward (optional)
    //  
    //  Produces numeric predictions [total_tokens, 1] parallel to text logits.
    //  Used for numeric supervision (e.g., predicting numeric values in code).
    //  
    //  NumericHead: y = encoder_output @ weights + bias
    //  Loss is computed separately and added to text loss.
    //  
    //  Uses TensorContract Tensor-based API with autograd grad_fn.
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (cfg->numeric_head_enabled && ts->numeric_head_weights.data) {
        AG_INFO("Step 5: Running NumericHead forward (autograd)...");
        
        // Create Tensor wrappers for weights and bias (leaf tensors)
        Tensor weights_tensor = Tensor::from_ptr(
            ts->numeric_head_weights.data,
            TensorContract::TensorShape::make_BSM(cfg->d_model, 1),
            false,  // doesn't own
            true    // requires_grad
        );
        weights_tensor.is_leaf = true;
        // ISSUE #59: Use share_grad() for proper shared_ptr semantics
        weights_tensor.share_grad(ts->numeric_head_weights);
        
        Tensor* bias_tensor_ptr = nullptr;
        Tensor bias_tensor;
        if (cfg->use_bias && ts->numeric_head_bias.data) {
            bias_tensor = Tensor::from_ptr(
                ts->numeric_head_bias.data,
                TensorContract::TensorShape::make_BSM(1, 1),
                false,  // doesn't own
                true    // requires_grad
            );
            bias_tensor.is_leaf = true;
            // ISSUE #59: Use share_grad()
            bias_tensor.share_grad(ts->numeric_head_bias);
            bias_tensor_ptr = &bias_tensor;
        }
        
        // Create encoder output Tensor (references ctx.encoder_output_tensor)
        Tensor encoder_for_numeric = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,  // doesn't own
            true    // requires_grad
        );
        encoder_for_numeric.is_leaf = false;
        // Link to encoder's grad_fn for chain continuation
        encoder_for_numeric.grad_fn = ctx.encoder_output_tensor.grad_fn;
        encoder_for_numeric.owns_grad_fn = false;  // ctx owns it
        
        // Call autograd numeric_head_forward
        ctx.numeric_head_output = numeric_head_forward(
            encoder_for_numeric,
            weights_tensor,
            bias_tensor_ptr,
            ctx.cublas_handle,
            ctx.stream
        );
        
        AG_INFO("Step 5: NumericHead complete (autograd), shape=[" << total_tokens << ", 1]");
    }
    
    AG_INFO("Forward complete: logits shape=[" << total_tokens << ", " << cfg->vocab_size << "]");
    
    result.success = true;
    return result;
}

//======================================================================
// Autograd Loss Computation
//======================================================================

LossResult computeAutogradLoss(
    AutogradContext& ctx,
    const int* targets,
    const float* valid_mask
) {
    LossResult result{};
    result.success = false;
    
    // RULE 20: Fail loud - validate context
    if (!ctx.isValid()) {
        throw std::runtime_error("computeAutogradLoss: Invalid context - missing required fields");
    }
    
    // RULE 20: Fail loud - validate logits tensor was populated by forward pass
    if (!ctx.logits_tensor.data) {
        throw std::runtime_error("computeAutogradLoss: Logits tensor not initialized - call executeAutogradForward() first");
    }
    
    // RULE 20: Fail loud - validate targets pointer
    if (!targets) {
        throw std::runtime_error("computeAutogradLoss: Targets pointer is NULL - caller MUST provide valid targets");
    }
    
    const int total_tokens = ctx.batch_size * ctx.seq_len;
    const int vocab_size = ctx.config->vocab_size;
    
    AG_INFO("Computing loss: tokens=" << total_tokens << " vocab=" << vocab_size);
    
    // Use our autograd cross-entropy loss
    // This returns a scalar Tensor with CrossEntropyLossGradFn attached
    Tensor loss_tensor = autograd::cross_entropy_loss(
        ctx.logits_tensor,
        targets,
        valid_mask,
        total_tokens,
        vocab_size,
        ctx.stream
    );
    
    // Move loss tensor to context (context owns it during backward)
    ctx.loss_tensor = std::move(loss_tensor);
    
    // Copy scalar loss to host for logging
    cudaMemcpyAsync(&result.loss_value, ctx.loss_tensor.data, sizeof(float), 
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    
    // Count valid tokens from mask (for logging and loss normalization)
    result.valid_tokens = countValidTokens(valid_mask, total_tokens, ctx.stream);
    
    // Create lightweight result view referencing context's tensor
    result.loss.data = ctx.loss_tensor.data;
    result.loss.shape = ctx.loss_tensor.shape;
    result.loss.requires_grad = ctx.loss_tensor.requires_grad;
    result.loss.is_leaf = ctx.loss_tensor.is_leaf;
    // CRITICAL: Do NOT share grad_fn - the context owns it.
    // Sharing would cause double-free when result is destroyed.
    result.loss.grad_fn = nullptr;  // Context owns the grad_fn
    result.loss.owns_data = false;
    // Note: owns_grad removed - ownership now managed by shared_ptr<Tensor> grad_
    
    AG_INFO("Loss computed: " << result.loss_value << " (valid_tokens=" << result.valid_tokens << ")");
    
    result.success = true;
    return result;
}

//======================================================================
// Autograd Backward Pass
//======================================================================

BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate
) {
    BackwardResult result{};
    result.success = false;
    result.grad_norm = 0.0f;
    
    if (!ctx.isValid()) {
        result.error_message = "Invalid context";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    
    if (!ctx.loss_tensor.data) {
        result.error_message = "Loss tensor not initialized (call computeLoss first)";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    
    if (!ctx.loss_tensor.grad_fn) {
        result.error_message = "Loss tensor has no grad_fn - autograd chain broken";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    
    AG_INFO("Executing backward pass (accumulate=" << accumulate << ", scale=" << ctx.grad_scale << ")");
    
    // Zero gradients if not accumulating
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (!accumulate) {
        auto* ts = ctx.training_state;
        
        // Zero all parameter gradients
        if (ts->embedding_weights.has_grad()) {
            cudaMemsetAsync(ts->embedding_weights.grad_data(), 0, 
                           ts->embedding_weights.numel() * sizeof(float), ctx.stream);
        }
        if (ts->position_embedding_weights.has_grad()) {
            cudaMemsetAsync(ts->position_embedding_weights.grad_data(), 0,
                           ts->position_embedding_weights.numel() * sizeof(float), ctx.stream);
        }
        if (ts->lm_head_weights.has_grad()) {
            cudaMemsetAsync(ts->lm_head_weights.grad_data(), 0,
                           ts->lm_head_weights.numel() * sizeof(float), ctx.stream);
        }
        if (ts->final_rms_gamma.has_grad()) {
            cudaMemsetAsync(ts->final_rms_gamma.grad_data(), 0,
                           ts->final_rms_gamma.numel() * sizeof(float), ctx.stream);
        }
        
        // NOTE: Encoder layer gradients are zeroed by the encoder itself during backward.
        // The encoder owns its weights/gradients internally, not TrainingState.
    }
    
    // Call backward on the loss tensor
    // This propagates through the entire computation graph via grad_fn nodes
    AG_INFO("Calling loss_tensor.backward()...");
    ctx.loss_tensor.backward();
    AG_INFO("loss_tensor.backward() returned successfully");
    
    // Copy gradients from Tensor.grad to TrainingState raw buffers
    // (for compatibility with optimizer that uses raw pointers)
    AG_INFO("Calling copyGradientsToTrainingState...");
    if (!copyGradientsToTrainingState(ctx)) {
        result.error_message = "Failed to copy gradients to TrainingState";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    AG_INFO("copyGradientsToTrainingState returned successfully");
    
    // Apply gradient scaling if needed
    // ISSUE #59: Use grad_data() accessor
    if (ctx.grad_scale != 1.0f) {
        auto* ts = ctx.training_state;
        const float scale = ctx.grad_scale;
        
        scaleGradBuffer(ts->embedding_weights.grad_data(), ts->embedding_weights.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->position_embedding_weights.grad_data(), ts->position_embedding_weights.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->lm_head_weights.grad_data(), ts->lm_head_weights.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->lm_head_bias.grad_data(), ts->lm_head_bias.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->numeric_head_weights.grad_data(), ts->numeric_head_weights.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->numeric_head_bias.grad_data(), ts->numeric_head_bias.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->final_rms_gamma.grad_data(), ts->final_rms_gamma.numel(), scale, ctx.stream);
        
        // NOTE: Encoder gradients are in encoder's internal Tensors.
        // The optimizer accesses them via enc->getAttnWqkvGrad() etc.
    }
    
    // Compute gradient norm
    result.grad_norm = computeGradientNorm(ctx);
    
    AG_INFO("Backward complete: grad_norm=" << result.grad_norm);
    
    result.success = true;
    return result;
}

//======================================================================
// Helper Functions
//======================================================================

bool copyGradientsToTrainingState(AutogradContext& ctx) {
    auto* ts = ctx.training_state;
    cudaStream_t stream = ctx.stream;
    
    // The autograd system stores gradients in Tensor.grad_ fields (shared_ptr<Tensor>)
    // The optimizer expects gradients in TrainingState raw buffers
    // 
    // CRITICAL: We MUST explicitly verify or copy gradients!
    // The "assume they're set up correctly" comment was WRONG and caused
    // frozen weights (gradients computed but never reaching optimizer).
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Embedding gradients (may be tied with LM head)
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ts->embedding_weights.data) {
        if (!ts->embedding_weights.has_grad()) {
            AG_WARN("embedding_weights.grad is NULL - gradients NOT flowing to optimizer!");
        } else {
            AG_INFO("Embedding gradients ready: " << ts->embedding_weights.numel() << " elements at " << ts->embedding_weights.grad_data());
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LM head gradients (may be tied to embedding)
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (ts->lm_head_weights.data) {
        if (!ts->lm_head_weights.has_grad()) {
            AG_WARN("lm_head_weights.grad is NULL - gradients NOT flowing to optimizer!");
        } else {
            // Check if tied to embedding (same underlying grad Tensor)
            if (ts->lm_head_weights.grad_data() == ts->embedding_weights.grad_data()) {
                AG_INFO("LM head gradients TIED to embedding: " << ts->lm_head_weights.numel() << " elements");
            } else {
                AG_INFO("LM head gradients SEPARATE: " << ts->lm_head_weights.numel() << " elements at " << ts->lm_head_weights.grad_data());
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Copy grad_logits from autograd tensor to legacy buffer
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (ctx.logits_tensor.has_grad() && ts->grad_logits_tensor.data) {
        const size_t logits_size = ctx.logits_tensor.numel() * sizeof(float);
        cudaMemcpyAsync(ts->grad_logits_tensor.data, ctx.logits_tensor.grad_data(), logits_size,
                        cudaMemcpyDeviceToDevice, stream);
        AG_INFO("Copied grad_logits: " << ctx.logits_tensor.numel() << " elements");
    }
    
    // NOTE: Encoder gradients are in encoder's internal Tensors, not TrainingState.
    // The optimizer accesses them via enc->getAttnWqkvGrad() etc.
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Final RMSNorm gamma
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ts->final_rms_gamma.data && !ts->final_rms_gamma.has_grad()) {
        AG_WARN("final_rms_gamma.grad is NULL - gradients NOT flowing!");
    }
    
    return true;
}

float computeGradientNorm(const AutogradContext& ctx) {
    auto* ts = ctx.training_state;
    
    // Allocate accumulator on device
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemsetAsync(d_sum, 0, sizeof(double), ctx.stream);
    
    // Accumulate sum of squared gradients from all parameter groups
    // Embeddings and LM head (may be tied)
    // ISSUE #59: Use grad_data() accessor
    computeSumSquared(ts->embedding_weights.grad_data(), ts->embedding_weights.numel(), d_sum, ctx.stream);
    if (ts->lm_head_weights.grad_data() != ts->embedding_weights.grad_data()) {
        // Only add if not tied (different pointers)
        computeSumSquared(ts->lm_head_weights.grad_data(), ts->lm_head_weights.numel(), d_sum, ctx.stream);
    }
    computeSumSquared(ts->position_embedding_weights.grad_data(), ts->position_embedding_weights.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->lm_head_bias.grad_data(), ts->lm_head_bias.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->final_rms_gamma.grad_data(), ts->final_rms_gamma.numel(), d_sum, ctx.stream);
    
    // NOTE: Encoder gradients need separate norm computation from encoder directly.
    // The optimizer accesses encoder gradients via enc->getAttnWqkvGrad() etc.
    // For now, we compute norm only from TrainingState-owned Tensors.
    // TODO: If needed, add gradient norm computation from encoder's internal Tensors.
    
    // Copy result to host and compute sqrt
    double h_sum = 0.0;
    cudaMemcpyAsync(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    cudaFree(d_sum);
    
    return static_cast<float>(std::sqrt(h_sum));
}

//======================================================================
// Main Entry Point
//======================================================================

float autogradTrainingStep(
    LanguageModel& model,
    TrainingState& training_state,
    int batch_size,
    int seq_len,
    bool accumulate,
    float grad_scale,
    uint64_t step
) {
    const auto& cfg = model.getConfig();
    
    // Get encoder for autograd forward
    GPUGrimEncoder* gpu_encoder = nullptr;
    try {
        gpu_encoder = &model.getGpuEncoder();
    } catch (...) {
        AG_ERROR("autogradTrainingStep: Failed to get encoder from model");
        return -1.0f;
    }
    
    if (!gpu_encoder) {
        AG_ERROR("autogradTrainingStep: Encoder is NULL");
        return -1.0f;
    }
    
    // Get ScratchBlock (optional - nullptr if not enabled)
    ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
    
    // NOTE: Encoder owns its weights internally; optimizer accesses gradients
    // via enc->getAttnWqkvGrad(), enc->getFFNW1Grad(), etc.
    // No linking needed - see buildParameterGroups() in LanguageModel_Training.cu
    
    // Initialize context
    AutogradContext ctx = initAutogradContext(
        &cfg,
        &training_state,
        gpu_encoder,
        scratch_block,
        training_state.cublas_handle,
        training_state.stream_ctrl.getPrimaryStream(),
        batch_size,
        seq_len,
        grad_scale,
        step
    );
    
    if (!ctx.isValid()) {
        AG_ERROR("autogradTrainingStep: Failed to initialize context");
        return -1.0f;
    }
    
    // Set ScratchBlock input buffers from TrainingState
    ctx.token_numeric_values = training_state.cached_token_numeric_values;
    ctx.token_numeric_mask = training_state.cached_token_numeric_mask;
    ctx.token_text_features = training_state.cached_token_text_features;
    ctx.token_text_mask = training_state.cached_token_text_mask;
    
    // Forward pass (runs entire model with autograd graph)
    ForwardResult fwd_result = executeAutogradForward(ctx);
    if (!fwd_result.success) {
        AG_ERROR("autogradTrainingStep: Forward failed - " << fwd_result.error_message);
        return -1.0f;
    }
    
    // Loss computation
    LossResult loss_result = computeAutogradLoss(ctx, training_state.cached_targets, nullptr);
    if (!loss_result.success) {
        AG_ERROR("autogradTrainingStep: Loss failed - " << loss_result.error_message);
        return -1.0f;
    }
    
    // Backward pass (propagates through entire graph)
    BackwardResult bwd_result = executeAutogradBackward(ctx, accumulate);
    if (!bwd_result.success) {
        AG_ERROR("autogradTrainingStep: Backward failed - " << bwd_result.error_message);
        return -1.0f;
    }
    
    // Clear intermediate tensors to free memory
    ctx.clearIntermediates();
    
    AG_INFO("Training step complete: loss=" << loss_result.loss_value 
            << " grad_norm=" << bwd_result.grad_norm);
    
    return loss_result.loss_value;
}

}  // namespace Autograd
}  // namespace GRIM
