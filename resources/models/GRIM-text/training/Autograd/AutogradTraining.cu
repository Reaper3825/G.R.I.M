//======================================================//
//  AutogradTraining.cu
//  Implementation of autograd-based training flow
//======================================================//

#include "AutogradTraining.hpp"

// MUST include full definition of GPUGrimEncoder for method calls
#include "../../GRIM/grim_language_model_cuda.hpp"
// RMSNorm_Kernel_GPU.hpp removed - using autograd::rms_norm in TensorContract_GPU instead
#include "../../Layers/grim_layer_gpu.hpp"                    // LayerWorkspace
#include "../../Layers/Encoding/Encoding_GPU.hpp"             // EncodingLayer::useExternalWeights
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"     // ScratchBlockLayer
#include "../../Layers/NumericHead/numeric_head_GPU.hpp"      // NumericHead forward/backward
#include "../../Layers/LMHead/lm_head_GPU.hpp"                // Issue #37/#43: launchCenterHiddenStates
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/TrainingState/TrainingTensors.hpp"     // GRIM::TrainingTensors definition

#include <iostream>
#include <cmath>
#include <algorithm>  // Rule 21 diagnostic: std::min_element, std::max_element
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../../Shared/Loss/NumericLoss/NumericLoss_GPU.hpp"  // launchNumericLoss, NumericLossInputs/Outputs
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
// Tensor& accessors (enc->attnWqkv().grad_data() etc.).
// See buildParameterGroups() in LanguageModel_Training.cu.

//======================================================================
// Context Initialization
//======================================================================

// Training overload — derives batch geometry from BatchPayload
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    ScratchBlockLayer* scratch_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const Batching::BatchPayload& payload,
    float grad_scale,
    uint64_t step,
    bool is_training
) {
    AutogradContext ctx{};
    ctx.config = config;
    ctx.training_state = training_state;
    ctx.gpu_encoder = gpu_encoder;
    ctx.scratch_block = scratch_block;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.payload = &payload;
    ctx.batch_size = payload.batch_size;
    ctx.seq_len = payload.max_seq_len;
    ctx.grad_scale = grad_scale;
    ctx.step = step;
    ctx.is_training = is_training;
    // Default ScratchBlock side-channel pointers from TrainingState caches.
    // Callers may overwrite these if they stage alternate buffers.
    ctx.token_numeric_values = training_state ? training_state->cached_token_numeric_values.data : nullptr;
    ctx.token_numeric_mask = training_state
        ? reinterpret_cast<const uint8_t*>(training_state->cached_token_numeric_mask.data)
        : nullptr;
    ctx.token_text_features = training_state
        ? reinterpret_cast<const uint16_t*>(training_state->cached_token_text_features.data)
        : nullptr;
    ctx.token_text_mask = training_state
        ? reinterpret_cast<const uint8_t*>(training_state->cached_token_text_mask.data)
        : nullptr;
    
    // Rule 20: Fail loud on invalid context
    ctx.validate("initAutogradContext(payload)");
    
    return ctx;
}

// Inference overload — batch_size/seq_len set directly (no payload)
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
    uint64_t step,
    bool is_training
) {
    AutogradContext ctx{};
    ctx.config = config;
    ctx.training_state = training_state;
    ctx.gpu_encoder = gpu_encoder;
    ctx.scratch_block = scratch_block;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.payload = nullptr;  // No payload for inference
    ctx.batch_size = batch_size;
    ctx.seq_len = seq_len;
    ctx.grad_scale = grad_scale;
    ctx.step = step;
    ctx.is_training = is_training;
    // Inference/sample paths also use ScratchBlock forward, which requires
    // numeric side-channel pointers when ScratchBlock is enabled.
    ctx.token_numeric_values = training_state ? training_state->cached_token_numeric_values.data : nullptr;
    ctx.token_numeric_mask = training_state
        ? reinterpret_cast<const uint8_t*>(training_state->cached_token_numeric_mask.data)
        : nullptr;
    ctx.token_text_features = training_state
        ? reinterpret_cast<const uint16_t*>(training_state->cached_token_text_features.data)
        : nullptr;
    ctx.token_text_mask = training_state
        ? reinterpret_cast<const uint8_t*>(training_state->cached_token_text_mask.data)
        : nullptr;
    
    // Rule 20: Fail loud on invalid context
    ctx.validate("initAutogradContext(inference)");
    
    return ctx;
}
  
//======================================================================
// Autograd Forward Pass
// PRODUCTION-READY: Runs entire model with autograd graph intact
//======================================================================

ForwardResult executeAutogradForward(AutogradContext& ctx) {
    ForwardResult result{};
    result.success = false;
    
    // Rule 20: Fail loud
    ctx.validate("executeAutogradForward");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    auto& intermediates = ts->autograd_intermediates;  // All intermediate tensors go HERE
    
    const int total_tokens = ctx.batch_size * ctx.seq_len;
    result.total_tokens = total_tokens;
    result.vocab_size = cfg->vocab_size;

    // Numeric head gradients come from an external loss kernel into grad_numeric_tensor.
    // Clear the active token window each forward so stale values never leak across steps.
    if (cfg->numeric_head_enabled && ts->grad_numeric_tensor.data) {
        const size_t numeric_elems = static_cast<size_t>(total_tokens);
        if (ts->grad_numeric_tensor.numel() < numeric_elems) {
            throw std::runtime_error("AutogradForward: grad_numeric_tensor capacity too small for batch");
        }
        cudaMemsetAsync(ts->grad_numeric_tensor.data, 0, numeric_elems * sizeof(float), ctx.stream);
    }
    
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
    // NOTE: cached_token_ids_tensor stores int32 data in float* buffer - cast when accessing
    int* token_ids = reinterpret_cast<int*>(ts->cached_token_ids_tensor.data);
    if (!token_ids) {
        throw std::runtime_error("AutogradForward: cached_token_ids_tensor.data is NULL");
    }
    
    // Embedding weights tensor (owned by TrainingTensors)
    Tensor& emb_weights = ts->tensors_->embedding_weights;
    if (!emb_weights.data) {
        throw std::runtime_error("AutogradForward: embedding_weights.data is NULL");
    }
    emb_weights.requires_grad = true;
    
    // Rule 20: Fail loud on invalid shape - caller MUST initialize correctly
    if (!emb_weights.shape.is_valid()) {
        throw std::runtime_error("[AutogradTraining] embedding_weights.shape is INVALID - caller MUST initialize with correct shape [vocab_size=" 
                                + std::to_string(cfg->vocab_size) + ", d_model=" + std::to_string(cfg->d_model) + "]");
    }
    
    // Use autograd::embedding for proper gradient tracking
    // This performs: output[i] = weight[token_ids[i]] * scale with gradient scatter-add backward
    //
    // Issue #140: REMOVED sqrt(d_model) embedding scaling.
    // AIAYN's sqrt(d_model) was designed for sinusoidal position encodings in the residual stream.
    // GRIM-text uses ALiBi/RoPE (position info INSIDE attention, NOT residual stream).
    // With tied weights, the 27.7x scaling creates gradient asymmetry:
    //   - Embedding backward: grad_W[tok] += grad_encoder[t] * 27.7  (amplified)
    //   - LM head backward:   grad_W = centered^T @ grad_logits      (raw, no scaling)
    // This asymmetry caused non-deterministic embedding gradient spikes (0.5 → 5.2)
    // because atomicAdd scatter order varies per run, and the 27.7x amplification
    // makes small ordering differences into large gradient magnitude differences.
    // Modern LLMs with tied weights (GPT-2, LLaMA, Mistral, Gemma) do NOT scale.
    const float embedding_scale = 1.0f;
    Tensor emb_output = autograd::embedding(
        emb_weights,
        token_ids,  // Use local variable from Tensor cast
        total_tokens,
        ctx.stream,
        embedding_scale  // Issue #140: No scaling (1.0f)
    );
    
    // Store embedding output in intermediates (keeps autograd graph alive)
    // Issue #57 FIX: Add position embeddings
    // PyTorch baseline: x = tok_emb->forward(idx) + pos_emb->forward(pos)
    // GRIM was MISSING this step, causing training plateau!
    //
    // Issue #96/103 FIX: ONLY add position embeddings for LEARNED positional encoding!
    // With ALIBI or ROPE, position embeddings are ISOTROPIC (all columns have same variance)
    // which causes GEMM coherent summation and QKV explosion.
    // Config says use_learned=false but code was ignoring it!
    // ═══════════════════════════════════════════════════════════════════════════
    const bool use_learned_pos_emb = (cfg->positional_encoding == HyperParameters::PositionalEncodingType::NONE);
    if (use_learned_pos_emb) {
        if (!ts->tensors_->position_embedding_weights.data) {
            throw std::runtime_error(
                "AutogradForward: positional_encoding=NONE requires position_embedding_weights.data, but it is NULL");
        }
        ts->tensors_->position_embedding_weights.requires_grad = true;
        
        // Ensure position embedding weights have correct shape [max_seq_len, d_model]
        if (!ts->tensors_->position_embedding_weights.shape.is_valid()) {
            ts->tensors_->position_embedding_weights.shape = TensorContract::TensorShape::make_BSM(
                cfg->max_seq_len, cfg->d_model);
        }
        
        // Allocate temporary buffer for position IDs on device
        int* d_position_ids = nullptr;
        cudaMallocAsync(&d_position_ids, total_tokens * sizeof(int), ctx.stream);
        
        // Generate position IDs: [0,1,2,...,seq_len-1] repeated for each batch
        generatePositionIds(d_position_ids, total_tokens, ctx.seq_len, ctx.stream);
        
        // Look up position embeddings with autograd tracking
        // Issue #140: Same scale as token embeddings (1.0f — no scaling)
        Tensor pos_emb_output = autograd::embedding(
            ts->tensors_->position_embedding_weights,
            d_position_ids,
            total_tokens,
            ctx.stream,
            embedding_scale  // Issue #140: No scaling (1.0f)
        );
        
        // Free temporary position IDs (embedding lookup already copied them)
        cudaFreeAsync(d_position_ids, ctx.stream);
        
        // Add token embeddings + position embeddings (both tracked by autograd)
        emb_output = autograd::add(emb_output, pos_emb_output, ctx.stream);
        
        AG_INFO("Step 1b: Position embeddings added (Issue #57 FIX)");
    } else {
        // ALiBi/RoPE: No position embedding added to residual stream.
        // Position information is injected directly inside attention via bias/rotary.
        AG_INFO("Step 1b: No position embeddings (using " 
                << HyperParameters::positionalEncodingTypeToString(cfg->positional_encoding)
                << " inside attention)");
    }
    
    // Store in intermediates for backward
    intermediates.embedding_tensor = std::move(emb_output);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  EMBEDDING DROPOUT (Issue #133)
    //  
    //  Apply dropout to embeddings BEFORE encoder layers. This breaks symmetry
    //  between hidden states and prevents mode collapse to dominant tokens.
    //  PyTorch transformers do this - we weren't, which caused mode collapse.
    // ═══════════════════════════════════════════════════════════════════════════
    if (cfg->dropout_rate > 0.0f) {
        // Vary seed per step so each batch sees a DIFFERENT dropout mask.
        const uint64_t emb_dropout_seed = ctx.step * 2654435761ULL + 500;
        intermediates.embedding_tensor = autograd::dropout(intermediates.embedding_tensor, cfg->dropout_rate, 
                                                  emb_dropout_seed, ctx.is_training, ctx.stream);
        AG_INFO("Step 1c: Embedding dropout " << (ctx.is_training ? "applied" : "skipped (eval mode)")
                << " (p=" << cfg->dropout_rate << ", step=" << ctx.step << ")");
    }
    
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
        
        // ═══════════════════════════════════════════════════════════════════════
        //  ISSUE #90 FIX: Operate directly on intermediates.embedding_tensor.data!
        //  ScratchBlock modifies the autograd buffer in-place.
        // ═══════════════════════════════════════════════════════════════════════
        sb_args.input = TensorContract::TensorView::make_BSM(
            intermediates.embedding_tensor.data, total_tokens, cfg->d_model, "sb_input");
        sb_args.output = TensorContract::TensorView::make_BSM(
            intermediates.embedding_tensor.data, total_tokens, cfg->d_model, "sb_output");
        
        sb_args.total_tokens = total_tokens;
        sb_args.seq_len = ctx.seq_len;
        sb_args.token_ids = reinterpret_cast<int*>(ts->cached_token_ids_tensor.data);
        
        // Numeric values and mask from DataLoader (passed via context)
        sb_args.token_numeric_values = ctx.token_numeric_values;
        sb_args.token_numeric_mask = ctx.token_numeric_mask;
        
        // GRMT v4: text features for atom injection
        sb_args.token_text_features = ctx.token_text_features;
        sb_args.token_text_mask = ctx.token_text_mask;
        
        sb_args.stream = ctx.stream;
        
        // Cache atom embeddings for backward
        if (ts->cached_scratch_block_embeddings.data) {
            sb_args.cache_atom_embeddings = TensorContract::TensorView::make_BSM(
                ts->cached_scratch_block_embeddings.data,
                ctx.scratch_block->config().max_atoms,
                ctx.scratch_block->config().atom_embedding_dim,
                "sb_cache_embeddings");
        }
        sb_args.cache_atom_positions = reinterpret_cast<int*>(ts->cached_scratch_block_positions.data);
        sb_args.cache_atom_types = reinterpret_cast<int*>(ts->cached_scratch_block_types.data);
        sb_args.cache_num_atoms = reinterpret_cast<int*>(ts->cached_scratch_block_num_atoms.data);
        
        // Run ScratchBlock forward (operates directly on intermediates.embedding_tensor.data)
        ctx.scratch_block->forward(sb_args);
        
        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("AutogradForward: ScratchBlock forward CUDA error: " + 
                                     std::string(cudaGetErrorString(cuda_err)));
        }
        
        AG_INFO("Step 1.5: ScratchBlock complete");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #91 DIAGNOSTIC: Dump embedding stats AFTER ScratchBlock, BEFORE encoder
    // (GUARDED: set ENABLE_EXPENSIVE_DIAGNOSTICS=true in VerboseLogging.hpp to enable)
    // ═══════════════════════════════════════════════════════════════════════════
    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        // This copies ~22MB to host - expensive!
        const int full_size = total_tokens * cfg->d_model;
        std::vector<float> h_emb(full_size);
        cudaMemcpy(h_emb.data(), intermediates.embedding_tensor.data, full_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        float emb_min = h_emb[0], emb_max = h_emb[0];
        double emb_sum = 0.0, emb_sum_sq = 0.0;
        for (int i = 0; i < full_size; i++) {
            emb_min = std::min(emb_min, h_emb[i]);
            emb_max = std::max(emb_max, h_emb[i]);
            emb_sum += h_emb[i];
            emb_sum_sq += h_emb[i] * h_emb[i];
        }
        float emb_mean = emb_sum / full_size;
        float emb_rms = sqrtf(emb_sum_sq / full_size);
        
        fprintf(stderr, "[Issue91-EMB-AFTER-SB] tokens=%d d_model=%d: min=%.10f max=%.10f mean=%.10f rms=%.10f\n",
                total_tokens, cfg->d_model, emb_min, emb_max, emb_mean, emb_rms);
        
        // ═══════════════════════════════════════════════════════════════════════
        // RULE 21 DIAGNOSTIC: Embedding Cosine Similarity (BEFORE encoder layers)
        // 
        // This measures how correlated token representations are BEFORE any
        // transformer layers process them. Without additive position embeddings
        // (Issue #103), same tokens at different positions have IDENTICAL
        // embeddings, causing avg_cos to approach 1.0 as token repetition increases.
        //
        // EQUATION: cosine(h_i, h_j) = (h_i · h_j) / (||h_i|| * ||h_j||)
        // EXPECTED: avg_cos ≈ 1/sqrt(d_model) ≈ 0.036 for random orthogonal vectors
        // ANOMALY: avg_cos > 0.5 indicates high correlation (representational collapse)
        // ═══════════════════════════════════════════════════════════════════════
        {
            const int d_model = cfg->d_model;
            const int sample_pairs = std::min(50, total_tokens / 2);  // Sample pairs for efficiency
            
            if (sample_pairs >= 2) {
                // Compute norms for each position
                std::vector<float> row_norms(total_tokens);
                for (int t = 0; t < total_tokens; t++) {
                    double norm_sq = 0.0;
                    for (int d = 0; d < d_model; d++) {
                        float v = h_emb[t * d_model + d];
                        norm_sq += v * v;
                    }
                    row_norms[t] = sqrtf(norm_sq);
                }
                
                // Compute pairwise cosine similarity for sampled pairs
                double cos_sum = 0.0;
                double cos_min = 2.0, cos_max = -2.0;
                int num_pairs = 0;
                int identical_token_pairs = 0;
                double identical_cos_sum = 0.0;
                
                // Sample evenly-spaced pairs throughout the sequence
                const int stride = std::max(1, total_tokens / sample_pairs);
                for (int i = 0; i < total_tokens && num_pairs < sample_pairs; i += stride) {
                    int j = (i + total_tokens / 2) % total_tokens;  // Pair with distant position
                    if (i == j || row_norms[i] < 1e-8f || row_norms[j] < 1e-8f) continue;
                    
                    // Compute dot product h_i · h_j
                    double dot = 0.0;
                    for (int d = 0; d < d_model; d++) {
                        dot += h_emb[i * d_model + d] * h_emb[j * d_model + d];
                    }
                    
                    double cosine = dot / (row_norms[i] * row_norms[j]);
                    cos_sum += cosine;
                    cos_min = std::min(cos_min, cosine);
                    cos_max = std::max(cos_max, cosine);
                    num_pairs++;
                    
                    // Track identical token pairs (from cached_token_ids_tensor if available)
                    if (ts->cached_token_ids_tensor.data) {
                        std::vector<int> h_tok_ids(total_tokens);
                        cudaMemcpy(h_tok_ids.data(), ts->cached_token_ids_tensor.data, 
                                   total_tokens * sizeof(int), cudaMemcpyDeviceToHost);
                        if (h_tok_ids[i] == h_tok_ids[j]) {
                            identical_token_pairs++;
                            identical_cos_sum += cosine;
                        }
                    }
                }
                
                const double avg_cos = (num_pairs > 0) ? cos_sum / num_pairs : 0.0;
                const double expected_cos = 1.0 / sqrt(static_cast<double>(d_model));  // ~0.036 for d=768
                
                fprintf(stderr, "[EMBED_COSINE_EQUATION] BEFORE_ENCODER: cosine(h_i, h_j) = (h_i · h_j) / (||h_i|| * ||h_j||)\n");
                fprintf(stderr, "  INPUT h (embeddings): shape=[%d, %d] row_norm_range=[%.6f, %.6f]\n",
                        total_tokens, d_model, 
                        *std::min_element(row_norms.begin(), row_norms.end()),
                        *std::max_element(row_norms.begin(), row_norms.end()));
                fprintf(stderr, "  PARAMETERS: sample_pairs=%d, stride=%d\n", num_pairs, stride);
                fprintf(stderr, "  EXPECTED avg_cos = 1/sqrt(%d) = %.6f (for random orthogonal vectors)\n", 
                        d_model, expected_cos);
                fprintf(stderr, "  ACTUAL avg_cos=%.6f min=%.6f max=%.6f\n", avg_cos, cos_min, cos_max);
                
                if (identical_token_pairs > 0) {
                    double avg_identical_cos = identical_cos_sum / identical_token_pairs;
                    fprintf(stderr, "  IDENTICAL_TOKEN_PAIRS: %d/%d pairs, avg_cos=%.6f\n",
                            identical_token_pairs, num_pairs, avg_identical_cos);
                    if (avg_identical_cos > 0.99) {
                        fprintf(stderr, "  [ANOMALY] Same tokens have cosine≈1.0 - NO position differentiation!\n");
                        fprintf(stderr, "  [ANOMALY] Without additive position embeddings, same tokens are IDENTICAL\n");
                    }
                }
                
                if (avg_cos > 0.5) {
                    fprintf(stderr, "  [ANOMALY] avg_cos=%.6f >> expected=%.6f (%.1fx larger!)\n",
                            avg_cos, expected_cos, avg_cos / expected_cos);
                    fprintf(stderr, "  [ANOMALY] High embedding correlation BEFORE encoder = representational collapse!\n");
                    fprintf(stderr, "  [ANOMALY] Root cause: Without additive pos_emb, same tokens have IDENTICAL representations\n");
                }
            }
        }
    }  // end ENABLE_EXPENSIVE_DIAGNOSTICS guard (Issue #91 embedding stats)

    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 2: Encoder Layers (transformer blocks)
    //  Encoder layer outputs stored in intermediates to keep autograd graph alive.
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (!ctx.gpu_encoder) {
        throw std::runtime_error("AutogradForward: gpu_encoder is NULL - pass encoder in context");
    }
    if (!ts->encoder_workspace.data) {
        throw std::runtime_error("AutogradForward: encoder_workspace is NULL - TrainingState MUST allocate workspace");
    }
    
    const int num_layers = ctx.gpu_encoder->getNumLayers();
    intermediates.encoder_layer_outputs.clear();
    intermediates.encoder_layer_outputs.reserve(num_layers);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  Issue #56: Keep all intermediate tensors alive until backward completes.
    // ═══════════════════════════════════════════════════════════════════════════
    intermediates.layer_intermediates.layers.clear();
    intermediates.layer_intermediates.layers.reserve(num_layers);
    
    intermediates.embedding_tensor.is_leaf = false;
    
    AG_INFO("Step 2: Running " << num_layers << " encoder layers with autograd...");
    AG_INFO("  embedding_tensor.grad_fn=" << (void*)intermediates.embedding_tensor.grad_fn.get() 
            << " requires_grad=" << intermediates.embedding_tensor.requires_grad);
    
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
        if (!enc_layer) {
            throw std::runtime_error("AutogradForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
        }
        
        // Setup workspace
        LayerWorkspace<float> ws{};
        ws.data = ts->encoder_workspace.data;
        ws.bytes = enc_layer->requiredWorkspaceBytes(total_tokens, ctx.seq_len);
        enc_layer->setWorkspace(ws.data, ws.bytes);
        
        // ═══════════════════════════════════════════════════════════════════════
        //  Issue #56 FIX: Create intermediates storage and pass to forward()
        //  
        //  forward() stores all intermediate tensors in this struct instead of
        //  local variables. This keeps the autograd graph alive until backward.
        // ═══════════════════════════════════════════════════════════════════════
        intermediates.layer_intermediates.layers.emplace_back();
        ForwardIntermediates& layer_storage = intermediates.layer_intermediates.layers.back();
        
        // First layer uses embedding_tensor directly (preserves grad_fn chain)
        // Subsequent layers use previous layer output
        Tensor& layer_input = (layer_idx == 0) 
            ? intermediates.embedding_tensor 
            : intermediates.encoder_layer_outputs.back();
        
        // Run layer forward with intermediates storage
        // Pass ctx.step for per-step attention dropout seed generation
        Tensor layer_output = enc_layer->forward(layer_input, ctx.seq_len, ctx.stream, layer_storage, ctx.step);
        
        // ═══════════════════════════════════════════════════════════════════════
        //  ENCODER LAYER DROPOUT (Issue #133)
        //  
        //  Apply dropout after each encoder layer. This is standard practice in
        //  transformer implementations and prevents mode collapse.
        // ═══════════════════════════════════════════════════════════════════════
        if (cfg->dropout_rate > 0.0f) {
            // Per-step dropout seed: varies each batch to ensure different dropout masks.
            // Fixed seed = same dropped features every batch = encoder learns around it = NOT real dropout.
            // Combine ctx.step (varies per batch) with layer offset (unique per layer) for maximum diversity.
            const uint64_t layer_dropout_seed = ctx.step * 2654435761ULL + (42 + 500 + layer_idx);
            layer_output = autograd::dropout(layer_output, cfg->dropout_rate,
                                             layer_dropout_seed, ctx.is_training, ctx.stream);
        }
        
        intermediates.encoder_layer_outputs.push_back(std::move(layer_output));
    }
    
    AG_INFO("Step 2: All " << num_layers << " encoder layers complete");
    
    // Final encoder output is the last layer's output
    float* encoder_output = intermediates.encoder_layer_outputs.back().data;
    result.encoder_output = encoder_output;
    
    // Copy to scratch buffer for diagnostics and inference
    if (ts->cached_encoder_output.data) {
        cudaMemcpyAsync(ts->cached_encoder_output.data, encoder_output,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 3: Final RMSNorm (before LM head)
    //  Uses autograd::rms_norm for gradient tracking
    // ═══════════════════════════════════════════════════════════════════════════
    
    Tensor normalized_output;
    if (ts->tensors_->final_rms_gamma.data) {
        ts->tensors_->final_rms_gamma.requires_grad = true;
        
        // Create input tensor from encoder output
        Tensor rms_input = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,
            true,
            "final_rms_input"
        );
        rms_input.is_leaf = false;
        rms_input.stream = ctx.stream;
        rms_input.grad_fn = intermediates.encoder_layer_outputs.back().grad_fn;
        
        // Apply autograd RMSNorm
        normalized_output = autograd::rms_norm(rms_input, ts->tensors_->final_rms_gamma, 
                                               cfg->rms_epsilon, ctx.stream);
        
        encoder_output = normalized_output.data;
        AG_INFO("Step 3: Final RMSNorm applied with autograd");
    } else {
        // No final RMSNorm, use encoder output directly
        normalized_output = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false, true, "encoder_output_passthrough"
        );
        normalized_output.is_leaf = false;
        normalized_output.stream = ctx.stream;
        normalized_output.grad_fn = intermediates.encoder_layer_outputs.back().grad_fn;
    }
    
    // Store for backward
    intermediates.encoder_output_tensor = std::move(normalized_output);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 4: LM Head → Logits
    //  Input: encoder_output [total_tokens, d_model]
    //  Weights: lm_head_weights [vocab_size, d_model]
    //  Output: logits [total_tokens, vocab_size]
    //   
    //  Using autograd::matmul for proper gradient tracking through the
    //  entire computation graph (encoder → final_rms → lm_head → loss)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // ════════════════════════════════════════════════════════════════════════
    // ISSUE #87 FIX: Use SAME Tensor object for tied weights!
    //
    // PyTorch uses ONE tensor (tok_emb->weight) for both embedding lookup AND
    // LM head matmul. Its autograd sees one tensor used twice and handles
    // gradient accumulation naturally.
    //
    // GRIM was using TWO Tensor objects (embedding_weights, lm_head_weights)
    // that happened to share data pointers. But they had separate grad_fn
    // chains, so autograd didn't know they were the same tensor!
    //
    // FIX: When tie_embeddings=true, use embedding_weights for BOTH operations.
    // This makes GRIM's autograd behave like PyTorch's.
    // ════════════════════════════════════════════════════════════════════════
    Tensor& lm_weights = cfg->tie_embeddings ? ts->tensors_->embedding_weights : ts->tensors_->lm_head_weights;
    lm_weights.requires_grad = true;  // Gradients for weight update

    // Issue #142: set_lm_head_grad_correction() removed.
    // Centering backward is handled by CenterRowsGradFn/CenterColumnsGradFn
    // inside the autograd graph (Issues #125/#132). No external correction needed.
    
    // RULE 20: Fail loud - validate cached_logits buffer
    float* logits_output = ts->cached_logits_tensor.data;
    if (!logits_output) {
        throw std::runtime_error("AutogradTraining: cached_logits_tensor buffer is NULL - TrainingState MUST allocate logits buffer");
    }
     
    // ════════════════════════════════════════════════════════════════════════
    // ISSUE #37/#43 FIX: Center hidden states BEFORE LM head projection!
    //
    // The autograd path was missing centering, causing mode collapse.
    // Without centering, non-zero mean hidden states create systematic bias
    // in weight gradients: negative_mean × negative_grad = POSITIVE update!
    // This caused Token 277 (SPACE) logits to explode: -0.08 → 3.11 in 3 batches.
    //
    // NOTE: Centering must happen BEFORE lm_input_tensor is created because
    // autograd::matmul will cache the input pointer for backward pass.
    // ════════════════════════════════════════════════════════════════════════
    float* lm_input_ptr = intermediates.encoder_output_tensor.data;  // Default: use encoder output directly
    
    // Check if centering is enabled in config and scratch buffer is available
    const bool use_centering = cfg->lm_head_center_hidden_states;
    
    if (use_centering) {
        // ISSUE #125 FIX: Column centering - removes common direction, reduces cos(h_i, h_j)
        // center_columns: centered[t,d] = hidden[t,d] - mean_t(hidden[:,d])
        // Result: Σ_t h[t,d] = 0 for each feature d
        intermediates.centered_encoder_output = autograd::center_columns(intermediates.encoder_output_tensor, ctx.stream);
        
        // ISSUE #132 FIX: Row centering - eliminates gradient sign flip!
        // center_rows: centered[t,d] = hidden[t,d] - mean_d(hidden[t,:])
        // Result: Σ_d h[t,d] = 0 for each position t
        // 
        // WHY: Gradient sign flip occurs when hidden_sum[t] × grad[t,277] creates
        // wrong-sign contributions. With row centering, hidden_sum[t] = 0 for all t,
        // eliminating the mechanism entirely.
        intermediates.centered_encoder_output = autograd::center_rows(intermediates.centered_encoder_output, ctx.stream);
        
        lm_input_ptr = intermediates.centered_encoder_output.data;
        
        // Also copy to centering_scratch_tensor for diagnostic reads
        if (ts->centering_scratch_tensor.data) {
            cudaMemcpyAsync(ts->centering_scratch_tensor.data, lm_input_ptr,
                           static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                           cudaMemcpyDeviceToDevice, ctx.stream);
        }
        
        AG_INFO("Step 4a: Hidden states COLUMN+ROW centered (Issue #125 + #132 fix)");
    }
    
    // Create input tensor referencing the (possibly centered) data
    // Link grad_fn to continue the backward chain
    // ISSUE #48: Store in intermediates so pointer remains valid until backward completes
    intermediates.lm_input_tensor = Tensor::from_ptr(
        lm_input_ptr,  // May point to centered scratch or raw encoder output
        TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
        false,  // doesn't own data
        true,    // requires_grad
        "lm_input"
    );
    intermediates.lm_input_tensor.is_leaf = false;
    intermediates.lm_input_tensor.stream = ctx.stream;
    
    // Link correct grad_fn based on whether centering is enabled
    if (use_centering) {
        intermediates.lm_input_tensor.grad_fn = intermediates.centered_encoder_output.grad_fn;
    } else {
        intermediates.lm_input_tensor.grad_fn = intermediates.encoder_output_tensor.grad_fn;
    }
    
    // RULE 20: Validate shapes
    intermediates.lm_input_tensor.shape.require("lm_input");
    lm_weights.shape = TensorContract::TensorShape::make_BSM(cfg->vocab_size, cfg->d_model);
    lm_weights.shape.require("lm_weights");
    
    // Execute autograd matmul: logits = centered_encoder @ lm_weights^T
    Tensor logits_tensor = autograd::matmul(
        intermediates.lm_input_tensor,
        lm_weights,
        ctx.stream,
        nullptr,
        nullptr,
        true      // transpose_b=true
    );
    
    // Issue #133: NO logit scaling. Softmax gradient (p - y) at uniform ≈ -1
    // regardless of logit magnitude. Scaling just amplifies backward by 27.7x.
    // The real fix is config: disable entropy_reg, reduce weight_decay, disable focal.
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  LOGIT CENTERING: Subtract mean across vocab for each position
    //  
    //  Forward:  centered[t,v] = logit[t,v] - mean_v(logit[t,:])
    //  Backward: grad_logit = grad_centered (centering is linear, same grad flows back)
    //  
    //  Mathematical effect:
    //  - Softmax is SHIFT-INVARIANT: softmax(x - c) = softmax(x) for any constant c
    //  - Therefore centering does NOT change predictions or loss
    //  - BUT it improves numerical stability by keeping logits near zero
    //  - AND it may help with gradient flow by removing DC bias
    //  
    //  Shape: [total_tokens, vocab_size] → each row centered to mean=0
    // ═══════════════════════════════════════════════════════════════════════════
    if (cfg->center_logits) {
        logits_tensor = autograd::center_rows(logits_tensor, ctx.stream);
        AG_INFO("Step 4a: Logits ROW-centered (each position's logit mean → 0)");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  RULE 21 DIAGNOSTIC: Why is token 277 (or any token) the argmax?
    //  (GUARDED: set ENABLE_EXPENSIVE_DIAGNOSTICS=true in VerboseLogging.hpp to enable)
    // ═══════════════════════════════════════════════════════════════════════════
    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        //  Equation: logit[v] = Σ_d encoder[pos, d] × W[v, d]
        //  
        //  For argmax analysis: compare logit[277] vs logit[other tokens]
        //  The argmax wins because: Σ_d h[d] × W[argmax, d] > Σ_d h[d] × W[v, d] for all v
        //  
        //  This diagnostic shows:
        //  1. Hidden state (encoder output) statistics at sample position
        //  2. Weight row statistics for top predicted tokens
        //  3. Dot product decomposition showing WHY argmax wins
        constexpr int kAnalysisToken = 277;
        constexpr int kSamplePositions = 5;  // Sample first 5 positions
        const int d_model = cfg->d_model;
        const int vocab_size_local = cfg->vocab_size;
        
        // Copy sample data to host
        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_weights_277(d_model);
        std::vector<float> h_logits(sample_size * vocab_size_local);
        
        cudaMemcpyAsync(h_encoder.data(), lm_input_ptr,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_weights_277.data(), 
                        lm_weights.data + static_cast<size_t>(kAnalysisToken) * d_model,
                        d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_logits.data(), logits_tensor.data,
                        sample_size * vocab_size_local * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        
        for (int pos = 0; pos < sample_size; ++pos) {
            const float* h = h_encoder.data() + pos * d_model;
            const float* logits_row = h_logits.data() + pos * vocab_size_local;
            
            // Compute hidden state statistics
            float h_sum = 0.0f, h_sum_sq = 0.0f, h_min = h[0], h_max = h[0];
            for (int d = 0; d < d_model; ++d) {
                h_sum += h[d];
                h_sum_sq += h[d] * h[d];
                h_min = std::min(h_min, h[d]);
                h_max = std::max(h_max, h[d]);
            }
            float h_mean = h_sum / d_model;
            float h_rms = std::sqrt(h_sum_sq / d_model);
            
            // Compute W[277] statistics
            float w_sum = 0.0f, w_sum_sq = 0.0f, w_min = h_weights_277[0], w_max = h_weights_277[0];
            for (int d = 0; d < d_model; ++d) {
                w_sum += h_weights_277[d];
                w_sum_sq += h_weights_277[d] * h_weights_277[d];
                w_min = std::min(w_min, h_weights_277[d]);
                w_max = std::max(w_max, h_weights_277[d]);
            }
            float w_mean = w_sum / d_model;
            float w_rms = std::sqrt(w_sum_sq / d_model);
            
            // Compute dot product decomposition for token 277
            float dot_product_277 = 0.0f;
            float positive_contrib = 0.0f, negative_contrib = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                float contrib = h[d] * h_weights_277[d];
                dot_product_277 += contrib;
                if (contrib > 0) positive_contrib += contrib;
                else negative_contrib += contrib;
            }
            
            // Find argmax and its logit
            int argmax_token = 0;
            float max_logit_val = logits_row[0];
            for (int v = 1; v < vocab_size_local; ++v) {
                if (logits_row[v] > max_logit_val) {
                    max_logit_val = logits_row[v];
                    argmax_token = v;
                }
            }
            
            // Compute logit[277] vs logit[argmax]
            float logit_277 = logits_row[kAnalysisToken];
            float logit_diff = max_logit_val - logit_277;  // How much 277 loses by
            
            // Compute cosine similarity between h and W[277]
            float h_norm = std::sqrt(h_sum_sq);
            float w_norm = std::sqrt(w_sum_sq);
            float cosine_sim = (h_norm > 1e-8f && w_norm > 1e-8f) 
                               ? (dot_product_277 / (h_norm * w_norm)) : 0.0f;
            
            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            fprintf(stderr, "[LOGIT_ANALYSIS] Position %d: Why does logit[v] = Σ_d h[d] × W[v,d] choose token %d?\n", pos, argmax_token);
            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            
            // Analyze hidden state properties
            fprintf(stderr, "HIDDEN STATE h[pos=%d]:\n", pos);
            fprintf(stderr, "  Statistics: mean=%.6f (offset) rms=%.6f (magnitude) range=[%.6f, %.6f]\n",
                    h_mean, h_rms, h_min, h_max);
            fprintf(stderr, "  ├─ mean≠0 → systematic bias in dot products (mean × W_sum term)\n");
            fprintf(stderr, "  ├─ rms=magnitude → scales all dot products proportionally\n");
            fprintf(stderr, "  └─ range→variation across dimensions affects different weight rows differently\n");
            
            // Analyze weight row for token 277 (often SPACE token causing mode collapse)
            fprintf(stderr, "WEIGHT ROW W[277] (target token):\n");
            fprintf(stderr, "  Statistics: mean=%.6f rms=%.6f range=[%.6f, %.6f]\n",
                    w_mean, w_rms, w_min, w_max);
            fprintf(stderr, "  ├─ Initialized to mean≈0 (ideal), rms controls sensitivity\n");
            fprintf(stderr, "  └─ Alignment with h determines dot product magnitude\n");
            
            // Decompose the dot product
            fprintf(stderr, "DOT_PRODUCT ANALYSIS Σ_d h[d]×W[277,d]:\n");
            fprintf(stderr, "  Raw computation: %.6f\n", dot_product_277);
            fprintf(stderr, "  ├─ Positive contributions (h×W>0): %.6f (%.1f%%)\n", 
                    positive_contrib, 100.0f * positive_contrib / (std::abs(dot_product_277) + 1e-8f));
            fprintf(stderr, "  ├─ Negative contributions (h×W<0): %.6f (%.1f%%)\n",
                    negative_contrib, 100.0f * std::abs(negative_contrib) / (std::abs(dot_product_277) + 1e-8f));
            fprintf(stderr, "  ├─ Cosine alignment: %.6f (1.0=perfect alignment, 0=orthogonal, -1=opposite)\n", cosine_sim);
            fprintf(stderr, "  └─ Interpretation: h and W[277] are %.1f%% aligned\n", 100.0f * cosine_sim);
            
            // Compare against argmax
            fprintf(stderr, "PREDICTIONS:\n");
            fprintf(stderr, "  logit[277]=%.6f (your token)\n", logit_277);
            fprintf(stderr, "  logit[%d]=%.6f (ARGMAX wins)\n", argmax_token, max_logit_val);
            fprintf(stderr, "  Margin: %.6f (277 is %.2f%% behind)\n", logit_diff, 100.0f * logit_diff / (std::abs(max_logit_val) + 1e-8f));
            
            // Reflection on what this means
            fprintf(stderr, "REFLECTION:\n");
            if (argmax_token == kAnalysisToken) {
                fprintf(stderr, "  ⚠️  TOKEN 277 IS ARGMAX!\n");
                fprintf(stderr, "  └─ This is the MODE COLLAPSE symptom:\n");
                fprintf(stderr, "     - 277 (SPACE) is winning most of the time\n");
                fprintf(stderr, "     - Hidden states are too correlated (high avg_cos)\n");
                fprintf(stderr, "     - All positions have similar h[d] values\n");
                fprintf(stderr, "     - W[277] captures the mode direction, beats other tokens\n");
                fprintf(stderr, "  └─ Root cause: Positions not sufficiently differentiated\n");
                fprintf(stderr, "     (Missing or weak position embeddings?)\n");
            } else {
                fprintf(stderr, "  ✓ Token 277 is NOT dominant (good)\n");
                if (cosine_sim > 0.3f) {
                    fprintf(stderr, "  ⚠️  But cosine_sim=%.6f is suspicious (>0.3):\n", cosine_sim);
                    fprintf(stderr, "     - h and W[277] are MORE aligned than expected\n");
                    fprintf(stderr, "     - If many positions have this alignment, mode collapse is forming\n");
                } else {
                    fprintf(stderr, "  ✓ h and W[277] are well-separated (cosine=%.6f, good)\n", cosine_sim);
                }
            }
            fprintf(stderr, "\n");
        }
    }  // end ENABLE_EXPENSIVE_DIAGNOSTICS guard
    
    // RULE 20: Validate logits output shape
    // Expected: [total_tokens, vocab_size] from matmul(lm_input, lm_weights^T)
    const auto expected_shape = TensorContract::TensorShape::make_LOGITS(total_tokens, cfg->vocab_size);
    const size_t logits_elements = logits_tensor.shape.total_elements();
    const size_t expected_elements = expected_shape.total_elements();
    if (logits_elements != expected_elements) {
        throw std::runtime_error(
            "AutogradTraining: Logits shape validation FAILED after matmul\n" +
            std::string("  Got: ") + std::to_string(logits_elements) + " elements (" +
            std::to_string(logits_tensor.shape.flat.rows) + "," +
            std::to_string(logits_tensor.shape.flat.cols) + ")\n" +
            std::string("  Expected: ") + std::to_string(expected_elements) + " elements (" +
            std::to_string(total_tokens) + "," + std::to_string(cfg->vocab_size) + ")\n" +
            "  Root cause: lm_input or lm_weights dimensions incorrect");
    }
    
    // Update logits shape to use LOGITS layout (semantic correctness)
    logits_tensor.shape = expected_shape;

    // LM head bias must stay inside autograd graph so text loss backpropagates
    // into lm_head_bias.grad_data() instead of bypassing parameter training.
    if (cfg->use_bias && ts->tensors_->lm_head_bias.data) {
        ts->tensors_->lm_head_bias.requires_grad = true;
        logits_tensor = autograd::broadcast_add(logits_tensor, ts->tensors_->lm_head_bias, ctx.stream);
        logits_tensor.shape = expected_shape;
        AG_INFO("Step 4b: Applied LM head bias with autograd::broadcast_add");
    }
    
    // Copy logits data to cached buffer (for diagnostics / compatibility paths)
    cudaMemcpyAsync(logits_output, logits_tensor.data,
                    logits_tensor.shape.total_elements() * sizeof(float),
                    cudaMemcpyDeviceToDevice, ctx.stream);
    
    // Move the autograd tensor to intermediates (preserves grad_fn chain)
    intermediates.logits_tensor = std::move(logits_tensor);
    
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
    
    if (cfg->numeric_head_enabled && ts->tensors_->numeric_head_weights.data) {
        AG_INFO("Step 5: Running NumericHead forward (autograd)...");
        
        // Create Tensor wrappers for weights and bias (leaf tensors)
        Tensor weights_tensor = Tensor::from_ptr(
            ts->tensors_->numeric_head_weights.data,
            TensorContract::TensorShape::make_BSM(cfg->d_model, 1),
            false,  // doesn't own
            true,    // requires_grad
            "numeric_head_weights_ref"
        );
        weights_tensor.is_leaf = true;
        // ISSUE #59: Use share_grad() for proper shared_ptr semantics
        weights_tensor.share_grad(ts->tensors_->numeric_head_weights);
        
        Tensor* bias_tensor_ptr = nullptr;
        Tensor bias_tensor;
        if (cfg->use_bias && ts->tensors_->numeric_head_bias.data) {
            bias_tensor = Tensor::from_ptr(
                ts->tensors_->numeric_head_bias.data,
                TensorContract::TensorShape::make_BSM(1, 1),
                false,  // doesn't own
                true,    // requires_grad
                "numeric_head_bias_ref"
            );
            bias_tensor.is_leaf = true;
            // ISSUE #59: Use share_grad()
            bias_tensor.share_grad(ts->tensors_->numeric_head_bias);
            bias_tensor_ptr = &bias_tensor;
        }
        
        // Create encoder output Tensor (references intermediates.encoder_output_tensor)
        Tensor encoder_for_numeric = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,  // doesn't own
            true,    // requires_grad
            "encoder_for_numeric"
        );
        encoder_for_numeric.is_leaf = false;
        encoder_for_numeric.stream = ctx.stream;
        // ISSUE #127: Use correct grad_fn based on centering
        if (use_centering && intermediates.centered_encoder_output.grad_fn) {
            encoder_for_numeric.grad_fn = intermediates.centered_encoder_output.grad_fn;
        } else {
            encoder_for_numeric.grad_fn = intermediates.encoder_output_tensor.grad_fn;
        }

        
        // Call autograd numeric_head_forward
        intermediates.numeric_head_output = numeric_head_forward(
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
// Loss Config Builder (single conversion point)
//======================================================================

autograd::LossConfig buildLossConfig(const LossContext::LossOptions& opts) {
    autograd::LossConfig lc{};
    lc.focal_enabled       = opts.focal_enabled;
    lc.focal_alpha         = opts.focal_enabled ? opts.focal_alpha : 1.0f;
    lc.focal_gamma         = opts.focal_enabled ? opts.focal_gamma : 0.0f;
    lc.smoothing_enabled   = opts.label_smoothing_enabled;
    lc.smoothing_epsilon   = opts.label_smoothing_enabled ? opts.label_smoothing_epsilon : 0.0f;
    lc.entropy_reg_enabled = opts.entropy_reg_enabled;
    lc.entropy_reg_lambda  = opts.entropy_reg_enabled ? opts.entropy_reg_lambda : 0.0f;
    return lc;
}

//======================================================================
// Autograd Loss Computation
//======================================================================

LossResult computeAutogradLoss(
    AutogradContext& ctx
) {
    LossResult result{};
    result.success = false;
    
    // RULE 20: Fail loud
    ctx.validate("computeAutogradLoss");
    if (!ctx.payload) {
        throw std::runtime_error("computeAutogradLoss: ctx.payload is NULL — training path MUST set payload via initAutogradContext(const BatchPayload&, ...)");
    }
    const auto& payload = *ctx.payload;
    payload.validate("computeAutogradLoss");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    
    // RULE 20: Fail loud - validate logits tensor was populated by forward pass
    auto& intermediates = ts->autograd_intermediates;
    if (!intermediates.logits_tensor.data) {
        throw std::runtime_error("computeAutogradLoss: Logits tensor not initialized - call executeAutogradForward() first");
    }
    
    // GPU-side targets are already in training_state (copied before forward pass)
    const int* targets = reinterpret_cast<const int*>(ts->cached_targets_tensor.data);
    if (!targets) {
        throw std::runtime_error("computeAutogradLoss: cached_targets_tensor.data is NULL - GPU copies must run before loss");
    }
    
    const int total_tokens = ctx.batch_size * ctx.seq_len;
    const int vocab_size = cfg->vocab_size;
    const int valid_tokens = payload.valid_tokens;
    
    AG_INFO("Computing loss: tokens=" << total_tokens << " vocab=" << vocab_size
            << " valid=" << valid_tokens);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 1. TEXT CROSS-ENTROPY LOSS (autograd::unified_loss)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Setup gradient buffer for logits (reuse pre-allocated buffer from TrainingState)
    if (!ts->grad_logits_tensor.data) {
        throw std::runtime_error("computeAutogradLoss: grad_logits_tensor.data not allocated - initTrainingState() must run first");
    }
    intermediates.logits_tensor.set_grad_from_buffer(
        ts->grad_logits_tensor.data
    );
    
    // CRITICAL: Release old loss tensor BEFORE unified_loss() allocates new one.
    // unified_loss() allocates ~4 GB (log_probs + grad_buffer + LogSoftmaxGradFn saved data).
    // Without releasing first, both old and new coexist → OOM on 12 GB GPU.
    intermediates.loss_tensor.release();
    
    // Compute text CE - returns scalar Tensor with NLLLossGradFn → LogSoftmaxGradFn chain
    Tensor loss_tensor = autograd::unified_loss(
        intermediates.logits_tensor,
        targets,
        nullptr,  // valid_mask not used - padding handled by target=-1
        total_tokens,
        vocab_size,
        ctx.loss_config,
        ctx.stream
    );
    
    // Move loss tensor to intermediates (TrainingState owns it during backward)
    intermediates.loss_tensor = std::move(loss_tensor);
    
    // Copy scalar loss to host
    float text_loss = 0.0f;
    cudaMemcpyAsync(&text_loss, intermediates.loss_tensor.data, sizeof(float), 
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    
    if (!std::isfinite(text_loss)) {
        throw std::runtime_error("computeAutogradLoss: text_loss is non-finite (" + std::to_string(text_loss) + ")");
    }
    
    result.text_loss = text_loss;
    result.valid_tokens = valid_tokens;
    ts->cached_loss_value = text_loss;
    ts->cached_text_loss = text_loss;
    ts->cached_valid_tokens = valid_tokens;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 2. NUMERIC REGRESSION LOSS (if enabled)
    // ═══════════════════════════════════════════════════════════════════════════
    
    float numeric_loss_sum = 0.0f;
    int numeric_loss_count = 0;
    
    if (cfg->numeric_head_enabled) {
        if (!ts->cached_numeric_predictions.data ||
            !ts->grad_numeric_tensor.data ||
            !ts->d_numeric_loss_sum.data ||
            !ts->d_numeric_loss_count.data) {
            throw std::runtime_error("computeAutogradLoss: numeric head enabled but buffers missing");
        }
        
        NumericLossInputs num_inputs{};
        num_inputs.predictions = ts->cached_numeric_predictions.data;
        num_inputs.token_numeric_values = ts->cached_token_numeric_values.data;
        num_inputs.token_numeric_mask = reinterpret_cast<uint8_t*>(
            const_cast<float*>(ts->cached_token_numeric_mask.data));
        num_inputs.targets = const_cast<int*>(targets);
        num_inputs.total_tokens = total_tokens;
        num_inputs.seq_len = ctx.seq_len;
        num_inputs.valid_text_tokens = valid_tokens;  // Issue #137: match text CE denominator
        num_inputs.huber_delta = cfg->numeric_head_huber_delta;
        num_inputs.log_scale = cfg->numeric_head_log_scale;
        num_inputs.loss_weight = cfg->numeric_head_loss_weight;
        
        NumericLossOutputs num_outputs{};
        num_outputs.loss_sum = ts->d_numeric_loss_sum.data;
        num_outputs.count = reinterpret_cast<int*>(ts->d_numeric_loss_count.data);
        num_outputs.grad_predictions = ts->grad_numeric_tensor.data;
        
        if (!launchNumericLoss(num_inputs, num_outputs, ctx.stream)) {
            throw std::runtime_error("computeAutogradLoss: numeric loss kernel launch failed");
        }
        
        cudaMemcpyAsync(&numeric_loss_sum, ts->d_numeric_loss_sum.data,
                        sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(&numeric_loss_count, reinterpret_cast<int*>(ts->d_numeric_loss_count.data),
                        sizeof(int), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        
        if (!std::isfinite(numeric_loss_sum)) {
            numeric_loss_sum = 0.0f;
            numeric_loss_count = 0;
        }
    }
    
    const float numeric_loss_avg = (numeric_loss_count > 0)
        ? (numeric_loss_sum / numeric_loss_count) : 0.0f;
    
    result.numeric_loss = numeric_loss_avg;
    result.numeric_count = numeric_loss_count;
    ts->cached_numeric_loss = numeric_loss_avg;
    ts->cached_numeric_count = numeric_loss_count;  // Issue #137: store for weight grad normalization
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 3. LEARNED LOSS WEIGHTING (homoscedastic uncertainty)
    //    L_total = L_text / (2σ²_text) + L_numeric / (2σ²_numeric)
    //              + 0.5·log(σ²_text) + 0.5·log(σ²_numeric)
    //    We learn log_var = log(σ²), so: L = L/(2·exp(log_var)) + 0.5·log_var
    // ═══════════════════════════════════════════════════════════════════════════
    
    float weight_text = 1.0f;
    float weight_numeric = cfg->numeric_head_loss_weight;
    float reg_text = 0.0f;
    float reg_numeric = 0.0f;
    
    const bool use_learned_weights = ts->log_var_text.data && ts->log_var_numeric.data;
    
    if (use_learned_weights) {
        float log_var_text_val = 0.0f;
        float log_var_numeric_val = 0.0f;
        
        cudaMemcpyAsync(&log_var_text_val, ts->log_var_text.data,
                        sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(&log_var_numeric_val, ts->log_var_numeric.data,
                        sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        
        // Clamp log_var to prevent numerical issues (σ² in [0.018, 54.6])
        log_var_text_val = std::clamp(log_var_text_val, -4.0f, 4.0f);
        log_var_numeric_val = std::clamp(log_var_numeric_val, -4.0f, 4.0f);
        
        weight_text = 0.5f * std::exp(-log_var_text_val);
        weight_numeric = 0.5f * std::exp(-log_var_numeric_val);
        reg_text = 0.5f * log_var_text_val;
        reg_numeric = 0.5f * log_var_numeric_val;
    }
    
    result.weight_text = weight_text;
    result.weight_numeric = weight_numeric;
    
    const float combined_loss = weight_text * text_loss + reg_text
                              + weight_numeric * numeric_loss_avg + reg_numeric;
    
    if (!std::isfinite(combined_loss)) {
        throw std::runtime_error("computeAutogradLoss: combined_loss is non-finite (text=" 
            + std::to_string(text_loss) + " numeric=" + std::to_string(numeric_loss_avg) + ")");
    }
    
    result.loss_value = combined_loss;
    
    fprintf(stderr, "[LossComponents] text_ce=%.4f (w=%.3f) numeric=%.4f (w=%.3f) reg=%.4f total=%.4f\n",
            text_loss, weight_text, numeric_loss_avg, weight_numeric,
            reg_text + reg_numeric, combined_loss);
    
    AG_INFO("Loss computed: combined=" << combined_loss << " text=" << text_loss 
            << " numeric=" << numeric_loss_avg << " valid_tokens=" << valid_tokens);
    
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
    
    ctx.validate("executeAutogradBackward");
    
    auto* ts = ctx.training_state;
    auto& intermediates = ts->autograd_intermediates;
    if (!intermediates.loss_tensor.data) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor not initialized - call computeAutogradLoss() first");
    }
    
    if (!intermediates.loss_tensor.grad_fn) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor has no grad_fn - autograd chain broken");
    }
    
    AG_INFO("Executing backward pass (accumulate=" << accumulate << ", scale=" << ctx.grad_scale << ")");

    // Match branch gradient scaling to the scalar objective from computeLossBatch:
    //   L_total = w_text * L_text + w_numeric * L_numeric + regularizers
    float text_branch_weight = 1.0f;
    float numeric_branch_weight = ctx.config->numeric_head_loss_weight;
    float log_var_text_val = 0.0f;
    float log_var_numeric_val = 0.0f;
    const bool use_learned_weights = ts->log_var_text.data && ts->log_var_numeric.data;

    if (use_learned_weights) {
        cudaMemcpyAsync(&log_var_text_val, ts->log_var_text.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(&log_var_numeric_val, ts->log_var_numeric.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);

        log_var_text_val = std::clamp(log_var_text_val, -4.0f, 4.0f);
        log_var_numeric_val = std::clamp(log_var_numeric_val, -4.0f, 4.0f);

        text_branch_weight = 0.5f * std::exp(-log_var_text_val);
        numeric_branch_weight = 0.5f * std::exp(-log_var_numeric_val);
    }

    AG_INFO("Branch weights: text=" << text_branch_weight
            << " numeric=" << numeric_branch_weight
            << (use_learned_weights ? " (learned)" : " (static)"));
    
    // Zero gradients if not accumulating
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (!accumulate) {
        // Top-level parameters
        ts->tensors_->embedding_weights.zero_grad(ctx.stream);
        ts->tensors_->position_embedding_weights.zero_grad(ctx.stream);
        ts->tensors_->lm_head_weights.zero_grad(ctx.stream);
        ts->tensors_->lm_head_bias.zero_grad(ctx.stream);
        ts->tensors_->numeric_head_weights.zero_grad(ctx.stream);
        ts->tensors_->numeric_head_bias.zero_grad(ctx.stream);
        ts->tensors_->final_rms_gamma.zero_grad(ctx.stream);
        ts->log_var_text.zero_grad(ctx.stream);
        ts->log_var_numeric.zero_grad(ctx.stream);

        // Encoder parameters
        if (ctx.gpu_encoder) {
            const int num_layers = ctx.gpu_encoder->getNumLayers();
            for (int layer = 0; layer < num_layers; ++layer) {
                auto* enc = ctx.gpu_encoder->getLayer(layer);
                if (!enc) continue;
                enc->rms1Gamma().zero_grad(ctx.stream);
                enc->rms2Gamma().zero_grad(ctx.stream);
                enc->rmsPostAttnGamma().zero_grad(ctx.stream);
                enc->rmsPostFfnGamma().zero_grad(ctx.stream);
                enc->attnWqkv().zero_grad(ctx.stream);
                enc->attnBqkv().zero_grad(ctx.stream);
                enc->attnWo().zero_grad(ctx.stream);
                enc->attnBo().zero_grad(ctx.stream);
                enc->ffnW1().zero_grad(ctx.stream);
                enc->ffnB1().zero_grad(ctx.stream);
                enc->ffnW2().zero_grad(ctx.stream);
                enc->ffnB2().zero_grad(ctx.stream);
                enc->layerScale1().zero_grad(ctx.stream);
                enc->layerScale2().zero_grad(ctx.stream);
            }
        }

        // ScratchBlock parameters
        if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
            ctx.scratch_block->atomTypeEmbeddings().zero_grad(ctx.stream);
            ctx.scratch_block->atomProjection().zero_grad(ctx.stream);
            ctx.scratch_block->textFeatureProjection().zero_grad(ctx.stream);
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Issue #141/#142 FIX: Set up gradient tap BEFORE loss_tensor.backward()
    // The embedding path GradFn (Dropout/Add/Embedding) will copy its grad_output
    // (= encoder input gradient) into this buffer before further propagation.
    // consumed by the autograd chain and inaccessible afterward.
    // ScratchBlock backward needs it to compute parameter gradients.
    // ═══════════════════════════════════════════════════════════════════════════
    const bool scratchblock_enabled = ctx.scratch_block && ctx.scratch_block->isEnabled();
    if (scratchblock_enabled && !ts->scratchblock_grad_tap.data) {
        throw std::runtime_error(
            "executeAutogradBackward: ScratchBlock is enabled but scratchblock_grad_tap.data is NULL");
    }
    const bool need_scratchblock_backward = scratchblock_enabled;
    if (need_scratchblock_backward) {
        if (!intermediates.embedding_tensor.grad_fn) {
            throw std::runtime_error(
                "executeAutogradBackward: ScratchBlock enabled but embedding_tensor.grad_fn is NULL; "
                "cannot capture encoder input gradient tap");
        }

        const size_t tap_elems = static_cast<size_t>(ctx.batch_size) * ctx.seq_len * ctx.config->d_model;
        auto* embedding_grad_fn = intermediates.embedding_tensor.grad_fn.get();
        embedding_grad_fn->grad_output_tap = ts->scratchblock_grad_tap.data;
        embedding_grad_fn->grad_output_tap_count = tap_elems;
        embedding_grad_fn->grad_output_tap_written = false;

        const cudaError_t tap_reset_err = cudaMemsetAsync(
            ts->scratchblock_grad_tap.data,
            0,
            tap_elems * sizeof(float),
            ctx.stream);
        if (tap_reset_err != cudaSuccess) {
            throw std::runtime_error(
                std::string("executeAutogradBackward: Failed to clear ScratchBlock gradient tap buffer: ")
                + cudaGetErrorString(tap_reset_err));
        }

        AG_INFO("ScratchBlock gradient tap armed on embedding_tensor.grad_fn op="
                << (embedding_grad_fn->op_name ? embedding_grad_fn->op_name : "<unknown>")
                << " (" << tap_elems << " elements)");
    }

    // Call backward on the text loss branch.
    // When branch weight != 1, seed scalar grad_output with that weight so
    // dL_total/dL_text is applied at the root of the autograd chain.
    AG_INFO("Calling loss_tensor.backward()...");
    if (std::abs(text_branch_weight - 1.0f) > 1e-7f) {
        if (!ts->d_loss_sum_scratch.data) {
            throw std::runtime_error("executeAutogradBackward: d_loss_sum_scratch is NULL - cannot seed weighted text gradient");
        }

        cudaMemcpyAsync(ts->d_loss_sum_scratch.data,
                        &text_branch_weight,
                        sizeof(float),
                        cudaMemcpyHostToDevice,
                        ctx.stream);

        Tensor text_grad_seed;
        text_grad_seed.data = ts->d_loss_sum_scratch.data;
        text_grad_seed.shape = intermediates.loss_tensor.shape;
        text_grad_seed.owns_data = false;
        text_grad_seed.stream = ctx.stream;
        intermediates.loss_tensor.backward(&text_grad_seed);
    } else {
        intermediates.loss_tensor.backward();
    }
    AG_INFO("loss_tensor.backward() returned successfully");

    // Numeric head is an auxiliary branch with its own gradient buffer
    // (dL_numeric/d(predictions)) produced by launchNumericLoss.
    if (ctx.config->numeric_head_enabled &&
        intermediates.numeric_head_output.data &&
        intermediates.numeric_head_output.grad_fn &&
        ts->grad_numeric_tensor.data) {
        // launchNumericLoss() seeds grad_numeric_tensor with cfg.numeric_head_loss_weight.
        // Rescale seed to match the effective branch weight used in scalar objective.
        const float kernel_numeric_weight = ctx.config->numeric_head_loss_weight;
        float numeric_seed_scale = 1.0f;
        if (std::abs(kernel_numeric_weight) > 1e-8f) {
            numeric_seed_scale = numeric_branch_weight / kernel_numeric_weight;
        } else {
            numeric_seed_scale = 0.0f;
            if (std::abs(numeric_branch_weight) > 1e-8f) {
                AG_WARN("numeric_head_loss_weight is 0 but desired numeric branch weight is "
                        << numeric_branch_weight
                        << "; numeric gradients are forced to 0 for this step");
            }
        }

        if (std::abs(numeric_seed_scale - 1.0f) > 1e-7f) {
            const size_t numeric_elems = intermediates.numeric_head_output.shape.total_elements();
            scaleGradBuffer(ts->grad_numeric_tensor.data, numeric_elems, numeric_seed_scale, ctx.stream);
            AG_INFO("Scaled grad_numeric_tensor by " << numeric_seed_scale << " to match branch weighting");
        }

        AG_INFO("Calling numeric_head_output.backward() with grad_numeric_tensor...");
        Tensor numeric_grad;
        numeric_grad.data = ts->grad_numeric_tensor.data;
        numeric_grad.shape = intermediates.numeric_head_output.shape;
        numeric_grad.owns_data = false;
        numeric_grad.stream = ctx.stream;
        intermediates.numeric_head_output.backward(&numeric_grad);
        AG_INFO("numeric_head_output.backward() returned successfully");

        // Issue #137: Normalize numeric head WEIGHT gradients for dense accumulation.
        // The matmul backward sums ~N_atoms gradient contributions into 768 params,
        // while the text LM head distributes across 50K vocab rows (sparse).
        // Scale weight+bias grads by sqrt(N_atoms / valid_tokens) to normalize
        // the dense accumulation variance to match text's effective accumulation.
        // This does NOT affect grad_encoder (already propagated by backward()).
        const int num_count = ts->cached_numeric_count;
        const int valid_tokens = ts->cached_valid_tokens;
        if (num_count > 0 && valid_tokens > 0) {
            const float dense_norm_scale = std::sqrt(
                static_cast<float>(num_count) / static_cast<float>(valid_tokens));
            scaleGradBuffer(ts->tensors_->numeric_head_weights.grad_data(),
                            ts->tensors_->numeric_head_weights.numel(),
                            dense_norm_scale, ctx.stream);
            if (ts->tensors_->numeric_head_bias.has_grad()) {
                scaleGradBuffer(ts->tensors_->numeric_head_bias.grad_data(),
                                ts->tensors_->numeric_head_bias.numel(),  
                                dense_norm_scale, ctx.stream);
            }
            AG_INFO("Post-scaled numeric weight grads by sqrt(" << num_count
                    << "/" << valid_tokens << ")=" << dense_norm_scale);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Issue #141/#142 FIX: ScratchBlock backward using gradient tap buffer
    //
    // ScratchBlock operates in-place on embeddings (additive injection).
    // Its backward needs the encoder input gradient to compute parameter gradients
    // for atom_projection_, atom_type_embeddings_, and text_feature_projection_.
    //
    // Previously guarded by intermediates.embedding_tensor.has_grad() which was
    // ALWAYS false (dropout output is non-leaf, autograd writes to GradFn-internal
    // buffers not tensor.grad_). ScratchBlock parameters were FROZEN.
    //
    // Now uses gradient tap: the embedding path GradFn copied its grad_output
    // (encoder input gradient) into scratchblock_grad_tap.
    // ═══════════════════════════════════════════════════════════════════════════
    if (need_scratchblock_backward) {
        auto* embedding_grad_fn = intermediates.embedding_tensor.grad_fn.get();
        if (!embedding_grad_fn || !embedding_grad_fn->grad_output_tap_written) {
            const char* op = (embedding_grad_fn && embedding_grad_fn->op_name)
                                 ? embedding_grad_fn->op_name
                                 : "<null>";
            throw std::runtime_error(
                "executeAutogradBackward: ScratchBlock gradient tap was not written by embedding grad_fn '" +
                std::string(op) + "'. This would use stale tap data.");
        }

        AG_INFO("Calling ScratchBlock backward() with gradient tap buffer...");
        const int total_tokens = ctx.batch_size * ctx.seq_len;
        ScratchBlockForwardArgs sb_args{};
        sb_args.input = TensorContract::TensorView::make_BSM(
            intermediates.embedding_tensor.data, total_tokens, ctx.config->d_model, "sb_input_bwd");
        sb_args.output = TensorContract::TensorView::make_BSM(
            intermediates.embedding_tensor.data, total_tokens, ctx.config->d_model, "sb_output_bwd");
        sb_args.total_tokens = total_tokens;
        sb_args.seq_len = ctx.seq_len;
        sb_args.token_ids = reinterpret_cast<int*>(ts->cached_token_ids_tensor.data);
        sb_args.token_numeric_values = ctx.token_numeric_values;
        sb_args.token_numeric_mask = ctx.token_numeric_mask;
        sb_args.token_text_features = ctx.token_text_features;
        sb_args.token_text_mask = ctx.token_text_mask;
        sb_args.stream = ctx.stream;
        if (ts->cached_scratch_block_embeddings.data) {
            sb_args.cache_atom_embeddings = TensorContract::TensorView::make_BSM(
                ts->cached_scratch_block_embeddings.data,
                ctx.scratch_block->config().max_atoms,
                ctx.scratch_block->config().atom_embedding_dim,
                "sb_cache_embeddings_bwd");
        }
        sb_args.cache_atom_positions = reinterpret_cast<int*>(ts->cached_scratch_block_positions.data);
        sb_args.cache_atom_types = reinterpret_cast<int*>(ts->cached_scratch_block_types.data);
        sb_args.cache_num_atoms = reinterpret_cast<int*>(ts->cached_scratch_block_num_atoms.data);

        // Use the gradient tap buffer (encoder input gradient captured before dropout)
        // Additive injection: grad_input = grad_output for pass-through, plus parameter grads
        ctx.scratch_block->backward(
            sb_args,
            ts->scratchblock_grad_tap.data,
            ts->scratchblock_grad_tap.data);
        AG_INFO("ScratchBlock backward() returned successfully");
    }
    
    // Compute gradients for learned loss weights (not in autograd graph)
    // L = L_task / (2*exp(log_var)) + 0.5*log_var
    // dL/d(log_var) = -L_task * 0.5*exp(-log_var) + 0.5
    // ts already declared at function start
    if (ts->log_var_text.data && ts->log_var_text.has_grad() &&
        ts->log_var_numeric.data && ts->log_var_numeric.has_grad()) {
        // grad = -L * weight + 0.5 (from regularization term)
        // Use the same branch weights that were applied to backward seeds.
        //
        // Issue #137 FIX: Raw gradient is O(loss) magnitude (~11.5 for loss=12).
        // These single scalars were dominating the NUMERIC_HEAD param group norm
        // (num=10.4 was almost entirely log_var, not actual numeric head weights).
        // Normalize by (1 + loss²) to bring contribution to ~0.1 (negligible vs weights).
        // At loss=12: scale = 1/145 → grad ≈ 0.08. Two params: sqrt(2×0.08²) ≈ 0.11.
        // At equilibrium: raw_grad=0 → normalized=0 (unaffected).
        // As loss decreases, normalization relaxes → faster convergence near optimum.
        const float grad_text_raw = -ts->cached_text_loss * text_branch_weight + 0.5f;
        const float grad_numeric_raw = -ts->cached_numeric_loss * numeric_branch_weight + 0.5f;
        const float text_loss_sq = ts->cached_text_loss * ts->cached_text_loss;
        const float numeric_loss_sq = ts->cached_numeric_loss * ts->cached_numeric_loss;
        const float grad_text = grad_text_raw / (1.0f + text_loss_sq);
        const float grad_numeric = grad_numeric_raw / (1.0f + numeric_loss_sq);
        
        // Write gradients to GPU
        cudaMemcpyAsync(ts->log_var_text.grad_data(), &grad_text, sizeof(float), cudaMemcpyHostToDevice, ctx.stream);
        cudaMemcpyAsync(ts->log_var_numeric.grad_data(), &grad_numeric, sizeof(float), cudaMemcpyHostToDevice, ctx.stream);
    }
    
    // Verify gradients are properly connected before optimizer runs
    AG_INFO("Verifying gradients are connected to optimizer...");
    if (!verifyGradientsAreConnected(ctx)) {
        result.error_message = "Gradient connectivity verification failed";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    AG_INFO("Gradient connectivity verified");
    
    // Apply gradient scaling if needed
    // ISSUE #59: Use grad_data() accessor
    if (ctx.grad_scale != 1.0f) {
        auto* ts = ctx.training_state;
        const float scale = ctx.grad_scale;
        
        scaleGradBuffer(ts->tensors_->embedding_weights.grad_data(), ts->tensors_->embedding_weights.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->tensors_->position_embedding_weights.grad_data(), ts->tensors_->position_embedding_weights.numel(), scale, ctx.stream);
        if (ts->tensors_->lm_head_weights.grad_data() != ts->tensors_->embedding_weights.grad_data()) {
            scaleGradBuffer(ts->tensors_->lm_head_weights.grad_data(), ts->tensors_->lm_head_weights.numel(), scale, ctx.stream);
        }
        scaleGradBuffer(ts->tensors_->lm_head_bias.grad_data(), ts->tensors_->lm_head_bias.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->tensors_->numeric_head_weights.grad_data(), ts->tensors_->numeric_head_weights.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->tensors_->numeric_head_bias.grad_data(), ts->tensors_->numeric_head_bias.numel(), scale, ctx.stream);
        scaleGradBuffer(ts->tensors_->final_rms_gamma.grad_data(), ts->tensors_->final_rms_gamma.numel(), scale, ctx.stream);
        
        if (ctx.gpu_encoder) {
            const int num_layers = ctx.gpu_encoder->getNumLayers();
            for (int layer = 0; layer < num_layers; ++layer) {
                auto* enc = ctx.gpu_encoder->getLayer(layer);
                if (!enc) continue;
                scaleGradBuffer(enc->rms1Gamma().grad_data(), enc->rms1Gamma().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->rms2Gamma().grad_data(), enc->rms2Gamma().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->rmsPostAttnGamma().grad_data(), enc->rmsPostAttnGamma().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->rmsPostFfnGamma().grad_data(), enc->rmsPostFfnGamma().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->attnWqkv().grad_data(), enc->attnWqkv().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->attnBqkv().grad_data(), enc->attnBqkv().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->attnWo().grad_data(), enc->attnWo().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->attnBo().grad_data(), enc->attnBo().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->ffnW1().grad_data(), enc->ffnW1().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->ffnB1().grad_data(), enc->ffnB1().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->ffnW2().grad_data(), enc->ffnW2().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->ffnB2().grad_data(), enc->ffnB2().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->layerScale1().grad_data(), enc->layerScale1().numel(), scale, ctx.stream);
                scaleGradBuffer(enc->layerScale2().grad_data(), enc->layerScale2().numel(), scale, ctx.stream);
            }
        }

        if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
            scaleGradBuffer(ctx.scratch_block->atomTypeEmbeddings().grad_data(),
                            ctx.scratch_block->atomTypeEmbeddings().numel(), scale, ctx.stream);
            scaleGradBuffer(ctx.scratch_block->atomProjection().grad_data(),
                            ctx.scratch_block->atomProjection().numel(), scale, ctx.stream);
            scaleGradBuffer(ctx.scratch_block->textFeatureProjection().grad_data(),
                            ctx.scratch_block->textFeatureProjection().numel(), scale, ctx.stream);
        }
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

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    bool ok = true;
    
    // The autograd system stores gradients in Tensor.grad_ fields (shared_ptr<Tensor>)
    // The optimizer accesses them directly via Tensor.grad_data() pointers.
    // 
    // This function verifies that gradients are properly allocated and accessible.
    // It does NOT copy — gradients are already wired up during initialization.
    // This is a diagnostic check to catch pointer setup bugs before optimizer runs.
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Embedding gradients (may be tied with LM head)
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ts->tensors_->embedding_weights.data) {
        if (!ts->tensors_->embedding_weights.has_grad()) {
            AG_WARN("embedding_weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            AG_INFO("Embedding gradients ready: " << ts->tensors_->embedding_weights.numel() << " elements at " << ts->tensors_->embedding_weights.grad_data());
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LM head gradients (may be tied to embedding)
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (ts->tensors_->lm_head_weights.data) {
        if (!ts->tensors_->lm_head_weights.has_grad()) {
            AG_WARN("lm_head_weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            // Check if tied to embedding (same underlying grad Tensor)
            if (ts->tensors_->lm_head_weights.grad_data() == ts->tensors_->embedding_weights.grad_data()) {
                AG_INFO("LM head gradients TIED to embedding: " << ts->tensors_->lm_head_weights.numel() << " elements");
            } else {
                AG_INFO("LM head gradients SEPARATE: " << ts->tensors_->lm_head_weights.numel() << " elements at " << ts->tensors_->lm_head_weights.grad_data());
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // DO NOT copy grad_logits back to TrainingState buffer!
    // ═══════════════════════════════════════════════════════════════════════════
    // CRITICAL FIX (Issue #136): The grad_logits_tensor.data IS THE STARTING GRADIENT BUFFER.
    // It's set via set_grad_from_buffer() at ComputeLossBatch.cu:882.
    // During backward, LogSoftmaxGradFn writes CE gradients directly to this buffer.
    // Then CenterRowsGradFn reads from it and produces its own centered output buffer.
    // If we copy intermediates.logits_tensor.grad_data() back to ts->grad_logits_tensor.data,
    // we OVERWRITE the CE gradients with centered versions from CenterRowsGradFn!
    // Result: Phase2 diagnostic reads CENTERED (negative) gradients instead of CE (positive) gradients.
    // 
    // Solution: SKIP THE COPY. The grad_logits_tensor.data is already correct from LogSoftmaxGradFn.
    // The autograd intermediates keep CenterRowsGradFn's output in their own buffer, which is fine.
    // Phase2 diagnostics read directly from ts->grad_logits_tensor.data (the CE gradients).
    //
    // DO NOT DO THIS:
    //   if (intermediates.logits_tensor.has_grad()) {
    //       cudaMemcpyAsync(ts->grad_logits_tensor.data, intermediates.logits_tensor.grad_data(), ...);
    //   }
    // This would CORRUPT the CE gradients with centered versions!
    
    // NOTE: Encoder gradients are in encoder's internal Tensors, not TrainingState.
    // The optimizer accesses them via Tensor& accessors (enc->attnWqkv() etc.).
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Final RMSNorm gamma
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ts->tensors_->final_rms_gamma.data && !ts->tensors_->final_rms_gamma.has_grad()) {
        AG_WARN("final_rms_gamma.grad is NULL - gradients NOT flowing!");
        ok = false;
    }

    if (cfg && cfg->use_bias && ts->tensors_->lm_head_bias.data && !ts->tensors_->lm_head_bias.has_grad()) {
        AG_WARN("lm_head_bias.grad is NULL - gradients NOT flowing!");
        ok = false;
    }

    if (cfg && cfg->numeric_head_enabled) {
        if (ts->tensors_->numeric_head_weights.data && !ts->tensors_->numeric_head_weights.has_grad()) {
            AG_WARN("numeric_head_weights.grad is NULL - gradients NOT flowing!");
            ok = false;
        }
        if (cfg->use_bias && ts->tensors_->numeric_head_bias.data && !ts->tensors_->numeric_head_bias.has_grad()) {
            AG_WARN("numeric_head_bias.grad is NULL - gradients NOT flowing!");
            ok = false;
        }
    }

    if (ctx.gpu_encoder) {
        const int num_layers = ctx.gpu_encoder->getNumLayers();
        for (int layer = 0; layer < num_layers; ++layer) {
            auto* enc = ctx.gpu_encoder->getLayer(layer);
            if (!enc) {
                AG_WARN("encoder layer " << layer << " is NULL during gradient verification");
                ok = false;
                continue;
            }
            auto check = [&](Tensor& t, const char* name) {
                if (t.data && !t.has_grad()) {
                    AG_WARN("layer " << layer << " " << name << ".grad is NULL");
                    ok = false;
                }
            };
            check(enc->rms1Gamma(), "rms1Gamma");
            check(enc->rms2Gamma(), "rms2Gamma");
            check(enc->rmsPostAttnGamma(), "rmsPostAttnGamma");
            check(enc->rmsPostFfnGamma(), "rmsPostFfnGamma");
            check(enc->attnWqkv(), "attnWqkv");
            check(enc->attnBqkv(), "attnBqkv");
            check(enc->attnWo(), "attnWo");
            check(enc->attnBo(), "attnBo");
            check(enc->ffnW1(), "ffnW1");
            check(enc->ffnB1(), "ffnB1");
            check(enc->ffnW2(), "ffnW2");
            check(enc->ffnB2(), "ffnB2");
            check(enc->layerScale1(), "layerScale1");
            check(enc->layerScale2(), "layerScale2");
        }
    }

    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        auto checkScratch = [&](Tensor& t, const char* name) {
            if (t.data && !t.has_grad()) {
                AG_WARN("scratch block " << name << ".grad is NULL");
                ok = false;
            }
        };
        checkScratch(ctx.scratch_block->atomTypeEmbeddings(), "atomTypeEmbeddings");
        checkScratch(ctx.scratch_block->atomProjection(), "atomProjection");
        checkScratch(ctx.scratch_block->textFeatureProjection(), "textFeatureProjection");
    }
    
    return ok;
}

float computeGradientNorm(const AutogradContext& ctx) {
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    
    // Allocate accumulator on device
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemsetAsync(d_sum, 0, sizeof(double), ctx.stream);
    
    // Accumulate sum of squared gradients from all parameter groups
    // Embeddings and LM head (may be tied)
    // ISSUE #59: Use grad_data() accessor
    computeSumSquared(ts->tensors_->embedding_weights.grad_data(), ts->tensors_->embedding_weights.numel(), d_sum, ctx.stream);
    if (ts->tensors_->lm_head_weights.grad_data() != ts->tensors_->embedding_weights.grad_data()) {
        // Only add if not tied (different pointers)
        computeSumSquared(ts->tensors_->lm_head_weights.grad_data(), ts->tensors_->lm_head_weights.numel(), d_sum, ctx.stream);
    }
    computeSumSquared(ts->tensors_->position_embedding_weights.grad_data(), ts->tensors_->position_embedding_weights.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->tensors_->lm_head_bias.grad_data(), ts->tensors_->lm_head_bias.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->tensors_->numeric_head_weights.grad_data(), ts->tensors_->numeric_head_weights.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->tensors_->numeric_head_bias.grad_data(), ts->tensors_->numeric_head_bias.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->tensors_->final_rms_gamma.grad_data(), ts->tensors_->final_rms_gamma.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->log_var_text.grad_data(), ts->log_var_text.numel(), d_sum, ctx.stream);
    computeSumSquared(ts->log_var_numeric.grad_data(), ts->log_var_numeric.numel(), d_sum, ctx.stream);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ENCODER GRADIENT NORM: Compute from encoder's internal Tensors
    // Uses Tensor& accessors + numel() — no manual size computation.
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.gpu_encoder && cfg) {
        const int num_layers = ctx.gpu_encoder->getNumLayers();
        
        // Helper: add gradient contribution from a Tensor if it has grad
        auto addGradNorm = [&](Tensor& tensor) {
            if (tensor.has_grad()) {
                computeSumSquared(tensor.grad_data(), tensor.numel(), d_sum, ctx.stream);
            }
        };
        
        for (int layer = 0; layer < num_layers; ++layer) {
            auto* enc = ctx.gpu_encoder->getLayer(layer);
            if (!enc) continue;
            
            addGradNorm(enc->rms1Gamma());
            addGradNorm(enc->rms2Gamma());
            addGradNorm(enc->rmsPostAttnGamma());
            addGradNorm(enc->rmsPostFfnGamma());
            addGradNorm(enc->attnWqkv());
            addGradNorm(enc->attnBqkv());
            addGradNorm(enc->attnWo());
            addGradNorm(enc->attnBo());
            addGradNorm(enc->ffnW1());
            addGradNorm(enc->ffnB1());
            addGradNorm(enc->ffnW2());
            addGradNorm(enc->ffnB2());
            addGradNorm(enc->layerScale1());
            addGradNorm(enc->layerScale2());
        }
    }

    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        computeSumSquared(ctx.scratch_block->atomTypeEmbeddings().grad_data(),
                          ctx.scratch_block->atomTypeEmbeddings().numel(), d_sum, ctx.stream);
        computeSumSquared(ctx.scratch_block->atomProjection().grad_data(),
                          ctx.scratch_block->atomProjection().numel(), d_sum, ctx.stream);
        computeSumSquared(ctx.scratch_block->textFeatureProjection().grad_data(),
                          ctx.scratch_block->textFeatureProjection().numel(), d_sum, ctx.stream);
    }
    
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

LossResult autogradTrainingStep(
    LanguageModel& model,
    TrainingState& training_state,
    const Batching::BatchPayload& payload,
    bool accumulate,
    float grad_scale,
    uint64_t step
) {
    payload.validate("autogradTrainingStep");
    
    const auto& cfg = model.getConfig();
    const int total_tokens = payload.total_tokens;
    
    // Get encoder for autograd forward
    GPUGrimEncoder& gpu_encoder = model.getGpuEncoder();
    ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU COPIES: Transfer padded data from payload to GPU tensors
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Validate buffer capacity
    const size_t logit_limit = training_state.max_logit_tokens > 0
        ? training_state.max_logit_tokens
        : training_state.max_cached_tokens;
    if (static_cast<size_t>(total_tokens) > logit_limit) {
        throw std::runtime_error(
            "autogradTrainingStep: total_tokens=" + std::to_string(total_tokens) +
            " exceeds logit buffer capacity=" + std::to_string(logit_limit));
    }
    
    // Token IDs
    if (!training_state.cached_token_ids_tensor.data) {
        throw std::runtime_error("autogradTrainingStep: cached_token_ids_tensor.data is NULL");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<int*>(training_state.cached_token_ids_tensor.data),
        payload.input_ids.data(),
        total_tokens * sizeof(int),
        cudaMemcpyHostToDevice, stream));
    
    // Targets
    if (!training_state.cached_targets_tensor.data) {
        throw std::runtime_error("autogradTrainingStep: cached_targets_tensor.data is NULL");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<int*>(training_state.cached_targets_tensor.data),
        payload.target_ids.data(),
        total_tokens * sizeof(int),
        cudaMemcpyHostToDevice, stream));
    
    // Numeric values
    if (!training_state.cached_token_numeric_values.data) {
        throw std::runtime_error("autogradTrainingStep: cached_token_numeric_values.data is NULL");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        training_state.cached_token_numeric_values.data,
        payload.numeric_values.data(),
        total_tokens * sizeof(float),
        cudaMemcpyHostToDevice, stream));
    
    // Numeric mask
    if (!training_state.cached_token_numeric_mask.data) {
        throw std::runtime_error("autogradTrainingStep: cached_token_numeric_mask.data is NULL");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<uint8_t*>(training_state.cached_token_numeric_mask.data),
        payload.numeric_mask.data(),
        total_tokens * sizeof(uint8_t),
        cudaMemcpyHostToDevice, stream));
    
    // Text features + mask (if buffers exist)
    constexpr int kTextFeatureDim = Batching::BatchPayload::kTextFeatureDim;
    if (training_state.cached_token_text_features.data && training_state.cached_token_text_mask.data) {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint16_t*>(training_state.cached_token_text_features.data),
            payload.text_features.data(),
            total_tokens * kTextFeatureDim * sizeof(uint16_t),
            cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint8_t*>(training_state.cached_token_text_mask.data),
            payload.text_mask.data(),
            total_tokens * sizeof(uint8_t),
            cudaMemcpyHostToDevice, stream));
    }
    
    // Store dimensions in TrainingState for downstream consumers
    training_state.cached_batch_size = payload.batch_size;
    training_state.cached_seq_len = payload.max_seq_len;
    training_state.cached_num_layers = cfg.num_layers;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOGRAD CONTEXT
    // ═══════════════════════════════════════════════════════════════════════════
    
    AutogradContext ctx = initAutogradContext(
        &cfg,
        &training_state,
        &gpu_encoder,
        scratch_block,
        training_state.cublas_handle,
        stream,
        payload,
        grad_scale,
        step,
        true
    );
    ctx.loss_config = buildLossConfig(model.getLossOptions());
    
    // ScratchBlock input buffers (now on GPU from copies above)
    ctx.token_numeric_values = training_state.cached_token_numeric_values.data;
    ctx.token_numeric_mask = reinterpret_cast<const uint8_t*>(training_state.cached_token_numeric_mask.data);
    ctx.token_text_features = reinterpret_cast<const uint16_t*>(training_state.cached_token_text_features.data);
    ctx.token_text_mask = reinterpret_cast<const uint8_t*>(training_state.cached_token_text_mask.data);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FORWARD → LOSS → BACKWARD
    // ═══════════════════════════════════════════════════════════════════════════
    
    ForwardResult fwd_result = executeAutogradForward(ctx);
    if (!fwd_result.success) {
        throw std::runtime_error("autogradTrainingStep: Forward failed - " + fwd_result.error_message);
    }
    
    LossResult loss_result = computeAutogradLoss(ctx);
    if (!loss_result.success) {
        loss_result.error_message = "autogradTrainingStep: Loss failed - " + loss_result.error_message;
        training_state.autograd_intermediates.clear();
        return loss_result;
    }
    
    // Rule 20: Non-finite loss means forward produced garbage.
    // Skip backward entirely — don't propagate NaN/Inf gradients.
    if (!std::isfinite(loss_result.loss_value)) {
        loss_result.success = false;
        loss_result.error_message = "Non-finite loss: " + std::to_string(loss_result.loss_value);
        training_state.autograd_intermediates.clear();
        return loss_result;
    }
    
    BackwardResult bwd_result = executeAutogradBackward(ctx, accumulate);
    if (!bwd_result.success) {
        loss_result.success = false;
        loss_result.error_message = "autogradTrainingStep: Backward failed - " + bwd_result.error_message;
        training_state.autograd_intermediates.clear();
        return loss_result;
    }
    
    // Post-backward cleanup (matches LanguageModel::backward() behavior)
    if (training_state.debug_gradient_attribution) {
        training_state.logGradientAttribution(static_cast<int>(step), stream);
    }
    training_state.sequence_weight_count = 0;
    
    // Clear intermediate tensors to free memory
    training_state.autograd_intermediates.clear();
    
    AG_INFO("Training step complete: loss=" << loss_result.loss_value 
            << " grad_norm=" << bwd_result.grad_norm);
    
    return loss_result;
}
}  // namespace Autograd
}  // namespace GRIM
