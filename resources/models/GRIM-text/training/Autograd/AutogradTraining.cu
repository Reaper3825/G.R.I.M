//======================================================//
//  AutogradTraining.cu
//  Implementation of autograd-based training flow
//======================================================//

#include "AutogradTraining.hpp"

// MUST include full definition of GPUGrimEncoder for method calls
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/grim_layer_gpu.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/Gradients/GradientCC_GPU.hpp"
#include "../../Shared/EquationLogging/EquationLogging.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>  // std::clamp, std::min, std::max (+ std::min_element/max_element in diagnostics)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../../Shared/VerboseLogging.hpp"

// Issue #142: DELETED kernelMaskNonContentLogits / launchMaskNonContentLogits.
// Setting special token logits to -inf is NON-STANDARD and was a workaround for
// mode collapse (tok1=PAD at 98% argmax). The standard approach is loss masking
// via target=-1 which is already implemented in:
//   - AutogradLoss.cu: forward skips target==-1, backward zeros grad for target==-1
//   - BatchPayload.cu: defense-masks non-content tokens with target=-1
//   - DataLoader.cu: masks non-content tokens during data loading
// The -inf masking was poisoning every diagnostic that read cached_logits_tensor
// (logit_min=-inf → logit_range=inf → logit_mean=-inf → logit_std=NaN).
// Inference-time masking in grim_language_model_gpu.cu is SEPARATE and stays.

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

// Thin wrapper around shared launchScaleGradients (GradientCC_GPU)
// Accepts size_t for convenience since .numel() returns size_t
[[maybe_unused]] inline void scaleGradBuffer(float* data, size_t n, float scale, cudaStream_t stream) {
    if (!data || n == 0) {
        return;
    }
    launchScaleGradients(data, static_cast<int>(n), scale, stream);
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

// Finding 1 (Rule 26): countValidTokensKernel/countValidTokens DELETED — zero callers
// Finding 2 (Rule 26): sumSquaredKernel/computeSumSquared DELETED — only caller was
//   computeGradientNorm() which is redundant with Phase2's computeGradNorm()

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
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
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
    ctx.embedding_layer = embedding_layer;
    ctx.lm_head = lm_head;
    ctx.scratch_block = scratch_block;
    ctx.reasoning_head = reasoning_head;
    ctx.execution_block = execution_block;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.payload = &payload;
    ctx.batch_size = payload.batch_size;
    ctx.seq_len = payload.max_seq_len;
    ctx.grad_scale = grad_scale;
    ctx.step = step;
    ctx.is_training = is_training;
    
    // Rule 20: Fail loud on invalid context
    ctx.validate("initAutogradContext(payload)");
    
    return ctx;
}

// Inference overload — batch_size/seq_len set directly (no payload)
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
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
    ctx.embedding_layer = embedding_layer;
    ctx.lm_head = lm_head;
    ctx.scratch_block = scratch_block;
    ctx.reasoning_head = reasoning_head;
    ctx.execution_block = execution_block;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.payload = nullptr;  // No payload for inference
    ctx.batch_size = batch_size;
    ctx.seq_len = seq_len;
    ctx.grad_scale = grad_scale;
    ctx.step = step;
    ctx.is_training = is_training;
    
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

    // Skip QKV_EQUATION D2H + fprintf on gradient-accumulation micro-batches (same weights, duplicate output)
    struct EquationLoggingScope {
        bool& ref;
        bool prev;
        explicit EquationLoggingScope(bool skip) : ref(GRIM::getEquationLoggingSkipThisPassRef()), prev(ref) { ref = skip; }
        ~EquationLoggingScope() { ref = prev; }
    };
    EquationLoggingScope eq_scope(ctx.skip_equation_logging);

    // Rule 20: Fail loud
    ctx.validate("executeAutogradForward");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    auto& intermediates = ts->autograd_intermediates;  // All intermediate tensors go HERE
    
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
    // NOTE: cached_token_ids_tensor stores int32 data in float* buffer - cast when accessing
    int* token_ids = reinterpret_cast<int*>(ts->cached_token_ids_tensor.data);
    if (!token_ids) {
        throw std::runtime_error("AutogradForward: cached_token_ids_tensor.data is NULL");
    }
    
    // Embedding weights tensor (owned by EmbeddingLayer — Pattern B)
    Tensor& emb_weights = ctx.embedding_layer->tokenWeights();
    if (!emb_weights.data) {
        throw std::runtime_error("AutogradForward: embedding token_weights.data is NULL");
    }
    emb_weights.requires_grad = ctx.is_training;
    
    // Rule 20: Fail loud on invalid shape - EmbeddingLayer constructor MUST initialize correctly
    if (!emb_weights.shape.is_valid()) {
        throw std::runtime_error("[AutogradTraining] embedding token_weights.shape is INVALID - EmbeddingLayer MUST initialize with correct shape [vocab_size=" 
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
        if (!ctx.embedding_layer->hasPositionEmbeddings()) {
            throw std::runtime_error(
                "AutogradForward: positional_encoding=NONE requires position embeddings, but EmbeddingLayer has none");
        }
        Tensor& pos_weights = ctx.embedding_layer->positionWeights();
        if (!pos_weights.data) {
            throw std::runtime_error(
                "AutogradForward: positional_encoding=NONE requires position_weights.data, but it is NULL");
        }
        pos_weights.requires_grad = ctx.is_training;
        
        // Ensure position embedding weights have correct shape [max_seq_len, d_model]
        if (!pos_weights.shape.is_valid()) {
            pos_weights.shape = TensorContract::TensorShape::make_BSM(
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
            pos_weights,
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
        AG_INFO("Step 1.5: Running ScratchBlock injection...");
        // Drain stale error from embedding/dropout so we only report ScratchBlock failures
        (void)cudaGetLastError();

        const bool exec_first_type_only =
            cfg->execution_block_enabled && cfg->scratch_block_execution_first_type_only;
        if (cfg->execution_block_enabled && !exec_first_type_only) {
            throw std::runtime_error(
                "executeAutogradForward: execution_block_enabled requires "
                "scratch_block_execution_first_type_only=true to prevent value leakage "
                "into hidden states on arithmetic batches");
        }
        intermediates.embedding_tensor = autograd::scratch_block_inject(
            intermediates.embedding_tensor,
            *ctx.scratch_block,
            reinterpret_cast<const int*>(ts->cached_token_ids_tensor.data),
            ts->cached_token_numeric_values.data,
            reinterpret_cast<const uint16_t*>(ts->cached_token_text_features.data),
            reinterpret_cast<const uint8_t*>(ts->cached_token_atom_mask.data),
            reinterpret_cast<const uint32_t*>(ts->cached_token_atom_flags.data),
            ts->cached_token_to_slot_map.data
                ? reinterpret_cast<const int32_t*>(ts->cached_token_to_slot_map.data)
                : nullptr,
            total_tokens,
            ctx.stream,
            exec_first_type_only);

        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("AutogradForward: ScratchBlock CUDA error: " +
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
        // EQUATION: cosine(h_i, h_j) = (h_i · h_j) / (h_rms_i * h_rms_j * d_model)
        // EXPECTED: avg_cos ≈ 1/sqrt(d_model) ≈ 0.036 for random orthogonal vectors
        // ANOMALY: avg_cos > 0.5 indicates high correlation (representational collapse)
        // ═══════════════════════════════════════════════════════════════════════
        {
            const int d_model = cfg->d_model;
            const int sample_pairs = std::min(50, total_tokens / 2);  // Sample pairs for efficiency
            
            if (sample_pairs >= 2) {
                // Compute RMS for each position
                std::vector<float> row_rms(total_tokens);
                for (int t = 0; t < total_tokens; t++) {
                    double norm_sq = 0.0;
                    for (int d = 0; d < d_model; d++) {
                        float v = h_emb[t * d_model + d];
                        norm_sq += v * v;
                    }
                    row_rms[t] = sqrtf(norm_sq / d_model);
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
                    if (i == j || row_rms[i] < 1e-8f || row_rms[j] < 1e-8f) continue;
                    
                    // Compute dot product h_i · h_j
                    double dot = 0.0;
                    for (int d = 0; d < d_model; d++) {
                        dot += h_emb[i * d_model + d] * h_emb[j * d_model + d];
                    }
                    
                    double cosine = dot / (static_cast<double>(row_rms[i]) * row_rms[j] * d_model);
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
                
                fprintf(stderr, "[EMBED_COSINE_EQUATION] BEFORE_ENCODER: cosine(h_i, h_j) = (h_i · h_j) / (h_rms_i * h_rms_j * d_model)\n");
                fprintf(stderr, "  INPUT h (embeddings): shape=[%d, %d] row_rms_range=[%.6f, %.6f]\n",
                        total_tokens, d_model, 
                        *std::min_element(row_rms.begin(), row_rms.end()),
                        *std::max_element(row_rms.begin(), row_rms.end()));
                fprintf(stderr, "  PARAMETERS: sample_pairs=%d, stride=%d\n", num_pairs, stride);
                fprintf(stderr, "  EXPECTED avg_cos = 1/sqrt(d_model)\n");
                fprintf(stderr, "                    = 1/sqrt(%d)\n", d_model);
                fprintf(stderr, "                    = %.6f (for random orthogonal vectors)\n", expected_cos);
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
    
    const int num_layers = ctx.gpu_encoder->getNumLayers();
    intermediates.encoder_layer_outputs.clear();
    intermediates.layer_intermediates.layers.clear();
    intermediates.embedding_tensor.is_leaf = false;
    
    float* encoder_output = nullptr;
    
    if (!ctx.is_training) {
        // ═══════════════════════════════════════════════════════════════════════
        //  NO-GRAD (validation): Do not store full autograd graph.
        //  Reuse a single layer's intermediates and one running tensor so we use
        //  O(1) memory instead of O(num_layers) — avoids training-level memory.
        // ═══════════════════════════════════════════════════════════════════════
        ForwardIntermediates no_grad_layer_storage;
        Tensor running;
        AG_INFO("Step 2: Running " << num_layers << " encoder layers (no_grad, validation)...");
        
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("AutogradForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }
            // Sync before clear so we never free Q/K buffers while previous layer's
            // kernels (RoPE, etc.) are still in flight — avoids illegal memory access.
            if (layer_idx > 0) {
                cudaError_t sync_err = cudaStreamSynchronize(ctx.stream);
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error("AutogradForward(no_grad): cudaStreamSynchronize failed after layer " +
                        std::to_string(layer_idx - 1) + ": " + cudaGetErrorString(sync_err));
                }
            }
            no_grad_layer_storage.clear();
            Tensor& layer_input = (layer_idx == 0) ? intermediates.embedding_tensor : running;
            running = enc_layer->forward(layer_input, ctx.seq_len, ctx.stream, no_grad_layer_storage, ctx.step, layer_idx);
            // Encoder returns a non-owning view of no_grad_layer_storage.output. That storage
            // is cleared next iteration (or destroyed when the loop exits). So running would
            // become a dangling pointer for the next layer OR for post-encoder use (final RMS,
            // LM head) — root cause of inference illegal memory access. Always make running own
            // a copy so we never hold a pointer into no_grad_layer_storage.
            {
                Tensor owned = Tensor::empty(running.shape, false, ctx.stream, "no_grad_layer_output");
                const size_t bytes = static_cast<size_t>(running.shape.total_elements()) * sizeof(float);
                cudaError_t cp_err = cudaMemcpyAsync(owned.data, running.data, bytes, cudaMemcpyDeviceToDevice, ctx.stream);
                if (cp_err != cudaSuccess) {
                    throw std::runtime_error("AutogradForward(no_grad): copy layer output failed: " +
                        std::string(cudaGetErrorString(cp_err)));
                }
                // Ensure copy completes before next iteration's clear() (or storage destructor) frees the source.
                cudaError_t sync_err = cudaStreamSynchronize(ctx.stream);
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error("AutogradForward(no_grad): sync after layer output copy failed: " +
                        std::string(cudaGetErrorString(sync_err)));
                }
                running = std::move(owned);
            }
        }
        
        // Surface any device error from the encoder loop before we use encoder_output.
        {
            cudaError_t enc_sync = cudaStreamSynchronize(ctx.stream);
            if (enc_sync != cudaSuccess) {
                throw std::runtime_error("AutogradForward(no_grad): CUDA error after encoder layers: " +
                    std::string(cudaGetErrorString(enc_sync)) + " (illegal access usually means a kernel wrote/read out of bounds)");
            }
        }
        
        encoder_output = running.data;
        intermediates.encoder_output_tensor = std::move(running);
        intermediates.encoder_output_tensor.requires_grad = false;
        intermediates.encoder_output_tensor.grad_fn.reset();
        intermediates.encoder_output_tensor.stream = ctx.stream;
        AG_INFO("Step 2: All " << num_layers << " encoder layers complete (no_grad)");
    } else {
        // ═══════════════════════════════════════════════════════════════════════
        //  Issue #56: Keep all intermediate tensors alive until backward completes.
        // ═══════════════════════════════════════════════════════════════════════
        intermediates.encoder_layer_outputs.reserve(num_layers);
        intermediates.layer_intermediates.layers.reserve(num_layers);
        
        AG_INFO("Step 2: Running " << num_layers << " encoder layers with autograd...");
        AG_INFO("  embedding_tensor.grad_fn=" << (void*)intermediates.embedding_tensor.grad_fn.get()
                << " requires_grad=" << intermediates.embedding_tensor.requires_grad);
        
        // Resolve execution block layer index
        int exec_layer = -1;
        int exec_K = 0;
        if (ctx.execution_block && cfg->execution_block_enabled && ctx.scratch_block && ctx.scratch_block->isEnabled()) {
            exec_layer = cfg->execution_block_layer;
            if (exec_layer < 0) exec_layer = num_layers - 2;
            if (exec_layer < 0) exec_layer = 0;
            if (exec_layer >= num_layers) exec_layer = num_layers - 1;
            exec_K = cfg->execution_block_num_steps;
        }

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("AutogradForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }
            
            intermediates.layer_intermediates.layers.emplace_back();
            ForwardIntermediates& layer_storage = intermediates.layer_intermediates.layers.back();
            
            Tensor& layer_input = (layer_idx == 0)
                ? intermediates.embedding_tensor
                : intermediates.encoder_layer_outputs.back();
            
            Tensor layer_output = enc_layer->forward(layer_input, ctx.seq_len, ctx.stream, layer_storage, ctx.step, layer_idx);
            
            if (cfg->residual_dropout_rate > 0.0f && ctx.is_training) {
                const uint64_t residual_drop_seed = ctx.step * 2654435761ULL + 7000 + static_cast<uint64_t>(layer_idx) * 131;
                layer_output = autograd::dropout(layer_output, cfg->residual_dropout_rate,
                                                 residual_drop_seed, ctx.is_training, ctx.stream);
            }

            // ExecutionBlock: run K execution steps at the configured layer
            // Per-row isolation: each batch row gets its own ExecutionMemory
            if (layer_idx == exec_layer && ctx.execution_block) {
                int num_atoms = 0;
                cudaMemcpyAsync(&num_atoms, ctx.scratch_block->numAtomsBuffer(),
                                sizeof(int), cudaMemcpyDeviceToHost, ctx.stream);
                cudaStreamSynchronize(ctx.stream);

                const int ae = cfg->scratch_block_atom_embedding_dim;
                const int V = cfg->execution_block_num_slots;
                const int dk = cfg->execution_block_d_key;
                const int dt = cfg->execution_block_d_type;
                const int B  = ctx.batch_size;
                const int sl = ctx.seq_len;

                intermediates.exec_memories.resize(B);
                intermediates.exec_outputs_per_row.resize(B);

                float T = cfg->execution_block_temp_start;

                const int32_t* d_slot_map_full = ts->cached_token_to_slot_map.data
                    ? reinterpret_cast<const int32_t*>(ts->cached_token_to_slot_map.data)
                    : nullptr;

                // Initialize persistent execution trace per row
                ts->execution_trace_by_row.resize(B);
                ts->trace_state_by_row.resize(B);
                for (int b = 0; b < B; ++b) {
                    ts->execution_trace_by_row[b].clear();
                    ts->trace_state_by_row[b] = Tensor::zeros({1, cfg->d_model}, ctx.stream, "trace_state_row");
                    ts->trace_state_by_row[b].requires_grad_();
                    ts->trace_state_by_row[b].ensure_grad();
                }

                // Teacher target buffer for causal loss (Fix 6)
                const bool have_exec_teacher = (ctx.payload && !ctx.payload->teacher_steps.empty());
                const int B_teacher = have_exec_teacher
                    ? static_cast<int>(ctx.payload->teacher_steps.size()) : 0;
                float* d_expected_target_buf = nullptr;
                CUDA_CHECK(cudaMalloc(&d_expected_target_buf, sizeof(float)));

                for (int b = 0; b < B; ++b) {
                    auto& M_b = intermediates.exec_memories[b];
                    M_b.allocate(V, ae, cfg->d_model, dk, dt, ctx.stream);
                    M_b.clear(ctx.stream);
                    intermediates.exec_outputs_per_row[b].steps.clear();

                    const int tok_off = b * sl;

                    if (d_slot_map_full && ts->cached_token_numeric_values.data) {
                        ctx.execution_block->bootstrapMemoryFromSlotMap(
                            M_b,
                            ts->cached_token_numeric_values.data + tok_off,
                            d_slot_map_full + tok_off,
                            sl, ctx.stream);
                    }

                    for (int step = 0; step < exec_K; ++step) {
                        ExecutionBlockStepOutput step_diag;

                        // Upload teacher expected_value for this step (Fix 6)
                        const float* d_expected_target = nullptr;
                        if (have_exec_teacher && b < B_teacher) {
                            const auto& teacher_row = ctx.payload->teacher_steps[b];
                            if (step < static_cast<int>(teacher_row.size())) {
                                float h_val = teacher_row[step].expected_value;
                                cudaMemcpyAsync(d_expected_target_buf, &h_val,
                                                sizeof(float), cudaMemcpyHostToDevice, ctx.stream);
                                d_expected_target = d_expected_target_buf;
                            }
                        }

                        ctx.execution_block->executeStep(
                            layer_output, M_b,
                            ctx.scratch_block->atomEmbeddingsBuffer(),
                            ctx.scratch_block->atomPositionsBuffer(),
                            d_slot_map_full,
                            num_atoms, total_tokens,
                            step, T, ctx.stream,
                            &step_diag,
                            tok_off, sl,
                            ts->trace_state_by_row[b],
                            ts->execution_trace_by_row[b],
                            d_expected_target);
                        ts->execution_trace_by_row[b].push_back(step_diag.record);
                        intermediates.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
                    }
                }
                cudaFreeAsync(d_expected_target_buf, ctx.stream);
            }

            // ExecutionBlock: gated cross-attention read at every layer >= exec_layer (per-row)
            if (exec_layer >= 0 && layer_idx >= exec_layer
                && ctx.execution_block
                && !intermediates.exec_memories.empty()) {
                const int B  = ctx.batch_size;
                const int sl = ctx.seq_len;
                for (int b = 0; b < B; ++b) {
                    ctx.execution_block->crossAttentionRead(
                        layer_output, intermediates.exec_memories[b],
                        total_tokens, ctx.stream,
                        b * sl, sl);
                }
            }
            
            intermediates.encoder_layer_outputs.push_back(std::move(layer_output));
        }
        
        AG_INFO("Step 2: All " << num_layers << " encoder layers complete");
        encoder_output = intermediates.encoder_layer_outputs.back().data;
    }
    
    // Final encoder output pointer for LM head and diagnostics
    result.encoder_output = encoder_output;
    
    // Copy to scratch buffer for diagnostics and inference
    if (ts->cached_encoder_output.data) {
        cudaMemcpyAsync(ts->cached_encoder_output.data, encoder_output,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 3+4: LM Head → Logits (via PERSISTENT LMHeadLayer)
    //
    //  Pattern B: LMHeadLayer is constructed ONCE in initGPU() and stored on
    //  LanguageModel. The layer self-allocates weights (or aliases embedding
    //  weights for tied config) and owns final_rms_gamma.
    //
    //  LMHeadLayer::forward() encapsulates the FULL pipeline:
    //    Step 0: RMSNorm (final_rms_gamma)
    //    Step 1: Optional column+row centering (Issues #125/#132)
    //    Step 2: Linear projection: logits = centered_encoder @ W^T
    //    Step 3: Optional logit centering (numerical stability)
    //    Step 4: Optional bias addition
    //
    //  Autograd graph built inside forward() - backward handled automatically.
    // ═══════════════════════════════════════════════════════════════════════════
    
    // When training: create encoder output tensor from last layer (preserves grad_fn chain).
    // When validation (no_grad): intermediates.encoder_output_tensor already set in no_grad path above.
    if (ctx.is_training) {
        const bool lmhead_track_grad = true;
        Tensor encoder_output_tensor = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,
            lmhead_track_grad,
            "encoder_output_for_lmhead"
        );
        encoder_output_tensor.is_leaf = false;
        encoder_output_tensor.stream = ctx.stream;
        encoder_output_tensor.grad_fn = intermediates.encoder_layer_outputs.back().grad_fn;
        intermediates.encoder_output_tensor = std::move(encoder_output_tensor);
    }
    // else: no_grad path already set intermediates.encoder_output_tensor and cleared grad_fn

    // Update persistent LM head with current stream/cublas (may differ between train/inference)
    ctx.lm_head->setStream(ctx.stream);
    ctx.lm_head->setCublasHandle(ctx.cublas_handle);

    // Inference path needs cached_logits for return value; training reads from autograd_intermediates directly
    float* logits_output = ctx.is_training ? nullptr : ts->cached_logits_tensor.data;
    if (!ctx.is_training && !logits_output) {
        throw std::runtime_error("AutogradTraining: cached_logits_tensor buffer is NULL - TrainingState MUST allocate logits buffer for inference");
    }
    
    // Forward pass: builds autograd graph through RMSNorm → centering → matmul → bias
    Tensor logits_tensor = ctx.lm_head->forward(intermediates.encoder_output_tensor, intermediates.centered_encoder_output);
    
    // Copy centered data to scratch buffer for diagnostics (Issue #115)
    if (cfg->lm_head_center_hidden_states && ts->centering_scratch_tensor.data && intermediates.centered_encoder_output.data) {
        cudaMemcpyAsync(ts->centering_scratch_tensor.data, intermediates.centered_encoder_output.data,
                       static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                       cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // Pointer for diagnostic reads (centered if enabled, raw encoder output otherwise)
    float* lm_input_ptr = cfg->lm_head_center_hidden_states 
        ? intermediates.centered_encoder_output.data 
        : intermediates.encoder_output_tensor.data;
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  RULE 21 DIAGNOSTIC: Why is token 277 (or any token) the argmax?
    //  (GUARDED: set ENABLE_EXPENSIVE_DIAGNOSTICS=true in VerboseLogging.hpp to enable)
    // ═══════════════════════════════════════════════════════════════════════════
    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        //  Equation: logit[v] = Σ_d encoder[pos, d] × W[v, d]
        //  
        //  Per-position argmax analysis: WHY did position choose its predicted token?
        //  The argmax wins because: Σ_d h[d] × W[argmax, d] > Σ_d h[d] × W[v, d] for all v
        //  
        //  This diagnostic shows:
        //  1. Hidden state (encoder output) statistics at sample position
        //  2. Weight row statistics for the predicted argmax token at that position  
        //  3. Dot product decomposition showing WHY argmax wins
        constexpr int kSamplePositions = 5;  // Sample first 5 positions
        const int d_model = cfg->d_model;
        const int vocab_size_local = cfg->vocab_size;
        
        // Copy sample data to host
        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_logits(sample_size * vocab_size_local);
        
        cudaMemcpyAsync(h_encoder.data(), lm_input_ptr,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_logits.data(), logits_tensor.data,
                        sample_size * vocab_size_local * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        
        for (int pos = 0; pos < sample_size; ++pos) {
            const float* h = h_encoder.data() + pos * d_model;
            const float* logits_row = h_logits.data() + pos * vocab_size_local;
            
            // Find argmax and its logit FIRST (since we need to fetch W[argmax])
            int argmax_token = 0;
            float max_logit_val = logits_row[0];
            for (int v = 1; v < vocab_size_local; ++v) {
                if (logits_row[v] > max_logit_val) {
                    max_logit_val = logits_row[v];
                    argmax_token = v;
                }
            }
            
            // Fetch W[argmax] row from device
            std::vector<float> h_weights_argmax(d_model);
            cudaMemcpyAsync(h_weights_argmax.data(), 
                            ctx.lm_head->weights().data + static_cast<size_t>(argmax_token) * d_model,
                            d_model * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            
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
            
            // Compute W[argmax] statistics
            float w_sum = 0.0f, w_sum_sq = 0.0f, w_min = h_weights_argmax[0], w_max = h_weights_argmax[0];
            for (int d = 0; d < d_model; ++d) {
                w_sum += h_weights_argmax[d];
                w_sum_sq += h_weights_argmax[d] * h_weights_argmax[d];
                w_min = std::min(w_min, h_weights_argmax[d]);
                w_max = std::max(w_max, h_weights_argmax[d]);
            }
            float w_mean = w_sum / d_model;
            float w_rms = std::sqrt(w_sum_sq / d_model);
            
            // Compute dot product decomposition for argmax token
            float dot_product_argmax = 0.0f;
            float positive_contrib = 0.0f, negative_contrib = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                float contrib = h[d] * h_weights_argmax[d];
                dot_product_argmax += contrib;
                if (contrib > 0) positive_contrib += contrib;
                else negative_contrib += contrib;
            }
            
            // Compute cosine similarity between h and W[argmax]
            // cos = dot / (rms_h * rms_w * d_model)  [equivalent to dot / (||h|| * ||w||)]
            float h_rms_val = std::sqrt(h_sum_sq / d_model);
            float w_rms_val = std::sqrt(w_sum_sq / d_model);
            float cosine_sim = (h_rms_val > 1e-8f && w_rms_val > 1e-8f) 
                               ? (dot_product_argmax / (h_rms_val * w_rms_val * d_model)) : 0.0f;
            
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
            
            // Analyze weight row for the argmax token (what was actually predicted)
            fprintf(stderr, "WEIGHT ROW W[%d] (predicted token):\n", argmax_token);
            fprintf(stderr, "  Statistics: mean=%.6f rms=%.6f range=[%.6f, %.6f]\n",
                    w_mean, w_rms, w_min, w_max);
            fprintf(stderr, "  ├─ Initialized to mean≈0 (ideal), rms controls sensitivity\n");
            fprintf(stderr, "  └─ Alignment with h determines dot product magnitude\n");
            
            // Decompose the dot product
            fprintf(stderr, "DOT_PRODUCT ANALYSIS Σ_d h[d]×W[%d,d]:\n", argmax_token);
            fprintf(stderr, "  Raw computation: %.6f\n", dot_product_argmax);
            fprintf(stderr, "  ├─ Positive contributions (h×W>0): %.6f (%.1f%%)\n", 
                    positive_contrib, 100.0f * positive_contrib / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Negative contributions (h×W<0): %.6f (%.1f%%)\n",
                    negative_contrib, 100.0f * std::abs(negative_contrib) / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Cosine alignment: %.6f (1.0=perfect alignment, 0=orthogonal, -1=opposite)\n", cosine_sim);
            fprintf(stderr, "  └─ Interpretation: h and W[%d] are %.1f%% aligned\n", argmax_token, 100.0f * cosine_sim);
            
            // Result summary
            fprintf(stderr, "RESULT:\n");
            fprintf(stderr, "  logit[%d]=%.6f (PREDICTED argmax token)\n", argmax_token, max_logit_val);
            fprintf(stderr, "  ├─ Token %d wins because: cos(h, W[%d])=%.3f (alignment) × magnitude\n", 
                    argmax_token, argmax_token, cosine_sim);
            fprintf(stderr, "  └─ Hidden state h[%d] points in direction captured by W[%d]\n", pos, argmax_token);
            
            // Reflection on what this means  
            fprintf(stderr, "REFLECTION:\n");
            if (cosine_sim > 0.5f) {
                fprintf(stderr, "  ⚠️  High alignment (cos=%.3f > 0.5) detected:\n", cosine_sim);
                fprintf(stderr, "     - Hidden state strongly aligned with predicted token weight\n");
                fprintf(stderr, "     - This is expected behavior for confident predictions\n");
                fprintf(stderr, "     - Mode collapse: check if SAME token wins across many positions\n");
            } else if (cosine_sim < 0.1f) {
                fprintf(stderr, "  ⚠️  Low alignment (cos=%.3f < 0.1) detected:\n", cosine_sim);
                fprintf(stderr, "     - Prediction driven more by magnitude than direction\n");
                fprintf(stderr, "     - May indicate weak position differentiation\n");
            } else {
                fprintf(stderr, "  ✓ Moderate alignment (cos=%.3f) is healthy\n", cosine_sim);
            }
            fprintf(stderr, "\n");
        }
    }  // end ENABLE_EXPENSIVE_DIAGNOSTICS guard
    
    // Shape validation, centering, bias addition all handled by LMHeadLayer::forward()

    // Inference: copy logits to pre-allocated buffer (inference return path reads cached_logits_tensor)
    // Training: skip D2D copy — Phase2 diagnostics read directly from autograd_intermediates.logits_tensor
    if (!ctx.is_training) {
        cudaMemcpyAsync(logits_output, logits_tensor.data,
                        logits_tensor.shape.total_elements() * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // Move the autograd tensor to intermediates (preserves grad_fn chain)
    intermediates.logits_tensor = std::move(logits_tensor);
    
    // Reasoning head: gated OFF when execution_block_enabled (single execution decision source)
    if (ctx.reasoning_head && ctx.scratch_block && ctx.scratch_block->isEnabled()
        && !cfg->execution_block_enabled) {
        // D2H num_atoms — numAtomsBuffer() is device memory
        int num_atoms = 0;
        cudaMemcpyAsync(&num_atoms, ctx.scratch_block->numAtomsBuffer(),
                        sizeof(int), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);

        if (num_atoms > 0) {
            const int atom_dim = cfg->scratch_block_atom_embedding_dim;

            // Create canonical copy-first atom embeddings tensor (AutogradIntermediates owns it)
            intermediates.scratch_atom_embeddings = Tensor::empty(
                TensorContract::TensorShape::make_BSM(num_atoms, atom_dim),
                ctx.is_training, ctx.stream, "scratch_atom_embeddings");
            cudaMemcpyAsync(
                intermediates.scratch_atom_embeddings.data,
                ctx.scratch_block->atomEmbeddingsBuffer(),
                static_cast<size_t>(num_atoms) * atom_dim * sizeof(float),
                cudaMemcpyDeviceToDevice, ctx.stream);

            ctx.reasoning_head->setStream(ctx.stream);
            ctx.reasoning_head->setCublasHandle(ctx.cublas_handle);

            ReasoningHeadOutput rh_out = ctx.reasoning_head->forward(
                intermediates.encoder_output_tensor,
                intermediates.scratch_atom_embeddings,
                ctx.scratch_block->atomPositionsBuffer(),
                num_atoms,
                total_tokens,
                ctx.stream);
            intermediates.reasoning_output = std::move(rh_out);
        }
    }
    
    AG_INFO("Forward complete: logits shape=[" << total_tokens << ", " << cfg->vocab_size << "]");
    
    result.success = true;
    return result;
}

//======================================================================
// Loss Config Builder (single conversion point)
//======================================================================

autograd::LossConfig buildLossConfig(const LossContext::LossOptions& opts, const float* d_class_weights) {
    autograd::LossConfig lc{};
    lc.focal_enabled       = opts.focal_enabled;
    lc.focal_alpha         = opts.focal_enabled ? opts.focal_alpha : 1.0f;
    lc.focal_gamma         = opts.focal_enabled ? opts.focal_gamma : 0.0f;
    lc.smoothing_enabled   = opts.label_smoothing_enabled;
    lc.smoothing_epsilon   = opts.label_smoothing_enabled ? opts.label_smoothing_epsilon : 0.0f;
    lc.entropy_reg_enabled = opts.entropy_reg_enabled;
    lc.entropy_reg_lambda  = opts.entropy_reg_enabled ? opts.entropy_reg_lambda : 0.0f;
    lc.class_balanced_enabled = opts.class_balanced_enabled;
    lc.d_class_weights     = opts.class_balanced_enabled ? d_class_weights : nullptr;
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

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. MTP (multi-token prediction) auxiliary losses: L_total += α/K * Σ_k L_k
    //
    // FIX (A1): MTP heads MUST consume the same representation as the LM head.
    // Previously MTP used raw encoder_output_tensor while the LM head applies
    // RMSNorm + optional center/PC1. That mismatch sent conflicting gradients to
    // the encoder. Now we use the exact same matmul input the LM head uses.
    // ═══════════════════════════════════════════════════════════════════════════
    ts->mtp_diagnostics.valid = false;
    if (ctx.model && cfg->mtp_enabled && cfg->mtp_k > 0 &&
        intermediates.encoder_output_tensor.data && ts->mtp_shifted_targets_tensor.data) {
        float L0_main = 0.0f;
        cudaMemcpyAsync(&L0_main, intermediates.loss_tensor.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        ts->mtp_diagnostics.L0_main = L0_main;
        if (!std::isfinite(L0_main)) {
            throw std::runtime_error("computeAutogradLoss: main CE loss (L0_main) is non-finite (" + std::to_string(L0_main) +
                ") — unified_loss failed before MTP. num_tokens=" + std::to_string(total_tokens) + " vocab=" + std::to_string(vocab_size));
        }
        const float alpha_effective = cfg->mtp_alpha * std::min(1.0f,
            static_cast<float>(ctx.step) / static_cast<float>(cfg->mtp_alpha_warmup_steps > 0 ? cfg->mtp_alpha_warmup_steps : 1));
        const int K = cfg->mtp_k;
        const float scale = (K > 0 && alpha_effective > 0.0f) ? (alpha_effective / static_cast<float>(K)) : 0.0f;
        intermediates.mtp_logits_tensors.clear();
        ts->mtp_diagnostics.head_loss.clear();
        ts->mtp_diagnostics.head_acc.clear();
        ts->mtp_diagnostics.alpha_effective = alpha_effective;

        // Resolve mtp_input: same representation as LM head matmul input (A1 fix)
        const Tensor* mtp_input = nullptr;
        if (intermediates.centered_encoder_output.data) {
            // LM head used center_hidden_states or project_out_pc1 — use that buffer
            mtp_input = &intermediates.centered_encoder_output;
        } else if (ctx.lm_head->config().has_final_rms_norm && ctx.lm_head->finalRmsGamma().data) {
            // LM head used only RMSNorm — apply it so MTP sees the same normalized representation
            intermediates.mtp_input_tensor = autograd::rms_norm(
                intermediates.encoder_output_tensor,
                ctx.lm_head->finalRmsGamma(),
                ctx.lm_head->config().rms_epsilon,
                ctx.stream
            );
            mtp_input = &intermediates.mtp_input_tensor;
        } else {
            mtp_input = &intermediates.encoder_output_tensor;
        }

        for (int k = 0; k < K && scale > 0.0f; ++k) {
            LanguageModel::MTPHead* head = ctx.model->getMtpHead(k);
            if (!head || !head->weight.data || !head->bias.data) continue;
            const int shift = k + 1;
            autograd::launchShiftTargetsKernel(
                targets,
                reinterpret_cast<int*>(ts->mtp_shifted_targets_tensor.data),
                total_tokens,
                ctx.seq_len,
                shift,
                ctx.stream
            );
            Tensor logits_k = autograd::matmul(
                *mtp_input,
                head->weight,
                ctx.stream,
                mtp_input->data,
                nullptr,
                true
            );
            logits_k = autograd::broadcast_add(logits_k, head->bias, ctx.stream);
            intermediates.mtp_logits_tensors.push_back(std::move(logits_k));
            Tensor loss_k = autograd::unified_loss(
                intermediates.mtp_logits_tensors.back(),
                reinterpret_cast<const int*>(ts->mtp_shifted_targets_tensor.data),
                nullptr,
                total_tokens,
                vocab_size,
                ctx.loss_config,
                ctx.stream
            );
            float h_loss_k = 0.0f;
            cudaMemcpyAsync(&h_loss_k, loss_k.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
            int* d_correct = nullptr;
            int* d_valid = nullptr;
            cudaMalloc(&d_correct, sizeof(int));
            cudaMalloc(&d_valid, sizeof(int));
            autograd::launchMTPAccuracyKernel(
                intermediates.mtp_logits_tensors.back().data,
                reinterpret_cast<const int*>(ts->mtp_shifted_targets_tensor.data),
                total_tokens,
                vocab_size,
                d_correct,
                d_valid,
                ctx.stream
            );
            cudaStreamSynchronize(ctx.stream);
            if (!std::isfinite(h_loss_k)) {
                throw std::runtime_error("computeAutogradLoss: MTP head k=" + std::to_string(k) +
                    " loss is non-finite (" + std::to_string(h_loss_k) + ") — shift=" + std::to_string(shift));
            }
            ts->mtp_diagnostics.head_loss.push_back(h_loss_k);
            int h_correct = 0, h_valid = 0;
            cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_correct);
            cudaFree(d_valid);
            float acc_k = (h_valid > 0) ? (static_cast<float>(h_correct) / static_cast<float>(h_valid)) * 100.0f : 0.0f;
            ts->mtp_diagnostics.head_acc.push_back(acc_k);
            Tensor scaled_k = autograd::scale_scalar(loss_k, scale, ctx.stream);
            intermediates.loss_tensor = autograd::add(intermediates.loss_tensor, scaled_k, ctx.stream);
        }
        ts->mtp_diagnostics.valid = !ts->mtp_diagnostics.head_loss.empty();
    }
    
    // Issue both loss D2H copies before a single sync (batch sync for text + numeric)
    float text_loss = 0.0f;
    cudaMemcpyAsync(&text_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);

    cudaStreamSynchronize(ctx.stream);

    if (ts->mtp_diagnostics.valid) {
        ts->mtp_diagnostics.L_total = text_loss;
    }

    if (!std::isfinite(text_loss)) {
        std::string msg = "computeAutogradLoss: text_loss is non-finite (" + std::to_string(text_loss) + ")";
        if (ts->mtp_diagnostics.valid) {
            msg += " L0_main=" + std::to_string(ts->mtp_diagnostics.L0_main);
            for (size_t i = 0; i < ts->mtp_diagnostics.head_loss.size(); ++i)
                msg += " head_loss[" + std::to_string(i) + "]=" + std::to_string(ts->mtp_diagnostics.head_loss[i]);
        }
        throw std::runtime_error(msg);
    }

    result.text_loss = text_loss;
    result.valid_tokens = valid_tokens;
    ts->cached_loss_value = text_loss;
    ts->cached_text_loss = text_loss;
    ts->cached_valid_tokens = valid_tokens;

    constexpr float numeric_loss = 0.0f;

    // ═══════════════════════════════════════════════════════════════════════════
    // STRUCTURED CE LOSS on ExecutionBlock outputs (REPLACES entropy as primary)
    // Per (batch_row, step): CE(p_op, op_id) + CE(p_arg1, arg1) + CE(p_arg2, arg2) + CE(p_write, write)
    // Step X/Y multipliers amplify CE terms on value mismatch.
    // ═══════════════════════════════════════════════════════════════════════════
    float exec_structured_ce = 0.0f;
    float exec_entropy_loss = 0.0f;
    float exec_causal_loss = 0.0f;

    if (cfg->execution_block_enabled && ctx.execution_block
        && !intermediates.exec_outputs_per_row.empty()) {

        const int V   = cfg->execution_block_num_slots;
        const int S   = 0; // TODO: read from ExecutionBlockConfig if scratch slots are used
        const int V_val = V - S;
        const int nop = cfg->execution_block_num_ops;
        const float m_x = cfg->step_x_multiplier;
        const float m_y = cfg->step_y_multiplier;
        const bool override_x = cfg->step_y_overrides_x;
        const float eps_match = cfg->value_match_epsilon;

        const bool have_teacher = (ctx.payload && !ctx.payload->teacher_steps.empty());

        float total_ce = 0.0f;
        int ce_count = 0;

        for (int b = 0; b < ctx.batch_size; ++b) {
            const auto& row_steps = intermediates.exec_outputs_per_row[b].steps;
            const auto* teacher_row = (have_teacher && b < static_cast<int>(ctx.payload->teacher_steps.size()))
                ? &ctx.payload->teacher_steps[b] : nullptr;

            for (int k = 0; k < static_cast<int>(row_steps.size()); ++k) {
                const auto& sout = row_steps[k];
                if (!sout.p_op.data || !sout.p_arg1.data || !sout.p_arg2.data || !sout.p_write.data)
                    continue;

                if (!teacher_row || k >= static_cast<int>(teacher_row->size()))
                    continue;

                const auto& ts_k = (*teacher_row)[k];

                // Host-side copies of probability distributions
                std::vector<float> h_p_op(nop), h_p_arg1(V_val), h_p_arg2(V_val), h_p_write(V);
                cudaMemcpyAsync(h_p_op.data(), sout.p_op.data, nop * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                cudaMemcpyAsync(h_p_arg1.data(), sout.p_arg1.data, V_val * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                cudaMemcpyAsync(h_p_arg2.data(), sout.p_arg2.data, V_val * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                cudaMemcpyAsync(h_p_write.data(), sout.p_write.data, V * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                cudaStreamSynchronize(ctx.stream);

                auto safe_nll = [](const std::vector<float>& p, int target) -> float {
                    if (target < 0 || target >= static_cast<int>(p.size())) return 20.0f;
                    float prob = p[target];
                    if (prob < 1e-10f) prob = 1e-10f;
                    return -logf(prob);
                };

                float ce_op = safe_nll(h_p_op, ts_k.op_id);
                float ce_arg1 = safe_nll(h_p_arg1, ts_k.arg1_slot - S);
                float ce_arg2 = safe_nll(h_p_arg2, ts_k.arg2_slot - S);
                float ce_write = safe_nll(h_p_write, ts_k.write_slot);

                // Value mismatch check for Step X / Y multipliers
                bool value_match = true;
                if (sout.state_after_values.data) {
                    float v_out_host = 0.0f;
                    cudaMemcpy(&v_out_host, sout.state_after_values.data + ts_k.write_slot,
                               sizeof(float), cudaMemcpyDeviceToHost);
                    float diff = fabsf(v_out_host - ts_k.expected_value);
                    if (ts_k.expected_value == truncf(ts_k.expected_value)) {
                        value_match = (v_out_host == ts_k.expected_value);
                    } else {
                        value_match = (diff <= eps_match);
                    }
                }

                if (!value_match) {
                    if (override_x) {
                        ce_op    *= m_y;
                        ce_arg1  *= m_y;
                        ce_arg2  *= m_y;
                        ce_write *= m_y;
                    } else {
                        ce_op    *= m_y;
                        ce_write *= m_y;
                        ce_arg1  *= m_x * m_y;
                        ce_arg2  *= m_x * m_y;
                    }
                }

                total_ce += ce_op + ce_arg1 + ce_arg2 + ce_write;
                ce_count++;
            }
        }

        if (ce_count > 0) {
            exec_structured_ce = total_ce / static_cast<float>(ce_count);
        }

        // Optional entropy auxiliary (non-differentiable monitoring term)
        if (cfg->entropy_aux_weight > 0.0f) {
            for (int b = 0; b < ctx.batch_size; ++b) {
                Tensor ent = ctx.execution_block->computeEntropyLoss(
                    intermediates.exec_outputs_per_row[b].steps,
                    cfg->entropy_aux_weight,
                    ctx.stream);
                float h_ent = 0.0f;
                cudaStreamSynchronize(ctx.stream);
                cudaMemcpy(&h_ent, ent.data, sizeof(float), cudaMemcpyDeviceToHost);
                exec_entropy_loss += h_ent;
            }
            if (ctx.batch_size > 0)
                exec_entropy_loss /= static_cast<float>(ctx.batch_size);
        }

        // L_exec: merge causal state losses into autograd loss_tensor (Fixes 1-9)
        float exec_causal_loss_sum = 0.0f;
        int exec_causal_count = 0;
        for (int b = 0; b < ctx.batch_size; ++b) {
            const auto& row_steps = intermediates.exec_outputs_per_row[b].steps;
            for (int k = 0; k < static_cast<int>(row_steps.size()); ++k) {
                const auto& sout = row_steps[k];

                // Add transition_loss to autograd graph (primary gradient signal)
                if (sout.transition_loss.data && sout.transition_loss.grad_fn) {
                    auto scaled = autograd::scale_scalar(
                        sout.transition_loss,
                        cfg->execution_block_causal_w1_transition,
                        ctx.stream);
                    intermediates.loss_tensor = autograd::add(
                        intermediates.loss_tensor, scaled, ctx.stream);
                }

            }
        }
        exec_causal_loss = (exec_causal_count > 0)
            ? exec_causal_loss_sum / static_cast<float>(exec_causal_count) : 0.0f;
    }

    // Optional Spec Step 9: final slot vs target MSE penalty
    float final_consistency_loss = 0.0f;
    if (cfg->final_slot_consistency_weight > 0.0f
        && cfg->execution_block_enabled && ctx.execution_block
        && !intermediates.exec_outputs_per_row.empty()
        && ctx.payload && !ctx.payload->teacher_steps.empty()) {
        float mse_sum = 0.0f;
        int mse_count = 0;
        for (int b = 0; b < ctx.batch_size; ++b) {
            const auto& row_steps = intermediates.exec_outputs_per_row[b].steps;
            const auto& teacher_row = ctx.payload->teacher_steps[b];
            if (row_steps.empty() || teacher_row.empty()) continue;
            const auto& last_step = row_steps.back();
            const auto& last_teacher = teacher_row.back();
            if (last_step.state_after_values.data) {
                float v_final = 0.0f;
                cudaMemcpy(&v_final, last_step.state_after_values.data + last_teacher.write_slot,
                           sizeof(float), cudaMemcpyDeviceToHost);
                float diff = v_final - last_teacher.expected_value;
                mse_sum += diff * diff;
                mse_count++;
            }
        }
        if (mse_count > 0) {
            final_consistency_loss = cfg->final_slot_consistency_weight *
                (mse_sum / static_cast<float>(mse_count));
        }
    }

    result.numeric_loss = numeric_loss;
    result.loss_value = text_loss + exec_structured_ce + exec_entropy_loss
                      + exec_causal_loss + final_consistency_loss;
    result.weight_text = 1.0f;
    
    if (!std::isfinite(result.loss_value)) {
        throw std::runtime_error("computeAutogradLoss: combined loss is non-finite (text=" 
            + std::to_string(text_loss) + " numeric=" + std::to_string(numeric_loss)
            + " exec_ce=" + std::to_string(exec_structured_ce)
            + " exec_entropy=" + std::to_string(exec_entropy_loss)
            + " exec_causal=" + std::to_string(exec_causal_loss)
            + " final_consistency=" + std::to_string(final_consistency_loss) + ")");
    }
    
    AG_INFO("Loss computed: text_ce=" << text_loss << " exec_ce=" << exec_structured_ce
            << " exec_causal=" << exec_causal_loss
            << " exec_entropy_aux=" << exec_entropy_loss << " valid_tokens=" << valid_tokens);
    
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
    result.grad_rms = 0.0f;
    
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

    // Zero gradients if not accumulating
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (!accumulate) {
        // Top-level parameters (Pattern B: owned by EmbeddingLayer)
        ctx.embedding_layer->tokenWeights().zero_grad(ctx.stream);
        if (ctx.embedding_layer->hasPositionEmbeddings()) {
            ctx.embedding_layer->positionWeights().zero_grad(ctx.stream);
        }
        
        // LM Head parameters (Pattern B: owned by persistent LMHeadLayer)
        ctx.lm_head->weights().zero_grad(ctx.stream);
        ctx.lm_head->bias().zero_grad(ctx.stream);
        ctx.lm_head->finalRmsGamma().zero_grad(ctx.stream);

        // Encoder parameters
        if (ctx.gpu_encoder) {
            const int num_layers = ctx.gpu_encoder->getNumLayers();
            for (int layer = 0; layer < num_layers; ++layer) {
                auto* enc = ctx.gpu_encoder->getLayer(layer);
                if (!enc) continue;
                enc->rms1Gamma().zero_grad(ctx.stream);
                enc->rms2Gamma().zero_grad(ctx.stream);
                // Issue #148: Sandwich norm gammas REMOVED
                enc->attnWqkv().zero_grad(ctx.stream);
                enc->attnBqkv().zero_grad(ctx.stream);
                enc->attnWo().zero_grad(ctx.stream);
                enc->attnBo().zero_grad(ctx.stream);
                enc->ffnWGate().zero_grad(ctx.stream);
                enc->ffnW1().zero_grad(ctx.stream);
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
        }

        // Numeric head parameters
        // MTP head parameters
        if (ctx.model && ctx.model->getMtpK() > 0) {
            for (int k = 0; k < ctx.model->getMtpK(); ++k) {
                LanguageModel::MTPHead* head = ctx.model->getMtpHead(k);
                if (head) {
                    if (head->weight.data) head->weight.zero_grad(ctx.stream);
                    if (head->bias.data) head->bias.zero_grad(ctx.stream);
                }
            }
        }
    }
    
    // Call backward on the text loss (single loss path)
    // Starting with ctx.grad_scale (usually 1/accumulation_steps)
    AG_INFO("Calling loss_tensor.backward(nullptr, " << ctx.grad_scale << ")...");
    intermediates.loss_tensor.backward(nullptr, ctx.grad_scale);
    AG_INFO("loss_tensor.backward() returned successfully");

    // ScratchBlock backward is now automatic via ScratchBlockGradFn in the autograd chain.
    // loss_tensor.backward() → ... → ScratchBlockGradFn::apply() handles parameter gradients.
    
    // Verify gradients are properly connected before optimizer runs
    AG_INFO("Verifying gradients are connected to optimizer...");
    if (!verifyGradientsAreConnected(ctx)) {
        result.error_message = "Gradient connectivity verification failed";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    AG_INFO("Gradient connectivity verified");
    
    // ISSUE #149: Manual parameter gradient scaling REMOVED.
    // We now scale the root gradient (the loss) at the start of backward()
    // which propagates the scale through the entire computation graph.
    // This is mathematically equivalent, more efficient (fewer kernels),
    // and safer against omission bugs when adding new layers.
    
    AG_INFO("Backward complete");
    
    result.success = true;
    return result;
}

//======================================================================
// Helper Functions
//======================================================================

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    (void)ctx.training_state;
    (void)ctx.config;
    bool ok = true;
    
    // The autograd system stores gradients in Tensor.grad_ fields (shared_ptr<Tensor>)
    // The optimizer accesses them directly via Tensor.grad_data() pointers.
    // 
    // This function verifies that gradients are properly allocated and accessible.
    // It does NOT copy — gradients are already wired up during initialization.
    // This is a diagnostic check to catch pointer setup bugs before optimizer runs.
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Embedding gradients (may be tied with LM head) — Pattern B: owned by EmbeddingLayer
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ctx.embedding_layer->tokenWeights().data) {
        if (!ctx.embedding_layer->tokenWeights().has_grad()) {
            AG_WARN("embedding token_weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            AG_INFO("Embedding gradients ready: " << ctx.embedding_layer->tokenWeights().numel() << " elements at " << ctx.embedding_layer->tokenWeights().grad_data());
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LM head gradients (Pattern B: owned by persistent LMHeadLayer)
    // May be tied to embedding (same underlying grad Tensor)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.lm_head->weights().data) {
        if (!ctx.lm_head->weights().has_grad()) {
            AG_WARN("lm_head weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            // Check if tied to embedding (same underlying grad Tensor)
            if (ctx.lm_head->weights().grad_data() == ctx.embedding_layer->tokenWeights().grad_data()) {
                AG_INFO("LM head gradients TIED to embedding: " << ctx.lm_head->weights().numel() << " elements");
            } else {
                AG_INFO("LM head gradients SEPARATE: " << ctx.lm_head->weights().numel() << " elements at " << ctx.lm_head->weights().grad_data());
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
    // Final RMSNorm gamma (Pattern B: owned by LMHeadLayer)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.lm_head->finalRmsGamma().data && !ctx.lm_head->finalRmsGamma().has_grad()) {
        AG_WARN("final_rms_gamma.grad is NULL - gradients NOT flowing!");
        ok = false;
    }

    if (ctx.lm_head->bias().data && !ctx.lm_head->bias().has_grad()) {
        AG_WARN("lm_head_bias.grad is NULL - gradients NOT flowing!");
        ok = false;
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
            // Issue #148: Sandwich norm gammas REMOVED
            check(enc->attnWqkv(), "attnWqkv");
            check(enc->attnBqkv(), "attnBqkv");
            check(enc->attnWo(), "attnWo");
            check(enc->attnBo(), "attnBo");
            check(enc->ffnWGate(), "ffnWGate");
            check(enc->ffnW1(), "ffnW1");
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
    }

    if (ctx.model && ctx.model->getMtpK() > 0) {
        for (int k = 0; k < ctx.model->getMtpK(); ++k) {
            LanguageModel::MTPHead* head = ctx.model->getMtpHead(k);
            if (head) {
                if (head->weight.data && !head->weight.has_grad()) {
                    AG_WARN("MTP head " << k << " weight.grad is NULL");
                    ok = false;
                }
                if (head->bias.data && !head->bias.has_grad()) {
                    AG_WARN("MTP head " << k << " bias.grad is NULL");
                    ok = false;
                }
            }
        }
    }
    
    return ok;
}

// Finding 2 (Rule 26): computeGradientNorm() DELETED — redundant with
// Phase2's ctx.model->computeGradNorm(true) which produces per-component breakdown.
// The old function duplicated a full L2 norm scan + cudaStreamSynchronize per batch
// whose result was only logged and never consumed by Phase2.

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
    EmbeddingLayer* embedding_layer = model.getEmbeddingLayer();
    LMHeadLayer* lm_head = model.getLmHeadLayer();
    ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
    ReasoningHeadLayer* reasoning_head = model.getReasoningHeadLayer();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU COPIES: Transfer padded data from payload to GPU tensors
    // Uses BatchPayload byte-size helpers for consistent transfer sizing.
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
        payload.inputIdBytes(),
        cudaMemcpyHostToDevice, stream));
    
    // Targets
    if (!training_state.cached_targets_tensor.data) {
        throw std::runtime_error("autogradTrainingStep: cached_targets_tensor.data is NULL");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<int*>(training_state.cached_targets_tensor.data),
        payload.target_ids.data(),
        payload.targetIdBytes(),
        cudaMemcpyHostToDevice, stream));
    
    // Numeric values
    if (!training_state.cached_token_numeric_values.data) {
        throw std::runtime_error("autogradTrainingStep: cached_token_numeric_values.data is NULL");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        training_state.cached_token_numeric_values.data,
        payload.numeric_values.data(),
        payload.numericValueBytes(),
        cudaMemcpyHostToDevice, stream));
    
    // Atom mask + text features
    (void)Batching::BatchPayload::kTextFeatureDim;  // reserved for validation
    if (training_state.cached_token_atom_mask.data) {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint8_t*>(training_state.cached_token_atom_mask.data),
            payload.atom_mask.data(),
            payload.atomMaskBytes(),
            cudaMemcpyHostToDevice, stream));
    }
    if (training_state.cached_token_text_features.data) {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint16_t*>(training_state.cached_token_text_features.data),
            payload.text_features.data(),
            payload.textFeatureBytes(),
            cudaMemcpyHostToDevice, stream));
    }
    // Atom flags (type-specific metadata from AtomTable)
    if (training_state.cached_token_atom_flags.data) {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<uint32_t*>(training_state.cached_token_atom_flags.data),
            payload.atom_flags.data(),
            payload.atomFlagBytes(),
            cudaMemcpyHostToDevice, stream));
    }
    // Execution slot map (runtime substrate metadata — stable for full forward/execute sequence)
    if (training_state.cached_token_to_slot_map.data) {
        CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<int32_t*>(training_state.cached_token_to_slot_map.data),
            payload.token_to_slot_map.data(),
            payload.slotMapBytes(),
            cudaMemcpyHostToDevice, stream));
    }
    
    // Store dimensions in TrainingState for downstream consumers
    training_state.cached_batch_size = payload.batch_size;
    training_state.cached_seq_len = payload.max_seq_len;
    training_state.cached_num_layers = cfg.num_layers;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOGRAD CONTEXT
    // ═══════════════════════════════════════════════════════════════════════════
    
    ExecutionBlockLayer* execution_block = model.getExecutionBlockLayer();

    AutogradContext ctx = initAutogradContext(
        &cfg,
        &training_state,
        &gpu_encoder,
        embedding_layer,
        lm_head,
        scratch_block,
        reasoning_head,
        execution_block,
        training_state.cublas_handle,
        stream,
        payload,
        grad_scale,
        step,
        true
    );
    ctx.loss_config = buildLossConfig(model.getLossOptions(), training_state.d_class_weights);
    ctx.skip_equation_logging = accumulate;  // Skip D2H + fprintf on accumulation micro-batches
    ctx.model = &model;  // For MTP head access in computeAutogradLoss

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
        training_state.logGradientAttribution(static_cast<int>(step), stream, ctx.embedding_layer);
    }
    training_state.sequence_weight_count = 0;
    
    // NOTE: Do NOT clear autograd_intermediates here — caller (processBatch) reads
    // intermediates.logits_tensor.data for diagnostics and MUST call clear() when done.
    
    AG_INFO("Training step complete: loss=" << loss_result.loss_value);
    
    return loss_result;
}
}  // namespace Autograd
}  // namespace GRIM
