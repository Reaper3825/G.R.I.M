//======================================================//
//  TrainingState_GPU.hpp
//  Standalone TrainingState declaration for GPU training
//======================================================//

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../TeacherLogits/TeacherLogits_GPU.hpp"
#include "../ScratchBlock/ScratchBlock_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../PBM/PositionalBiasMethod.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

// Forward declaration for autograd tensor system
namespace GRIM {
    struct TrainingTensors;
    namespace Autograd {
        struct AutogradContext;  // Issue #47: Forward context persists for backward
    }
}

namespace GRIM {

struct EncoderLayerCache;

struct TrainingState {
	TrainingState();
	~TrainingState();

	TrainingState(const TrainingState&) = delete;
	TrainingState& operator=(const TrainingState&) = delete;
	TrainingState(TrainingState&&) = delete;
	TrainingState& operator=(TrainingState&&) = delete;

	//======================================================//
	//  PARAMETER TENSORS (weights + gradients via autograd)
	//======================================================//
	// Rule 20: NO raw float* for gradients - use GRIM::Tensor with autograd
	
	// Embedding weights [vocab_size, d_model]
	// NOTE: When tie_embeddings=true, lm_head_weights shares data with embedding_weights
	Tensor embedding_weights;
	
	// Position embedding weights [max_seq_len, d_model]
	// Issue #36 FIX: Position embeddings MUST be trainable to match PyTorch baseline
	Tensor position_embedding_weights;

	// LM head weights [vocab_size, d_model] and optional bias [vocab_size]
	Tensor lm_head_weights;
	Tensor lm_head_bias;
	
	// Numeric head for number prediction
	Tensor numeric_head_weights;  // [d_model]
	Tensor numeric_head_bias;     // [1]
	
	// Learned loss weighting (homoscedastic uncertainty)
	// log_var = log(σ²), loss = L / (2*exp(log_var)) + 0.5*log_var
	Tensor log_var_text;     // [1] - learned log-variance for text CE loss
	Tensor log_var_numeric;  // [1] - learned log-variance for numeric loss

	// Final RMSNorm before LM head (Issue #33 fix)
	Tensor final_rms_gamma;  // [d_model]
	
	// NOTE: Encoder layer weights are owned by GPUGrimEncoder, not TrainingState.
	// Access encoder gradients via enc->getAttnWqkvGrad(), enc->getFFNW1Grad(), etc.
	// See buildParameterGroups() in LanguageModel_Training.cu for optimizer integration.
	
	// ═══════════════════════════════════════════════════════════════
	// AUTOGRAD MIGRATION COMPLETE - Legacy vectors REMOVED
	// ═══════════════════════════════════════════════════════════════
	// FFN/Attention/RMSNorm gradients now use encoder's Tensor.grad:
	//   - enc->getFFNW1Grad(), enc->getFFNW2Grad()
	//   - enc->getAttnWqkvGrad(), enc->getAttnWoGrad()
	//   - enc->getRMS1GammaGrad(), enc->getRMS2GammaGrad()
	// See BackwardPhase2_Encoder.cu for usage.
	
	// QK-norm learned scales (nGPT-style) - weights and gradients
	std::vector<float*> attn_alpha_q;       // [num_heads] per layer
	std::vector<float*> attn_alpha_k;       // [num_kv_heads] per layer
	std::vector<float*> attn_alpha_q_grads;
	std::vector<float*> attn_alpha_k_grads;
	
	// Helpers to access raw gradient pointers for legacy code (during migration)
	float* embedding_grads() { return embedding_weights.grad_data(); }  // ISSUE #59: Use accessor
	float* position_embedding_grads() { return position_embedding_weights.grad_data(); }  // ISSUE #59
	float* lm_head_weight_grads() { return lm_head_weights.grad_data(); }  // ISSUE #59
	float* final_rms_gamma_grads() { return final_rms_gamma.grad_data(); }  // ISSUE #59
	
	// GQA configuration (stored for cache sizing)
	int num_heads = 0;           // Q heads
	int num_kv_heads = 0;        // K,V heads (GQA: num_kv_heads < num_heads)

	// Activation cache for backward pass
	float* cached_embeddings = nullptr;

	// Per-layer activation cache (allocated per-layer)
	std::vector<float*> cached_ln1_outputs;        // After first layer norm
	std::vector<float*> cached_attn_outputs;       // After attention
	std::vector<float*> cached_residual1_outputs;  // After first residual
	std::vector<float*> cached_ln2_outputs;        // After second layer norm
	std::vector<float*> cached_ffn_pre_gelu;       // Before GELU activation
	std::vector<float*> cached_ffn_outputs;        // After FFN
	std::vector<float*> cached_layer_outputs;      // Final layer output (after residual2)
	EncoderLayerCache* forward_layer_caches = nullptr;
	int forward_layer_cache_count = 0;

	// QKV cache for attention backward
	// GQA: Q has num_heads, K/V have num_kv_heads (reduced size)
	std::vector<float*> cached_Q;                  // BHSD format [batch, num_heads, seq, head_dim]
	std::vector<float*> cached_K;                  // BHSD format [batch, num_kv_heads, seq, head_dim]
	std::vector<float*> cached_V;                  // BHSD format [batch, num_kv_heads, seq, head_dim]
	std::vector<float*> cached_attn_inputs;        // Input to attention (after LN1)
	std::vector<float*> cached_attn_bhsd;          // BHSD format - attention output before W_o projection
	std::vector<float*> cached_softmax_lse;        // [batch, num_heads, seq] FP32 (dense, no padding)

	float* cached_encoder_outputs = nullptr;  // Final encoder output
	float* cached_final_rms_input = nullptr;  // Issue #33: Input to final RMSNorm (for backward gamma gradients)
	float* cached_logits = nullptr;
	float* cached_numeric_predictions = nullptr;  // [max_cached_tokens]
	int* cached_targets = nullptr;
	int* cached_token_ids = nullptr;  // For embedding backward
	float* cached_token_numeric_values = nullptr;  // [max_cached_tokens]
	uint8_t* cached_token_numeric_mask = nullptr;  // [max_cached_tokens]
	// GRMT v4: text features for ScratchBlock
	uint16_t* cached_token_text_features = nullptr;  // [max_cached_tokens * kTextFeatureDim] FP16
	uint8_t* cached_token_text_mask = nullptr;       // [max_cached_tokens]
	// GRMT v6: per-token byte lengths for loss weighting (atoms cost what they represent)
	uint16_t* cached_token_byte_lengths = nullptr;   // [max_cached_tokens]
	int cached_batch_size = 0;
	int cached_seq_len = 0;
	int cached_valid_tokens = 0;
	
	// =========================================================================
	// INCREMENTAL KV CACHE STATE (for autoregressive generation)
	// =========================================================================
	// kv_cache_len tracks how many tokens have K,V cached across all layers.
	// During forwardInit(): set to prompt length after computing all K,V
	// During forwardStep(): increment by 1 after appending new K,V
	// During resetKVCache(): reset to 0
	// =========================================================================
	int kv_cache_len = 0;           // Number of tokens with valid K,V in cache
	int kv_cache_capacity = 0;      // Maximum tokens the cache can hold (= max_cached_seq_len)
	
	// Single-token buffers for incremental generation (allocated in initInferenceState)
	// single_token_logits: [vocab_size] - logits for one token (vs seq_len * vocab_size)
	// single_token_hidden: [d_model] - hidden state for one token
	// single_token_embedding: [d_model] - embedding for one token
	float* single_token_logits = nullptr;
	float* single_token_hidden = nullptr;
	float* single_token_embedding = nullptr;
	
	int cached_num_layers = 0;
	
	int max_cached_batch = 0;
	int max_cached_seq_len = 0;
	size_t max_cached_tokens = 0;
	size_t max_logit_tokens = 0;
	TeacherLogits::Buffer teacher_logits;
	TeacherLogits::Buffer reference_logits;
	float* sequence_weights = nullptr;
	int sequence_weight_count = 0;
	int sequence_weight_capacity = 0;
	
	// Issue #38 FIX: Per-token class weighting to prevent mode collapse on frequent tokens
	// Weight indexed by TOKEN ID (not position) - frequent tokens like SPACE get lower weight
	// Computed from inverse token frequency in training corpus
	float* token_weights = nullptr;       // [vocab_size] - on GPU
	int token_weights_count = 0;          // Should equal vocab_size when set

	// Issue #39 FIX: Output logit bias correction to prevent mode collapse
	// Tracks EMA of mean logit per vocabulary token across training batches.
	// Subtracted from logits BEFORE softmax to prevent any token from having
	// systematically higher pre-softmax activation (e.g., SPACE token 277).
	// Formula: corrected_logit[v] = raw_logit[v] - logit_bias[v]
	// EMA update: logit_bias[v] = (1-α)*logit_bias[v] + α*batch_mean_logit[v]
	float* logit_bias = nullptr;          // [vocab_size] - EMA of per-token mean logit
	float* logit_bias_update = nullptr;   // [vocab_size] - scratch for batch mean computation
	int logit_bias_count = 0;             // Should equal vocab_size when set
	uint64_t logit_bias_update_step = 0;  // Number of EMA updates (for warm-up)

	// ═══════════════════════════════════════════════════════════════
	//  AUTOGRAD LOSS TENSOR (Issue #46 FIX: Unified loss/gradient system)
	// ═══════════════════════════════════════════════════════════════
	// The loss tensor computed during forward pass. Stores:
	// - Scalar loss value (for logging)
	// - grad_fn (CrossEntropyLossGradFn) for backward pass
	// This unifies loss computation and gradient calculation into one system.
	// Forward: computeLossBatch() computes loss and attaches grad_fn
	// Backward: backward() calls loss_tensor.backward() → grad_fn->apply()
	Tensor loss_tensor;                   // Scalar [1] - loss value + grad_fn
	Tensor logits_tensor;                 // [total_tokens, vocab_size] - wraps cached_logits
	float cached_loss_value = 0.0f;       // Host copy of total loss (for return value)
	float cached_text_loss = 0.0f;        // Host copy of text CE loss (for learned weighting backward)
	float cached_numeric_loss = 0.0f;     // Host copy of numeric loss (for learned weighting backward)
	
	// ═══════════════════════════════════════════════════════════════
	//  AUTOGRAD CONTEXT (Issue #47: Full computation graph for backward)
	// ═══════════════════════════════════════════════════════════════
	// Persists the autograd context from forward pass to backward pass.
	// Contains intermediate Tensors (encoder_layer_outputs, etc.) that keep
	// the grad_fn chain alive. Without this, backward() would have dangling pointers.
	// Created in computeLossBatch(), used in backward(), cleared after optimizer step.
	std::unique_ptr<Autograd::AutogradContext> autograd_ctx;

	// ═══════════════════════════════════════════════════════════════
	//  INTERMEDIATE GRADIENT TENSORS (Issue #45 FIX: Proper autograd)
	// ═══════════════════════════════════════════════════════════════
	// Memory is owned by Tensor objects. Use tensor.data to access raw pointer.
	// Tensors provide zero_grad(stream) for proper gradient zeroing.
	// Rule 20: NO BACKWARDS COMPATIBILITY - use tensor.data directly.
	
	Tensor grad_logits_tensor;            // [max_logit_tokens, vocab_size] LOGITS layout
	Tensor grad_numeric_tensor;           // [max_logit_tokens] for numeric head
	Tensor grad_encoder_tensor;           // [max_tokens, d_model] encoder output grad
	Tensor grad_ffn_input_tensor;         // [max_tokens, d_model]
	Tensor grad_ffn_hidden_tensor;        // [max_tokens, d_ff]
	Tensor grad_attn_input_tensor;        // [max_tokens, d_model]
	Tensor grad_attn_out_proj_tensor;     // [max_tokens, d_model] pre-W_o
	Tensor grad_attn_out_bhsd_tensor;     // [max_tokens, d_model] reshaped BHSD
	Tensor grad_q_tensor;                 // [batch, heads, seq, head_dim]
	Tensor grad_k_tensor;                 // [batch, kv_heads, seq, head_dim]
	Tensor grad_v_tensor;                 // [batch, kv_heads, seq, head_dim]
	Tensor grad_qkv_concat_tensor;        // [max_tokens, 3*d_model]
	Tensor grad_qkv_input_tensor;         // [max_tokens, d_model]
	Tensor grad_attn_bsm_tensor;          // [max_tokens, d_model] BSM scratch
	
	/// Zero all intermediate gradient tensors (call at start of accumulation window)
	void zeroIntermediateGrads(cudaStream_t stream);
	// Flash Attention backward workspace (per-step scratch, fp32)
	float* fa_dq_accum = nullptr;
	float* fa_dsoftmax_sum = nullptr;
	size_t fa_dq_accum_bytes = 0;
	size_t fa_dsoftmax_sum_bytes = 0;

	// Flash Attention BF16 scratch (float <-> bf16 conversion buffers)
	// Layout uses BSHD for FA v2 and converts back to BHSD float.
	__nv_bfloat16* fa_q_bf16 = nullptr;
	__nv_bfloat16* fa_k_bf16 = nullptr;
	__nv_bfloat16* fa_v_bf16 = nullptr;
	__nv_bfloat16* fa_out_bf16 = nullptr;
	__nv_bfloat16* fa_dout_bf16 = nullptr;
	__nv_bfloat16* fa_dq_bf16 = nullptr;
	__nv_bfloat16* fa_dk_bf16 = nullptr;
	__nv_bfloat16* fa_dv_bf16 = nullptr;
	size_t fa_q_bf16_elems = 0;
	size_t fa_kv_bf16_elems = 0;
	
	// ═══════════════════════════════════════════════════════════════
	//  Issue #43 FIX: Encoder Weight Gradient Centering Scratch Buffer
	// ═══════════════════════════════════════════════════════════════
	// Migrated to Tensor: centering_scratch_tensor
	// See grad_attn_bsm_tensor (already defined above in INTERMEDIATE GRADIENT TENSORS section)
	Tensor centering_scratch_tensor;      // max(d_model, d_ff) width for Issue #43
	size_t centering_scratch_elems() const { return centering_scratch_tensor.numel(); }
	float* centered_activation_scratch() const { return centering_scratch_tensor.data; }

	// Scratch buffers for loss computation (pre-allocated to avoid malloc/free per sequence)
	float* d_loss_scratch = nullptr;      // Per-token losses
	float* d_loss_sum_scratch = nullptr;  // Reduced loss sum
	size_t loss_scratch_capacity = 0;     // Current allocation size
	float* d_numeric_loss_sum = nullptr;  // Numeric loss sum (device scalar)
	int* d_numeric_loss_count = nullptr;  // Numeric loss count (device scalar)
	
	// Attention entropy output buffer (per forward pass)
	// Size: [batch_size * num_heads] floats
	// Populated by Flash Attention forward kernel, averaged across Q blocks
	float* d_entropy_output = nullptr;
	size_t entropy_output_capacity = 0;   // Current allocation (num elements, not bytes)

	// Encoder workspace for GPU-native forward pass
	// Used for intermediate computations in GPUEncoderLayer::forwardGPU()
	float* encoder_workspace = nullptr;
	size_t encoder_workspace_size = 0;    // Size in bytes

	// ═══════════════════════════════════════════════════════════════
	//  STREAM & GRADIENT MANAGEMENT (Centralized Controllers)
	// ═══════════════════════════════════════════════════════════════
	// NO RAW STREAMS: All stream operations go through stream_ctrl
	// Gradients now managed via autograd system (GRIM::Tensor)
	StreamController stream_ctrl;  // Owns all CUDA streams
	GradNorm::GradNormController gradnorm_ctrl;  // GPU-resident gradient norm computation

	// cuBLAS handle bound to stream_ctrl.getPrimaryStream()
	// NOTE: Handle does NOT own a stream - it's bound to Primary
	cublasHandle_t cublas_handle = nullptr;

	// ═══════════════════════════════════════════════════════════════
	//  POSITIONAL ENCODING STATE (Unified ALiBi+RoPE Hybrid)
	// ═══════════════════════════════════════════════════════════════
	// PBM = Positional Bias Method: ALWAYS applies both ALiBi and RoPE together.
	// - RoPE: Rotary Position Embedding rotates Q,K to encode position in magnitude/phase
	// - ALiBi: Attention-Linear-Biases adds position-dependent bias to attention scores
	// Without this, attention becomes uniform (all positions look identical → plateau!)
	PBM::PBMState pbm_state;   // Owns device memory for slopes + inv_freq
	PBM::PBMSpec pbm_spec;     // Cached spec for passing to layers
	bool pbm_initialized = false;

	// Scratch block pool for pinned memory transfers (togglable)
	ScratchBlock::ScratchBlockPool* scratch_pool = nullptr;
	bool scratch_enabled = true;  // Can be toggled for multi-model orchestration

	// ScratchBlock reasoning layer activation caches
	float* cached_scratch_block_embeddings = nullptr;  // [max_atoms, atom_embedding_dim]
	int* cached_scratch_block_positions = nullptr;     // [max_atoms] - atom token positions
	int* cached_scratch_block_types = nullptr;         // [max_atoms] - atom type for each atom
	int* cached_scratch_block_num_atoms = nullptr;     // Scalar

	// ═══════════════════════════════════════════════════════════════
	//  AUTOGRAD SYSTEM (Migration Complete - Jan 2026)
	// ═══════════════════════════════════════════════════════════════
	// Encoder gradients now use encoder's own Tensors via:
	//   enc->getFFNW1Grad(), enc->getAttnWqkvGrad(), etc.
	// 
	// tensors_ is DEPRECATED and set to nullptr - encoder owns all gradients.
	// use_autograd_tensors enables the autograd hybrid backward path.
	std::unique_ptr<TrainingTensors> tensors_;  // DEPRECATED - always nullptr
	bool use_autograd_tensors = false;  // Initialized by initializeAutogradTensors()
	
	// Initialize autograd system (just sets use_autograd_tensors = true)
	// Issue #96: Added positional_encoding to control position embedding allocation
	void initializeAutogradTensors(int vocab_size, int d_model, int d_ff,
	                               int num_layers, int num_heads, int num_kv_heads,
	                               int max_seq_len, bool tie_embeddings, bool use_bias,
	                               HyperParameters::PositionalEncodingType positional_encoding,
	                               cudaStream_t stream = nullptr);

	// ═══════════════════════════════════════════════════════════════
	//  OPTIMIZER STATE BUFFERS (AdamW momentum/velocity)
	// ═══════════════════════════════════════════════════════════════
	// Centralized ownership: Allocated/freed in TrainingState lifecycle
	// ParameterGroup holds pointers into these vectors, does NOT allocate
	std::vector<float*> optimizer_m_states;  // First moment (momentum) per param group
	std::vector<float*> optimizer_v_states;  // Second moment (velocity) per param group
	std::vector<size_t> optimizer_state_sizes;  // Size of each buffer (for cleanup)
	bool optimizer_states_allocated = false;
	
	// Allocate optimizer states for N parameter groups with given sizes
	void allocateOptimizerStates(const std::vector<size_t>& sizes);
	// Free all optimizer state buffers
	void freeOptimizerStates();

	// Flag to track if training state is initialized
	bool initialized = false;
	
	// Architecture config hash for detecting when parameter groups need rebuild
	// Computed from: num_layers, num_heads, num_kv_heads, d_model, d_ff, tie_embeddings, scratch_block_enabled
	// If this changes, optimizer state must be reset.
	uint64_t architecture_config_hash = 0;
	
	// ═══════════════════════════════════════════════════════════════
	//  DEBUG: GRADIENT ATTRIBUTION BUFFERS (Issue #60 Investigation)
	// ═══════════════════════════════════════════════════════════════
	// Separate buffers to isolate LM head vs embedding backward contributions
	// With tied weights, both write to the SAME shared gradient buffer.
	// This lets us see each source independently for debugging.
	//
	// Method A (Current): shared_grad = LM_head_grad + embedding_grad (accumulated)
	// Method B (Debug):   lm_only_grad, emb_only_grad (separate buffers)
	//
	// Set debug_gradient_attribution=true to enable logging of both sources.
	bool debug_gradient_attribution = false;  // Enable via ai_config.json
	float* debug_lm_head_only_grad = nullptr; // [vocab_size * d_model] - LM head backward contribution only
	float* debug_embedding_only_grad = nullptr; // [vocab_size * d_model] - Embedding backward contribution only
	size_t debug_grad_buffer_size = 0;        // vocab_size * d_model
	
	// ═══════════════════════════════════════════════════════════════
	//  ISSUE #60 FIX: PCGRAD BUFFER FOR TIED WEIGHTS
	// ═══════════════════════════════════════════════════════════════
	// When tie_embeddings=true, LM head and embedding backward produce OPPOSITE
	// gradients that cancel when combined! PCGrad projects out the conflicting
	// component: g_final = g_lm + (g_emb - proj_{g_lm}(g_emb))
	//
	// This buffer holds the embedding gradient BEFORE combining with LM head.
	float* pcgrad_temp_buffer = nullptr;  // [vocab_size * d_model]
	size_t pcgrad_buffer_size = 0;
	
	// Allocate/free PCGrad buffer
	void allocatePCGradBuffer(int vocab_size, int d_model);
	void freePCGradBuffer();
	
	// Allocate debug gradient attribution buffers
	void allocateDebugGradBuffers(int vocab_size, int d_model);
	// Free debug gradient attribution buffers
	void freeDebugGradBuffers();
	// Log comparison of gradient sources for token 277 (SPACE)
	void logGradientAttribution(int batch_idx, cudaStream_t stream);

	// ═══════════════════════════════════════════════════════════════
	//  GUESS CACHE BUFFERS (GRIM-TS Integration - Rule 22 Compliant)
	// ═══════════════════════════════════════════════════════════════
	// Centralized ownership: Allocated/freed in TrainingState lifecycle
	// GRIM-TS receives pointers to these buffers, does NOT allocate GPU memory
	struct GuessCacheBuffers {
		// Main cache structures
		void* records = nullptr;           // GuessRecord array [capacity]
		uint64_t* keys = nullptr;          // Hash keys [capacity]
		unsigned int* size = nullptr;      // Entry count (single value)
		unsigned int* evict_cursor = nullptr;  // Eviction position (single value)
		
		// Optional diversity bloom filter
		uint32_t* diversity_bloom = nullptr;
		size_t bloom_words = 0;
		
		// Calibration
		float* calibration_offset = nullptr;
		
		// Single-item transfer buffers
		void* single_meta_buffer = nullptr;    // Single GuessMetadata
		float* single_reward_buffer = nullptr; // Single float reward
		
		// Pinned host memory for async transfers
		void* pinned_meta = nullptr;
		float* pinned_rewards = nullptr;
		size_t pinned_capacity = 0;
		
		// Capacity tracking
		size_t capacity = 0;
		bool allocated = false;
	};
	GuessCacheBuffers guess_cache_buffers;
	
	// Allocate GRIM-TS guess cache buffers
	// Returns false and logs error on failure (fail-loud)
	bool allocateGuessCacheBuffers(size_t capacity, bool enable_diversity, size_t diversity_bloom_bits, size_t pinned_buffer_size);
	// Free all guess cache buffers
	void freeGuessCacheBuffers();
	
	// ═══════════════════════════════════════════════════════════════
	//  BATCH PREPARATION BUFFERS (CPU-side - Rule 22 Compliant)
	// ═══════════════════════════════════════════════════════════════
	// Pre-allocated CPU buffers to avoid 7-8s reallocation per batch
	// Used by prepareLossBatchInputs() in ComputeLossBatch.cu
	std::vector<int> batch_prep_input_ids;
	std::vector<int> batch_prep_target_ids;
	std::vector<float> batch_prep_numeric_values;
	std::vector<uint8_t> batch_prep_numeric_mask;
	std::vector<uint16_t> batch_prep_text_features;
	std::vector<uint8_t> batch_prep_text_mask;
	std::vector<uint16_t> batch_prep_byte_lengths;  // GRMT v6
	std::vector<int> batch_prep_sequence_lengths;
	size_t batch_prep_capacity = 0;  // Track total_tokens capacity
};

} // namespace GRIM

#else

namespace GRIM {
struct TrainingState {
	TrainingState() = default;
	~TrainingState() = default;
};
} // namespace GRIM

#endif  // USE_CUDA

