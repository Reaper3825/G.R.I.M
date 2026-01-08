//======================================================//
//  TrainingState_GPU.hpp
//  Standalone TrainingState declaration for GPU training
//======================================================//

#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../TeacherLogits/TeacherLogits_GPU.hpp"
#include "../ScratchBlock/ScratchBlock_GPU.hpp"
#include "../StreamController/StreamController_GPU.hpp"
#include "../GradAccumulationController/GradAccumulationController_Integration.hpp"
#include "../GradNorm/GradNormGPU.hpp"
#include "../PBM/PositionalBiasMethod.hpp"

namespace GRIM {

struct EncoderLayerCache;

struct TrainingState {
	TrainingState();
	~TrainingState();

	TrainingState(const TrainingState&) = delete;
	TrainingState& operator=(const TrainingState&) = delete;
	TrainingState(TrainingState&&) = delete;
	TrainingState& operator=(TrainingState&&) = delete;

	// Gradient buffers for parameters
	// NOTE: When tie_embeddings=true, embedding_grads == lm_head_weight_grads (aliased)
	float* embedding_grads = nullptr;
	size_t embedding_grad_size = 0;

	float* lm_head_weight_grads = nullptr;
	float* lm_head_bias_grads = nullptr;
	float* numeric_head_weight_grads = nullptr;
	float* numeric_head_bias_grads = nullptr;

	// LM head weights (actual parameters on GPU)
	// NOTE: When tie_embeddings=true, lm_head_weights points to EmbeddingRuntime::token_buffer
	// (NOT owned by TrainingState - don't free!)
	float* lm_head_weights = nullptr;  // [vocab_size, d_model]
	float* lm_head_bias = nullptr;     // [vocab_size]
	bool lm_head_weights_owned = false; // True if we allocated, false if aliased
	float* numeric_head_weights = nullptr;  // [d_model]
	float* numeric_head_bias = nullptr;     // [1]

	// Encoder layer gradients (allocated per-layer)
	std::vector<float*> rms1_gamma_grads;
	std::vector<float*> rms2_gamma_grads;
	std::vector<float*> attn_qkv_weight_grads;
	std::vector<float*> attn_qkv_bias_grads;
	std::vector<float*> attn_out_weight_grads;
	std::vector<float*> attn_out_bias_grads;
	std::vector<float*> ffn_w1_grads;
	std::vector<float*> ffn_b1_grads;
	std::vector<float*> ffn_w2_grads;
	std::vector<float*> ffn_b2_grads;
	
	// Learnable QK-norm scales (nGPT-style) - per layer, per head
	// alpha_q[layer][head], alpha_k[layer][head]
	// Forward: q̂ = alpha_q * (q / ||q||), buffers the 1/||q|| division
	// GQA: alpha_q has num_heads entries, alpha_k has num_kv_heads entries
	std::vector<float*> attn_alpha_q;        // [num_layers] arrays of [num_heads]
	std::vector<float*> attn_alpha_k;        // [num_layers] arrays of [num_kv_heads]
	std::vector<float*> attn_alpha_q_grads;  // Gradients for alpha_q [num_heads]
	std::vector<float*> attn_alpha_k_grads;  // Gradients for alpha_k [num_kv_heads]
	
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
	float* cached_logits = nullptr;
	float* cached_numeric_predictions = nullptr;  // [max_cached_tokens]
	int* cached_targets = nullptr;
	int* cached_token_ids = nullptr;  // For embedding backward
	float* cached_token_numeric_values = nullptr;  // [max_cached_tokens]
	uint8_t* cached_token_numeric_mask = nullptr;  // [max_cached_tokens]
	// GRMT v4: text features for ScratchBlock
	uint16_t* cached_token_text_features = nullptr;  // [max_cached_tokens * kTextFeatureDim] FP16
	uint8_t* cached_token_text_mask = nullptr;       // [max_cached_tokens]
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

	// Intermediate gradient buffers
	float* grad_logits = nullptr;
	float* grad_numeric_predictions = nullptr;  // [max_logit_tokens]
	float* grad_encoder_out = nullptr;
	// Reusable backward temporaries (allocated once at init)
	float* grad_ffn_input = nullptr;
	float* grad_ffn_hidden = nullptr;
	float* grad_attn_input = nullptr;
	float* grad_attn_out_before_proj = nullptr;
	float* grad_attn_out_reshaped = nullptr;
	float* grad_q = nullptr;
	float* grad_k = nullptr;
	float* grad_v = nullptr;
	float* grad_qkv_concat = nullptr;
	float* grad_qkv_input = nullptr;
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
	
	// DEDICATED scratch buffer for attention output BSM conversion (W_o gradient computation)
	// CRITICAL: Do NOT reuse grad_qkv_input - that buffer is needed later in the same backward pass.
	// This prevents temporal aliasing bugs when operations are parallelized or refactored.
	float* grad_attn_bsm_scratch = nullptr;

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
	// NO RAW GRAD POINTERS: All grad operations go through grad_ctrl
	StreamController stream_ctrl;  // Owns all CUDA streams
	ModelGradAccumulationController grad_ctrl;  // Manages grad accumulation
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

