//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

// Include grim_language_model_cuda.hpp for GPUGrimEncoder, FlashAttentionBF16Scratch, etc.
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "TrainingState_GPU.hpp"
#include "TrainingTensors.hpp"  // Autograd tensor system
#include "../../training/Autograd/AutogradTraining.hpp"  

#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

TrainingState::TrainingState() = default;

TrainingState::~TrainingState() {
	// ═══════════════════════════════════════════════════════════════
	// ALL MEMORY NOW TENSOR-MANAGED
	// ═══════════════════════════════════════════════════════════════
	// After Rule 20 migration, all float* buffers are GRIM::Tensor objects.
	// Tensor::~Tensor() calls release() to free GPU memory.
	// DO NOT manually cudaFree any Tensor member - they own their data.
	
	// encoder_layer_caches is std::vector<EncoderLayerCacheTensors>
	// Each EncoderLayerCacheTensors contains Tensor members that auto-cleanup.
	encoder_layer_caches.clear();
	
	// TeacherLogits has its own release function (not Tensor-based yet)
	TeacherLogits::release(teacher_logits);
	TeacherLogits::release(reference_logits);
	
	// Free optimizer states (std::vector<Tensor> clears automatically)
	freeOptimizerStates();
	
	// Free guess cache buffers (GRIM-TS - not Tensor-based, uses raw cudaMalloc)
	freeGuessCacheBuffers();
	
	// Issue #60: Free debug gradient attribution buffers
	freeDebugGradBuffers();
	
	// Issue #60 FIX: Free PCGrad buffer for tied weights
	freePCGradBuffer();
	
	// StreamController owns and destroys streams - no manual cleanup needed
	if (cublas_handle) cublasDestroy(cublas_handle);
}

//======================================================//
//  Optimizer State Management (Tensor-based)
//======================================================//

void TrainingState::allocateOptimizerStates(const std::vector<size_t>& sizes, cudaStream_t stream) {
	// Free any existing states first
	freeOptimizerStates();
	
	// Get stream from centralized controller if not provided
	cudaStream_t primary_stream = stream ? stream :
		(stream_ctrl.isInitialized() ? stream_ctrl.getPrimaryStream() : nullptr);
	StreamController::fatalIfDefaultStream(primary_stream, "TrainingState::allocateOptimizerStates");
	
	// Allocate Tensor objects for each parameter group
	optimizer_m_states.reserve(sizes.size());
	optimizer_v_states.reserve(sizes.size());
	
	for (size_t i = 0; i < sizes.size(); ++i) {
		if (sizes[i] > 0) {
			// Create Tensor with flat shape and zero-initialize
			optimizer_m_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, primary_stream));
			optimizer_v_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, primary_stream));
		} else {
			// Empty placeholder Tensor for zero-size groups
			optimizer_m_states.emplace_back();
			optimizer_v_states.emplace_back();
		}
	}
	optimizer_states_allocated = true;
}

void TrainingState::freeOptimizerStates() {
	// Tensors auto-cleanup via destructor - just clear the vectors
	optimizer_m_states.clear();
	optimizer_v_states.clear();
	optimizer_states_allocated = false;
}

//======================================================//
//  Guess Cache Buffer Management (GRIM-TS Rule 22)
//======================================================//

bool TrainingState::allocateGuessCacheBuffers(
	size_t capacity, 
	bool enable_diversity, 
	size_t diversity_bloom_bits,
	size_t pinned_buffer_size) 
{
	// Rule 20: Fail loud - throw if already allocated
	if (guess_cache_buffers.allocated) {
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] buffers already allocated! "
			"Call freeGuessCacheBuffers() first. capacity=" + std::to_string(capacity));
	}
	
	// Rule 20: Fail loud - throw on zero capacity
	if (capacity == 0) {
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] capacity cannot be zero!");
	}
	
	// GuessRecord is defined in grim-ts.hpp, we need its size
	// For now, use a conservative estimate: 128 bytes per record
	// CRITICAL: This MUST match sizeof(GRIMTS::GuessRecord) exactly
	constexpr size_t GUESS_RECORD_SIZE = 128;  // sizeof(GuessMetadata) + sizeof(GuessRewardStats)
	constexpr size_t GUESS_METADATA_SIZE = 32; // sizeof(GuessMetadata)
	
	cudaStream_t primary_stream = stream_ctrl.isInitialized() 
		? stream_ctrl.getPrimaryStream() 
		: nullptr;
	StreamController::fatalIfDefaultStream(primary_stream, "TrainingState::allocateGuessCacheBuffers");
	
	// Allocate records
	cudaError_t err = cudaMalloc(&guess_cache_buffers.records, capacity * GUESS_RECORD_SIZE);
	if (err != cudaSuccess) {
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc records failed! "
			"capacity=" + std::to_string(capacity) + ", size=" + 
			std::to_string((capacity * GUESS_RECORD_SIZE) / (1024*1024)) + "MB, error=" + cudaGetErrorString(err));
	}
	
	// Allocate keys
	err = cudaMalloc(&guess_cache_buffers.keys, capacity * sizeof(uint64_t));
	if (err != cudaSuccess) {
		freeGuessCacheBuffers();
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc keys failed! error=" +
			std::string(cudaGetErrorString(err)));
	}
	
	// Allocate size counter
	err = cudaMalloc(&guess_cache_buffers.size, sizeof(unsigned int));
	if (err != cudaSuccess) {
		freeGuessCacheBuffers();
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc size failed! error=" +
			std::string(cudaGetErrorString(err)));
	}
	
	// Allocate evict cursor
	err = cudaMalloc(&guess_cache_buffers.evict_cursor, sizeof(unsigned int));
	if (err != cudaSuccess) {
		freeGuessCacheBuffers();
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc evict_cursor failed! error=" +
			std::string(cudaGetErrorString(err)));
	}
	
	// Allocate diversity bloom filter (optional)
	if (enable_diversity && diversity_bloom_bits > 0) {
		guess_cache_buffers.bloom_words = (diversity_bloom_bits + 31) / 32;
		err = cudaMalloc(&guess_cache_buffers.diversity_bloom, 
		                 guess_cache_buffers.bloom_words * sizeof(uint32_t));
		if (err != cudaSuccess) {
			freeGuessCacheBuffers();
			throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc diversity_bloom failed! "
				"bits=" + std::to_string(diversity_bloom_bits) + " error=" + cudaGetErrorString(err));
		}
	}
	
	// Allocate calibration offset
	err = cudaMalloc(&guess_cache_buffers.calibration_offset, sizeof(float));
	if (err != cudaSuccess) {
		freeGuessCacheBuffers();
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc calibration_offset failed! error=" +
			std::string(cudaGetErrorString(err)));
	}
	
	// Allocate single-item transfer buffers
	err = cudaMalloc(&guess_cache_buffers.single_meta_buffer, GUESS_METADATA_SIZE);
	if (err != cudaSuccess) {
		freeGuessCacheBuffers();
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc single_meta_buffer failed! error=" +
			std::string(cudaGetErrorString(err)));
	}
	
	err = cudaMalloc(&guess_cache_buffers.single_reward_buffer, sizeof(float));
	if (err != cudaSuccess) {
		freeGuessCacheBuffers();
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc single_reward_buffer failed! error=" +
			std::string(cudaGetErrorString(err)));
	}
	
	// Allocate pinned host memory for async transfers
	if (pinned_buffer_size > 0) {
		err = cudaMallocHost(&guess_cache_buffers.pinned_meta, pinned_buffer_size * GUESS_METADATA_SIZE);
		if (err != cudaSuccess) {
			freeGuessCacheBuffers();
			throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMallocHost pinned_meta failed! error=" +
				std::string(cudaGetErrorString(err)));
		}
		
		err = cudaMallocHost(&guess_cache_buffers.pinned_rewards, pinned_buffer_size * sizeof(float));
		if (err != cudaSuccess) {
			freeGuessCacheBuffers();
			throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMallocHost pinned_rewards failed! error=" +
				std::string(cudaGetErrorString(err)));
		}
		guess_cache_buffers.pinned_capacity = pinned_buffer_size;
	}
	
	// Initialize memory on primary stream
	cudaMemsetAsync(guess_cache_buffers.size, 0, sizeof(unsigned int), primary_stream);
	cudaMemsetAsync(guess_cache_buffers.keys, 0xFF, capacity * sizeof(uint64_t), primary_stream);  // kEmptyKey = 0xFF...
	cudaMemsetAsync(guess_cache_buffers.records, 0, capacity * GUESS_RECORD_SIZE, primary_stream);
	cudaMemsetAsync(guess_cache_buffers.evict_cursor, 0, sizeof(unsigned int), primary_stream);
	if (guess_cache_buffers.diversity_bloom) {
		cudaMemsetAsync(guess_cache_buffers.diversity_bloom, 0, 
		                guess_cache_buffers.bloom_words * sizeof(uint32_t), primary_stream);
	}
	float zero_cal = 0.0f;
	cudaMemcpyAsync(guess_cache_buffers.calibration_offset, &zero_cal, sizeof(float), 
	                cudaMemcpyHostToDevice, primary_stream);
	
	guess_cache_buffers.capacity = capacity;
	guess_cache_buffers.allocated = true;
	
	// Info log - success case only
	fprintf(stdout, "[INFO] TrainingState: Guess cache buffers allocated. capacity=%zu, "
	        "diversity=%s, bloom_bits=%zu, pinned=%zu\n",
	        capacity, enable_diversity ? "ON" : "OFF", diversity_bloom_bits, pinned_buffer_size);
	
	return true;  // Return kept for API compatibility, but failures throw
}

void TrainingState::freeGuessCacheBuffers() {
	if (guess_cache_buffers.records) { cudaFree(guess_cache_buffers.records); guess_cache_buffers.records = nullptr; }
	if (guess_cache_buffers.keys) { cudaFree(guess_cache_buffers.keys); guess_cache_buffers.keys = nullptr; }
	if (guess_cache_buffers.size) { cudaFree(guess_cache_buffers.size); guess_cache_buffers.size = nullptr; }
	if (guess_cache_buffers.evict_cursor) { cudaFree(guess_cache_buffers.evict_cursor); guess_cache_buffers.evict_cursor = nullptr; }
	if (guess_cache_buffers.diversity_bloom) { cudaFree(guess_cache_buffers.diversity_bloom); guess_cache_buffers.diversity_bloom = nullptr; }
	if (guess_cache_buffers.calibration_offset) { cudaFree(guess_cache_buffers.calibration_offset); guess_cache_buffers.calibration_offset = nullptr; }
	if (guess_cache_buffers.single_meta_buffer) { cudaFree(guess_cache_buffers.single_meta_buffer); guess_cache_buffers.single_meta_buffer = nullptr; }
	if (guess_cache_buffers.single_reward_buffer) { cudaFree(guess_cache_buffers.single_reward_buffer); guess_cache_buffers.single_reward_buffer = nullptr; }
	
	// Pinned memory uses cudaFreeHost
	if (guess_cache_buffers.pinned_meta) { cudaFreeHost(guess_cache_buffers.pinned_meta); guess_cache_buffers.pinned_meta = nullptr; }
	if (guess_cache_buffers.pinned_rewards) { cudaFreeHost(guess_cache_buffers.pinned_rewards); guess_cache_buffers.pinned_rewards = nullptr; }
	
	guess_cache_buffers.capacity = 0;
	guess_cache_buffers.bloom_words = 0;
	guess_cache_buffers.pinned_capacity = 0;
	guess_cache_buffers.allocated = false;
}

//======================================================//
//  Autograd Tensor System (GRIM::Tensor migration)
//======================================================//

void TrainingState::initializeAutogradTensors(
	int vocab_sz, int d_mod, int d_ffn,
	int n_layers, int n_heads, int n_kv_heads,
	int max_seq, bool tie_emb, bool bias,
	HyperParameters::PositionalEncodingType positional_encoding,
	bool use_layer_scale,
	float layer_scale_init,
	uint64_t seed,
	cudaStream_t stream
) {
	// Rule 20: Fail loud if already initialized
	if (use_autograd_tensors) {
		throw std::runtime_error("[TrainingState::initializeAutogradTensors] already initialized! "
			"use_autograd_tensors is already true");
	}
	
	// ═══════════════════════════════════════════════════════════════
	// PROPER FIX: TrainingTensors OWNS all parameter memory
	// ═══════════════════════════════════════════════════════════════
	// TrainingTensors creates Tensors via Tensor::zeros() which OWNS GPU memory.
	// This replaces the broken pattern of:
	//   EmbeddingRuntime: cudaMalloc → token_buffer
	//   InitTrainingState: Tensor::from_ptr(token_buffer) ← WRAPPING!
	//
	// After this, embedding_weights/position_embedding_weights/lm_head_weights
	// can be accessed via tensors_->embedding_weights etc.
	tensors_ = std::make_unique<TrainingTensors>();
	tensors_->initializeParams(
		vocab_sz, d_mod, d_ffn,
		n_layers, n_heads, n_kv_heads,
		max_seq, tie_emb, bias,
		positional_encoding,
		use_layer_scale,
		layer_scale_init,
		seed,
		stream
	);
	
	// Also copy references to the TrainingState Tensor members for backwards compatibility
	// with code that accesses training_state_.embedding_weights directly
	embedding_weights = Tensor::from_ptr(
		tensors_->embedding_weights.data,
		tensors_->embedding_weights.shape,
		false,  // doesn't take ownership (TrainingTensors owns it)
		true    // requires_grad
	);
	// ISSUE #59: Share the grad Tensor object (shared_ptr) for proper reference counting
	embedding_weights.share_grad(tensors_->embedding_weights);
	
	// ISSUE #96 FIX: Only copy position embeddings if they were allocated
	// (they are NULL when using ALIBI/ROPE/ALIBI_ROPE positional encoding)
	if (tensors_->position_embedding_weights.data) {
		position_embedding_weights = Tensor::from_ptr(
			tensors_->position_embedding_weights.data,
			tensors_->position_embedding_weights.shape,
			false,
			true
		);
		// ISSUE #59: Share the grad Tensor object
		position_embedding_weights.share_grad(tensors_->position_embedding_weights);
	} else {
		// Leave position_embedding_weights uninitialized (data=nullptr)
		// AutogradTraining.cu will check this and skip position embedding addition
		fprintf(stdout, "[TrainingState] position_embedding_weights: SKIPPED (not using learned position embeddings)\n");
	}
	
	lm_head_weights = Tensor::from_ptr(
		tensors_->lm_head_weights.data,
		tensors_->lm_head_weights.shape,
		false,
		true
	);
	// BUG FIX: When tie_embeddings=true, lm_head_weights.grad MUST alias embedding_weights.grad!
	// TrainingTensors sets this up, but we must preserve it when copying to TrainingState.
	if (tie_emb) {
		// Weight tying: share grad Tensor with embedding_weights (which we just set above)
		// ISSUE #59: Use share_grad() for proper shared_ptr semantics
		lm_head_weights.share_grad(embedding_weights);
	} else {
		lm_head_weights.share_grad(tensors_->lm_head_weights);
	}
	
	use_autograd_tensors = true;
	
	fprintf(stdout, "[INFO] TrainingState: TrainingTensors initialized (%d params, %zu layers). "
	        "Memory owned by Tensor::zeros(), not EmbeddingRuntime.\n",
	        vocab_sz, static_cast<size_t>(n_layers));
}

//======================================================//
//  Intermediate Gradient Zeroing (Issue #45 FIX)
//======================================================//

void TrainingState::zeroIntermediateGrads(cudaStream_t stream) {
	// RULE 20: Fail loud on NULL stream - it causes race conditions
	StreamController::fatalIfDefaultStream(stream, "TrainingState::zeroIntermediateGrads");
	
	// Zero all intermediate gradient tensors at start of accumulation window
	// This replaces the GradAccumulationController registration approach
	// NOTE: Only zero tensors that are actually allocated (data != nullptr)
	// Use try-catch to prevent crashes from uninitialized tensors
	
	// BUG FIX Issue #45: zero_grad() zeros tensor.grad, but intermediate gradient tensors
	// store their gradient DATA in tensor.data (NOT tensor.grad)!
	// Must use cudaMemsetAsync on tensor.data instead.
	auto safe_zero = [stream](Tensor& t, const char* name) {
		try {
			if (t.data && t.numel() > 0) {
				// Zero the DATA field, not the grad field!
				cudaMemsetAsync(t.data, 0, t.numel() * sizeof(float), stream);
			}
		} catch (...) {
			// Silently skip - tensor not properly initialized
		}
	};
	
	// Loss backward → encoder
	safe_zero(grad_logits_tensor, "grad_logits");
	safe_zero(grad_numeric_tensor, "grad_numeric");
	safe_zero(grad_encoder_tensor, "grad_encoder");
	
	// Encoder backward temporaries
	safe_zero(grad_ffn_input_tensor, "grad_ffn_input");
	safe_zero(grad_ffn_hidden_tensor, "grad_ffn_hidden");
	safe_zero(grad_attn_input_tensor, "grad_attn_input");
	safe_zero(grad_attn_out_proj_tensor, "grad_attn_out_proj");
	safe_zero(grad_attn_out_bhsd_tensor, "grad_attn_out_bhsd");
	safe_zero(grad_q_tensor, "grad_q");
	safe_zero(grad_k_tensor, "grad_k");
	safe_zero(grad_v_tensor, "grad_v");
	safe_zero(grad_qkv_concat_tensor, "grad_qkv_concat");
	safe_zero(grad_qkv_input_tensor, "grad_qkv_input");
	safe_zero(grad_attn_bsm_tensor, "grad_attn_bsm");
	
	// Issue #43: Centering scratch (not strictly a gradient, but needs zeroing)
	safe_zero(centering_scratch_tensor, "centering_scratch");
}

//======================================================//
//  ISSUE #60 FIX: PCGrad Buffer for Tied Weights (Tensor-based)
//======================================================//

void TrainingState::allocatePCGradBuffer(int vocab_size, int d_model, cudaStream_t stream) {
	if (pcgrad_temp_buffer.data) {
		freePCGradBuffer();
	}
	
	// Get stream from centralized controller if not provided
	cudaStream_t primary_stream = stream ? stream :
		(stream_ctrl.isInitialized() ? stream_ctrl.getPrimaryStream() : nullptr);
	
	// Allocate as Tensor [vocab_size, d_model]
	pcgrad_temp_buffer = Tensor::zeros({vocab_size, d_model}, primary_stream);
	
	// Set global pointers for EmbeddingGradFn to use
	extern float* g_pcgrad_temp_buffer;
	extern size_t g_pcgrad_buffer_size;
	g_pcgrad_temp_buffer = pcgrad_temp_buffer.data;
	g_pcgrad_buffer_size = static_cast<size_t>(vocab_size) * d_model;
	
	fprintf(stdout, "[PCGRAD] Allocated PCGrad buffer: %zu elements (%zu MB)\n",
	        g_pcgrad_buffer_size, g_pcgrad_buffer_size * sizeof(float) / (1024 * 1024));
}

void TrainingState::freePCGradBuffer() {
	if (pcgrad_temp_buffer.data) {
		// Clear global pointers BEFORE releasing tensor
		extern float* g_pcgrad_temp_buffer;
		extern size_t g_pcgrad_buffer_size;
		g_pcgrad_temp_buffer = nullptr;
		g_pcgrad_buffer_size = 0;
		
		// Tensor::release() called via default destruction or explicit clear
		pcgrad_temp_buffer = Tensor();  // Replace with empty tensor
	}
}

//======================================================//
//  DEBUG: Gradient Attribution Buffers (Issue #60) - Tensor-based
//======================================================//

void TrainingState::allocateDebugGradBuffers(int vocab_size, int d_model, cudaStream_t stream) {
	if (debug_lm_head_only_grad.data || debug_embedding_only_grad.data) {
		freeDebugGradBuffers();  // Clean up any existing buffers
	}
	
	// Get stream from centralized controller if not provided
	cudaStream_t primary_stream = stream ? stream :
		(stream_ctrl.isInitialized() ? stream_ctrl.getPrimaryStream() : nullptr);
	
	// Allocate as Tensors [vocab_size, d_model]
	debug_lm_head_only_grad = Tensor::zeros({vocab_size, d_model}, primary_stream);
	debug_embedding_only_grad = Tensor::zeros({vocab_size, d_model}, primary_stream);
	
	const size_t buffer_size = static_cast<size_t>(vocab_size) * d_model;
	fprintf(stdout, "[DEBUG] Allocated gradient attribution buffers: %zu elements (%zu MB each)\n",
	        buffer_size, buffer_size * sizeof(float) / (1024 * 1024));
}

void TrainingState::freeDebugGradBuffers() {
	// Tensor auto-cleanup - just replace with empty tensors
	debug_lm_head_only_grad = Tensor();
	debug_embedding_only_grad = Tensor();
}

void TrainingState::logGradientAttribution(int batch_idx, cudaStream_t stream) {
	if (!debug_gradient_attribution || !debug_lm_head_only_grad.data || !debug_embedding_only_grad.data) {
		return;  // Debug mode not enabled or buffers not allocated
	}
	
	// TOKEN 277 = SPACE (the one causing mode collapse)
	constexpr int TARGET_TOKEN = 277;
	
	// Get d_model from the embedding shape - BSM layout uses Shape2D with (rows=vocab, cols=d_model)
	int d_model = 768;  // Default fallback
	if (embedding_weights.shape.is_flat()) {
		d_model = embedding_weights.shape.flat.cols;
	}
	const int vocab_size = static_cast<int>(debug_lm_head_only_grad.numel() / d_model);
	
	if (TARGET_TOKEN >= vocab_size) {
		fprintf(stderr, "[DEBUG] Token %d out of range (vocab=%d)\n", TARGET_TOKEN, vocab_size);
		return;
	}
	
	// Sync stream to ensure backward pass is complete
	cudaStreamSynchronize(stream);
	
	// Offset for token 277's row in the weight gradient matrix
	const size_t row_offset = static_cast<size_t>(TARGET_TOKEN) * d_model;
	const size_t row_bytes = d_model * sizeof(float);
	
	// Copy row 277 from each debug buffer to host
	// NOTE: debug_lm_head_only_grad = LM head contribution only (captured AFTER LM head backward)
	//       debug_embedding_only_grad = Raw embedding contribution (captured BEFORE PCGrad projection)
	// With PCGrad: final_grad = g_lm + orthogonal(g_emb)
	std::vector<float> lm_grad_277(d_model);
	std::vector<float> raw_emb_grad_277(d_model);  // Pre-projection embedding gradient
	
	cudaMemcpy(lm_grad_277.data(), debug_lm_head_only_grad.data + row_offset, row_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(raw_emb_grad_277.data(), debug_embedding_only_grad.data + row_offset, row_bytes, cudaMemcpyDeviceToHost);
	
	// Get the ACTUAL final gradient from the shared buffer (this is post-PCGrad!)
	std::vector<float> final_grad_277(d_model);
	float* shared_grad_ptr = embedding_weights.grad_data();
	if (shared_grad_ptr) {
		cudaMemcpy(final_grad_277.data(), shared_grad_ptr + row_offset, row_bytes, cudaMemcpyDeviceToHost);
	}
	
	// Compute statistics for LM head gradient
	float lm_sum = 0, lm_sq_sum = 0;
	for (int i = 0; i < d_model; ++i) {
		lm_sum += lm_grad_277[i];
		lm_sq_sum += lm_grad_277[i] * lm_grad_277[i];
	}
	const float lm_norm = sqrtf(lm_sq_sum);
	
	// Compute statistics for raw embedding gradient (pre-projection)
	float raw_emb_sum = 0, raw_emb_sq_sum = 0;
	for (int i = 0; i < d_model; ++i) {
		raw_emb_sum += raw_emb_grad_277[i];
		raw_emb_sq_sum += raw_emb_grad_277[i] * raw_emb_grad_277[i];
	}
	const float raw_emb_norm = sqrtf(raw_emb_sq_sum);
	
	// Compute statistics for FINAL gradient (post-PCGrad)
	float final_sum = 0, final_sq_sum = 0;
	for (int i = 0; i < d_model; ++i) {
		final_sum += final_grad_277[i];
		final_sq_sum += final_grad_277[i] * final_grad_277[i];
	}
	const float final_norm = sqrtf(final_sq_sum);
	
	// Compute cosine similarity between LM and raw embedding gradients
	float lm_emb_dot = 0;
	for (int i = 0; i < d_model; ++i) {
		lm_emb_dot += lm_grad_277[i] * raw_emb_grad_277[i];
	}
	const float cosine_sim = (lm_norm > 1e-8f && raw_emb_norm > 1e-8f) 
	                       ? lm_emb_dot / (lm_norm * raw_emb_norm) : 0.0f;
	
	// Check if they're canceling or reinforcing
	const char* interaction = (cosine_sim > 0.5f) ? "REINFORCING" 
	                        : (cosine_sim < -0.5f) ? "CANCELING" 
	                        : "ORTHOGONAL";
	
	// Check if PCGrad preserved the gradient (final should ≈ lm when emb opposes)
	const char* pcgrad_status = (final_norm > lm_norm * 0.5f) ? "PRESERVED" : "LOST";
	
	fprintf(stdout, "\n[GRAD_ATTRIB_TOKEN277] batch=%d SPACE_TOKEN gradient conflict analysis:\n", batch_idx);
	fprintf(stdout, "  ├─ LM_HEAD_GRAD:        sum=%+.10f  norm=%.10f  mean=%+.10e\n", 
	        lm_sum, lm_norm, lm_sum / d_model);
	fprintf(stdout, "  ├─ EMBEDDING_RAW_GRAD:  sum=%+.10f  norm=%.10f  mean=%+.10e\n", 
	        raw_emb_sum, raw_emb_norm, raw_emb_sum / d_model);
	fprintf(stdout, "  ├─ COSINE_SIMILARITY:   %.10f [%s]\n", cosine_sim, interaction);
	fprintf(stdout, "  ├─ FINAL_GRAD_POSTPCGR: sum=%+.10f  norm=%.10f  mean=%+.10e [%s]\n", 
	        final_sum, final_norm, final_sum / d_model, pcgrad_status);
	fprintf(stdout, "  └─ UPDATE_DIRECTION:    LM→%s  EMB→%s  (W_new = W - lr*grad)\n",
	        lm_sum > 0 ? "DECREASE" : "INCREASE",
	        raw_emb_sum > 0 ? "DECREASE" : "INCREASE");
	fprintf(stdout, "\n");
	fflush(stdout);
}

} // namespace GRIM

#endif  // USE_CUDA