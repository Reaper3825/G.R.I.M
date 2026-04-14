//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

// Include grim_language_model_cuda.hpp for GPUGrimEncoder, FlashAttentionBF16Scratch, etc.
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "TrainingState_GPU.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"  

#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

//======================================================//
//  Weight tensor accessors
//  Session 6: Embedding accessors DELETED — weights now owned by EmbeddingLayer (Pattern B).
//  Access via LanguageModel::getEmbeddingLayer()->tokenWeights() / positionWeights().
//======================================================//

TrainingState::TrainingState() = default;

TrainingState::~TrainingState() {
	// ═══════════════════════════════════════════════════════════════
	// ALL MEMORY NOW TENSOR-MANAGED
	// ═══════════════════════════════════════════════════════════════
	// After Rule 20 migration, all float* buffers are GRIM::Tensor objects.
	// Tensor::~Tensor() calls release() to free GPU memory.
	// DO NOT manually cudaFree any Tensor member - they own their data.
	
	// Rule 20: NO encoder_layer_caches - intermediate tensors now managed via AutogradIntermediates
	// (owned and zero'd separately during training, NOT in destructor)

	// Release PBM GPU buffers (alibi_slopes, rope_inv_freq) so they are freed before Tensor members destruct
	GRIM::PBM::releasePBM(pbm_state);
	pbm_initialized = false;

	// TeacherLogits has its own release function (not Tensor-based yet)
	TeacherLogits::release(teacher_logits);
	TeacherLogits::release(reference_logits);
	
	// Free optimizer states (std::vector<Tensor> clears automatically)
	freeOptimizerStates();
	
	// Free guess cache buffers (GRIM-TS - not Tensor-based, uses raw cudaMalloc)
	freeGuessCacheBuffers();
	
	// Issue #60: Free debug gradient attribution buffers
	freeDebugGradBuffers();
	
	// Free class-balanced loss weights (raw cudaMalloc)
	if (d_class_weights) { cudaFree(d_class_weights); d_class_weights = nullptr; }
	
	// Free per-layer KV cache (raw cudaMalloc, BF16)
	for (auto& ptr : kv_cache_k) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }
	for (auto& ptr : kv_cache_v) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }
	kv_cache_k.clear();
	kv_cache_v.clear();
	if (kv_cache_softmax_lse) { cudaFree(kv_cache_softmax_lse); kv_cache_softmax_lse = nullptr; }
	
	// Free decode scratch buffers (raw cudaMalloc, BF16/FP32)
	if (decode_q_bf16) { cudaFree(decode_q_bf16); decode_q_bf16 = nullptr; }
	if (decode_kv_bf16) { cudaFree(decode_kv_bf16); decode_kv_bf16 = nullptr; }
	if (decode_attn_out_bf16) { cudaFree(decode_attn_out_bf16); decode_attn_out_bf16 = nullptr; }
	if (decode_attn_out_fp32) { cudaFree(decode_attn_out_fp32); decode_attn_out_fp32 = nullptr; }
	
	// Free ScratchBlockPool (pinned memory blocks)
	if (scratch_pool) {
		delete scratch_pool;
		scratch_pool = nullptr;
	}
	
	// Free gradient norm scratch buffers
	GradNorm::freeGradNormScratch(grad_norm_scratch);
	
	// StreamController owns and destroys streams - no manual cleanup needed
	if (cublas_handle) cublasDestroy(cublas_handle);
}

//======================================================//
//  Optimizer State Management (Tensor-based)
//======================================================//

void TrainingState::allocateOptimizerStates(const std::vector<size_t>& sizes, cudaStream_t stream) {
	// Free any existing states first
	freeOptimizerStates();
	
	// Rule 22: Get stream from centralized controller if not provided.
	// getPrimaryStream() throws if not initialized (Rule 20).
	cudaStream_t primary_stream = stream ? stream : stream_ctrl.getPrimaryStream();
	
	// Allocate Tensor objects for each parameter group
	optimizer_m_states.reserve(sizes.size());
	optimizer_v_states.reserve(sizes.size());
	
	for (size_t i = 0; i < sizes.size(); ++i) {
		if (sizes[i] > 0) {
			// Create Tensor with flat shape and zero-initialize
			optimizer_m_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, primary_stream, "optimizer_m"));
			optimizer_v_states.push_back(Tensor::zeros({static_cast<int>(sizes[i])}, primary_stream, "optimizer_v"));
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

void TrainingState::allocateGuessCacheBuffers(
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
	
	// GuessRecord is defined in GRIM-TS.hpp, we need its size.
	// CRITICAL: This MUST match sizeof(GRIMTS::GuessRecord) exactly.
	// A static_assert in GRIM-TS.hpp enforces this at compile time.
	// GuessMetadata: 8+8+4+2+2+4 = 28 → padded to 32 (alignment 8)
	// GuessRewardStats: 15×4+1 = 61 → padded to 64 (alignment 4)
	// GuessRecord: 32+64 = 96
	constexpr size_t GUESS_RECORD_SIZE = 96;
	constexpr size_t GUESS_METADATA_SIZE = 32; // sizeof(GuessMetadata)
	
	// Rule 22: getPrimaryStream() throws if not initialized (Rule 20)
	cudaStream_t primary_stream = stream_ctrl.getPrimaryStream();
	
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
//  Autograd Seed Storage (Session 7: TrainingTensors deleted)
//======================================================//

void TrainingState::initializeAutogradSeed(uint64_t seed) {
	// Rule 20: Fail loud if already initialized
	if (seed_initialized_) {
		throw std::runtime_error("[TrainingState::initializeAutogradSeed] already initialized!");
	}
	
	weight_init_seed = seed;
	seed_initialized_ = true;
	
	fprintf(stdout, "[INFO] TrainingState: autograd seed stored (seed=%llu). "
	        "All weights managed by Pattern B layers.\n",
	        static_cast<unsigned long long>(seed));
}

//======================================================//
//  Intermediate Gradient Zeroing (Issue #45 FIX)
//======================================================//

void TrainingState::zeroIntermediateGrads(cudaStream_t stream) {
	// RULE 20: Fail loud on NULL stream - it causes race conditions
	StreamController::fatalIfDefaultStream(stream, "FATAL:TrainingState::zeroIntermediateGrads Default stream detected!");

	auto safe_zero = [stream](Tensor& t, const char* name) {
		if (t.data && t.numel() > 0) {
			// Zero the DATA field, not the grad field!
			cudaMemsetAsync(t.data, 0, t.numel() * sizeof(float), stream);
		}
	};
	
	// Loss backward → encoder
	safe_zero(grad_logits_tensor, "grad_logits");
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


//======================================================//
//  DEBUG: Gradient Attribution Buffers (Issue #60) - Tensor-based
//======================================================//

void TrainingState::allocateDebugGradBuffers(int vocab_size, int d_model, cudaStream_t stream) {
	if (debug_lm_head_only_grad.data || debug_embedding_only_grad.data) {
		freeDebugGradBuffers();  // Clean up any existing buffers
	}
	
	// Rule 22: getPrimaryStream() throws if not initialized (Rule 20)
	cudaStream_t primary_stream = stream ? stream : stream_ctrl.getPrimaryStream();
	
	// Allocate as Tensors [vocab_size, d_model]
	debug_lm_head_only_grad = Tensor::zeros({vocab_size, d_model}, primary_stream, "debug_lm_head_only_grad");
	debug_embedding_only_grad = Tensor::zeros({vocab_size, d_model}, primary_stream, "debug_embedding_only_grad");
	
	const size_t buffer_size = static_cast<size_t>(vocab_size) * d_model;
	fprintf(stdout, "[DEBUG] Allocated gradient attribution buffers: %zu elements (%zu MB each)\n",
	        buffer_size, buffer_size * sizeof(float) / (1024 * 1024));
}

void TrainingState::freeDebugGradBuffers() {
	// Tensor auto-cleanup - just replace with empty tensors
	debug_lm_head_only_grad = Tensor();
	debug_embedding_only_grad = Tensor();
}

} // namespace GRIM

#endif  // USE_CUDA