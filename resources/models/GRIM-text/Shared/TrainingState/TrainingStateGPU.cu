//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

#include "TrainingState_GPU.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"

#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

TrainingState::TrainingState() = default;

TrainingState::~TrainingState() {
	// WEIGHT TYING: When tied, embedding_grads == lm_head_weight_grads (same pointer)
	// Only free ONCE to avoid double-free crash
	const bool grads_are_tied = (embedding_grads == lm_head_weight_grads) && 
	                            (embedding_grads != nullptr);
	
	if (grads_are_tied) {
		// Tied: free once via lm_head_weight_grads, null out embedding_grads
		if (lm_head_weight_grads) cudaFree(lm_head_weight_grads);
		// Don't free embedding_grads - it's the same pointer!
	} else {
		// Untied: free both separately
		if (embedding_grads) cudaFree(embedding_grads);
		if (lm_head_weight_grads) cudaFree(lm_head_weight_grads);
	}
	
	if (lm_head_bias_grads) cudaFree(lm_head_bias_grads);
	if (numeric_head_weight_grads) cudaFree(numeric_head_weight_grads);
	if (numeric_head_bias_grads) cudaFree(numeric_head_bias_grads);
	
	// Only free lm_head_weights if we own it (allocated separately)
	// When tie_embeddings=true, it points to EmbeddingRuntime::token_buffer
	if (lm_head_weights_owned && lm_head_weights) {
		cudaFree(lm_head_weights);
	}
	if (lm_head_bias) cudaFree(lm_head_bias);
	if (numeric_head_weights) cudaFree(numeric_head_weights);
	if (numeric_head_bias) cudaFree(numeric_head_bias);

	// Issue #33: Final RMSNorm cleanup
	if (final_rms_gamma) cudaFree(final_rms_gamma);
	if (final_rms_gamma_grads) cudaFree(final_rms_gamma_grads);
	if (cached_final_rms_input) cudaFree(cached_final_rms_input);

	auto freeVector = [](std::vector<float*>& buffers) {
		for (auto ptr : buffers) {
			if (ptr) {
				cudaFree(ptr);
			}
		}
	};

	freeVector(rms1_gamma_grads);
	freeVector(rms2_gamma_grads);
	freeVector(attn_qkv_weight_grads);
	freeVector(attn_qkv_bias_grads);
	freeVector(attn_out_weight_grads);
	freeVector(attn_out_bias_grads);
	freeVector(ffn_w1_grads);
	freeVector(ffn_b1_grads);
	freeVector(ffn_w2_grads);
	freeVector(ffn_b2_grads);

	if (cached_embeddings) cudaFree(cached_embeddings);
	freeVector(cached_ln1_outputs);
	freeVector(cached_attn_outputs);
	freeVector(cached_residual1_outputs);
	freeVector(cached_ln2_outputs);
	freeVector(cached_ffn_pre_gelu);
	freeVector(cached_ffn_outputs);
	freeVector(cached_layer_outputs);
	if (forward_layer_caches) {
		delete[] forward_layer_caches;
		forward_layer_caches = nullptr;
		forward_layer_cache_count = 0;
	}

	freeVector(cached_Q);
	freeVector(cached_K);
	freeVector(cached_V);
	freeVector(cached_attn_inputs);
	freeVector(cached_attn_bhsd);
	freeVector(cached_softmax_lse);

	if (cached_encoder_outputs) cudaFree(cached_encoder_outputs);

	if (cached_logits) cudaFree(cached_logits);
	if (cached_numeric_predictions) cudaFree(cached_numeric_predictions);
	if (cached_targets) cudaFree(cached_targets);
	if (cached_token_ids) cudaFree(cached_token_ids);
	if (cached_token_numeric_values) cudaFree(cached_token_numeric_values);
	if (cached_token_numeric_mask) cudaFree(cached_token_numeric_mask);
	// GRMT v4: text feature buffers
	if (cached_token_text_features) cudaFree(cached_token_text_features);
	if (cached_token_text_mask) cudaFree(cached_token_text_mask);
	if (sequence_weights) cudaFree(sequence_weights);
	if (grad_logits) cudaFree(grad_logits);
	if (grad_numeric_predictions) cudaFree(grad_numeric_predictions);
	if (grad_encoder_out) cudaFree(grad_encoder_out);
	if (fa_dq_accum) cudaFree(fa_dq_accum);
	if (fa_dsoftmax_sum) cudaFree(fa_dsoftmax_sum);
	if (fa_q_bf16) cudaFree(fa_q_bf16);
	if (fa_k_bf16) cudaFree(fa_k_bf16);
	if (fa_v_bf16) cudaFree(fa_v_bf16);
	if (fa_out_bf16) cudaFree(fa_out_bf16);
	if (fa_dout_bf16) cudaFree(fa_dout_bf16);
	if (fa_dq_bf16) cudaFree(fa_dq_bf16);
	if (fa_dk_bf16) cudaFree(fa_dk_bf16);
	if (fa_dv_bf16) cudaFree(fa_dv_bf16);
	if (encoder_workspace) cudaFree(encoder_workspace);
	if (d_loss_scratch) cudaFree(d_loss_scratch);
	if (d_loss_sum_scratch) cudaFree(d_loss_sum_scratch);
	if (d_numeric_loss_sum) cudaFree(d_numeric_loss_sum);
	if (d_numeric_loss_count) cudaFree(d_numeric_loss_count);
	if (d_entropy_output) cudaFree(d_entropy_output);  // Free entropy buffer
	TeacherLogits::release(teacher_logits);
	TeacherLogits::release(reference_logits);
	
	// Free optimizer states
	freeOptimizerStates();
	
	// Free guess cache buffers (GRIM-TS)
	freeGuessCacheBuffers();
	
	// StreamController owns and destroys streams - no manual cleanup needed
	if (cublas_handle) cublasDestroy(cublas_handle);
}

//======================================================//
//  Optimizer State Management (Centralized)
//======================================================//

void TrainingState::allocateOptimizerStates(const std::vector<size_t>& sizes) {
	// Free any existing states first
	freeOptimizerStates();
	
	optimizer_m_states.resize(sizes.size(), nullptr);
	optimizer_v_states.resize(sizes.size(), nullptr);
	optimizer_state_sizes = sizes;
	
	// Get stream from centralized controller per Rule 22
	cudaStream_t primary_stream = stream_ctrl.isInitialized() 
		? stream_ctrl.getPrimaryStream() 
		: nullptr;
	StreamController::fatalIfDefaultStream(primary_stream, "TrainingState::allocateOptimizerStates");
	
	for (size_t i = 0; i < sizes.size(); ++i) {
		if (sizes[i] > 0) {
			// Rule 20: Check cudaMalloc and throw on failure
			cudaError_t err_m = cudaMalloc(&optimizer_m_states[i], sizes[i] * sizeof(float));
			if (err_m != cudaSuccess) {
				freeOptimizerStates();
				throw std::runtime_error("[TrainingState::allocateOptimizerStates] cudaMalloc m_states[" +
					std::to_string(i) + "] failed: size=" + std::to_string(sizes[i]) +
					" error=" + cudaGetErrorString(err_m));
			}
			
			cudaError_t err_v = cudaMalloc(&optimizer_v_states[i], sizes[i] * sizeof(float));
			if (err_v != cudaSuccess) {
				freeOptimizerStates();
				throw std::runtime_error("[TrainingState::allocateOptimizerStates] cudaMalloc v_states[" +
					std::to_string(i) + "] failed: size=" + std::to_string(sizes[i]) +
					" error=" + cudaGetErrorString(err_v));
			}
			
			// Async zero using centralized stream
			cudaMemsetAsync(optimizer_m_states[i], 0, sizes[i] * sizeof(float), primary_stream);
			cudaMemsetAsync(optimizer_v_states[i], 0, sizes[i] * sizeof(float), primary_stream);
		}
	}
	optimizer_states_allocated = true;
}

void TrainingState::freeOptimizerStates() {
	for (auto ptr : optimizer_m_states) {
		if (ptr) cudaFree(ptr);
	}
	for (auto ptr : optimizer_v_states) {
		if (ptr) cudaFree(ptr);
	}
	optimizer_m_states.clear();
	optimizer_v_states.clear();
	optimizer_state_sizes.clear();
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

} // namespace GRIM

#endif  // USE_CUDA

