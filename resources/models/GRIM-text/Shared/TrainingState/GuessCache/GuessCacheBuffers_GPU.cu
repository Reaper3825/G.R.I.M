//======================================================//
//  GuessCacheBuffers_GPU.cu
//  TrainingState-owned GRIM-TS guess cache buffers
//======================================================//

#include "../TrainingState_GPU.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

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

	// GuessRecord is defined in GRIM-TS.hpp; these sizes must match its static_asserts.
	constexpr size_t GUESS_RECORD_SIZE = 96;
	constexpr size_t GUESS_METADATA_SIZE = 32;

	// Rule 22: getPrimaryStream() throws if not initialized (Rule 20)
	cudaStream_t primary_stream = stream_ctrl.getPrimaryStream();

	// Allocate records
	cudaError_t err = cudaMalloc(&guess_cache_buffers.records, capacity * GUESS_RECORD_SIZE);
	if (err != cudaSuccess) {
		throw std::runtime_error("[TrainingState::allocateGuessCacheBuffers] cudaMalloc records failed! "
			"capacity=" + std::to_string(capacity) + ", size=" +
			std::to_string((capacity * GUESS_RECORD_SIZE) / (1024 * 1024)) + "MB, error=" + cudaGetErrorString(err));
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
	cudaMemsetAsync(guess_cache_buffers.keys, 0xFF, capacity * sizeof(uint64_t), primary_stream);
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

} // namespace GRIM

#endif  // USE_CUDA
