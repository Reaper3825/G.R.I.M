#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace GRIM {

// Shared cache limits so staging helpers and training state stay aligned.
inline constexpr size_t kMaxCachedBatch = 4;
inline constexpr size_t kMaxCachedSeqLen = 8192;

// Holds host-side tensors needed to stage a batch into the GPU caches.
struct BatchPreparationResult {
	bool fits_in_cache = true;
	size_t batch_size = 0;
	size_t max_seq_len = 0;
	std::vector<int> padded_input_ids;
	std::vector<int> padded_target_ids;
	std::vector<float> padded_numeric_values;
	std::vector<uint8_t> padded_numeric_mask;
	// GRMT v4: text features
	std::vector<uint16_t> padded_text_features;  // [batch_size * max_seq_len * kTextFeatureDim]
	std::vector<uint8_t> padded_text_mask;       // [batch_size * max_seq_len]
	// GRMT v6: per-token byte lengths for loss weighting
	std::vector<uint16_t> padded_byte_lengths;   // [batch_size * max_seq_len]
	std::vector<int> sequence_lengths;
};

// Prepares padded tensors for a batch, enforcing cache capacity limits.
BatchPreparationResult prepareLossBatchInputs(
	const std::vector<std::vector<int>>& batch_input_ids,
	const std::vector<std::vector<int>>& batch_target_ids,
	const std::vector<std::vector<float>>& batch_numeric_values,
	const std::vector<std::vector<uint8_t>>& batch_numeric_mask,
	const std::vector<std::vector<uint16_t>>& batch_text_features,  // GRMT v4
	const std::vector<std::vector<uint8_t>>& batch_text_mask,       // GRMT v4
	const std::vector<std::vector<uint16_t>>& batch_byte_lengths,   // GRMT v6
	size_t max_cached_batch,
	size_t max_cached_seq_len);

}  // namespace GRIM
