//======================================================//
//  Token-normalized_clipping.hpp
//  Token-aware gradient clipping helpers (host side)
//======================================================//

#pragma once

#include <cstdint>
#include <vector>

namespace GRIM {
namespace TNC {

struct BatchTokenStats {
	std::int64_t total_tokens = 0;
	int max_sequence_length = 0;
	int batch_size = 0;
};

enum class ClipMode {
	FixedBase,
	TokenNormalized,
	LongSequenceOverride
};

struct ClipSelection {
	float per_token_limit = 0.0f;
	float effective_clip_norm = 0.0f;
	ClipMode mode = ClipMode::FixedBase;
	BatchTokenStats stats{};
};

// Derive aggregate batch statistics from tokenized sequences.
BatchTokenStats computeBatchTokenStats(const std::vector<std::vector<int>>& sequences);

// Produce the per-token and effective clip thresholds using token-normalized logic.
ClipSelection computeClipSelection(float base_clip,
						   const BatchTokenStats& stats,
						   float long_sequence_limit = 1.0f,
						   int long_sequence_threshold = 1152);

// Convenience helper for logging/decision making.
float computeNormalizedGrad(float grad_norm, const BatchTokenStats& stats);

// String label for diagnostics/logging.
const char* clipModeToString(ClipMode mode);

} // namespace TNC
} // namespace GRIM

