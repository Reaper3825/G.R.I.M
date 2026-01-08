//======================================================//
//  Token-normalized_clipping.cu
//  Token-aware gradient clipping helpers implementation
//======================================================//

#include "Token-normalized_clipping.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <algorithm>
#include <cmath>

namespace GRIM {
namespace TNC {

namespace {

constexpr float kMinClip = HyperParameters::EPSILON_GRADIENT_CLIP;

inline float safeTokenCount(const BatchTokenStats& stats) {
	return static_cast<float>(std::max<std::int64_t>(stats.total_tokens, 1));
}

} // namespace

BatchTokenStats computeBatchTokenStats(const std::vector<std::vector<int>>& sequences) {
	BatchTokenStats stats{};
	stats.batch_size = static_cast<int>(sequences.size());

	for (const auto& seq : sequences) {
		const int seq_len = static_cast<int>(seq.size());
		stats.total_tokens += seq_len;
		stats.max_sequence_length = std::max(stats.max_sequence_length, seq_len);
	}

	return stats;
}

ClipSelection computeClipSelection(float base_clip,
						   const BatchTokenStats& stats,
						   float long_sequence_limit,
						   int long_sequence_threshold) {
	ClipSelection selection{};
	selection.stats = stats;

	float per_token_limit = std::max(base_clip, kMinClip);
	ClipMode mode = (stats.total_tokens > 0) ? ClipMode::TokenNormalized : ClipMode::FixedBase;

	// Do not boost long sequences; we only flag them for diagnostics.
	if (stats.max_sequence_length > long_sequence_threshold) {
		mode = ClipMode::LongSequenceOverride;
	}

	const float token_count = safeTokenCount(stats);
	selection.per_token_limit = per_token_limit;
	const float clip_scale = token_count;  // strictly token-normalized
	selection.effective_clip_norm = std::max(per_token_limit * clip_scale, kMinClip);
	selection.mode = mode;
	return selection;
}

float computeNormalizedGrad(float grad_norm, const BatchTokenStats& stats) {
	// Gradients are no longer pre-scaled by 1/valid_tokens (Issue #25 fix).
	// Return raw grad_norm - clipping threshold should match this scale.
	(void)stats;
	return grad_norm;
}

const char* clipModeToString(ClipMode mode) {
	switch (mode) {
	case ClipMode::FixedBase:
		return "fixed";
	case ClipMode::TokenNormalized:
		return "token_norm";
	case ClipMode::LongSequenceOverride:
		return "long_seq";
	default:
		return "unknown";
	}
}

} // namespace TNC
} // namespace GRIM

