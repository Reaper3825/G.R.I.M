#pragma once

#include <cstddef>
#include <cstdint>

#include <array>
#include <string_view>

#include <cuda_runtime_api.h>
#include "../HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM::Loss {

enum class Term : std::uint8_t {
	CrossEntropy = 0,
	LabelSmoothing,
	DistillationKL,
	PreferenceKL,
	Focal,
	Masked,
	Custom
};

struct LabelSmoothingConfig {
	bool enabled = HyperParameters::DEFAULT_LOSS_LABEL_SMOOTHING_ENABLED;
	float epsilon = HyperParameters::DEFAULT_LOSS_LABEL_SMOOTHING_EPSILON;
};

struct DistillationConfig {
	bool enabled = HyperParameters::DEFAULT_LOSS_DISTILLATION_ENABLED;
	float temperature = HyperParameters::DEFAULT_LOSS_DISTILLATION_TEMPERATURE;
	float lambda = HyperParameters::DEFAULT_LOSS_DISTILLATION_LAMBDA;
};

struct PreferenceKLConfig {
	bool enabled = HyperParameters::DEFAULT_LOSS_PREFERENCE_ENABLED;
	float beta = HyperParameters::DEFAULT_LOSS_PREFERENCE_BETA;
};

struct FocalLossConfig {
	bool enabled = HyperParameters::DEFAULT_LOSS_FOCAL_ENABLED;
	float gamma = HyperParameters::DEFAULT_LOSS_FOCAL_GAMMA;
	float alpha = HyperParameters::DEFAULT_LOSS_FOCAL_ALPHA;
};

struct MaskConfig {
	bool enabled = HyperParameters::DEFAULT_LOSS_MASKING_ENABLED;
	std::string_view tag{};
};

struct GuessFeedbackConfig {
	bool enabled = false;
	float lambda = 0.0f;
	float min_confidence = 0.0f;
};

struct LimitsConfig {
	// Maximum tokens the loss pipeline is allowed to process in one call.
	// 0 means "no explicit cap" (caller should fill this for safety).
	std::size_t max_tokens = 0;
};

// Issue #44 FIX: Entropy regularization to prevent mode collapse
// Penalizes logit concentration: reg = λ * Σ_v p_v² (where p = softmax(z))
// This directly attacks the mode collapse by penalizing when one token dominates.
struct EntropyRegConfig {
	bool enabled = false;
	float lambda = 0.0f;  // Regularization strength (try 0.1-1.0)
};

struct LossConfig {
	LabelSmoothingConfig label_smoothing{};
	DistillationConfig distillation{};
	PreferenceKLConfig preference{};
	FocalLossConfig focal{};
	MaskConfig masking{};
	GuessFeedbackConfig guess_feedback{};
	LimitsConfig limits{};
	EntropyRegConfig entropy_reg{};  // Issue #44: Entropy regularization
};

struct LossContext {
	const float* logits = nullptr;
	const int* targets = nullptr;
	const float* teacher_logits = nullptr;
	const float* reference_logits = nullptr;
	const float* token_mask = nullptr;
	const float* sequence_weights = nullptr;
	int sequence_weight_count = 0;
	// Issue #38 FIX: Per-token class weighting to prevent mode collapse on frequent tokens
	// weight[token_id] = inverse frequency based weight (frequent tokens get lower weight)
	const float* token_weights = nullptr;  // [vocab_size] - Per-token weight (nullptr = 1.0 for all)
	// GRMT v6: Per-position byte length weight for loss (prevents atoms being "free")
	const uint16_t* position_byte_lengths = nullptr;  // [batch * seq] - Byte length per position
	// Issue #39 FIX: Output logit bias correction to prevent mode collapse
	// Subtracts running EMA of mean logit per token BEFORE softmax.
	// This prevents tokens like SPACE from having systematically higher logits.
	float* logit_bias = nullptr;           // [vocab_size] - EMA of per-token mean logit (mutable for update)
	float* logit_bias_update = nullptr;    // [vocab_size] - scratch for batch mean computation
	float logit_bias_ema_alpha = 0.05f;    // EMA decay rate (0.05 = slow adapt)
	int valid_tokens = 0; // optional override for valid tokens (excludes masked/padded)
	int batch_size = 0;
	int seq_len = 0;
	int vocab_size = 0;
	cudaStream_t stream = nullptr;
};

struct LossBreakdown {
	float cross_entropy = 0.0f;
	float label_smoothing = 0.0f;
	float distillation_kl = 0.0f;
	float preference_kl = 0.0f;
	float focal = 0.0f;
	float masked = 0.0f;
	float custom = 0.0f;
	float total = 0.0f;
};

struct DeviceBuffers {
	float* token_losses = nullptr;
	float* grad_logits = nullptr;
	float* scratch = nullptr;
};

struct AuxiliaryBatchViews {
	const float* guess_confidence = nullptr;
	const float* per_token_mask = nullptr;
	const float* reward_scores = nullptr;
	int sample_count = 0;
};

void validate(const LossContext& ctx);

LossBreakdown computeLossTerms(const LossContext& ctx,
							   const LossConfig& cfg,
							   DeviceBuffers buffers);

void applyLabelSmoothing(const LossContext& ctx,
						 const LabelSmoothingConfig& cfg,
						 DeviceBuffers buffers,
						 LossBreakdown& out_loss);

void accumulateDistillationKL(const LossContext& ctx,
							  const DistillationConfig& cfg,
							  DeviceBuffers buffers,
							  LossBreakdown& out_loss);

void accumulatePreferenceKL(const LossContext& ctx,
							const PreferenceKLConfig& cfg,
							DeviceBuffers buffers,
							LossBreakdown& out_loss);

void applyFocalLossScaling(const LossContext& ctx,
						   const FocalLossConfig& cfg,
						   DeviceBuffers buffers,
						   LossBreakdown& out_loss);

void applyTokenMasking(const LossContext& ctx,
					   const MaskConfig& cfg,
					   DeviceBuffers buffers,
					   LossBreakdown& out_loss);

void blendGuessFeedback(const AuxiliaryBatchViews& aux,
						const GuessFeedbackConfig& cfg,
						float* sequence_weights,
						LossBreakdown& out_loss);

float computeInfoNCELoss(const float* anchors,
						 const float* positives,
						 const float* negatives,
						 int feature_dim,
						 int negative_count,
						 float temperature,
						 cudaStream_t stream);

void computeCosineSimilarityLoss(const float* a,
								 const float* b,
								 int feature_dim,
								 float margin,
								 cudaStream_t stream,
								 float* out_value,
								 float* out_gradient = nullptr);

} // namespace GRIM::Loss

