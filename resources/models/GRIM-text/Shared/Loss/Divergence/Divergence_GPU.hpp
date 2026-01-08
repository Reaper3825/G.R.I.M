#pragma once

#include "Shared/Loss/Loss.hpp"

namespace GRIM::Loss {

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
								 float* out_gradient);

} // namespace GRIM::Loss

