#pragma once

#include "Shared/Loss/Loss.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace GRIM {

struct LossComputationInputs {
	Loss::LossContext context{};
	Loss::LossConfig config{};
	float* grad_logits = nullptr;  // Pre-allocated [tokens * vocab_size] from training_state
	// Optional override for the number of valid tokens (excludes masked/padded positions).
	// When 0, total_tokens = batch_size * seq_len is used.
	std::size_t valid_token_count = 0;
	// Optional diagnostics (disabled by default to avoid sync costs).
	bool enable_diagnostics = false;
	cudaStream_t diagnostic_stream = nullptr;
};

struct LossScratch {
	float* loss_values = nullptr;
	float* loss_accumulator = nullptr;
	size_t capacity = 0; // number of tokens we can store losses for
};

struct LossComputationResult {
	float total_loss = 0.0f;
	float average_loss = 0.0f;
	Loss::LossBreakdown breakdown{};
	bool success = false;
};

struct LossGradientInputs {
	Loss::LossContext context{};
	float* grad_logits = nullptr;
	std::size_t max_tokens = 0;
};

bool ensureLossScratchCapacity(LossScratch& scratch,
	size_t required_tokens,
	cudaStream_t stream);

LossComputationResult computeLossHost(
	const LossComputationInputs& inputs,
	LossScratch& scratch);

bool computeCrossEntropyGradientHost(
	const LossGradientInputs& inputs);

} // namespace GRIM

