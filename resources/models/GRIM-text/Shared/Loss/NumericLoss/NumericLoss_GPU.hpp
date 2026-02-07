//======================================================//
//  NumericLoss_GPU.hpp
//  Side-channel regression loss for numeric atoms
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM {

struct NumericLossInputs {
	const float* predictions = nullptr;           // [total_tokens]
	const float* token_numeric_values = nullptr;  // [total_tokens]
	const uint8_t* token_numeric_mask = nullptr;  // [total_tokens]
	const int* targets = nullptr;                 // [total_tokens]
	int total_tokens = 0;
	int seq_len = 0;
	int valid_text_tokens = 0;                    // FOR ISSUE #136: To compensate numeric gradient scale
	float huber_delta = 1.0f;
	bool log_scale = true;
	float log_max = 20.0f;
	float loss_weight = 1.0f;
};

struct NumericLossOutputs {
	float* loss_sum = nullptr;           // [1] device scalar
	int* count = nullptr;                // [1] device scalar
	float* grad_predictions = nullptr;   // [total_tokens]
};

// Launch numeric loss + gradient kernel.
// Returns false if inputs are invalid.
bool launchNumericLoss(const NumericLossInputs& inputs,
                       const NumericLossOutputs& outputs,
                       cudaStream_t stream);

} // namespace GRIM
