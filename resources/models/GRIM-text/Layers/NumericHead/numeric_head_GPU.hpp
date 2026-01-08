//======================================================//
//  Numeric Head GPU Helpers
//  Side-channel regression head for numeric atoms
//======================================================//

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GRIM {

struct NumericHeadForwardParams {
	const float* encoder_output = nullptr;   // [total_tokens, d_model]
	float* predictions = nullptr;            // [total_tokens]
	const float* weights = nullptr;          // [d_model]
	const float* bias = nullptr;             // [1]
	int total_tokens = 0;
	int d_model = 0;
	bool use_bias = false;
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
};

struct NumericHeadBackwardParams {
	const float* grad_predictions = nullptr; // [total_tokens]
	const float* encoder_output = nullptr;   // [total_tokens, d_model]
	float* grad_encoder = nullptr;           // [total_tokens, d_model]
	float* grad_weight = nullptr;            // [d_model]
	float* grad_bias = nullptr;              // [1]
	const float* weights = nullptr;          // [d_model]
	int total_tokens = 0;
	int d_model = 0;
	bool accumulate = false;
	bool use_bias = false;
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
};

// Executes numeric head forward projection on GPU (predictions per token).
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchNumericHeadForward(const NumericHeadForwardParams& params);

// Computes numeric head gradients (encoder, weights, bias).
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchNumericHeadBackward(const NumericHeadBackwardParams& params);

} // namespace GRIM
