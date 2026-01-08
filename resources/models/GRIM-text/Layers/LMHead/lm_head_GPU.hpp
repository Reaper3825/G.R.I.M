//======================================================//
//  LM Head GPU Helpers
//  Shared CUDA helpers for projection + gradients
//======================================================//

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GRIM {

struct LMHeadForwardParams {
	const float* encoder_output = nullptr;   // [total_tokens, d_model]
	float* logits = nullptr;                 // [total_tokens, vocab_size]
	const float* weights = nullptr;          // [vocab_size, d_model] row-major
	const float* bias = nullptr;             // [vocab_size]
	int batch_size = 0;
	int seq_len = 0;
	int d_model = 0;
	int vocab_size = 0;
	bool use_bias = false;
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
};

struct LMHeadBackwardParams {
	const float* grad_logits = nullptr;      // [total_tokens, vocab_size]
	const float* encoder_output = nullptr;   // [total_tokens, d_model]
	float* grad_encoder = nullptr;           // [total_tokens, d_model]
	float* grad_weight = nullptr;            // [vocab_size, d_model]
	float* grad_bias = nullptr;              // [vocab_size]
	const float* weights = nullptr;          // [vocab_size, d_model]
	int batch_size = 0;
	int seq_len = 0;
	int d_model = 0;
	int vocab_size = 0;
	bool accumulate = false;
	bool use_bias = false;
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
};

// Executes LM head forward projection on GPU using cuBLAS.
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchLMHeadForward(const LMHeadForwardParams& params);

// Computes LM head gradients (encoder, weights, bias) with cuBLAS helpers.
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchLMHeadBackward(const LMHeadBackwardParams& params);

} // namespace GRIM
