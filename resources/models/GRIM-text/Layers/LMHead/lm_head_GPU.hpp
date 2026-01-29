//======================================================//
//  LM Head GPU Helpers
//  Shared CUDA helpers for projection + gradients
//======================================================//
//
//  REFACTORED: Uses TensorContract::TensorView instead of raw float*
//  All tensor views validated via require() (Rule 20: Fail Loud)
//
//======================================================//

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

//======================================================//
//  LMHeadForwardParams - Type-safe tensor views
//======================================================//
struct LMHeadForwardParams {
	// Input tensors (const views)
	TensorContract::TensorView encoder_output;   // [total_tokens, d_model] BSM layout
	TensorContract::TensorView weights;          // [vocab_size, d_model] BSM layout (row-major)
	TensorContract::TensorView bias;             // [vocab_size, 1] BSM layout (optional, may be null)
	
	// Output tensors (mutable views)
	TensorContract::TensorView logits;           // [total_tokens, vocab_size] LOGITS layout
	TensorContract::TensorView centered_scratch; // [total_tokens, d_model] BSM (scratch for Issue #37, may be null)
	
	// Operation flags
	bool use_bias = false;
	bool use_centering = true;                   // Enable zero-mean centering (Issue #37 fix)
	
	// Execution context
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	
	// RULE 20: Fail loud validation
	void validate(const char* context) const {
		encoder_output.require(context);
		weights.require(context);
		logits.require(context);
		
		if (use_bias && !bias.is_valid()) {
			throw std::runtime_error(std::string(context) + ": use_bias=true but bias view is invalid");
		}
		if (use_centering && !centered_scratch.is_valid()) {
			throw std::runtime_error(std::string(context) + ": use_centering=true but centered_scratch is invalid");
		}
		if (!handle) {
			throw std::runtime_error(std::string(context) + ": cuBLAS handle is NULL");
		}
		
		// Layout validation
		if (encoder_output.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": encoder_output must have BSM layout");
		}
		if (logits.layout() != TensorContract::Layout::LOGITS) {
			throw std::runtime_error(std::string(context) + ": logits must have LOGITS layout");
		}
	}
	
	// Convenience: extract dimensions from tensor shapes
	int total_tokens() const { return encoder_output.shape.as_2d().rows; }
	int d_model() const { return encoder_output.shape.as_2d().cols; }
	int vocab_size() const { return logits.shape.as_2d().cols; }
};

// Executes LM head forward projection on GPU using cuBLAS.
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchLMHeadForward(const LMHeadForwardParams& params);

// NOTE: launchLMHeadBackward() REMOVED (Issue #58 cleanup)
// Production training uses autograd::matmul() which has its own MatMulGradFn
// that handles backward pass via TensorContract_GPU.cu operations.
// See AutogradTraining.cu for usage.

// ============================================================================
// ISSUE #37/#43 FIX: Center hidden states (zero-mean each token's features)
// 
// This function centers the encoder output before LM head projection.
// It MUST be called before autograd::matmul() in the autograd training path!
//
// Why: Without centering, non-zero mean hidden states cause systematic bias
// in weight gradients, leading to mode collapse (all predictions converge
// to most frequent token 277 = SPACE).
//
// Math: centered[t,i] = hidden[t,i] - mean_t  where mean_t = (1/d_model) Σ_i hidden[t,i]
// After centering: Σ_i centered[t,i] = 0 for all positions t
// ============================================================================
void launchCenterHiddenStates(
    const float* input,    // [total_tokens, d_model] - encoder output
    float* output,         // [total_tokens, d_model] - scratch buffer for centered data
    int d_model,
    int total_tokens,
    cudaStream_t stream
);

} // namespace GRIM
