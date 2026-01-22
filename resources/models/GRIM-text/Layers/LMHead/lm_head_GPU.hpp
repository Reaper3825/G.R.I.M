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

//======================================================//
//  LMHeadBackwardParams - Type-safe tensor views
//======================================================//
struct LMHeadBackwardParams {
	// Input tensors (const views)
	TensorContract::TensorView grad_logits;      // [total_tokens, vocab_size] LOGITS layout
	TensorContract::TensorView encoder_output;   // [total_tokens, d_model] BSM (original input)
	TensorContract::TensorView centered_encoder; // [total_tokens, d_model] BSM (centered, Issue #37)
	TensorContract::TensorView weights;          // [vocab_size, d_model] BSM (for grad_encoder)
	
	// Output tensors (mutable views, gradients)
	TensorContract::TensorView grad_encoder;     // [total_tokens, d_model] BSM
	TensorContract::TensorView grad_weight;      // [vocab_size, d_model] BSM
	TensorContract::TensorView grad_bias;        // [vocab_size, 1] BSM (optional)
	
	// Operation flags
	bool accumulate = false;
	bool use_bias = false;
	bool use_centering = true;                   // Whether centering was applied in forward (Issue #37)
	bool recenter_gradients = true;              // Recenter grad rows after GEMM (Issue #40)
	
	// Execution context
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	
	// RULE 20: Fail loud validation
	void validate(const char* context) const {
		grad_logits.require(context);
		encoder_output.require(context);
		grad_encoder.require(context);
		grad_weight.require(context);
		weights.require(context);
		
		if (use_centering && !centered_encoder.is_valid()) {
			throw std::runtime_error(std::string(context) + ": use_centering=true but centered_encoder is invalid");
		}
		if (use_bias && !grad_bias.is_valid()) {
			throw std::runtime_error(std::string(context) + ": use_bias=true but grad_bias is invalid");
		}
		if (!handle) {
			throw std::runtime_error(std::string(context) + ": cuBLAS handle is NULL");
		}
		
		// Layout validation
		if (grad_logits.layout() != TensorContract::Layout::LOGITS) {
			throw std::runtime_error(std::string(context) + ": grad_logits must have LOGITS layout");
		}
		if (encoder_output.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": encoder_output must have BSM layout");
		}
	}
	
	// Convenience: extract dimensions from tensor shapes
	int total_tokens() const { return grad_logits.shape.as_2d().rows; }
	int d_model() const { return encoder_output.shape.as_2d().cols; }
	int vocab_size() const { return grad_logits.shape.as_2d().cols; }
};

// Executes LM head forward projection on GPU using cuBLAS.
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchLMHeadForward(const LMHeadForwardParams& params);

// Computes LM head gradients (encoder, weights, bias) with cuBLAS helpers.
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchLMHeadBackward(const LMHeadBackwardParams& params);

} // namespace GRIM
