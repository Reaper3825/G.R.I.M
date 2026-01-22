//======================================================//
//  Numeric Head GPU Helpers
//  Side-channel regression head for numeric atoms
//======================================================//

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

/**
 * @brief Forward pass parameters for NumericHead
 * 
 * NumericHead is a simple regression layer: predictions = encoder_output @ weights + bias
 * Uses TensorContract::TensorView for Rule 20 fail-loud validation.
 */
struct NumericHeadForwardParams {
	// Input tensors (read-only views)
	TensorContract::TensorView encoder_output;   // [total_tokens, d_model] BSM layout
	TensorContract::TensorView weights;          // [d_model, 1] BSM layout (column vector)
	TensorContract::TensorView bias;             // [1, 1] BSM layout (scalar, optional)
	
	// Output tensor (mutable view)
	TensorContract::TensorView predictions;      // [total_tokens, 1] BSM layout
	
	// Operation flags
	bool use_bias = false;
	
	// Execution context
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	
	// RULE 20: Fail loud validation
	void validate(const char* context) const {
		encoder_output.require(context);
		weights.require(context);
		predictions.require(context);  // Output tensor validation
		
		if (use_bias && !bias.is_valid()) {
			throw std::runtime_error(std::string(context) + ": use_bias=true but bias is invalid");
		}
		if (!handle) {
			throw std::runtime_error(std::string(context) + ": cuBLAS handle is NULL");
		}
	}
	
	// Convenience: extract dimensions from tensor shapes
	int total_tokens() const { return encoder_output.shape.as_2d().rows; }
	int d_model() const { return encoder_output.shape.as_2d().cols; }
};

/**
 * @brief Backward pass parameters for NumericHead
 * 
 * Computes:
 * - grad_encoder = weights @ grad_predictions^T (outer product broadcast)
 * - grad_weight = encoder_output^T @ grad_predictions
 * - grad_bias = sum(grad_predictions) if use_bias
 */
struct NumericHeadBackwardParams {
	// Input tensors (read-only views)
	TensorContract::TensorView grad_predictions; // [total_tokens, 1] BSM layout
	TensorContract::TensorView encoder_output;   // [total_tokens, d_model] BSM layout
	TensorContract::TensorView weights;          // [d_model, 1] BSM layout
	
	// Output tensors (mutable views)
	TensorContract::TensorView grad_encoder;     // [total_tokens, d_model] BSM layout
	TensorContract::TensorView grad_weight;      // [d_model, 1] BSM layout
	TensorContract::TensorView grad_bias;        // [1, 1] BSM layout (optional)
	
	// Operation flags
	bool accumulate = false;
	bool use_bias = false;
	
	// Execution context
	cublasHandle_t handle = nullptr;
	cudaStream_t stream = nullptr;
	
	// RULE 20: Fail loud validation
	void validate(const char* context) const {
		grad_predictions.require(context);
		encoder_output.require(context);
		weights.require(context);
		grad_encoder.require(context);   // Output tensor validation
		grad_weight.require(context);    // Output tensor validation
		
		if (use_bias && !grad_bias.is_valid()) {
			throw std::runtime_error(std::string(context) + ": use_bias=true but grad_bias is invalid");
		}
		if (!handle) {
			throw std::runtime_error(std::string(context) + ": cuBLAS handle is NULL");
		}
	}
	
	// Convenience: extract dimensions from tensor shapes
	int total_tokens() const { return encoder_output.shape.as_2d().rows; }
	int d_model() const { return encoder_output.shape.as_2d().cols; }
};

// Executes numeric head forward projection on GPU (predictions per token).
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchNumericHeadForward(const NumericHeadForwardParams& params);

// Computes numeric head gradients (encoder, weights, bias).
// Throws std::runtime_error on invalid params or cuBLAS failure.
void launchNumericHeadBackward(const NumericHeadBackwardParams& params);

} // namespace GRIM
