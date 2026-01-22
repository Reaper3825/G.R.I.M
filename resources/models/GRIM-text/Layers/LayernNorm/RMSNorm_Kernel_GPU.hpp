//======================================================//
//  RMSNorm_Kernel_GPU.hpp
//  CUDA launchers for root-mean-square normalization
//
//  REFACTORED: Uses TensorContract::TensorView instead of raw float*
//  All tensor views validated via require() (Rule 20: Fail Loud)
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

//======================================================//
//  RMSNormForwardParams - Type-safe tensor views
//======================================================//
struct RMSNormForwardParams {
	// Input tensors (const views)
	TensorContract::TensorView input;    // [tokens, hidden_dim] BSM layout
	TensorContract::TensorView gamma;    // [hidden_dim] as BSM [1, hidden_dim]
	
	// Output tensors (mutable view)
	TensorContract::TensorView output;   // [tokens, hidden_dim] BSM layout
	
	// Config
	float epsilon = 1e-5f;
	cudaStream_t stream = nullptr;
	
	// RULE 20: Fail loud validation
	void validate(const char* context) const {
		input.require(context);
		gamma.require(context);
		output.require(context);
		
		if (input.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": input must have BSM layout");
		}
		if (output.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": output must have BSM layout");
		}
		
		const auto& in_shape = input.shape.as_2d();
		const auto& out_shape = output.shape.as_2d();
		if (in_shape.rows != out_shape.rows || in_shape.cols != out_shape.cols) {
			throw std::runtime_error(std::string(context) + ": input/output shape mismatch");
		}
	}
	
	// Convenience dimensions
	int tokens() const { return input.shape.as_2d().rows; }
	int hidden_dim() const { return input.shape.as_2d().cols; }
};

//======================================================//
//  RMSNormBackwardParams - Type-safe tensor views
//======================================================//
struct RMSNormBackwardParams {
	// Input tensors (const views)
	TensorContract::TensorView input;        // Original forward input [tokens, hidden_dim] BSM
	TensorContract::TensorView grad_output;  // dL/dy [tokens, hidden_dim] BSM
	TensorContract::TensorView gamma;        // Scale parameter [1, hidden_dim] BSM
	
	// Output tensors (mutable views)
	TensorContract::TensorView grad_input;   // dL/dx [tokens, hidden_dim] BSM
	TensorContract::TensorView grad_gamma;   // dL/dgamma [1, hidden_dim] BSM (may be null if not tracking)
	
	// Config
	float epsilon = 1e-5f;
	cudaStream_t stream = nullptr;
	
	// RULE 20: Fail loud validation
	void validate(const char* context) const {
		input.require(context);
		grad_output.require(context);
		gamma.require(context);
		grad_input.require(context);
		// grad_gamma may be null (optional)
		
		if (input.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": input must have BSM layout");
		}
		if (grad_output.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": grad_output must have BSM layout");
		}
		if (grad_input.layout() != TensorContract::Layout::BSM) {
			throw std::runtime_error(std::string(context) + ": grad_input must have BSM layout");
		}
		
		const auto& in_shape = input.shape.as_2d();
		const auto& go_shape = grad_output.shape.as_2d();
		const auto& gi_shape = grad_input.shape.as_2d();
		
		if (in_shape != go_shape || go_shape != gi_shape) {
			throw std::runtime_error(std::string(context) + ": input/grad_output/grad_input shape mismatch");
		}
	}
	
	// Convenience dimensions
	int tokens() const { return input.shape.as_2d().rows; }
	int hidden_dim() const { return input.shape.as_2d().cols; }
};

//======================================================//
//  Kernel launchers with TensorView (PREFERRED)
//======================================================//

// Forward pass with TensorView
void launchRMSNormForward(const RMSNormForwardParams& params);

// Backward pass with TensorView
void launchRMSNormBackward(const RMSNormBackwardParams& params);

//======================================================//
//  Legacy kernel launchers (raw float*) - for backwards compat
//======================================================//

void launchRMSNorm(const float* input,
				      float* output,
				      const float* gamma,
				      int batch_size,
				      int hidden_dim,
				      float eps,
				      cudaStream_t stream);



} // namespace GRIM