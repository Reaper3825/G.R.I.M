//======================================================//
//  RMSNorm_GPU.hpp
//  RMSNorm configuration and weight structs
//  
//  NOTE: RMSNormLayer class DELETED per Rule 20 (dead code)
//        Production uses launchRMSNorm/launchRMSNormBackward directly
//        from RMSNorm_Kernel_GPU.hpp with TensorView params
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>

namespace GRIM {

//======================================================//
//  Configuration
//======================================================//
struct RMSNormConfig {
	int hidden_dim = 0;
	float epsilon = 1e-5f;
	cudaStream_t stream = nullptr;
};

//======================================================//
//  Weight Management (used by EncodingLayer, TrainingState)
//======================================================//
struct RMSNormWeights {
	float* gamma = nullptr;       // Scale parameter [hidden_dim]
	float* gamma_grad = nullptr;  // Gradient buffer [hidden_dim]
	int size = 0;                 // Must equal hidden_dim
	
	void validate(const char* context) const {
		if (!gamma) {
			throw std::runtime_error(std::string(context) + ": gamma is NULL");
		}
		if (size <= 0) {
			throw std::runtime_error(std::string(context) + ": size must be > 0");
		}
	}
};

} // namespace GRIM
