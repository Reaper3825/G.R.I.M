//======================================================//
//  RMSNorm_GPU.hpp
//  Production RMSNorm layer - controller-managed optimization
//  
//  ARCHITECTURE: 
//    - Layer owns: forward/backward logic, weight pointers
//    - Controller owns: optimizer state (m, v), gradient updates
//    - Layer exposes: weightPtr(), gradPtr(), paramCount() for controller
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <stdexcept>

#include "../grim_layer_gpu.hpp"

namespace GRIM {

struct RMSNormConfig {
	int hidden_dim = 0;
	float epsilon = 1e-5f;
	cudaStream_t stream = nullptr;
};

struct RMSNormWeights {
	float* gamma = nullptr;       // Scale parameter [hidden_dim]
	float* gamma_grad = nullptr;  // Gradient buffer [hidden_dim]
	int size = 0;                 // Must equal hidden_dim
};

struct RMSNormForwardArgs {
	const float* input = nullptr;
	float* output = nullptr;
	int tokens = 0;
	int hidden_dim = 0;
	cudaStream_t stream = nullptr;
};

struct RMSNormBackwardArgs {
	const float* input = nullptr;       // Original input x (required for correct gradients)
	const float* grad_output = nullptr; // dL/dy
	float* grad_input = nullptr;        // dL/dx
	int tokens = 0;
	int hidden_dim = 0;
	cudaStream_t stream = nullptr;
};

class RMSNormLayer final : public Layer<RMSNormLayer, float> {
public:
	static constexpr LayerType layer_type = LayerType::kRMSNorm;

	RMSNormLayer() = delete;  // Must provide config
	
	explicit RMSNormLayer(const RMSNormConfig& config) : config_(config) {
		if (config.hidden_dim <= 0) {
			throw std::invalid_argument("RMSNormLayer: hidden_dim must be > 0");
		}
	}
	
	RMSNormLayer(const Dimensions& dims, const RMSNormConfig& config)
		: Layer(dims), config_(config) {
		if (config.hidden_dim <= 0) {
			throw std::invalid_argument("RMSNormLayer: hidden_dim must be > 0");
		}
	}

	// Config is immutable after construction
	const RMSNormConfig& config() const noexcept { return config_; }

	// Weight management - controller allocates buffers, layer uses them
	void setWeights(const RMSNormWeights& weights);
	const RMSNormWeights& weights() const noexcept { return weights_; }
	
	// Accessors for controller to build ParameterGroup
	float* weightPtr() const noexcept { return weights_.gamma; }
	float* gradPtr() const noexcept { return weights_.gamma_grad; }
	int paramCount() const noexcept { return weights_.size; }

	// Forward pass - caches input for backward
	void forward(const RMSNormForwardArgs& args);

	// Backward pass - accumulates into gamma_grad
	void backward(const RMSNormBackwardArgs& args);

	// Layer base class interface
	void onConfigure(const Dimensions& dims);
	void forwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace);
	void backwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace);

	// Cached state accessors (for external backward if needed)
	const float* lastInput() const noexcept { return last_input_; }
	int lastTokens() const noexcept { return last_tokens_; }
	int lastHiddenDim() const noexcept { return last_hidden_; }

private:
	RMSNormConfig config_;
	RMSNormWeights weights_{};
	
	// Cached from forward for backward
	const float* last_input_ = nullptr;
	int last_tokens_ = 0;
	int last_hidden_ = 0;
};

} // namespace GRIM
