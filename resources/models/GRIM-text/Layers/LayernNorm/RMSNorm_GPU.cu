//======================================================//
//  RMSNorm_GPU.cu
//  Production RMSNorm implementation
//  
//  ARCHITECTURE: Layer owns forward/backward, controller owns optimizer
//======================================================//

#include "RMSNorm_GPU.hpp"
#include "RMSNorm_Kernel_GPU.hpp"

#include <limits>
#include <stdexcept>

namespace {

int resolvePositive(int primary, int secondary, int fallback) {
	if (primary > 0) return primary;
	if (secondary > 0) return secondary;
	if (fallback > 0) return fallback;
	throw std::runtime_error("RMSNorm: all dimension sources are zero/negative");
}

int resolveTokenCount(const GRIM::RMSNormForwardArgs& args,
				      const GRIM::Dimensions& dims) {
	if (args.tokens > 0) return args.tokens;
	if (dims.input > 0) return dims.input;
	if (dims.output > 0) return dims.output;
	throw std::runtime_error("RMSNorm: token count unresolved from all sources");
}

int castTokens(std::size_t tokens) {
	if (tokens == 0) {
		throw std::runtime_error("RMSNorm: token count is zero");
	}
	if (tokens > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
		throw std::runtime_error("RMSNorm: token count exceeds int range");
	}
	return static_cast<int>(tokens);
}

} // namespace

namespace GRIM {

void RMSNormLayer::setWeights(const RMSNormWeights& weights) {
	if (!weights.gamma) {
		throw std::invalid_argument("RMSNormLayer::setWeights: gamma is null");
	}
	if (!weights.gamma_grad) {
		throw std::invalid_argument("RMSNormLayer::setWeights: gamma_grad is null");
	}
	if (weights.size <= 0) {
		throw std::invalid_argument("RMSNormLayer::setWeights: size must be > 0");
	}
	if (weights.size != config_.hidden_dim) {
		throw std::invalid_argument("RMSNormLayer::setWeights: size != hidden_dim");
	}
	weights_ = weights;
}

void RMSNormLayer::forward(const RMSNormForwardArgs& args) {
	if (!args.input) {
		throw std::invalid_argument("RMSNormLayer::forward: input is null");
	}
	if (!args.output) {
		throw std::invalid_argument("RMSNormLayer::forward: output is null");
	}
	if (!weights_.gamma) {
		throw std::runtime_error("RMSNormLayer::forward: weights not set");
	}

	const int tokens = resolveTokenCount(args, dims());
	const int hidden_dim = resolvePositive(args.hidden_dim, config_.hidden_dim, dims().output);

	if (hidden_dim > weights_.size) {
		throw std::runtime_error("RMSNormLayer::forward: hidden_dim exceeds weight size");
	}

	const cudaStream_t stream = args.stream ? args.stream : config_.stream;
	
	launchRMSNorm(args.input,
			      args.output,
			      weights_.gamma,
			      tokens,
			      hidden_dim,
			      config_.epsilon,
			      stream);

	// Cache for backward
	last_input_ = args.input;
	last_tokens_ = tokens;
	last_hidden_ = hidden_dim;
}

void RMSNormLayer::backward(const RMSNormBackwardArgs& args) {
	if (!args.grad_output) {
		throw std::invalid_argument("RMSNormLayer::backward: grad_output is null");
	}
	if (!args.grad_input) {
		throw std::invalid_argument("RMSNormLayer::backward: grad_input is null");
	}
	if (!weights_.gamma) {
		throw std::runtime_error("RMSNormLayer::backward: weights not set");
	}
	if (!weights_.gamma_grad) {
		throw std::runtime_error("RMSNormLayer::backward: gamma_grad not set");
	}

	// Prefer provided input; fall back to cached forward input
	const float* input = args.input ? args.input : last_input_;
	if (!input) {
		throw std::runtime_error("RMSNormLayer::backward: input missing (provide args.input or call forward first)");
	}

	const int tokens = resolvePositive(args.tokens, last_tokens_, 0);
	const int hidden_dim = resolvePositive(args.hidden_dim, config_.hidden_dim, last_hidden_);

	const cudaStream_t stream = args.stream ? args.stream : config_.stream;
	
	launchRMSNormBackward(input,
	                      args.grad_output,
	                      weights_.gamma,
	                      args.grad_input,
	                      weights_.gamma_grad,
	                      tokens,
	                      hidden_dim,
	                      config_.epsilon,
	                      stream);
}

void RMSNormLayer::onConfigure(const Dimensions& dims) {
	setDimensions(dims);
	// Don't mutate config after construction - just validate
	if (dims.output != config_.hidden_dim && config_.hidden_dim > 0) {
		throw std::runtime_error("RMSNormLayer::onConfigure: dims.output != config.hidden_dim");
	}
}

void RMSNormLayer::forwardImpl(const LayerIO<float>& io,
					           LayerWorkspace<float>* /*workspace*/) {
	RMSNormForwardArgs args;
	args.input = io.input;
	args.output = io.output;
	args.tokens = castTokens(io.tokens);
	args.hidden_dim = config_.hidden_dim;
	args.stream = config_.stream;
	forward(args);
}

void RMSNormLayer::backwardImpl(const LayerIO<float>& io,
					            LayerWorkspace<float>* /*workspace*/) {
	RMSNormBackwardArgs args{};
	args.grad_output = io.input;   // LayerIO convention: input is upstream gradient
	args.grad_input = io.output;   // LayerIO convention: output is downstream gradient
	args.input = last_input_;      // Use cached forward input
	args.tokens = castTokens(io.tokens);
	args.hidden_dim = config_.hidden_dim;
	args.stream = config_.stream;
	backward(args);
}

} // namespace GRIM
