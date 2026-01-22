#define USE_CUDA

#include "numeric_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <stdexcept>
#include <sstream>

// External kernel declaration (C++ linkage - can throw exceptions)
void launchBiasSumGradient(const float* grad_output, float* grad_bias,
                          int total_tokens, int hidden_dim, cudaStream_t stream);

namespace GRIM {

namespace {

__global__ void addScalarBiasKernel(float* predictions,
                                    const float* bias,
                                    int total_tokens) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens) {
		return;
	}
	predictions[idx] += bias[0];
}

inline bool isDevicePtr(const void* ptr) {
	cudaPointerAttributes attr{};
	return ptr &&
	       cudaPointerGetAttributes(&attr, ptr) == cudaSuccess &&
	       attr.type == cudaMemoryTypeDevice;
}

inline void addScalarBias(float* predictions,
                          const float* bias,
                          int total_tokens,
                          cudaStream_t stream) {
	if (!predictions) {
		throw std::runtime_error("[NumericHead::addBias] predictions is NULL");
	}
	if (!bias) {
		throw std::runtime_error("[NumericHead::addBias] bias is NULL");
	}
	if (total_tokens <= 0) {
		throw std::runtime_error("[NumericHead::addBias] total_tokens must be > 0");
	}
	constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
	const int grid = (total_tokens + kBlockSize - 1) / kBlockSize;
	addScalarBiasKernel<<<grid, kBlockSize, 0, stream>>>(predictions, bias, total_tokens);
}

} // namespace

void launchNumericHeadForward(const NumericHeadForwardParams& params) {
	// Rule 20: Fail loud validation
	params.validate("NumericHead::forward");
	
	// Extract dimensions and raw pointers from TensorViews
	const int total_tokens = params.total_tokens();
	const int d_model = params.d_model();
	const float* encoder_output = params.encoder_output.ptr;
	const float* weights = params.weights.ptr;
	float* predictions = params.predictions.ptr;
	cublasHandle_t handle = params.handle;
	cudaStream_t stream = params.stream;

	// Validate device pointers (Rule 20: crash if not device memory)
	if (!isDevicePtr(encoder_output)) {
		throw std::runtime_error("[NumericHead::forward] encoder_output must be device pointer");
	}
	if (!isDevicePtr(weights)) {
		throw std::runtime_error("[NumericHead::forward] weights must be device pointer");
	}
	if (!isDevicePtr(predictions)) {
		throw std::runtime_error("[NumericHead::forward] predictions must be device pointer");
	}

	if (stream) {
		cublasSetStream(handle, stream);
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	// Treat encoder_output as column-major [d_model, total_tokens]
	const cublasStatus_t status = cublasSgemv(
		handle,
		CUBLAS_OP_T,
		d_model,
		total_tokens,
		&alpha,
		encoder_output,
		d_model,
		weights,
		1,
		&beta,
		predictions,
		1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[NumericHead::forward] cublasSgemv failed status=" << status
		    << " m=" << d_model
		    << " n=" << total_tokens
		    << " lda=" << d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && params.bias.is_valid()) {
		addScalarBias(predictions, params.bias.ptr, total_tokens, stream);
	}
}

void launchNumericHeadBackward(const NumericHeadBackwardParams& params) {
	// Rule 20: Fail loud validation
	params.validate("NumericHead::backward");
	
	// Extract dimensions and raw pointers from TensorViews
	const int total_tokens = params.total_tokens();
	const int d_model = params.d_model();
	const float* grad_predictions = params.grad_predictions.ptr;
	const float* encoder_output = params.encoder_output.ptr;
	const float* weights = params.weights.ptr;
	float* grad_encoder = params.grad_encoder.ptr;
	float* grad_weight = params.grad_weight.ptr;
	cublasHandle_t handle = params.handle;
	cudaStream_t stream = params.stream;
	
	// Validate device pointers (Rule 20: crash if not device memory)
	if (!isDevicePtr(grad_predictions)) {
		throw std::runtime_error("[NumericHead::backward] grad_predictions must be device pointer");
	}
	if (!isDevicePtr(grad_encoder)) {
		throw std::runtime_error("[NumericHead::backward] grad_encoder must be device pointer");
	}
	if (!isDevicePtr(weights)) {
		throw std::runtime_error("[NumericHead::backward] weights must be device pointer");
	}
	if (!isDevicePtr(grad_weight)) {
		throw std::runtime_error("[NumericHead::backward] grad_weight must be device pointer");
	}
	if (!isDevicePtr(encoder_output)) {
		throw std::runtime_error("[NumericHead::backward] encoder_output must be device pointer");
	}
	if (params.use_bias && params.grad_bias.is_valid() && !isDevicePtr(params.grad_bias.ptr)) {
		throw std::runtime_error("[NumericHead::backward] grad_bias must be device pointer");
	}

	const float alpha = 1.0f;
	const float beta = params.accumulate ? 1.0f : 0.0f;

	// grad_weight = encoder_output^T * grad_predictions
	const cublasStatus_t w_status = cublasSgemv(
		handle,
		CUBLAS_OP_N,
		d_model,
		total_tokens,
		&alpha,
		encoder_output,
		d_model,
		grad_predictions,
		1,
		&beta,
		grad_weight,
		1);

	if (w_status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[NumericHead::backward] cublasSgemv grad_weight failed status=" << w_status
		    << " m=" << d_model
		    << " n=" << total_tokens
		    << " lda=" << d_model;
		throw std::runtime_error(oss.str());
	}

	// grad_encoder += weights * grad_predictions^T
	const cublasStatus_t enc_status = cublasSger(
		handle,
		d_model,
		total_tokens,
		&alpha,
		weights,
		1,
		grad_predictions,
		1,
		grad_encoder,
		d_model);
	if (enc_status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[NumericHead::backward] cublasSger grad_encoder failed status=" << enc_status
		    << " m=" << d_model
		    << " n=" << total_tokens
		    << " lda=" << d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && params.grad_bias.is_valid()) {
		if (!params.accumulate) {
			cudaMemsetAsync(params.grad_bias.ptr, 0, sizeof(float), stream);
		}
		launchBiasSumGradient(grad_predictions,
		                      params.grad_bias.ptr,
		                      total_tokens,
		                      1,
		                      stream);
	}
}

} // namespace GRIM
