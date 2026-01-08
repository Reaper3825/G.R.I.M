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

inline bool validateDims(int total_tokens, int d_model) {
	return total_tokens > 0 && d_model > 0;
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
	if (!validateDims(params.total_tokens, params.d_model)) {
		std::ostringstream oss;
		oss << "[NumericHead::forward] Invalid dimensions: total_tokens=" << params.total_tokens
		    << " d_model=" << params.d_model << " (must be > 0)";
		throw std::runtime_error(oss.str());
	}
	if (!params.handle) {
		throw std::runtime_error("[NumericHead::forward] cuBLAS handle is NULL");
	}
	if (!params.encoder_output) {
		throw std::runtime_error("[NumericHead::forward] encoder_output is NULL");
	}
	if (!params.weights) {
		throw std::runtime_error("[NumericHead::forward] weights is NULL");
	}
	if (!params.predictions) {
		throw std::runtime_error("[NumericHead::forward] predictions is NULL");
	}
	if (!isDevicePtr(params.encoder_output)) {
		throw std::runtime_error("[NumericHead::forward] encoder_output must be device pointer");
	}
	if (!isDevicePtr(params.weights)) {
		throw std::runtime_error("[NumericHead::forward] weights must be device pointer");
	}
	if (!isDevicePtr(params.predictions)) {
		throw std::runtime_error("[NumericHead::forward] predictions must be device pointer");
	}

	if (params.stream) {
		cublasSetStream(params.handle, params.stream);
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	// Treat encoder_output as column-major [d_model, total_tokens]
	const cublasStatus_t status = cublasSgemv(
		params.handle,
		CUBLAS_OP_T,
		params.d_model,
		params.total_tokens,
		&alpha,
		params.encoder_output,
		params.d_model,
		params.weights,
		1,
		&beta,
		params.predictions,
		1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[NumericHead::forward] cublasSgemv failed status=" << status
		    << " m=" << params.d_model
		    << " n=" << params.total_tokens
		    << " lda=" << params.d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && params.bias) {
		addScalarBias(params.predictions, params.bias, params.total_tokens, params.stream);
	}
}

void launchNumericHeadBackward(const NumericHeadBackwardParams& params) {
	if (!validateDims(params.total_tokens, params.d_model)) {
		std::ostringstream oss;
		oss << "[NumericHead::backward] Invalid dimensions: total_tokens=" << params.total_tokens
		    << " d_model=" << params.d_model << " (must be > 0)";
		throw std::runtime_error(oss.str());
	}
	if (!params.handle) {
		throw std::runtime_error("[NumericHead::backward] cuBLAS handle is NULL");
	}
	if (!params.grad_predictions) {
		throw std::runtime_error("[NumericHead::backward] grad_predictions is NULL");
	}
	if (!params.grad_encoder) {
		throw std::runtime_error("[NumericHead::backward] grad_encoder is NULL");
	}
	if (!params.weights) {
		throw std::runtime_error("[NumericHead::backward] weights is NULL");
	}
	if (!isDevicePtr(params.grad_predictions)) {
		throw std::runtime_error("[NumericHead::backward] grad_predictions must be device pointer");
	}
	if (!isDevicePtr(params.grad_encoder)) {
		throw std::runtime_error("[NumericHead::backward] grad_encoder must be device pointer");
	}
	if (!isDevicePtr(params.weights)) {
		throw std::runtime_error("[NumericHead::backward] weights must be device pointer");
	}
	if (params.grad_weight && !isDevicePtr(params.grad_weight)) {
		throw std::runtime_error("[NumericHead::backward] grad_weight must be device pointer");
	}
	if (params.grad_bias && !isDevicePtr(params.grad_bias)) {
		throw std::runtime_error("[NumericHead::backward] grad_bias must be device pointer");
	}
	if (params.encoder_output && !isDevicePtr(params.encoder_output)) {
		throw std::runtime_error("[NumericHead::backward] encoder_output must be device pointer");
	}

	// REMOVED cublasSetStream - handle already bound to stream in InitTrainingState.cu

	const float alpha = 1.0f;
	const float beta = params.accumulate ? 1.0f : 0.0f;

	if (params.grad_weight && params.encoder_output) {
		// grad_weight = encoder_output^T * grad_predictions
		const cublasStatus_t w_status = cublasSgemv(
			params.handle,
			CUBLAS_OP_N,
			params.d_model,
			params.total_tokens,
			&alpha,
			params.encoder_output,
			params.d_model,
			params.grad_predictions,
			1,
			&beta,
			params.grad_weight,
			1);

		if (w_status != CUBLAS_STATUS_SUCCESS) {
			std::ostringstream oss;
			oss << "[NumericHead::backward] cublasSgemv grad_weight failed status=" << w_status
			    << " m=" << params.d_model
			    << " n=" << params.total_tokens
			    << " lda=" << params.d_model;
			throw std::runtime_error(oss.str());
		}
	}

	// grad_encoder += weights * grad_predictions^T
	const cublasStatus_t enc_status = cublasSger(
		params.handle,
		params.d_model,
		params.total_tokens,
		&alpha,
		params.weights,
		1,
		params.grad_predictions,
		1,
		params.grad_encoder,
		params.d_model);
	if (enc_status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[NumericHead::backward] cublasSger grad_encoder failed status=" << enc_status
		    << " m=" << params.d_model
		    << " n=" << params.total_tokens
		    << " lda=" << params.d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && params.grad_bias) {
		if (!params.accumulate) {
			cudaMemsetAsync(params.grad_bias, 0, sizeof(float), params.stream);
		}
		launchBiasSumGradient(params.grad_predictions,
		                      params.grad_bias,
		                      params.total_tokens,
		                      1,
		                      params.stream);
	}
}

} // namespace GRIM
