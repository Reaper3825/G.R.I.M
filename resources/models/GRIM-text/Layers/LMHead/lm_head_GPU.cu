#include "lm_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <stdexcept>
#include <sstream>

// External kernel declaration (defined in BackwardKernels.cu)
void launchBiasSumGradient(const float* grad_output,
                          float* grad_bias,
                          int total_tokens,
                          int hidden_dim,
                          cudaStream_t stream);

namespace GRIM {

namespace {

__global__ void addBiasKernel(float* logits,
							  const float* bias,
							  int vocab_size,
							  int total_tokens) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = vocab_size * total_tokens;
	if (idx >= total) {
		return;
	}
	const int vocab_idx = idx % vocab_size;
	logits[idx] += bias[vocab_idx];
}

inline bool validateCommonDims(int batch_size,
							   int seq_len,
							   int d_model,
							   int vocab_size) {
	return batch_size > 0 && seq_len > 0 && d_model > 0 && vocab_size > 0;
}

inline void addBias(float* logits,
					const float* bias,
					int vocab_size,
					int total_tokens,
					cudaStream_t stream) {
	if (!logits) {
		throw std::runtime_error("[LMHead::addBias] logits is NULL");
	}
	if (!bias) {
		throw std::runtime_error("[LMHead::addBias] bias is NULL");
	}
	if (vocab_size <= 0) {
		throw std::runtime_error("[LMHead::addBias] vocab_size must be > 0, got " + std::to_string(vocab_size));
	}
	if (total_tokens <= 0) {
		throw std::runtime_error("[LMHead::addBias] total_tokens must be > 0, got " + std::to_string(total_tokens));
	}
	constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
	const int total = vocab_size * total_tokens;
	const int grid = (total + kBlockSize - 1) / kBlockSize;
	addBiasKernel<<<grid, kBlockSize, 0, stream>>>(logits, bias, vocab_size, total_tokens);
	
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("[LMHead::addBias] kernel launch failed: ") + cudaGetErrorString(err));
	}
}

inline bool isDevicePtr(const void* ptr) {
	cudaPointerAttributes attr{};
	return ptr &&
	       cudaPointerGetAttributes(&attr, ptr) == cudaSuccess &&
	       attr.type == cudaMemoryTypeDevice;
}

} // namespace

void launchLMHeadForward(const LMHeadForwardParams& params) {
	if (!validateCommonDims(params.batch_size, params.seq_len,
							params.d_model, params.vocab_size)) {
		std::ostringstream oss;
		oss << "[LMHead::forward] Invalid dimensions: batch_size=" << params.batch_size
		    << " seq_len=" << params.seq_len << " d_model=" << params.d_model
		    << " vocab_size=" << params.vocab_size << " (all must be > 0)";
		throw std::runtime_error(oss.str());
	}

	if (!params.handle) {
		throw std::runtime_error("[LMHead::forward] cuBLAS handle is NULL");
	}
	if (!params.encoder_output) {
		throw std::runtime_error("[LMHead::forward] encoder_output is NULL");
	}
	if (!params.weights) {
		throw std::runtime_error("[LMHead::forward] weights is NULL");
	}
	if (!params.logits) {
		throw std::runtime_error("[LMHead::forward] logits output buffer is NULL");
	}
	if (!isDevicePtr(params.encoder_output)) {
		throw std::runtime_error("[LMHead::forward] encoder_output must be device pointer");
	}
	if (!isDevicePtr(params.weights)) {
		throw std::runtime_error("[LMHead::forward] weights must be device pointer");
	}
	if (!isDevicePtr(params.logits)) {
		throw std::runtime_error("[LMHead::forward] logits must be device pointer");
	}

	const int total_tokens = params.batch_size * params.seq_len;
	if (total_tokens == 0) {
		return;
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasStatus_t status = cublasSgemm(
		params.handle,
		CUBLAS_OP_T,               // weights: row-major [vocab, d_model] -> transpose for GEMM
		CUBLAS_OP_N,               // encoder: row-major [tokens, d_model]
		params.vocab_size,         // m
		total_tokens,              // n
		params.d_model,            // k
		&alpha,
		params.weights,
		params.d_model,            // lda
		params.encoder_output,
		params.d_model,            // ldb
		&beta,
		params.logits,
		params.vocab_size);        // ldc

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[LMHead::forward] cublasSgemm failed status=" << status
		    << " m=" << params.vocab_size << " n=" << total_tokens
		    << " k=" << params.d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && params.bias) {
		addBias(params.logits, params.bias, params.vocab_size, total_tokens, params.stream);
	}
}

void launchLMHeadBackward(const LMHeadBackwardParams& params) {
	if (!validateCommonDims(params.batch_size, params.seq_len,
							params.d_model, params.vocab_size)) {
		std::ostringstream oss;
		oss << "[LMHead::backward] Invalid dimensions: batch_size=" << params.batch_size
		    << " seq_len=" << params.seq_len << " d_model=" << params.d_model
		    << " vocab_size=" << params.vocab_size << " (all must be > 0)";
		throw std::runtime_error(oss.str());
	}

	if (!params.handle) {
		throw std::runtime_error("[LMHead::backward] cuBLAS handle is NULL");
	}
	if (!params.grad_logits) {
		throw std::runtime_error("[LMHead::backward] grad_logits is NULL");
	}
	if (!params.grad_encoder) {
		throw std::runtime_error("[LMHead::backward] grad_encoder is NULL");
	}
	if (!params.weights) {
		throw std::runtime_error("[LMHead::backward] weights is NULL");
	}
	if (!isDevicePtr(params.grad_logits)) {
		throw std::runtime_error("[LMHead::backward] grad_logits must be device pointer");
	}
	if (!isDevicePtr(params.grad_encoder)) {
		throw std::runtime_error("[LMHead::backward] grad_encoder must be device pointer");
	}
	if (!isDevicePtr(params.weights)) {
		throw std::runtime_error("[LMHead::backward] weights must be device pointer");
	}
	if (params.grad_weight && !isDevicePtr(params.grad_weight)) {
		throw std::runtime_error("[LMHead::backward] grad_weight must be device pointer");
	}
	if (params.grad_bias && !isDevicePtr(params.grad_bias)) {
		throw std::runtime_error("[LMHead::backward] grad_bias must be device pointer");
	}
	if (params.encoder_output && !isDevicePtr(params.encoder_output)) {
		throw std::runtime_error("[LMHead::backward] encoder_output must be device pointer");
	}

	const int total_tokens = params.batch_size * params.seq_len;
	if (total_tokens == 0) {
		return;
	}

	const float alpha = 1.0f;
	const float beta = params.accumulate ? 1.0f : 0.0f;

	// grad_encoder = weights @ grad_logits
	cublasStatus_t status = cublasSgemm(
		params.handle,
		CUBLAS_OP_N,              // weights: [d_model, vocab]
		CUBLAS_OP_N,              // grad_logits: [vocab, tokens]
		params.d_model,           // m
		total_tokens,             // n
		params.vocab_size,        // k
		&alpha,
		params.weights,
		params.d_model,           // lda
		params.grad_logits,
		params.vocab_size,        // ldb
		&beta,
		params.grad_encoder,
		params.d_model);          // ldc

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[LMHead::backward] cublasSgemm grad_encoder failed status=" << status;
		throw std::runtime_error(oss.str());
	}

	// grad_weight = encoder_output @ grad_logits^T
	if (params.grad_weight && params.encoder_output) {
		status = cublasSgemm(
			params.handle,
			CUBLAS_OP_N,           // encoder_output: [d_model, tokens]
			CUBLAS_OP_T,           // grad_logits: [vocab, tokens]^T
			params.d_model,        // m
			params.vocab_size,     // n
			total_tokens,          // k
			&alpha,
			params.encoder_output,
			params.d_model,        // lda
			params.grad_logits,
			params.vocab_size,     // ldb
			&beta,
			params.grad_weight,
			params.d_model);       // ldc

		if (status != CUBLAS_STATUS_SUCCESS) {
			std::ostringstream oss;
			oss << "[LMHead::backward] cublasSgemm grad_weight failed status=" << status;
			throw std::runtime_error(oss.str());
		}
	}

	if (params.use_bias && params.grad_bias) {
		launchBiasSumGradient(
			params.grad_logits,
			params.grad_bias,
			total_tokens,
			params.vocab_size,
			params.stream);
	}
}

} // namespace GRIM
