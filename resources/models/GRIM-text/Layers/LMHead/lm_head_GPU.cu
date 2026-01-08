#define USE_CUDA

#include "lm_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <cmath>
#include <stdexcept>
#include <sstream>
#include "../Shared/LogRecorder/LogRecorder.hpp"

// External kernel declaration (global scope - defined in BackwardKernels.cu)
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
		return;  // No-op for empty batch
	}

	{
		std::ostringstream oss;
		oss << "[LMHead fwd] m=" << params.vocab_size
		    << " n=" << total_tokens
		    << " k=" << params.d_model
		    << " lda=" << params.d_model      // weights leading dim (col-major d_model x vocab)
		    << " ldb=" << params.d_model      // encoder leading dim (col-major d_model x tokens)
		    << " ldc=" << params.vocab_size;  // logits leading dim (col-major vocab x tokens)
		GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::ForwardPass, oss.str());
	}


	// REMOVED cublasSetStream - handle already bound to stream in InitTrainingState.cu

	// FIXED: LM head should NOT scale by 1/sqrt(d_model) - that's for attention!
	// Use alpha=1.0 for standard logits output
	const float alpha = 1.0f;  // Was: 1.0f / std::sqrt(static_cast<float>(params.d_model))
	const float beta_zero = 0.0f;

	cublasStatus_t status = cublasSgemm(
		params.handle,
		CUBLAS_OP_T,               // weights: row-major [vocab, d_model] -> col-major [d_model, vocab], transpose to [vocab, d_model]
		CUBLAS_OP_N,               // encoder: row-major [tokens, d_model] -> col-major [d_model, tokens]
		params.vocab_size,         // m
		total_tokens,              // n
		params.d_model,            // k
		&alpha,
		params.weights,
		params.d_model,            // lda
		params.encoder_output,
		params.d_model,            // ldb
		&beta_zero,
		params.logits,
		params.vocab_size);        // ldc


	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[LMHead::forward] cublasSgemm failed status=" << status
		    << " m=" << params.vocab_size << " n=" << total_tokens
		    << " k=" << params.d_model
		    << " lda=" << params.d_model
		    << " ldb=" << params.d_model
		    << " ldc=" << params.vocab_size;
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
		return;  // No-op for empty batch
	}

	{
		std::ostringstream oss;
		oss << "[LMHead bwd grad_encoder] m=" << params.d_model
		    << " n=" << total_tokens
		    << " k=" << params.vocab_size
		    << " lda=" << params.d_model
		    << " ldb=" << params.vocab_size
		    << " ldc=" << params.d_model
		    << " accumulate=" << params.accumulate;
		GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::BackwardPass, oss.str());
	}

	// REMOVED cublasSetStream - handle already bound to stream in InitTrainingState.cu

	// FIXED: LM head backward should also use alpha=1.0 (not scaled)
	const float alpha = 1.0f;  // Was: 1.0f / std::sqrt(static_cast<float>(params.d_model))
	const float beta_zero = 0.0f;
	const float beta = params.accumulate ? 1.0f : 0.0f;

	cublasStatus_t status = cublasSgemm(
		params.handle,
		CUBLAS_OP_N,              // weights: row-major [vocab, d_model] -> col-major [d_model, vocab]
		CUBLAS_OP_N,              // grad_logits: row-major [tokens, vocab] -> col-major [vocab, tokens]
		params.d_model,           // M = d_model
		total_tokens,             // N = tokens
		params.vocab_size,        // K = vocab_size
		&alpha,
		params.weights,           // A = weights
		params.d_model,           // lda = leading dim of weights
		params.grad_logits,       // B = grad_logits
		params.vocab_size,        // ldb = leading dim of grad_logits
		&beta_zero,
		params.grad_encoder,      // C = grad_encoder
		params.d_model);          // ldc = leading dim of grad_encoder



	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[LMHead::backward] cublasSgemm grad_encoder failed status=" << status
		    << " m=" << params.d_model << " n=" << total_tokens
		    << " k=" << params.vocab_size
		    << " lda=" << params.d_model
		    << " ldb=" << params.vocab_size
		    << " ldc=" << params.d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.grad_weight && params.encoder_output) {
		std::ostringstream oss;
		oss << "[LMHead bwd grad_weight] m=" << params.d_model
		    << " n=" << params.vocab_size
		    << " k=" << total_tokens
		    << " lda=" << params.d_model
		    << " ldb=" << params.vocab_size
		    << " ldc=" << params.d_model
		    << " accumulate=" << params.accumulate;
		GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::BackwardPass, oss.str());

		status = cublasSgemm(
			params.handle,
			CUBLAS_OP_N,           // encoder_output: col-major [d_model, tokens]
			CUBLAS_OP_T,           // grad_logits: col-major [vocab, tokens] -> [tokens, vocab]
			params.d_model,        // m
			params.vocab_size,     // n
			total_tokens,          // k
			&alpha,
			params.encoder_output, // A
			params.d_model,        // lda
			params.grad_logits,    // B
			params.vocab_size,     // ldb
			&beta,
			params.grad_weight,    // C
			params.d_model);       // ldc

		if (status != CUBLAS_STATUS_SUCCESS) {
			std::ostringstream oss;
			oss << "[LMHead::backward] cublasSgemm grad_weight failed status=" << status
			    << " m=" << params.d_model << " n=" << params.vocab_size
			    << " k=" << total_tokens
			    << " lda=" << params.d_model
			    << " ldb=" << params.vocab_size
			    << " ldc=" << params.d_model;
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
