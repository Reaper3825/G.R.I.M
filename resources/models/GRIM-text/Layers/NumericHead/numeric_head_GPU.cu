#define USE_CUDA

#include "numeric_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

#include <stdexcept>
#include <sstream>
#include <memory>

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

__global__ void sumGradBiasKernel(const float* grad_output, float* grad_bias, int total_tokens) {
	__shared__ float shared[256];
	const int tid = threadIdx.x;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Load and accumulate
	float sum = 0.0f;
	for (int i = idx; i < total_tokens; i += blockDim.x * gridDim.x) {
		sum += grad_output[i];
	}
	shared[tid] = sum;
	__syncthreads();
	
	// Warp reduction
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared[tid] += shared[tid + s];
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		atomicAdd(grad_bias, shared[0]);
	}
}

inline bool isDevicePtr(const void* ptr) {
	cudaPointerAttributes attr{};
	return ptr &&
	       cudaPointerGetAttributes(&attr, ptr) == cudaSuccess &&
	       attr.type == cudaMemoryTypeDevice;
}

// Helper to queue GPU memory for deferred cleanup (avoid destroying in destructor)
void queueForDeferredNumericHeadCleanup(float* ptr) {
	if (ptr) {
		cudaFree(ptr);  // Simple sync free - can be improved with stream-ordered allocator
	}
}

} // namespace

//======================================================//
//  NumericHeadGradFn - Autograd backward node
//======================================================//

struct NumericHeadGradFn : public GradFn {
	// Cached data for backward
	std::shared_ptr<float> owned_cache_encoder;   // Copy of encoder_output for grad_weight
	const float* cached_encoder = nullptr;
	
	// Weight tensor data (for grad_encoder computation)
	const float* weights_data = nullptr;
	TensorContract::TensorShape weights_shape;
	
	// Grad buffers
	std::shared_ptr<float> owned_grad_encoder;
	float* grad_encoder = nullptr;
	float* grad_weight = nullptr;   // Points to leaf tensor's grad buffer
	float* grad_bias = nullptr;     // Points to leaf tensor's grad buffer (optional)
	
	// Chain continuation
	std::shared_ptr<GradFn> encoder_grad_fn;
	TensorContract::TensorShape encoder_shape;
	
	// Dimensions
	int total_tokens = 0;
	int d_model = 0;
	bool use_bias = false;
	
	// Execution context
	cublasHandle_t cublas_handle = nullptr;
	cudaStream_t stream = nullptr;
	
	NumericHeadGradFn() { op_name = "numeric_head"; }
	
	~NumericHeadGradFn() {
		encoder_grad_fn.reset();
	}
	
	void capture_inputs(
		Tensor& encoder_output,
		Tensor& weights,
		Tensor* bias,
		cublasHandle_t handle,
		cudaStream_t str
	) {
		total_tokens = encoder_output.shape.as_2d().rows;
		d_model = encoder_output.shape.as_2d().cols;
		encoder_shape = encoder_output.shape;
		weights_shape = weights.shape;
		cublas_handle = handle;
		stream = str;
		use_bias = (bias != nullptr);
		
		// Cache encoder output (need for grad_weight = encoder^T @ grad_pred)
		const size_t encoder_numel = static_cast<size_t>(total_tokens) * d_model;
		float* cache_buf = nullptr;
		cudaMalloc(&cache_buf, encoder_numel * sizeof(float));
		cudaMemcpyAsync(cache_buf, encoder_output.data, encoder_numel * sizeof(float),
		                cudaMemcpyDeviceToDevice, stream);
		owned_cache_encoder = std::shared_ptr<float>(cache_buf, [](float* p) {
			queueForDeferredNumericHeadCleanup(p);
		});
		cached_encoder = owned_cache_encoder.get();
		
		// Store weights pointer (weights is leaf, persists)
		weights_data = weights.data;
		
		// Allocate grad buffer for encoder (non-leaf intermediate)
		if (encoder_output.requires_grad) {
			float* grad_enc_buf = nullptr;
			cudaMalloc(&grad_enc_buf, encoder_numel * sizeof(float));
			cudaMemsetAsync(grad_enc_buf, 0, encoder_numel * sizeof(float), stream);
			owned_grad_encoder = std::shared_ptr<float>(grad_enc_buf, [](float* p) {
				queueForDeferredNumericHeadCleanup(p);
			});
			grad_encoder = owned_grad_encoder.get();
		}
		
		// Grad buffers for leaf weights/bias (persistent, use directly)
		// ISSUE #59: Use grad_data() accessor
		weights.ensure_grad();
		grad_weight = weights.grad_data();
		
		if (bias) {
			bias->ensure_grad();
			grad_bias = bias->grad_data();
		}
		
		// Capture encoder's grad_fn for chain continuation
		encoder_grad_fn = encoder_output.grad_fn;
	}
	
	void apply(const Tensor& grad_output, cudaStream_t str) override {
		if (applied) return;
		applied = true;
		
		cudaStream_t use_stream = str ? str : stream;
		if (use_stream) {
			cublasSetStream(cublas_handle, use_stream);
		}
		
		const float* grad_pred = grad_output.data;
		const float alpha = 1.0f;
		const float beta = 0.0f;  // Overwrite (first micro-batch) - accumulation handled by caller
		
		// grad_weight = encoder^T @ grad_predictions
		// cublasSgemv: y = alpha * A * x + beta * y
		// We want: grad_weight[d_model] = encoder[total_tokens, d_model]^T @ grad_pred[total_tokens]
		cublasStatus_t w_status = cublasSgemv(
			cublas_handle,
			CUBLAS_OP_N,      // No transpose (row-major encoder becomes col-major)
			d_model,          // M
			total_tokens,     // N
			&alpha,
			cached_encoder,
			d_model,          // lda
			grad_pred,
			1,
			&beta,
			grad_weight,
			1);
		
		if (w_status != CUBLAS_STATUS_SUCCESS) {
			throw std::runtime_error("[NumericHeadGradFn] cublasSgemv grad_weight failed");
		}
		
		// grad_encoder = weights @ grad_predictions^T (outer product broadcast)
		// Each row i of grad_encoder gets: grad_encoder[i,:] = weights * grad_pred[i]
		// cublasSger: A = alpha * x * y^T + A
		if (grad_encoder) {
			cudaMemsetAsync(grad_encoder, 0, static_cast<size_t>(total_tokens) * d_model * sizeof(float), use_stream);
			
			cublasStatus_t enc_status = cublasSger(
				cublas_handle,
				d_model,          // M
				total_tokens,     // N
				&alpha,
				weights_data,     // x [d_model]
				1,
				grad_pred,        // y [total_tokens]
				1,
				grad_encoder,     // A [d_model, total_tokens] col-major = [total_tokens, d_model] row-major
				d_model);
			
			if (enc_status != CUBLAS_STATUS_SUCCESS) {
				throw std::runtime_error("[NumericHeadGradFn] cublasSger grad_encoder failed");
			}
		}
		
		// grad_bias = sum(grad_predictions)
		if (use_bias && grad_bias) {
			cudaMemsetAsync(grad_bias, 0, sizeof(float), use_stream);
			constexpr int kBlockSize = 256;
			const int grid = (total_tokens + kBlockSize - 1) / kBlockSize;
			sumGradBiasKernel<<<grid, kBlockSize, 0, use_stream>>>(grad_pred, grad_bias, total_tokens);
		}
		
		// Continue autograd chain through encoder
		if (encoder_grad_fn && grad_encoder) {
			Tensor grad_enc_tensor;
			grad_enc_tensor.data = grad_encoder;
			grad_enc_tensor.shape = encoder_shape;
			grad_enc_tensor.owns_data = false;
			grad_enc_tensor.stream = use_stream;
			
			encoder_grad_fn->apply(grad_enc_tensor, use_stream);
			encoder_grad_fn->release_saved();
		}
	}
	
	void release_saved() override {
		if (released_) return;
		released_ = true;
		owned_cache_encoder.reset();
		cached_encoder = nullptr;
	}
};

//======================================================//
//  Autograd NumericHead Forward
//======================================================//

Tensor numeric_head_forward(
    Tensor& encoder_output,
    Tensor& weights,
    Tensor* bias,
    cublasHandle_t handle,
    cudaStream_t stream
) {
	// Rule 20: Fail loud validation
	if (!encoder_output.data) {
		throw std::runtime_error("[numeric_head_forward] encoder_output.data is NULL");
	}
	if (!weights.data) {
		throw std::runtime_error("[numeric_head_forward] weights.data is NULL");
	}
	if (!handle) {
		throw std::runtime_error("[numeric_head_forward] cuBLAS handle is NULL");
	}
	
	const int total_tokens = encoder_output.shape.as_2d().rows;
	const int d_model = encoder_output.shape.as_2d().cols;
	
	// Allocate output tensor
	float* pred_data = nullptr;
	cudaMalloc(&pred_data, static_cast<size_t>(total_tokens) * sizeof(float));
	
	Tensor predictions;
	predictions.data = pred_data;
	predictions.shape = TensorContract::TensorShape::make_BSM(total_tokens, 1);
	predictions.owns_data = true;
	predictions.requires_grad = encoder_output.requires_grad || weights.requires_grad;
	predictions.is_leaf = false;
	predictions.stream = stream;
	
	// Forward: predictions = encoder_output @ weights
	if (stream) {
		cublasSetStream(handle, stream);
	}
	
	const float alpha = 1.0f;
	const float beta = 0.0f;
	
	// cublasSgemv: y = alpha * A^T * x + beta * y
	// encoder_output[total_tokens, d_model] @ weights[d_model, 1] -> predictions[total_tokens, 1]
	cublasStatus_t status = cublasSgemv(
		handle,
		CUBLAS_OP_T,       // Transpose A
		d_model,           // M (rows of A)
		total_tokens,      // N (cols of A in row-major = rows in col-major after transpose)
		&alpha,
		encoder_output.data,
		d_model,           // lda
		weights.data,
		1,
		&beta,
		predictions.data,
		1);
	
	if (status != CUBLAS_STATUS_SUCCESS) {
		cudaFree(pred_data);
		throw std::runtime_error("[numeric_head_forward] cublasSgemv failed");
	}
	
	// Add bias if present
	if (bias && bias->data) {
		constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
		const int grid = (total_tokens + kBlockSize - 1) / kBlockSize;
		addScalarBiasKernel<<<grid, kBlockSize, 0, stream>>>(predictions.data, bias->data, total_tokens);
	}
	
	// Setup autograd if needed
	if (predictions.requires_grad) {
		auto grad_fn = std::make_shared<NumericHeadGradFn>();
		grad_fn->capture_inputs(encoder_output, weights, bias, handle, stream);
		predictions.grad_fn = grad_fn;
	}
	
	return predictions;
}

} // namespace GRIM
