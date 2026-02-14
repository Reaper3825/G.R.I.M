#include "lm_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/EquationLogging/EquationLogging.hpp"

#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <vector>
#include <cmath>

// External kernel declaration (defined in BackwardKernels.cu)
void launchBiasSumGradient(const float* grad_output,
                          float* grad_bias,
                          int total_tokens,
                          int hidden_dim,
                          cudaStream_t stream);

namespace GRIM {

// ============================================================================
// Token 277 (SPACE) Mode Collapse Diagnostic — host-side, via EQ_LOG
// ============================================================================
// NOTE: logToken277Inputs (fprintf), kernelToken277LogitDiagnostic (device ring buffer),
// and launchToken277LogitDiagnostic (device buffer accessor) were DELETED per Rule 26
// (zero callers / dead code). If kernel-side Token277 diagnostics are needed in the
// future, implement as a host-side cudaMemcpy + EQ_LOG pattern.

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

// ============================================================================
// Zero-Mean Centering Kernel (Issue #37 Fix)
// Centers each hidden state vector to have zero mean before LM head projection.
// This prevents the mode collapse where negative hidden_sum × negative grad = positive.
// 
// Math: centered[t,i] = hidden[t,i] - mean_t  where mean_t = (1/d_model) Σ_i hidden[t,i]
// After centering: Σ_i centered[t,i] = 0 for all positions t
// ============================================================================

__global__ void centerHiddenStatesKernel(
    const float* __restrict__ input,   // [total_tokens, d_model]
    float* __restrict__ output,        // [total_tokens, d_model]  
    int d_model,
    int total_tokens
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    const float* in_row = input + static_cast<size_t>(token_idx) * d_model;
    float* out_row = output + static_cast<size_t>(token_idx) * d_model;
    
    // Compute mean using all threads in block via reduction
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Each thread sums a subset of elements
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum += in_row[i];
    }
    
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // First thread in each warp adds to shared memory
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    const float mean = s_sum / static_cast<float>(d_model);
    
    // Subtract mean from each element
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        out_row[i] = in_row[i] - mean;
    }
}

inline void centerHiddenStates(
    const float* input,
    float* output,
    int d_model,
    int total_tokens,
    cudaStream_t stream
) {
    if (total_tokens == 0) return;
    
    // One block per token, 256 threads per block
    constexpr int kBlockSize = 256;
    centerHiddenStatesKernel<<<total_tokens, kBlockSize, 0, stream>>>(
        input, output, d_model, total_tokens);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[centerHiddenStates] kernel launch failed: ") + 
                                 cudaGetErrorString(err));
    }
}

} // namespace

// ============================================================================
// ISSUE #37/#43 FIX: Public API for centering hidden states
// Called by AutogradTraining.cu BEFORE autograd::matmul() for LM head
// ============================================================================
void launchCenterHiddenStates(
    const float* input,
    float* output,
    int d_model,
    int total_tokens,
    cudaStream_t stream
) {
    // RULE 20: Fail loud
    if (!input) {
        throw std::runtime_error("[launchCenterHiddenStates] input is NULL");
    }
    if (!output) {
        throw std::runtime_error("[launchCenterHiddenStates] output is NULL");
    }
    if (d_model <= 0) {
        throw std::runtime_error("[launchCenterHiddenStates] d_model must be > 0, got " + std::to_string(d_model));
    }
    if (total_tokens <= 0) {
        return;  // Nothing to do
    }
    
    centerHiddenStates(input, output, d_model, total_tokens, stream);
}

void launchLMHeadForward(const LMHeadForwardParams& params) {
	// RULE 20: Fail loud - validate all tensor views upfront
	params.validate("LMHead::forward");
	
	const int total_tokens = params.total_tokens();
	const int d_model = params.d_model();
	const int vocab_size = params.vocab_size();
	
	if (total_tokens == 0) {
		return;
	}

	// Extract raw pointers from TensorViews for CUDA kernels and cuBLAS
	const float* encoder_output_ptr = params.encoder_output.ptr;
	const float* weights_ptr = params.weights.ptr;
	float* logits_ptr = params.logits.ptr;
	float* centered_scratch_ptr = params.centered_scratch.is_valid() ? params.centered_scratch.ptr : nullptr;
	const float* bias_ptr = params.bias.is_valid() ? params.bias.ptr : nullptr;
	
	// Extract execution context
	cublasHandle_t handle = params.handle;
	cudaStream_t stream = params.stream;

	// CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
	if (stream) {
		cublasSetStream(handle, stream);
	}

	// === Issue #37 FIX: Zero-mean centering before projection ===
	// Centers each hidden state vector to have zero mean, preventing mode collapse.
	// Without centering: grad_W[tok] = Σ_t (hidden_sum[t] × grad[t,tok])
	// If hidden_sum has same sign as grad, the contribution is wrong!
	// With centering: hidden_sum = 0 for each position, so gradients depend
	// only on hidden state direction, not offset.
	const float* projection_input = encoder_output_ptr;
	
	if (params.use_centering && centered_scratch_ptr) {
		centerHiddenStates(
			encoder_output_ptr,
			centered_scratch_ptr,
			d_model,
			total_tokens,
			stream
		);
		projection_input = centered_scratch_ptr;
		
		// Diagnostic: Verify centering worked (first 5 calls only)
		static int s_centering_diag_count = 0;
		if (++s_centering_diag_count <= 5) {
			cudaStreamSynchronize(stream);
			
			// Sample position 0
			std::vector<float> orig(d_model), centered(d_model);
			cudaMemcpy(orig.data(), encoder_output_ptr, d_model * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(centered.data(), centered_scratch_ptr, d_model * sizeof(float), cudaMemcpyDeviceToHost);
			
			float orig_sum = 0, centered_sum = 0;
			for (int i = 0; i < d_model; ++i) {
				orig_sum += orig[i];
				centered_sum += centered[i];
			}
			float orig_mean = orig_sum / d_model;
			float centered_mean = centered_sum / d_model;
			
			std::ostringstream eq;
			eq << "[CENTERING_VERIFY] call=" << s_centering_diag_count 
			   << " tokens=" << total_tokens
			   << " pos0: orig_mean=" << orig_mean 
			   << " centered_mean=" << centered_mean << " (should be ~0)\n";
			EQ_LOG("CENTERING_VERIFY", eq.str(), 0, -1, 0, GRIM::EquationPhase::LM_HEAD_PROJECTION);
		}
	}

	const float alpha = 1.0f;
	const float beta = 0.0f;

	cublasStatus_t status = cublasSgemm(
		handle,
		CUBLAS_OP_T,               // weights: row-major [vocab, d_model] -> transpose for GEMM
		CUBLAS_OP_N,               // encoder: row-major [tokens, d_model]
		vocab_size,                // m
		total_tokens,              // n
		d_model,                   // k
		&alpha,
		weights_ptr,
		d_model,                   // lda
		projection_input,          // Use centered hidden states (Issue #37)
		d_model,                   // ldb
		&beta,
		logits_ptr,
		vocab_size);               // ldc

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[LMHead::forward] cublasSgemm FAILED status=" << status
		    << " m=" << vocab_size << " n=" << total_tokens
		    << " k=" << d_model
		    << " projection_input=" << projection_input
		    << " beta=" << beta;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && bias_ptr) {
		addBias(logits_ptr, bias_ptr, vocab_size, total_tokens, stream);
	}

	// Token 277 equation diagnostic removed (Rule 26: dead code behind unset #ifdef).
	// Host-side Token277 diagnostics are in Phase2_TrainingLoop.cu via
	// computeToken277Diagnostic() + formatToken277Diagnostic() + EQ_LOG.
}

// NOTE: launchLMHeadBackward() REMOVED (Issue #58 cleanup)
// Production training uses autograd::matmul() which has its own MatMulGradFn 
// that handles backward pass via TensorContract_GPU.cu operations.
//
// The centering fixes (Issue #37, #40) are no longer needed because the autograd
// system handles gradient computation differently - it uses the exact same cuBLAS
// operations but tracked through the autograd graph.
//
// If you need LM head backward in the future, use:
//   Tensor logits = autograd::matmul(encoder_output, lm_weights, stream, ...)
//   Tensor loss = autograd::unified_loss(logits, targets, ...LossConfig...)
//   loss.backward()  // Gradients flow automatically to lm_weights.grad

} // namespace GRIM
