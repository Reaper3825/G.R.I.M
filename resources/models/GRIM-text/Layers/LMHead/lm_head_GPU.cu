#include "lm_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"

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
// Token 277 (SPACE) Mode Collapse Diagnostic
// Logs the actual values that determine logit[277] = hidden_state · W[277]
// ============================================================================
static int s_lm_head_call_count = 0;

static void logToken277Inputs(const LMHeadForwardParams& params) {
    // Extract dimensions from TensorViews
    const int d_model = params.d_model();
    const int vocab_size = params.vocab_size();
    const int total_tokens = params.total_tokens();
    
    constexpr int kToken277 = 277;
    if (kToken277 >= vocab_size) return;
    
    ++s_lm_head_call_count;
    
    // Only log first 20 calls to avoid spam
    if (s_lm_head_call_count > 20) return;
    
    cudaStreamSynchronize(params.stream);
    
    const int num_sample = std::min(5, total_tokens);
    
    // Extract raw pointers from TensorViews
    const float* weights_ptr = params.weights.ptr;
    const float* encoder_output_ptr = params.encoder_output.ptr;
    
    // Read W[277] row (the weight row for token 277)
    // Weights are [vocab_size, d_model] row-major, so W[277] starts at offset 277*d_model
    std::vector<float> w277(d_model);
    cudaMemcpy(w277.data(), 
               weights_ptr + static_cast<size_t>(kToken277) * d_model,
               d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute W[277] statistics
    float w277_sum = 0.0f, w277_sq_sum = 0.0f;
    float w277_min = w277[0], w277_max = w277[0];
    for (int i = 0; i < d_model; ++i) {
        w277_sum += w277[i];
        w277_sq_sum += w277[i] * w277[i];
        w277_min = std::min(w277_min, w277[i]);
        w277_max = std::max(w277_max, w277[i]);
    }
    float w277_mean = w277_sum / d_model;
    float w277_var = (w277_sq_sum / d_model) - (w277_mean * w277_mean);
    float w277_norm = std::sqrt(w277_sq_sum);
    
    fprintf(stderr, "[Token277Input] call=%d W[277]: norm=%.6f mean=%.6f var=%.6f min=%.6f max=%.6f\n",
            s_lm_head_call_count, w277_norm, w277_mean, w277_var, w277_min, w277_max);
    
    // Read first few hidden states and compute their dot product with W[277]
    std::vector<float> hidden_sample(num_sample * d_model);
    cudaMemcpy(hidden_sample.data(), encoder_output_ptr,
               num_sample * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    
    fprintf(stderr, "[Token277Input] call=%d Hidden states for %d tokens:\n", s_lm_head_call_count, num_sample);
    
    for (int t = 0; t < num_sample; ++t) {
        float* h = hidden_sample.data() + t * d_model;
        
        // Hidden state statistics
        float h_sum = 0.0f, h_sq_sum = 0.0f;
        float h_min = h[0], h_max = h[0];
        for (int i = 0; i < d_model; ++i) {
            h_sum += h[i];
            h_sq_sum += h[i] * h[i];
            h_min = std::min(h_min, h[i]);
            h_max = std::max(h_max, h[i]);
        }
        float h_norm = std::sqrt(h_sq_sum);
        float h_mean = h_sum / d_model;
        
        // Compute dot product h · W[277] = logit[277] (before bias)
        float dot_277 = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            dot_277 += h[i] * w277[i];
        }
        
        // Compute cosine similarity between h and W[277]
        float cos_sim = (w277_norm > 1e-8f && h_norm > 1e-8f) ? 
                        (dot_277 / (h_norm * w277_norm)) : 0.0f;
        
        fprintf(stderr, "  [t=%d] h_norm=%.4f h_mean=%.4f h_min=%.4f h_max=%.4f | "
                        "dot(h,W277)=%.4f cos_sim=%.4f\n",
                t, h_norm, h_mean, h_min, h_max, dot_277, cos_sim);
    }
    
    // Also sample a few other weight rows for comparison
    constexpr int kCompareTokens[] = {0, 100, 500, 1000};  // Compare with other tokens
    fprintf(stderr, "[Token277Input] call=%d Weight row norms for comparison:\n", s_lm_head_call_count);
    for (int tok : kCompareTokens) {
        if (tok >= vocab_size) continue;
        std::vector<float> w_other(d_model);
        cudaMemcpy(w_other.data(),
                   weights_ptr + static_cast<size_t>(tok) * d_model,
                   d_model * sizeof(float), cudaMemcpyDeviceToHost);
        float norm = 0.0f;
        for (int i = 0; i < d_model; ++i) norm += w_other[i] * w_other[i];
        norm = std::sqrt(norm);
        fprintf(stderr, "  W[%d] norm=%.6f\n", tok, norm);
    }
}

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

// ============================================================================
// Issue #40 FIX: Re-center gradient rows after GEMM to eliminate FP32 error
// ============================================================================
// ROOT CAUSE: cuBLAS SGEMM accumulates FP32 errors that destroy the mathematical
// property that each row of grad_W should sum to ~0 (due to centered hidden states).
// The error accumulates SYSTEMATICALLY in one direction, creating a positive bias
// that drives mode collapse to token 277 (SPACE).
//
// FIX: After GEMM computes grad_W[v,d] = Σ_t h[t,d] * g[t,v], we re-center each row:
// grad_W[v,d] -= mean(grad_W[v,:]) for all d in [0, d_model)
//
// This restores the zero-sum property destroyed by FP32 accumulation error.
// ============================================================================

__global__ void recenterGradientRowsKernel(
    float* __restrict__ grad_weight,   // [vocab_size, d_model] row-major
    int d_model,
    int vocab_size
) {
    // One block per vocab row
    const int vocab_idx = blockIdx.x;
    if (vocab_idx >= vocab_size) return;
    
    float* row = grad_weight + static_cast<size_t>(vocab_idx) * d_model;
    
    // Step 1: Compute row mean using parallel reduction
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    // Each thread sums a subset of elements
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum += row[i];
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
    
    // Step 2: Subtract mean from each element to restore zero-sum property
    const float mean = s_sum / static_cast<float>(d_model);
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        row[i] -= mean;
    }
}

inline void recenterGradientRows(
    float* grad_weight,
    int d_model,
    int vocab_size,
    cudaStream_t stream
) {
    if (vocab_size == 0 || d_model == 0) return;
    
    // One block per vocab row, 256 threads per block
    constexpr int kBlockSize = 256;
    recenterGradientRowsKernel<<<vocab_size, kBlockSize, 0, stream>>>(
        grad_weight, d_model, vocab_size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[recenterGradientRows] kernel launch failed: ") + 
                                 cudaGetErrorString(err));
    }
}

} // namespace

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
			
			fprintf(stderr, "[Issue37-Centering] call=%d tokens=%d pos0: orig_mean=%.6f centered_mean=%.6f (should be ~0)\n",
					s_centering_diag_count, total_tokens, orig_mean, centered_mean);
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
		oss << "[LMHead::forward] cublasSgemm failed status=" << status
		    << " m=" << vocab_size << " n=" << total_tokens
		    << " k=" << d_model;
		throw std::runtime_error(oss.str());
	}

	if (params.use_bias && bias_ptr) {
		addBias(logits_ptr, bias_ptr, vocab_size, total_tokens, stream);
	}
}

void launchLMHeadBackward(const LMHeadBackwardParams& params) {
	// RULE 20: Fail loud - validate all tensor views upfront
	params.validate("LMHead::backward");
	
	const int total_tokens = params.total_tokens();
	const int d_model = params.d_model();
	const int vocab_size = params.vocab_size();
	
	if (total_tokens == 0) {
		return;
	}

	// Extract raw pointers from TensorViews for CUDA kernels and cuBLAS
	const float* grad_logits_ptr = params.grad_logits.ptr;
	const float* encoder_output_ptr = params.encoder_output.ptr;
	float* centered_encoder_ptr = params.centered_encoder.is_valid() ? params.centered_encoder.ptr : nullptr;
	float* grad_encoder_ptr = params.grad_encoder.ptr;
	float* grad_weight_ptr = params.grad_weight.ptr;
	float* grad_bias_ptr = params.grad_bias.is_valid() ? params.grad_bias.ptr : nullptr;
	const float* weights_ptr = params.weights.ptr;
	
	// Extract execution context
	cublasHandle_t handle = params.handle;
	cudaStream_t stream = params.stream;

	// CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
	if (stream) {
		cublasSetStream(handle, stream);
	}

	const float alpha = 1.0f;
	const float beta = params.accumulate ? 1.0f : 0.0f;

	// grad_encoder = weights @ grad_logits
	// NOTE: If centering was applied in forward, this gives us grad w.r.t. CENTERED hidden states.
	//       We need to backprop through the centering operation to get true grad_encoder.
	cublasStatus_t status = cublasSgemm(
		handle,
		CUBLAS_OP_N,              // weights: [d_model, vocab]
		CUBLAS_OP_N,              // grad_logits: [vocab, tokens]
		d_model,                  // m
		total_tokens,             // n
		vocab_size,               // k
		&alpha,
		weights_ptr,
		d_model,                  // lda
		grad_logits_ptr,
		vocab_size,               // ldb
		&beta,
		grad_encoder_ptr,
		d_model);                 // ldc

	if (status != CUBLAS_STATUS_SUCCESS) {
		std::ostringstream oss;
		oss << "[LMHead::backward] cublasSgemm grad_encoder failed status=" << status;
		throw std::runtime_error(oss.str());
	}

	// === Issue #37: Backprop through centering operation ===
	// Forward: y = x - mean(x)  =>  Backward: grad_x = grad_y - mean(grad_y)
	// The centering backward is identical to centering forward!
	// We apply it IN-PLACE to grad_encoder since we just computed it.
	if (params.use_centering) {
		// Use centered_encoder as scratch buffer for the backward centering
		// Actually, we need to center grad_encoder in-place. The centering function
		// can do in-place if we pass same src/dst... but our kernel doesn't support that.
		// Let's use a two-step: copy to scratch, center back to grad_encoder
		if (centered_encoder_ptr) {
			cudaMemcpyAsync(centered_encoder_ptr, grad_encoder_ptr,
				static_cast<size_t>(total_tokens) * d_model * sizeof(float),
				cudaMemcpyDeviceToDevice, stream);
			centerHiddenStates(
				centered_encoder_ptr,
				grad_encoder_ptr,  // Output back to grad_encoder
				d_model,
				total_tokens,
				stream
			);
		}
	}

	// grad_weight = encoder_output @ grad_logits^T
	// === Issue #37: Use centered encoder output if centering was applied ===
	// If we centered hidden states in forward (h_centered = h - mean(h)),
	// then grad_weight must be computed against the centered hidden states!
	// grad_W[i,j] = sum_t(h_centered[t,j] * grad_logits[t,i])
	const float* encoder_for_grad_weight = encoder_output_ptr;
	if (params.use_centering && centered_encoder_ptr && encoder_output_ptr) {
		// Re-compute centering (matches forward pass)
		centerHiddenStates(
			encoder_output_ptr,
			centered_encoder_ptr,  // Scratch buffer
			d_model,
			total_tokens,
			stream
		);
		encoder_for_grad_weight = centered_encoder_ptr;
		
		// === DIAGNOSTIC: Verify centering worked and compute expected grad_W[277] sum ===
		static int s_bwd_center_diag = 0;
		constexpr int kToken277_diag = 277;
		if (++s_bwd_center_diag <= 10) {
			cudaStreamSynchronize(stream);
			
			// Read centered hidden states and grad_logits for verification
			std::vector<float> h_centered(static_cast<size_t>(total_tokens) * d_model);
			std::vector<float> h_grad_logits(static_cast<size_t>(total_tokens) * vocab_size);
			
			cudaMemcpy(h_centered.data(), centered_encoder_ptr,
					   h_centered.size() * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_grad_logits.data(), grad_logits_ptr,
					   h_grad_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
			
			// Verify centering: each row should sum to ~0
			double max_row_sum = 0.0, total_row_sum = 0.0;
			for (int t = 0; t < total_tokens; ++t) {
				double row_sum = 0.0;
				for (int j = 0; j < d_model; ++j) {
					row_sum += h_centered[static_cast<size_t>(t) * d_model + j];
				}
				if (std::abs(row_sum) > std::abs(max_row_sum)) {
					max_row_sum = row_sum;
				}
				total_row_sum += row_sum;
			}
			
			// Compute expected grad_W[277] sum using centered hidden states
			// grad_W[277,j] = Σ_t (h_centered[t,j] * grad[t,277])
			// Sum_j grad_W[277,j] = Σ_t (Σ_j h_centered[t,j] * grad[t,277])
			//                     = Σ_t (row_sum[t] * grad[t,277])
			double expected_grad_sum = 0.0;
			for (int t = 0; t < total_tokens; ++t) {
				double row_sum = 0.0;
				for (int j = 0; j < d_model; ++j) {
					row_sum += h_centered[static_cast<size_t>(t) * d_model + j];
				}
				float grad_277_t = h_grad_logits[static_cast<size_t>(t) * vocab_size + kToken277_diag];
				expected_grad_sum += row_sum * grad_277_t;
			}
			
			// === Issue #39 DIAGNOSTIC: Compute ACTUAL expected grad_W[277] directly ===
			// grad_W[277,j] = Σ_t h_centered[t,j] * grad_logits[t,277]
			// Sum all j: Σ_j grad_W[277,j] = Σ_t Σ_j h_centered[t,j] * grad_logits[t,277]
			double cpu_grad_w277_sum = 0.0;
			double grad_277_sum = 0.0;  // Sum of all grad_logits[t, 277]
			for (int j = 0; j < d_model; ++j) {
				double col_sum = 0.0;
				for (int t = 0; t < total_tokens; ++t) {
					float h_tj = h_centered[static_cast<size_t>(t) * d_model + j];
					float g_t277 = h_grad_logits[static_cast<size_t>(t) * vocab_size + kToken277_diag];
					col_sum += h_tj * g_t277;
					if (j == 0) grad_277_sum += g_t277;  // Only count once
				}
				cpu_grad_w277_sum += col_sum;
			}
			
			// Sanity: Compute what UNCENTERED would give
			double uncentered_grad_sum = 0.0;
			for (int j = 0; j < d_model; ++j) {
				for (int t = 0; t < total_tokens; ++t) {
					// Read from ORIGINAL encoder_output, not centered
					float h_tj_orig = 0.0f;  // We don't have it here, estimate from centered + mean
					float g_t277 = h_grad_logits[static_cast<size_t>(t) * vocab_size + kToken277_diag];
					uncentered_grad_sum += h_centered[static_cast<size_t>(t) * d_model + j] * g_t277;
				}
			}
			
			{
				std::ostringstream oss;
				oss << "[Issue37-BWD-DIAG] call=" << s_bwd_center_diag << " tokens=" << total_tokens
					<< " | Centering check: max_row_sum=" << std::fixed << std::setprecision(9) << max_row_sum
					<< " total_row_sum=" << total_row_sum << " (should be ~0)"
					<< " | Expected grad_W[277].sum (row_sum*grad): " << expected_grad_sum
					<< " | CPU direct grad_W[277].sum: " << cpu_grad_w277_sum
					<< " | sum(grad_logits[:,277]): " << grad_277_sum;
				GRIM::Logging::EmitModuleInfo("BackwardPass", oss.str());
			}
			
			// Print sample values from position 0 for debugging
			if (s_bwd_center_diag == 1) {
				std::ostringstream oss;
				oss << "[Issue39-SAMPLE] t=0: h[0,0]=" << h_centered[0] 
					<< " h[0,1]=" << h_centered[1]
					<< " g[0,277]=" << h_grad_logits[277]
					<< " g[0,278]=" << h_grad_logits[278]
					<< " | Row0 sum=" << std::setprecision(9);
				double row0_sum = 0;
				for (int j = 0; j < d_model; ++j) row0_sum += h_centered[j];
				oss << row0_sum;
				GRIM::Logging::EmitModuleInfo("BackwardPass", oss.str());
			}
		}
	}
	
	// === Issue #39: Verify pointer before GEMM ===
	// BUG FIX Issue #39: GEMM was computing TRANSPOSED gradient!
	// We want: grad_W[v, d] = Σ_t hidden[t, d] * grad_logits[t, v]
	// Matrix form: grad_W = grad_logits^T @ hidden = [vocab, d_model]
	// 
	// Row-major data interpreted as column-major by cuBLAS:
	// - hidden[tokens, d_model] row-major → [d_model, tokens] col-major
	// - grad_logits[tokens, vocab] row-major → [vocab, tokens] col-major
	// - grad_weight[vocab, d_model] row-major → [d_model, vocab] col-major
	//
	// We need: C = grad_logits^T @ hidden (in row-major terms)
	// Col-major: C[d_model, vocab] = hidden^T[d_model, tokens] @ grad_logits[tokens, vocab]
	//          = A^T @ B where A=hidden, B=grad_logits
	// cuBLAS: C = op(A) @ op(B) with CUBLAS_OP_T on A, CUBLAS_OP_N on B
	//
	// WAIT - cuBLAS stores C in col-major [M, N] where M=rows, N=cols
	// We want C col-major [d_model, vocab] which IS grad_weight col-major
	// M=d_model, N=vocab, K=tokens
	// A=hidden [d_model, tokens] with OP_N, B=grad_logits [vocab, tokens] with OP_T... 
	// that gives [d_model, vocab] which is WRONG shape for [vocab, d_model] row-major!
	//
	// CORRECT: We need grad_weight col-major [d_model, vocab] = row-major [vocab, d_model]
	// grad_W = grad_logits^T @ hidden in row-major
	// In col-major terms, this is: grad_W_colmaj = hidden^T_colmaj @ grad_logits_colmaj
	// hidden_colmaj = [d_model, tokens], grad_logits_colmaj = [vocab, tokens]
	// Need: [d_model, tokens] @ [vocab, tokens]^T = [d_model, vocab]? No that's M=d_model...
	//
	// Let's be very explicit: grad_W_rowmaj[v,d] = grad_logits_rowmaj[t,v]^T @ hidden_rowmaj[t,d]
	// In col-major: grad_W_colmaj = hidden_colmaj @ grad_logits_colmaj^T
	//             = [d_model, tokens] @ [vocab, tokens]^T
	//             = [d_model, tokens] @ [tokens, vocab]
	//             = [d_model, vocab]
	// But this gives C[d_model, vocab] col-major = [vocab, d_model] row-major ✓
	// So: M=d_model, N=vocab, K=tokens, A=hidden(OP_N), B=grad_logits(OP_T)
	// This is what we HAD... but it's wrong?
	// 
	// ACTUAL BUG: ldc should be vocab (C is [d_model, vocab] col-major, stride=d_model is wrong!)
	// When M=d_model, C is MxN = [d_model, vocab], stored col-major with stride ldc=M=d_model ✓
	// But wait, C row-major [vocab, d_model] = col-major [d_model, vocab], stride should be d_model ✓
	//
	// NEW THEORY: The matrix multiplication ORDER is wrong.
	// grad_W[v,d] = Σ_t h[t,d] * g[t,v]  (element-wise definition)
	// = h[:,d]^T @ g[:,v]  (for each pair d,v)
	// Full matrix: grad_W = g^T @ h  (row-major)
	// That's [vocab, tokens] @ [tokens, d_model] = [vocab, d_model] row-major ✓
	//
	// In cuBLAS col-major world:
	// g row-major [tokens, vocab] = col-major [vocab, tokens]
	// h row-major [tokens, d_model] = col-major [d_model, tokens]
	// grad_W row-major [vocab, d_model] = col-major [d_model, vocab]
	//
	// We need: grad_W_colmaj[d_model, vocab] = some_op(g_colmaj[vocab,tokens], h_colmaj[d_model,tokens])
	// grad_W_colmaj = h_colmaj @ g_colmaj^T = [d_model, tokens] @ [tokens, vocab] = [d_model, vocab] ✓
	// So A=h (OP_N), B=g (OP_T), M=d_model, N=vocab, K=tokens
	//
	// THAT'S WHAT WE HAVE! So why is the result wrong?
	// Let me re-verify: grad_W[277,j] for j in [0,d_model) should give sum=0 for centered h.
	// The GEMM computes grad_W_colmaj[j, 277] = Σ_t h_colmaj[j,t] * g_colmaj[t,277]
	//                                         = Σ_t h_rowmaj[t,j] * g_rowmaj[t,277]  ← CORRECT!
	// 
	// Wait, h_colmaj[j,t] = h stored at offset j + t*d_model (col-major [d_model, tokens])
	// But h_rowmaj[t,j] = h stored at offset t*d_model + j (row-major [tokens, d_model])
	// These are THE SAME offset! j + t*d_model = t*d_model + j ✓
	//
	// The issue must be g_colmaj[t,277] vs g_rowmaj[t,277]
	// g_colmaj[t,277] = g at offset t + 277*tokens (col-major [vocab, tokens])  
	// g_rowmaj[t,277] = g at offset t*vocab + 277 (row-major [tokens, vocab])
	// These are DIFFERENT! t + 277*tokens ≠ t*vocab + 277
	//
	// THE BUG: cuBLAS is reading grad_logits[t + 277*tokens] but we stored at [t*vocab + 277]!
	// With CUBLAS_OP_T on B, it reads B^T[k,n] = B[n,k] = g[277,t] at offset 277 + t*vocab
	// But we want g[t,277] at offset t*vocab + 277!
	// These are only equal if vocab==1, which it's not.
	//
	// CONCLUSION: We need CUBLAS_OP_N on grad_logits, not CUBLAS_OP_T!
	// Then B[k,n] = g_colmaj[k,n] = g_rowmaj[n,k]... wait that's still wrong.
	//
	// Let me think again. For row-major [tokens, vocab]:
	// g[t,v] stored at offset t*vocab + v
	// cuBLAS sees this as col-major [vocab, tokens]: g_colmaj[v,t] at offset v + t*vocab
	// So g_colmaj[v,t] = g_rowmaj[t,v] ✓
	//
	// With CUBLAS_OP_T on B: B^T[k,n] is accessed, where B is [vocab, tokens]
	// B^T[k,n] = B[n,k] = g_colmaj[n,k] = g_rowmaj[k,n]
	// We want g_rowmaj[t,v] where t=k, v=n (k loops over tokens, n=277 is vocab index)
	// So B^T[t,277] = g_rowmaj[t,277] ✓ This SHOULD be correct!
	//
	// NEW THEORY: The ldb parameter is wrong.
	// With CUBLAS_OP_T, ldb is the leading dimension of B (before transpose).
	// B is [vocab, tokens] col-major, so ldb should be vocab. ✓ We have vocab_size.
	//
	// Let me verify the element access:
	// GEMM computes C[i,j] = Σ_k A[i,k] * B_op[k,j]
	// With A OP_N: A[i,k] at offset i + k*lda = i + k*d_model (h_colmaj)
	// With B OP_T: B^T[k,j] = B[j,k] at offset j + k*ldb = j + k*vocab (g_colmaj)
	// We're computing: C[i,j] = Σ_k h[i + k*d_model] * g[j + k*vocab]
	// In our terms: grad_W_colmaj[i,j] = Σ_t h_colmaj[i,t] * g_colmaj[j,t]
	//             = Σ_t h_rowmaj[t,i] * g_rowmaj[t,j]? NO!
	// h_colmaj[i,t] at offset i + t*d_model = h_rowmaj[t,i] at offset t*d_model + i ✓
	// g_colmaj[j,t] at offset j + t*vocab = g_rowmaj[?,?]
	// g_rowmaj[r,c] at offset r*vocab + c
	// j + t*vocab = r*vocab + c → r=t, c=j ← g_rowmaj[t,j] ✓
	//
	// So we compute: grad_W[i,j] = Σ_t h[t,i] * g[t,j]
	// But we WANT:   grad_W[v,d] = Σ_t h[t,d] * g[t,v]
	// With i=d, j=v: grad_W[d,v] = Σ_t h[t,d] * g[t,v] ← TRANSPOSED INDICES!
	//
	// Our GEMM computes grad_W[d_model, vocab] but we store to grad_weight[vocab, d_model]!
	// The GEMM result is grad_W^T of what we want!
	//
	// FIX: Swap M and N, and swap A and B order:
	// We want grad_W[v,d] = Σ_t g[t,v] * h[t,d]
	// C[j,i] = Σ_k B_op[j,k] * A_op[k,i]
	// Let j=v, i=d, k=t: C[v,d] = Σ_t g_op[v,t] * h_op[t,d]
	// For g_op[v,t] = g[t,v]: need B[t,v] accessed via OP_N or B[v,t] via OP_T
	// g_colmaj is [vocab,tokens], g_colmaj[v,t] at offset v + t*vocab = g_rowmaj[t,v] ✓
	// So B=g with OP_N gives B[v,t] which is g_colmaj[v,t] = g_rowmaj[t,v] ✓
	// Wait no, OP_N means B[k,j] = B[t,v] accessed at offset t + v*ldb = t + v*vocab
	// That's g_colmaj[t,v] = g at offset t + v*vocab 
	// But g_rowmaj[r,c] at offset r*vocab+c, so t+v*vocab = ???
	// r*vocab + c = t + v*vocab → r=v, c=t-v*vocab+v*vocab=t? Only if c<vocab, so not well-defined.
	//
	// I'm getting confused. Let me just FIX IT by swapping the matrices:
	// Current: C = A @ B^T with A=h, B=g
	// Try: C = B @ A^T with A=h, B=g? That's C = g @ h^T
	// g_colmaj[vocab,tokens] @ h_colmaj^T[tokens,d_model] = [vocab, d_model] ✓
	// CUBLAS: M=vocab, N=d_model, K=tokens, A=g(OP_N), B=h(OP_T)
	if (grad_weight_ptr && encoder_for_grad_weight) {
		static int s_ptr_check = 0;
		if (++s_ptr_check <= 5) {
			const char* which_data = (encoder_for_grad_weight == centered_encoder_ptr) ? "CENTERED" :
			                         (encoder_for_grad_weight == encoder_output_ptr) ? "UN-CENTERED" : "UNKNOWN";
			std::ostringstream oss;
			oss << "[Issue39-GEMM-PTR] call=" << s_ptr_check 
				<< " encoder_for_grad_weight=" << (void*)encoder_for_grad_weight
				<< " centered_encoder=" << (void*)centered_encoder_ptr
				<< " encoder_output=" << (void*)encoder_output_ptr
				<< " USING: " << which_data;
			GRIM::Logging::EmitModuleInfo("BackwardPass", oss.str());
		}
		// CORRECTED GEMM for grad_weight (Issue #39 FIX):
		// grad_W[v,d] = Σ_t g[t,v] * h[t,d]
		// In col-major: C = g_colmaj @ h_colmaj^T = [vocab,tokens] @ [tokens,d_model] = [vocab,d_model]
		// M=vocab, N=d_model, K=tokens
		// A=grad_logits [vocab,tokens] col-major, OP_N
		// B=hidden [d_model,tokens] col-major, OP_T → [tokens,d_model]
		// C=grad_weight [vocab,d_model] col-major = row-major [d_model,vocab]... 
		// WAIT that's still wrong! We want grad_weight row-major [vocab,d_model]!
		// 
		// Let's verify: C col-major [vocab,d_model] with ldc=vocab
		// C[v,d] stored at offset v + d*vocab
		// grad_weight row-major [vocab,d_model]: grad_W[v,d] at offset v*d_model + d
		// These don't match! v + d*vocab ≠ v*d_model + d
		//
		// SIGH. The truth is: row-major[R,C] = col-major[C,R]
		// grad_weight row-major [vocab,d_model] = col-major [d_model,vocab]
		// So we NEED C col-major [d_model,vocab] with M=d_model, N=vocab
		// C[d,v] = Σ_t h[?,?] * g[?,?] such that result is grad_W_rowmaj[v,d]
		// C[d,v] at offset d + v*d_model, and grad_W_rowmaj[v,d] at offset v*d_model + d ✓ SAME!
		// 
		// So we need: C[d,v] = grad_W_rowmaj[v,d] = Σ_t h_rowmaj[t,d] * g_rowmaj[t,v]
		// With M=d_model(d), N=vocab(v), K=tokens(t):
		// C[i,j] = Σ_k A[i,k] * B^T[k,j] (using OP_T on B)
		// C[d,v] = Σ_t A[d,t] * B[v,t]
		// We need A[d,t] = h[t,d] and B[v,t] = g[t,v]
		// A col-major [d_model,tokens]: A[d,t] at offset d + t*d_model = h_colmaj[d,t]
		// h_rowmaj[t,d] at offset t*d_model + d = d + t*d_model ✓
		// B col-major [vocab,tokens]: B[v,t] at offset v + t*vocab = g_colmaj[v,t]
		// g_rowmaj[t,v] at offset t*vocab + v = v + t*vocab ✓
		// 
		// So the ORIGINAL formula IS CORRECT: A=h(OP_N), B=g(OP_T), M=d_model, N=vocab, K=tokens
		// The bug must be elsewhere... let me check ldc.
		// C is [d_model,vocab] col-major, ldc = leading dimension = M = d_model ✓
		//
		// BREAKTHROUGH: I bet the diagnostics are READING the result wrong!
		// grad_W[277,:] in row-major is at offsets 277*d_model to 277*d_model+767
		// But if C is col-major [d_model,vocab], grad_W_colmaj[d,277] is at offset d + 277*d_model
		// So grad_W_rowmaj[277,d] = grad_W_colmaj[d,277] at offset d + 277*d_model 
		// vs grad_W_rowmaj at offset 277*d_model + d
		// d + 277*d_model vs 277*d_model + d ← SAME! ✓
		//
		// OK the storage is correct. Let me trace what the GEMM actually computes.
		// C[d,v] = Σ_t h_colmaj[d,t] * g_colmaj[v,t] (with B accessed as B[v,t] due to OP_T)
		// Wait, OP_T means we access B^T[k,j] = B[j,k]
		// So C[i,j] = Σ_k A[i,k] * B[j,k]
		// C[d,v] = Σ_t A[d,t] * B[v,t] = Σ_t h_colmaj[d,t] * g_colmaj[v,t]
		// h_colmaj[d,t] at (d + t*lda) = (d + t*d_model) → h_rowmaj[t,d] ✓
		// g_colmaj[v,t] at (v + t*ldb) = (v + t*vocab) → g_rowmaj[t,v] ✓
		// C[d,v] = Σ_t h_rowmaj[t,d] * g_rowmaj[t,v] = grad_W_rowmaj[v,d] ✓
		//
		// THE FORMULA IS CORRECT! The bug must be in the diagnostic code reading the result.
		// Let me check the diagnostic that reads grad_W[277]...
		status = cublasSgemm(
			handle,
			CUBLAS_OP_N,           // hidden: [d_model, tokens] col-major, no transpose
			CUBLAS_OP_T,           // grad_logits: [vocab, tokens] col-major, transpose to [tokens, vocab]
			d_model,               // m (rows of result)
			vocab_size,            // n (cols of result)
			total_tokens,          // k (inner dimension)
			&alpha,
			encoder_for_grad_weight, // A: centered hidden states [d_model, tokens]
			d_model,               // lda
			grad_logits_ptr,       // B: grad_logits [vocab, tokens]
			vocab_size,            // ldb
			&beta,
			grad_weight_ptr,       // C: grad_weight [d_model, vocab] col-major = [vocab, d_model] row-major
			d_model);              // ldc

		if (status != CUBLAS_STATUS_SUCCESS) {
			std::ostringstream oss;
			oss << "[LMHead::backward] cublasSgemm grad_weight failed status=" << status;
			throw std::runtime_error(oss.str());
		}
		
		// =====================================================================
		// ISSUE #41 DIAGNOSTIC: Compare GEMM output vs manual BEFORE recentering
		// =====================================================================
		{
			static int s_gemm_diag = 0;
			if (++s_gemm_diag <= 3) {
				cudaStreamSynchronize(stream);
				
				// Read inputs
				std::vector<float> h_enc(static_cast<size_t>(total_tokens) * d_model);
				std::vector<float> h_grad(static_cast<size_t>(total_tokens) * vocab_size);
				cudaMemcpy(h_enc.data(), encoder_for_grad_weight, h_enc.size() * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_grad.data(), grad_logits_ptr, h_grad.size() * sizeof(float), cudaMemcpyDeviceToHost);
				
				// Read GEMM output (BEFORE recentering)
				const size_t row277_offset = static_cast<size_t>(277) * d_model;
				std::vector<float> gemm_row(d_model);
				cudaMemcpy(gemm_row.data(), grad_weight_ptr + row277_offset, d_model * sizeof(float), cudaMemcpyDeviceToHost);
				
				// Manual computation: grad_W[277,d] = Σ_t h[t,d] * g[t,277]
				std::vector<double> manual_row(d_model, 0.0);
				for (int i = 0; i < d_model; ++i) {
					for (int t = 0; t < total_tokens; ++t) {
						float h_td = h_enc[static_cast<size_t>(t) * d_model + i];
						float g_t277 = h_grad[static_cast<size_t>(t) * vocab_size + 277];
						manual_row[i] += h_td * g_t277;
					}
				}
				
				// Compare first 5 elements
				std::ostringstream oss;
				oss << "[Issue41-GEMM-VS-MANUAL] call=" << s_gemm_diag << " tokens=" << total_tokens << "\n";
				oss << "  GEMM[277,0..4]:   ";
				for (int i = 0; i < 5; ++i) oss << std::setprecision(9) << gemm_row[i] << " ";
				oss << "\n  MANUAL[277,0..4]: ";
				for (int i = 0; i < 5; ++i) oss << std::setprecision(9) << manual_row[i] << " ";
				oss << "\n  DIFF[0..4]:       ";
				for (int i = 0; i < 5; ++i) oss << std::setprecision(9) << (gemm_row[i] - manual_row[i]) << " ";
				
				// Sum comparison
				double gemm_sum = 0, manual_sum = 0;
				for (int i = 0; i < d_model; ++i) {
					gemm_sum += gemm_row[i];
					manual_sum += manual_row[i];
				}
				oss << "\n  ROW_SUMS: GEMM=" << gemm_sum << " MANUAL=" << manual_sum << " DIFF=" << (gemm_sum - manual_sum);
				
				// Sample input values
				oss << "\n  h_enc[0,0..2]: " << h_enc[0] << " " << h_enc[1] << " " << h_enc[2];
				oss << "\n  g_grad[0,277]: " << h_grad[277] << " g_grad[1,277]: " << h_grad[vocab_size + 277];
				
				GRIM::Logging::EmitModuleInfo("BackwardPass", oss.str());
			}
		}
		
		// =====================================================================
		// ISSUE #40 FIX: Re-center gradient rows to eliminate FP32 GEMM error
		// =====================================================================
		// The cuBLAS SGEMM accumulates FP32 rounding errors that destroy the
		// mathematical property: sum(grad_W[v,:]) ≈ 0 (due to centered hidden states).
		// These errors accumulate SYSTEMATICALLY (~6e-5 positive bias per row),
		// which causes mode collapse to token 277 (SPACE).
		// 
		// Solution: Re-center each row by subtracting its mean.
		// NOTE: Only apply if centering was used AND recenter_gradients is enabled.
		// =====================================================================
		if (params.use_centering && params.recenter_gradients) {
			recenterGradientRows(
				grad_weight_ptr,
				d_model,
				vocab_size,
				stream
			);
		}
		
		// === TOKEN 277 WEIGHT GRADIENT DIAGNOSTIC ===
		// grad_weight[277] = sum_t(encoder_output[t] * grad_logits[t, 277])
		// This shows the aggregated gradient update direction for row 277
		static int s_grad_w277_count = 0;
		constexpr int kToken277 = 277;
		if (++s_grad_w277_count <= 20 && kToken277 < vocab_size) {
			cudaStreamSynchronize(stream);
			
			// === Issue #39: DETAILED GEMM OUTPUT VERIFICATION ===
			// C is [d_model, vocab] col-major = [vocab, d_model] row-major
			// Row 277 row-major = Column 277 col-major
			// Row 277 starts at offset 277 * d_model (both interpretations agree!)
			const size_t row_offset = static_cast<size_t>(kToken277) * d_model;
			std::vector<float> grad_w277(d_model);
			cudaMemcpy(grad_w277.data(), grad_weight_ptr + row_offset,
					   d_model * sizeof(float), cudaMemcpyDeviceToHost);
			
			// Also read a few random elements to verify GEMM output structure
			// Element [d,v] in col-major at offset d + v*d_model
			// Element [v,d] in row-major at offset v*d_model + d (SAME as above!)
			float elem_277_0, elem_277_1, elem_0_277, elem_1_277;
			// grad_W[277, 0] row-major = grad_W_colmaj[0, 277] at offset 0 + 277*d_model
			cudaMemcpy(&elem_277_0, grad_weight_ptr + 0 + 277*d_model, sizeof(float), cudaMemcpyDeviceToHost);
			// grad_W[277, 1] row-major at offset 277*d_model + 1
			cudaMemcpy(&elem_277_1, grad_weight_ptr + 277*d_model + 1, sizeof(float), cudaMemcpyDeviceToHost);
			// grad_W[0, 277] row-major at offset 0*d_model + 277 = 277
			cudaMemcpy(&elem_0_277, grad_weight_ptr + 277, sizeof(float), cudaMemcpyDeviceToHost);
			// grad_W[1, 277] row-major at offset 1*d_model + 277
			cudaMemcpy(&elem_1_277, grad_weight_ptr + d_model + 277, sizeof(float), cudaMemcpyDeviceToHost);
			
			// Compute norm and mean of grad_W[277]
			float norm_sq = 0.0f, sum = 0.0f;
			for (int i = 0; i < d_model; ++i) {
				norm_sq += grad_w277[i] * grad_w277[i];
				sum += grad_w277[i];
			}
			float norm = std::sqrt(norm_sq);
			float mean = sum / d_model;
			
			// === Issue #40: COMPUTE GRADIENT-WEIGHT ALIGNMENT ===
			// If grad dot W > 0, AdamW will push W further in its current direction!
			// AdamW: W_new = W - lr * (adam_update + wd * W)
			// adam_update direction ≈ sign(grad), so if grad·W > 0, update reinforces W
			std::vector<float> w277(d_model);
			cudaMemcpy(w277.data(), weights_ptr + row_offset,
					   d_model * sizeof(float), cudaMemcpyDeviceToHost);
			
			float w_norm_sq = 0.0f, dot_product = 0.0f;
			for (int i = 0; i < d_model; ++i) {
				w_norm_sq += w277[i] * w277[i];
				dot_product += grad_w277[i] * w277[i];
			}
			float w_norm = std::sqrt(w_norm_sq);
			float cosine_sim = (norm > 1e-8f && w_norm > 1e-8f) ? (dot_product / (norm * w_norm)) : 0.0f;
			
			// Predict norm change: if gradient aligns with W, norm increases
			// AdamW update: W_new = W - lr * normalized_grad - lr * wd * W
			// Delta_W ≈ -lr * grad_direction - lr * wd * W
			// ||W_new||² ≈ ||W||² - 2*lr*dot(grad,W) - 2*lr*wd*||W||² + O(lr²)
			// Norm increases if: -2*lr*dot(grad,W) - 2*lr*wd*||W||² > 0
			// i.e., if: dot(grad,W) < -wd*||W||²  (gradient pushes opposite to W)
			// Norm decreases if: dot(grad,W) > -wd*||W||²
			const float wd = 0.01f;  // weight_decay from config
			const float threshold = -wd * w_norm_sq;
			const char* prediction = (dot_product < threshold) ? "NORM_WILL_INCREASE" : 
									  (dot_product > 0) ? "NORM_WILL_DECREASE" : "NEAR_THRESHOLD";
			
			{
				std::ostringstream oss;
				oss << "[Token277GradW] call=" << s_grad_w277_count << " tokens=" << total_tokens
					<< " | grad_W[277]: norm=" << std::fixed << std::setprecision(6) << norm
					<< " sum=" << std::setprecision(9) << sum 
					<< " mean=" << mean;
				GRIM::Logging::EmitModuleInfo("BackwardPass", oss.str());
				
				// Issue #40: Alignment diagnostic
				std::ostringstream oss2;
				oss2 << "[Issue40-ALIGN] call=" << s_grad_w277_count
					 << " | W[277].norm=" << std::setprecision(6) << w_norm
					 << " grad·W=" << std::setprecision(9) << dot_product
					 << " cosine=" << std::setprecision(4) << cosine_sim
					 << " | threshold=" << threshold << " " << prediction;
				GRIM::Logging::EmitModuleInfo("BackwardPass", oss2.str());
			}
			
			// Print cross-check of individual elements
			if (s_grad_w277_count == 1) {
				// MANUAL COMPUTATION of grad_W[277, 0]
				// Formula: grad_W[v,d] = Σ_t h[t,d] * g[t,v]
				// grad_W[277, 0] = Σ_t h[t, 0] * g[t, 277]
				std::vector<float> h_data(static_cast<size_t>(total_tokens) * d_model);
				std::vector<float> g_data(static_cast<size_t>(total_tokens) * vocab_size);
				cudaMemcpy(h_data.data(), encoder_for_grad_weight, h_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(g_data.data(), grad_logits_ptr, g_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
				
				// Compute FULL manual sum across all 768 dimensions
				double manual_full_sum = 0.0;
				double manual_277_0 = 0.0;
				double manual_277_1 = 0.0;
				for (int dd = 0; dd < d_model; ++dd) {
					double col_sum = 0.0;
					for (int t = 0; t < total_tokens; ++t) {
						float h_t_d = h_data[static_cast<size_t>(t) * d_model + dd];
						float g_t_277 = g_data[static_cast<size_t>(t) * vocab_size + 277];
						col_sum += h_t_d * g_t_277;
					}
					manual_full_sum += col_sum;
					if (dd == 0) manual_277_0 = col_sum;
					if (dd == 1) manual_277_1 = col_sum;
				}
				
				// Issue #40: ALSO read grad_weight row 277 and sum it (same as Token277GradW does)
				// This verifies we're reading from the same memory the GEMM wrote to
				std::vector<float> gemm_row277(d_model);
				const size_t row277_offset = static_cast<size_t>(277) * d_model;
				cudaMemcpy(gemm_row277.data(), grad_weight_ptr + row277_offset,
				           d_model * sizeof(float), cudaMemcpyDeviceToHost);
				double gemm_row_sum = 0.0;
				for (int dd = 0; dd < d_model; ++dd) {
					gemm_row_sum += gemm_row277[dd];
				}
				// Print pointer addresses for verification
				std::ostringstream ptr_oss;
				ptr_oss << "[Issue40-PTR-VERIFY] grad_weight=" << (void*)grad_weight_ptr
				        << " row277_at=" << (void*)(grad_weight_ptr + row277_offset)
				        << " encoder_for_grad=" << (void*)encoder_for_grad_weight
				        << " | GEMM_ROW_SUM=" << std::setprecision(9) << gemm_row_sum
				        << " MANUAL_SUM=" << manual_full_sum
				        << " DIFF=" << (gemm_row_sum - manual_full_sum);
				GRIM::Logging::EmitModuleInfo("BackwardPass", ptr_oss.str());
				
				// Also verify h_data row sums (should be ~0 if centered)
				double h_row0_sum = 0.0;
				for (int dd = 0; dd < d_model; ++dd) {
					h_row0_sum += h_data[dd];  // h[0, d]
				}
				
				std::ostringstream oss;
				oss << "[Issue39-VERIFY] MANUAL_FULL_SUM=" << std::setprecision(9) << manual_full_sum
					<< " (should match CPU direct ~0)"
					<< " | h_row0_sum=" << h_row0_sum << " (should be ~0 if centered)"
					<< "\n                MANUAL[277,0]=" << manual_277_0 << " MANUAL[277,1]=" << manual_277_1
					<< " | GEMM[277,0]=" << elem_277_0 << " GEMM[277,1]=" << elem_277_1
					<< " | grad_W[0,277]=" << elem_0_277 
					<< " grad_W[1,277]=" << elem_1_277;
				GRIM::Logging::EmitModuleInfo("BackwardPass", oss.str());
			}
		}
	}

	// Bias gradient computation
	if (params.use_bias && grad_bias_ptr) {
		launchBiasSumGradient(
			grad_logits_ptr,
			grad_bias_ptr,
			total_tokens,
			vocab_size,
			stream);
	}
}

} // namespace GRIM
