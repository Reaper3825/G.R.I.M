#include "lm_head_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <stdexcept>
#include <sstream>
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
    constexpr int kToken277 = 277;
    if (kToken277 >= params.vocab_size) return;
    
    ++s_lm_head_call_count;
    
    // Only log first 20 calls to avoid spam
    if (s_lm_head_call_count > 20) return;
    
    cudaStreamSynchronize(params.stream);
    
    const int d_model = params.d_model;
    const int total_tokens = params.batch_size * params.seq_len;
    const int num_sample = std::min(5, total_tokens);
    
    // Read W[277] row (the weight row for token 277)
    // Weights are [vocab_size, d_model] row-major, so W[277] starts at offset 277*d_model
    std::vector<float> w277(d_model);
    cudaMemcpy(w277.data(), 
               params.weights + static_cast<size_t>(kToken277) * d_model,
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
    cudaMemcpy(hidden_sample.data(), params.encoder_output,
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
        if (tok >= params.vocab_size) continue;
        std::vector<float> w_other(d_model);
        cudaMemcpy(w_other.data(),
                   params.weights + static_cast<size_t>(tok) * d_model,
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

	// === TOKEN 277 DIAGNOSTIC: Log inputs BEFORE computing logits ===
	logToken277Inputs(params);

	// CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
	if (params.stream) {
		cublasSetStream(params.handle, params.stream);
	}

	// === Issue #37 FIX: Zero-mean centering before projection ===
	// Centers each hidden state vector to have zero mean, preventing mode collapse.
	// Without centering: grad_W[tok] = Σ_t (hidden_sum[t] × grad[t,tok])
	// If hidden_sum has same sign as grad, the contribution is wrong!
	// With centering: hidden_sum = 0 for each position, so gradients depend
	// only on hidden state direction, not offset.
	const float* projection_input = params.encoder_output;
	
	if (params.use_centering && params.centered_scratch) {
		centerHiddenStates(
			params.encoder_output,
			params.centered_scratch,
			params.d_model,
			total_tokens,
			params.stream
		);
		projection_input = params.centered_scratch;
		
		// Diagnostic: Verify centering worked (first 5 calls only)
		static int s_centering_diag_count = 0;
		if (++s_centering_diag_count <= 5) {
			cudaStreamSynchronize(params.stream);
			
			// Sample position 0
			std::vector<float> orig(params.d_model), centered(params.d_model);
			cudaMemcpy(orig.data(), params.encoder_output, params.d_model * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(centered.data(), params.centered_scratch, params.d_model * sizeof(float), cudaMemcpyDeviceToHost);
			
			float orig_sum = 0, centered_sum = 0;
			for (int i = 0; i < params.d_model; ++i) {
				orig_sum += orig[i];
				centered_sum += centered[i];
			}
			float orig_mean = orig_sum / params.d_model;
			float centered_mean = centered_sum / params.d_model;
			
			fprintf(stderr, "[Issue37-Centering] call=%d tokens=%d pos0: orig_mean=%.6f centered_mean=%.6f (should be ~0)\n",
					s_centering_diag_count, total_tokens, orig_mean, centered_mean);
		}
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
		projection_input,          // Use centered hidden states (Issue #37)
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

	// CRITICAL: Always rebind stream before cuBLAS ops - NumericHead may have changed it
	if (params.stream) {
		cublasSetStream(params.handle, params.stream);
	}

	const float alpha = 1.0f;
	const float beta = params.accumulate ? 1.0f : 0.0f;

	// grad_encoder = weights @ grad_logits
	// NOTE: If centering was applied in forward, this gives us grad w.r.t. CENTERED hidden states.
	//       We need to backprop through the centering operation to get true grad_encoder.
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

	// === Issue #37: Backprop through centering operation ===
	// Forward: y = x - mean(x)  =>  Backward: grad_x = grad_y - mean(grad_y)
	// The centering backward is identical to centering forward!
	// We apply it IN-PLACE to grad_encoder since we just computed it.
	if (params.use_centering) {
		// Use centered_encoder as scratch buffer for the backward centering
		// Actually, we need to center grad_encoder in-place. The centering function
		// can do in-place if we pass same src/dst... but our kernel doesn't support that.
		// Let's use a two-step: copy to scratch, center back to grad_encoder
		if (params.centered_encoder) {
			cudaMemcpyAsync(const_cast<float*>(params.centered_encoder), params.grad_encoder,
				static_cast<size_t>(total_tokens) * params.d_model * sizeof(float),
				cudaMemcpyDeviceToDevice, params.stream);
			centerHiddenStates(
				params.centered_encoder,
				params.grad_encoder,  // Output back to grad_encoder
				params.d_model,
				total_tokens,
				params.stream
			);
		}
	}

	// grad_weight = encoder_output @ grad_logits^T
	// === Issue #37: Use centered encoder output if centering was applied ===
	// If we centered hidden states in forward (h_centered = h - mean(h)),
	// then grad_weight must be computed against the centered hidden states!
	// grad_W[i,j] = sum_t(h_centered[t,j] * grad_logits[t,i])
	const float* encoder_for_grad_weight = params.encoder_output;
	if (params.use_centering && params.centered_encoder && params.encoder_output) {
		// Re-compute centering (matches forward pass)
		centerHiddenStates(
			params.encoder_output,
			const_cast<float*>(params.centered_encoder),  // Scratch buffer
			params.d_model,
			total_tokens,
			params.stream
		);
		encoder_for_grad_weight = params.centered_encoder;
		
		// === DIAGNOSTIC: Verify centering worked and compute expected grad_W[277] sum ===
		static int s_bwd_center_diag = 0;
		constexpr int kToken277_diag = 277;
		if (++s_bwd_center_diag <= 10) {
			cudaStreamSynchronize(params.stream);
			
			// Read centered hidden states and grad_logits for verification
			std::vector<float> h_centered(static_cast<size_t>(total_tokens) * params.d_model);
			std::vector<float> h_grad_logits(static_cast<size_t>(total_tokens) * params.vocab_size);
			
			cudaMemcpy(h_centered.data(), params.centered_encoder,
					   h_centered.size() * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_grad_logits.data(), params.grad_logits,
					   h_grad_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
			
			// Verify centering: each row should sum to ~0
			double max_row_sum = 0.0, total_row_sum = 0.0;
			for (int t = 0; t < total_tokens; ++t) {
				double row_sum = 0.0;
				for (int j = 0; j < params.d_model; ++j) {
					row_sum += h_centered[static_cast<size_t>(t) * params.d_model + j];
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
				for (int j = 0; j < params.d_model; ++j) {
					row_sum += h_centered[static_cast<size_t>(t) * params.d_model + j];
				}
				float grad_277_t = h_grad_logits[static_cast<size_t>(t) * params.vocab_size + kToken277_diag];
				expected_grad_sum += row_sum * grad_277_t;
			}
			
			fprintf(stderr, "[Issue37-BWD-DIAG] call=%d tokens=%d\n", s_bwd_center_diag, total_tokens);
			fprintf(stderr, "[Issue37-BWD-DIAG] Centering check: max_row_sum=%.9f total_row_sum=%.9f (should be ~0)\n",
					max_row_sum, total_row_sum);
			fprintf(stderr, "[Issue37-BWD-DIAG] Expected grad_W[277].sum from centered_h: %.9f\n", expected_grad_sum);
		}
	}
	
	if (params.grad_weight && encoder_for_grad_weight) {
		status = cublasSgemm(
			params.handle,
			CUBLAS_OP_N,           // encoder_output: [d_model, tokens]
			CUBLAS_OP_T,           // grad_logits: [vocab, tokens]^T
			params.d_model,        // m
			params.vocab_size,     // n
			total_tokens,          // k
			&alpha,
			encoder_for_grad_weight, // Use centered hidden states (Issue #37)
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
		
		// === TOKEN 277 WEIGHT GRADIENT DIAGNOSTIC ===
		// grad_weight[277] = sum_t(encoder_output[t] * grad_logits[t, 277])
		// This shows the aggregated gradient update direction for row 277
		static int s_grad_w277_count = 0;
		constexpr int kToken277 = 277;
		if (++s_grad_w277_count <= 20 && kToken277 < params.vocab_size) {
			cudaStreamSynchronize(params.stream);
			
			// Row 277 of grad_weight: starts at [d_model * 277]
			const size_t row_offset = static_cast<size_t>(kToken277) * params.d_model;
			std::vector<float> grad_w277(params.d_model);
			cudaMemcpy(grad_w277.data(), params.grad_weight + row_offset,
					   params.d_model * sizeof(float), cudaMemcpyDeviceToHost);
			
			// Compute norm and mean of grad_W[277]
			float norm_sq = 0.0f, sum = 0.0f;
			for (int i = 0; i < params.d_model; ++i) {
				norm_sq += grad_w277[i] * grad_w277[i];
				sum += grad_w277[i];
			}
			float norm = std::sqrt(norm_sq);
			float mean = sum / params.d_model;
			
			// Compare with a few other rows
			std::vector<float> grad_w0(params.d_model), grad_w100(params.d_model);
			cudaMemcpy(grad_w0.data(), params.grad_weight, params.d_model * sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(grad_w100.data(), params.grad_weight + 100 * params.d_model, 
					   params.d_model * sizeof(float), cudaMemcpyDeviceToHost);
			
			float norm_w0_sq = 0.0f, norm_w100_sq = 0.0f;
			for (int i = 0; i < params.d_model; ++i) {
				norm_w0_sq += grad_w0[i] * grad_w0[i];
				norm_w100_sq += grad_w100[i] * grad_w100[i];
			}
			
			fprintf(stderr, "[Token277GradW] call=%d tokens=%d\n", s_grad_w277_count, total_tokens);
			fprintf(stderr, "[Token277GradW] grad_W[277]: norm=%.6f sum=%.9f mean=%.9f\n", norm, sum, mean);
			fprintf(stderr, "[Token277GradW] first_5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
					grad_w277[0], grad_w277[1], grad_w277[2], grad_w277[3], grad_w277[4]);
			fprintf(stderr, "[Token277GradW] comparison: grad_W[0]_norm=%.6f grad_W[100]_norm=%.6f\n",
					std::sqrt(norm_w0_sq), std::sqrt(norm_w100_sq));
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
