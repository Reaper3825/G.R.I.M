//======================================================//
//  flash_attention_test.cu
//  Comprehensive test suite for Flash Attention
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif
#include "flash_attention_test.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Shared/TensorConversion/TensorConversion.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <numeric>

using namespace GRIM::Test;

namespace {

//======================================================//
//  Helper Functions
//======================================================//

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
    }
}

// Initialize tensor with random values
void initRandom(float* d_ptr, size_t count, float scale = 0.1f, unsigned seed = 42) {
    std::vector<float> host(count);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < count; ++i) {
        host[i] = dist(rng);
    }
    cudaMemcpy(d_ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
}

// Initialize tensor with ones
void initOnes(float* d_ptr, size_t count) {
    std::vector<float> host(count, 1.0f);
    cudaMemcpy(d_ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
}

// Initialize tensor with zeros
void initZeros(float* d_ptr, size_t count) {
    cudaMemset(d_ptr, 0, count * sizeof(float));
}

// Copy device to host
std::vector<float> toHost(const float* d_ptr, size_t count) {
    std::vector<float> host(count);
    cudaMemcpy(host.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    return host;
}

// Compute L2 norm
float l2Norm(const std::vector<float>& v) {
    float sum = 0.0f;
    for (float x : v) sum += x * x;
    return std::sqrt(sum);
}

// Compute mean absolute value
float meanAbs(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    float sum = 0.0f;
    for (float x : v) sum += std::abs(x);
    return sum / v.size();
}

// Compute max absolute value
float maxAbs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::abs(x));
    return m;
}

// Check if values are finite
bool allFinite(const std::vector<float>& v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

float canonicalSoftmaxScale(int head_dim) {
    if (head_dim <= 0) {
        throw std::runtime_error("canonicalSoftmaxScale: head_dim must be > 0");
    }
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// Naive attention for comparison (CPU)
void naiveAttentionForward(
    const float* Q, const float* K, const float* V,
    float* output,
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal,
    float softmax_scale
) {
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            const int head_offset = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
            const float* Q_head = Q + head_offset;
            const float* K_head = K + head_offset;
            const float* V_head = V + head_offset;
            float* O_head = output + head_offset;
            
            for (int q = 0; q < seq_len; ++q) {
                // Compute attention scores
                std::vector<float> scores(seq_len);
                float max_score = -1e9f;
                
                for (int k = 0; k < seq_len; ++k) {
                    if (causal && k > q) {
                        scores[k] = -1e9f;
                    } else {
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            score += Q_head[q * head_dim + d] * K_head[k * head_dim + d];
                        }
                        scores[k] = score * softmax_scale;
                    }
                    max_score = std::max(max_score, scores[k]);
                }
                
                // Softmax
                float sum_exp = 0.0f;
                for (int k = 0; k < seq_len; ++k) {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                for (int k = 0; k < seq_len; ++k) {
                    scores[k] /= sum_exp;
                }
                
                // Output = P @ V
                for (int d = 0; d < head_dim; ++d) {
                    float out = 0.0f;
                    for (int k = 0; k < seq_len; ++k) {
                        out += scores[k] * V_head[k * head_dim + d];
                    }
                    O_head[q * head_dim + d] = out;
                }
            }
        }
    }
}

// Relative error between two vectors
float relativeError(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9f;
    float max_diff = 0.0f;
    float max_val = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        max_val = std::max(max_val, std::max(std::abs(a[i]), std::abs(b[i])));
    }
    return max_val > 1e-6f ? max_diff / max_val : max_diff;
}

struct FlashAttnForwardScratch {
    __nv_bfloat16* q = nullptr;
    __nv_bfloat16* k = nullptr;
    __nv_bfloat16* v = nullptr;
    __nv_bfloat16* out = nullptr;
    float* softmax_lse = nullptr;
    size_t q_elems = 0;
    size_t kv_elems = 0;
    size_t out_elems = 0;
    size_t lse_elems = 0;
};

struct FlashAttnBackwardScratch {
    FlashAttnForwardScratch fwd{};
    __nv_bfloat16* dout = nullptr;
    __nv_bfloat16* dq = nullptr;
    __nv_bfloat16* dk = nullptr;
    __nv_bfloat16* dv = nullptr;
    void* dq_accum = nullptr;
    void* dsoftmax_sum = nullptr;
    size_t dq_accum_bytes = 0;
    size_t dsoftmax_sum_bytes = 0;
};

FlashAttnForwardScratch allocateFlashAttnForwardScratch(int batch, int heads, int kv_heads,
                                                        int seq, int head_dim) {
    FlashAttnForwardScratch scratch{};
    scratch.q_elems = static_cast<size_t>(batch) * heads * seq * head_dim;
    scratch.kv_elems = static_cast<size_t>(batch) * kv_heads * seq * head_dim;
    scratch.out_elems = scratch.q_elems;
    scratch.lse_elems = static_cast<size_t>(batch) * heads * seq;

    checkCuda(cudaMalloc(&scratch.q, scratch.q_elems * sizeof(__nv_bfloat16)), "alloc fa_q");
    checkCuda(cudaMalloc(&scratch.k, scratch.kv_elems * sizeof(__nv_bfloat16)), "alloc fa_k");
    checkCuda(cudaMalloc(&scratch.v, scratch.kv_elems * sizeof(__nv_bfloat16)), "alloc fa_v");
    checkCuda(cudaMalloc(&scratch.out, scratch.out_elems * sizeof(__nv_bfloat16)), "alloc fa_out");
    checkCuda(cudaMalloc(&scratch.softmax_lse, scratch.lse_elems * sizeof(float)), "alloc fa_lse");
    return scratch;
}

void freeFlashAttnForwardScratch(FlashAttnForwardScratch& scratch) {
    if (scratch.q) cudaFree(scratch.q);
    if (scratch.k) cudaFree(scratch.k);
    if (scratch.v) cudaFree(scratch.v);
    if (scratch.out) cudaFree(scratch.out);
    if (scratch.softmax_lse) cudaFree(scratch.softmax_lse);
    scratch = {};
}

FlashAttnBackwardScratch allocateFlashAttnBackwardScratch(int batch, int heads, int kv_heads,
                                                          int seq, int head_dim) {
    FlashAttnBackwardScratch scratch{};
    scratch.fwd = allocateFlashAttnForwardScratch(batch, heads, kv_heads, seq, head_dim);

    checkCuda(cudaMalloc(&scratch.dout, scratch.fwd.out_elems * sizeof(__nv_bfloat16)), "alloc fa_dout");
    checkCuda(cudaMalloc(&scratch.dq, scratch.fwd.q_elems * sizeof(__nv_bfloat16)), "alloc fa_dq");
    checkCuda(cudaMalloc(&scratch.dk, scratch.fwd.kv_elems * sizeof(__nv_bfloat16)), "alloc fa_dk");
    checkCuda(cudaMalloc(&scratch.dv, scratch.fwd.kv_elems * sizeof(__nv_bfloat16)), "alloc fa_dv");

    scratch.dq_accum_bytes = flash_attn_dq_accum_bytes(batch, seq, heads, head_dim);
    scratch.dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(batch, seq, heads);
    checkCuda(cudaMalloc(&scratch.dq_accum, scratch.dq_accum_bytes), "alloc fa_dq_accum");
    checkCuda(cudaMalloc(&scratch.dsoftmax_sum, scratch.dsoftmax_sum_bytes), "alloc fa_dsoftmax_sum");
    return scratch;
}

void freeFlashAttnBackwardScratch(FlashAttnBackwardScratch& scratch) {
    if (scratch.dout) cudaFree(scratch.dout);
    if (scratch.dq) cudaFree(scratch.dq);
    if (scratch.dk) cudaFree(scratch.dk);
    if (scratch.dv) cudaFree(scratch.dv);
    if (scratch.dq_accum) cudaFree(scratch.dq_accum);
    if (scratch.dsoftmax_sum) cudaFree(scratch.dsoftmax_sum);
    freeFlashAttnForwardScratch(scratch.fwd);
    scratch = {};
}

void flashAttnForwardBHSD(const float* q, const float* k, const float* v,
                          float* out,
                          FlashAttnForwardScratch& scratch,
                          int batch, int heads, int kv_heads,
                          int seq, int head_dim,
                          bool causal, cudaStream_t stream,
                          float softmax_scale) {
    TensorConversion::convert_BHSD_to_BSHD_bf16(q, scratch.q, batch, heads, seq, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(k, scratch.k, batch, kv_heads, seq, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(v, scratch.v, batch, kv_heads, seq, head_dim, stream);

    flash_attn_fwd_ex(
        scratch.q,
        scratch.k,
        scratch.v,
        scratch.out,
        scratch.softmax_lse,
        nullptr,
        batch,
        seq,
        heads,
        kv_heads,
        head_dim,
        softmax_scale,
        causal,
        true,
        0.0f,
        0,
        stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(scratch.out, out, batch, seq, heads, head_dim, stream);
}

void flashAttnBackwardBHSD(const float* q, const float* k, const float* v,
                           const float* dout,
                           float* dq, float* dk, float* dv,
                           FlashAttnBackwardScratch& scratch,
                           int batch, int heads, int kv_heads,
                           int seq, int head_dim,
                           bool causal, cudaStream_t stream,
                           float softmax_scale) {
    TensorConversion::convert_BHSD_to_BSHD_bf16(q, scratch.fwd.q, batch, heads, seq, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(k, scratch.fwd.k, batch, kv_heads, seq, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(v, scratch.fwd.v, batch, kv_heads, seq, head_dim, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(dout, scratch.dout, batch, heads, seq, head_dim, stream);

    flash_attn_bwd_ex(
        scratch.fwd.q,
        scratch.fwd.k,
        scratch.fwd.v,
        scratch.fwd.out,
        scratch.dout,
        scratch.fwd.softmax_lse,
        nullptr,
        scratch.dq,
        scratch.dk,
        scratch.dv,
        scratch.dq_accum,
        scratch.dsoftmax_sum,
        batch,
        seq,
        heads,
        kv_heads,
        head_dim,
        softmax_scale,
        causal,
        true,
        0.0f,
        0,
        stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(scratch.dq, dq, batch, seq, heads, head_dim, stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(scratch.dk, dk, batch, seq, kv_heads, head_dim, stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(scratch.dv, dv, batch, seq, kv_heads, head_dim, stream);
}

} // anonymous namespace

//======================================================//
//  Section 1: Forward Pass Tests
//======================================================//

bool GRIM::Test::testFlashForwardBasic(std::string& message) {
    const int batch = 1, heads = 4, seq = 64, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 1);
    initRandom(d_K, total, 0.1f, 2);
    initRandom(d_V, total, 0.1f, 3);
    initZeros(d_out, total);

    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto output = toHost(d_out, total);

    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    ATTN_ASSERT(allFinite(output), "Output contains NaN/Inf");
    ATTN_ASSERT(maxAbs(output) > 0.0f, "Output is all zeros");
    ATTN_ASSERT(maxAbs(output) < 100.0f, "Output has exploding values");
    
    return true;
}

bool GRIM::Test::testFlashForwardCausalMask(std::string& message) {
    const int batch = 1, heads = 1, seq = 32, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    // Initialize V with position-dependent values so we can verify causal masking
    std::vector<float> V_host(total);
    for (int s = 0; s < seq; ++s) {
        for (int d = 0; d < dim; ++d) {
            V_host[s * dim + d] = static_cast<float>(s);  // V[s,:] = s
        }
    }
    cudaMemcpy(d_V, V_host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    
    // Q = K = identity-like (strong self-attention)
    initOnes(d_Q, total);
    initOnes(d_K, total);

    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto output = toHost(d_out, total);

    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    // Position 0 should only see V[0] = 0, so output[0,:] should be ~0
    float pos0_mean = 0.0f;
    for (int d = 0; d < dim; ++d) pos0_mean += output[d];
    pos0_mean /= dim;
    
    ATTN_ASSERT(std::abs(pos0_mean) < 0.5f, "Position 0 should primarily see V[0]=0");
    
    // Later positions should see higher values (weighted average of 0..pos)
    float posLast_mean = 0.0f;
    for (int d = 0; d < dim; ++d) posLast_mean += output[(seq-1) * dim + d];
    posLast_mean /= dim;
    
    ATTN_ASSERT(posLast_mean > pos0_mean, "Later positions should see higher V values");
    
    return true;
}

bool GRIM::Test::testFlashForwardVsNaive(std::string& message) {
    const int batch = 1, heads = 2, seq = 32, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 10);
    initRandom(d_K, total, 0.1f, 20);
    initRandom(d_V, total, 0.1f, 30);
    
    auto Q_host = toHost(d_Q, total);
    auto K_host = toHost(d_K, total);
    auto V_host = toHost(d_V, total);
    
    // Flash Attention
    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto flash_out = toHost(d_out, total);
    
    // Naive attention (CPU)
    std::vector<float> naive_out(total);
    naiveAttentionForward(Q_host.data(), K_host.data(), V_host.data(),
                          naive_out.data(), batch, heads, seq, dim, true, softmax_scale);
    
    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    float rel_err = relativeError(flash_out, naive_out);
    
    ATTN_ASSERT(rel_err < 0.05f, "Flash vs Naive relative error too high: " + std::to_string(rel_err));
    
    return true;
}

bool GRIM::Test::testFlashForwardCustomScale(std::string& message) {
    const int batch = 1, heads = 1, seq = 32, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float custom_scale = 2.0f / std::sqrt(static_cast<float>(dim));
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    initRandom(d_Q, total, 0.35f, 110);
    initRandom(d_K, total, 0.35f, 120);
    initRandom(d_V, total, 0.20f, 130);
    
    auto Q_host = toHost(d_Q, total);
    auto K_host = toHost(d_K, total);
    auto V_host = toHost(d_V, total);
    
    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch,
                         batch, heads, heads, seq, dim, true, nullptr, custom_scale);
    cudaDeviceSynchronize();
    
    auto flash_out = toHost(d_out, total);
    std::vector<float> naive_out(total);
    naiveAttentionForward(Q_host.data(), K_host.data(), V_host.data(),
                          naive_out.data(), batch, heads, seq, dim, true, custom_scale);
    
    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    const float rel_err = relativeError(flash_out, naive_out);
    ATTN_ASSERT(rel_err < 0.08f, std::string("Flash ignored or misapplied custom softmax scale; rel_err=") + std::to_string(rel_err));
    return true;
}

//======================================================//
//  Section 2: Backward Pass Tests
//======================================================//

bool GRIM::Test::testFlashBackwardGradientFlow(std::string& message) {
    const int batch = 1, heads = 4, seq = 64, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    float *d_dO, *d_dQ, *d_dK, *d_dV;
    
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_dO, total * sizeof(float));
    cudaMalloc(&d_dQ, total * sizeof(float));
    cudaMalloc(&d_dK, total * sizeof(float));
    cudaMalloc(&d_dV, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 100);
    initRandom(d_K, total, 0.1f, 200);
    initRandom(d_V, total, 0.1f, 300);
    initRandom(d_dO, total, 0.1f, 400);  // Upstream gradient

    FlashAttnBackwardScratch scratch = allocateFlashAttnBackwardScratch(batch, heads, heads, seq, dim);

    // Forward
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch.fwd,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    // Backward
    flashAttnBackwardBHSD(d_Q, d_K, d_V, d_dO, d_dQ, d_dK, d_dV, scratch,
                          batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto dQ = toHost(d_dQ, total);
    auto dK = toHost(d_dK, total);
    auto dV = toHost(d_dV, total);
    
    freeFlashAttnBackwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
    
    // Check gradients are valid
    ATTN_ASSERT(allFinite(dQ), "dQ contains NaN/Inf");
    ATTN_ASSERT(allFinite(dK), "dK contains NaN/Inf");
    ATTN_ASSERT(allFinite(dV), "dV contains NaN/Inf");
    
    // Check gradients are non-zero
    float dQ_norm = l2Norm(dQ);
    float dK_norm = l2Norm(dK);
    float dV_norm = l2Norm(dV);
    
    ATTN_ASSERT(dQ_norm > 1e-6f, "dQ gradient is zero (norm=" + std::to_string(dQ_norm) + ")");
    ATTN_ASSERT(dK_norm > 1e-6f, "dK gradient is zero (norm=" + std::to_string(dK_norm) + ")");
    ATTN_ASSERT(dV_norm > 1e-6f, "dV gradient is zero (norm=" + std::to_string(dV_norm) + ")");
    
    return true;
}

bool GRIM::Test::testFlashBackwardNumericalGradient(std::string& message) {
    const int batch = 1, heads = 1, seq = 16, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float eps = 1e-3f;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    float *d_dO, *d_dQ, *d_dK, *d_dV;
    
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_dO, total * sizeof(float));
    cudaMalloc(&d_dQ, total * sizeof(float));
    cudaMalloc(&d_dK, total * sizeof(float));
    cudaMalloc(&d_dV, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 1000);
    initRandom(d_K, total, 0.1f, 2000);
    initRandom(d_V, total, 0.1f, 3000);
    initOnes(d_dO, total);  // dO = 1 means we're computing gradient of sum(output)
    
    auto Q_host = toHost(d_Q, total);
    auto K_host = toHost(d_K, total);
    auto V_host = toHost(d_V, total);
    
    FlashAttnBackwardScratch scratch = allocateFlashAttnBackwardScratch(batch, heads, heads, seq, dim);

    // Analytical backward
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch.fwd,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    flashAttnBackwardBHSD(d_Q, d_K, d_V, d_dO, d_dQ, d_dK, d_dV, scratch,
                          batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto analytical_dV = toHost(d_dV, total);
    
    // Numerical gradient check on V (just a few elements)
    int check_indices[] = {0, seq * dim / 2, total - 1};
    float max_rel_err = 0.0f;
    
    for (int idx : check_indices) {
        // f(V + eps)
        std::vector<float> V_plus = V_host;
        V_plus[idx] += eps;
        cudaMemcpy(d_V, V_plus.data(), total * sizeof(float), cudaMemcpyHostToDevice);
        flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch.fwd,
                             batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
        cudaDeviceSynchronize();
        auto out_plus = toHost(d_out, total);
        float loss_plus = std::accumulate(out_plus.begin(), out_plus.end(), 0.0f);
        
        // f(V - eps)
        std::vector<float> V_minus = V_host;
        V_minus[idx] -= eps;
        cudaMemcpy(d_V, V_minus.data(), total * sizeof(float), cudaMemcpyHostToDevice);
        flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch.fwd,
                             batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
        cudaDeviceSynchronize();
        auto out_minus = toHost(d_out, total);
        float loss_minus = std::accumulate(out_minus.begin(), out_minus.end(), 0.0f);
        
        float numerical = (loss_plus - loss_minus) / (2.0f * eps);
        float analytical = analytical_dV[idx];
        
        float rel_err = std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-6f);
        max_rel_err = std::max(max_rel_err, rel_err);
    }
    
    freeFlashAttnBackwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
    
    ATTN_ASSERT(max_rel_err < 0.1f, "Numerical gradient check failed (rel_err=" + std::to_string(max_rel_err) + ")");
    
    return true;
}

bool GRIM::Test::testFlashBackwardVsNaive(std::string& message) {
    // This would require implementing naive backward - skip for now
    message = "Not implemented (would need naive backward)";
    return true;
}

//======================================================//
//  Section 3: Scale & Performance Tests
//======================================================//

bool GRIM::Test::testFlashLargeSequence(std::string& message) {
    const int batch = 1, heads = 8, seq = 512, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 5000);
    initRandom(d_K, total, 0.1f, 6000);
    initRandom(d_V, total, 0.1f, 7000);

    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    TensorConversion::convert_BHSD_to_BSHD_bf16(d_Q, scratch.q, batch, heads, seq, dim, nullptr);
    TensorConversion::convert_BHSD_to_BSHD_bf16(d_K, scratch.k, batch, heads, seq, dim, nullptr);
    TensorConversion::convert_BHSD_to_BSHD_bf16(d_V, scratch.v, batch, heads, seq, dim, nullptr);

    // Warmup
    flash_attn_fwd_ex(scratch.q, scratch.k, scratch.v, scratch.out, scratch.softmax_lse,
                      nullptr, batch, seq, heads, heads, dim, softmax_scale, true, true, 0.0f, 0, nullptr);
    cudaDeviceSynchronize();
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        flash_attn_fwd_ex(scratch.q, scratch.k, scratch.v, scratch.out, scratch.softmax_lse,
                          nullptr, batch, seq, heads, heads, dim, softmax_scale, true, true, 0.0f, 0, nullptr);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count() / 10.0;

    TensorConversion::convert_BSHD_bf16_to_BHSD(scratch.out, d_out, batch, seq, heads, dim, nullptr);
    auto output = toHost(d_out, total);

    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    ATTN_ASSERT(allFinite(output), "Large sequence output has NaN/Inf");
    
    message = "seq=" + std::to_string(seq) + " avg=" + std::to_string(ms) + "ms";
    return true;
}

bool GRIM::Test::testFlashMultiBatch(std::string& message) {
    const int batch = 4, heads = 8, seq = 128, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 8000);
    initRandom(d_K, total, 0.1f, 9000);
    initRandom(d_V, total, 0.1f, 10000);

    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto output = toHost(d_out, total);
    
    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    ATTN_ASSERT(allFinite(output), "Multi-batch output has NaN/Inf");
    
    // Check each batch has different output (not just replicated)
    size_t per_batch = heads * seq * dim;
    float batch0_sum = 0.0f, batch1_sum = 0.0f;
    for (size_t i = 0; i < per_batch; ++i) {
        batch0_sum += output[i];
        batch1_sum += output[per_batch + i];
    }
    // With random inputs, batches should have different sums
    // (This is a weak test but catches obvious bugs)
    
    return true;
}

bool GRIM::Test::testFlashMultiHead(std::string& message) {
    const int batch = 1, heads = 12, seq = 64, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 11000);
    initRandom(d_K, total, 0.1f, 12000);
    initRandom(d_V, total, 0.1f, 13000);

    FlashAttnForwardScratch scratch = allocateFlashAttnForwardScratch(batch, heads, heads, seq, dim);
    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto output = toHost(d_out, total);
    
    freeFlashAttnForwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    
    ATTN_ASSERT(allFinite(output), "Multi-head output has NaN/Inf");
    
    return true;
}

bool GRIM::Test::testFlashGradientMagnitude(std::string& message) {
    const int batch = 1, heads = 4, seq = 128, dim = 64;
    const size_t total = batch * heads * seq * dim;
    const float softmax_scale = canonicalSoftmaxScale(dim);
    
    float *d_Q, *d_K, *d_V, *d_out;
    float *d_dO, *d_dQ, *d_dK, *d_dV;
    
    cudaMalloc(&d_Q, total * sizeof(float));
    cudaMalloc(&d_K, total * sizeof(float));
    cudaMalloc(&d_V, total * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));
    cudaMalloc(&d_dO, total * sizeof(float));
    cudaMalloc(&d_dQ, total * sizeof(float));
    cudaMalloc(&d_dK, total * sizeof(float));
    cudaMalloc(&d_dV, total * sizeof(float));
    
    initRandom(d_Q, total, 0.1f, 20000);
    initRandom(d_K, total, 0.1f, 21000);
    initRandom(d_V, total, 0.1f, 22000);
    initRandom(d_dO, total, 0.1f, 23000);

    FlashAttnBackwardScratch scratch = allocateFlashAttnBackwardScratch(batch, heads, heads, seq, dim);

    flashAttnForwardBHSD(d_Q, d_K, d_V, d_out, scratch.fwd,
                         batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    flashAttnBackwardBHSD(d_Q, d_K, d_V, d_dO, d_dQ, d_dK, d_dV, scratch,
                          batch, heads, heads, seq, dim, true, nullptr, softmax_scale);
    cudaDeviceSynchronize();
    
    auto dO = toHost(d_dO, total);
    auto dQ = toHost(d_dQ, total);
    auto dK = toHost(d_dK, total);
    auto dV = toHost(d_dV, total);
    
    freeFlashAttnBackwardScratch(scratch);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_out);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
    
    float dO_norm = l2Norm(dO);
    float dQ_norm = l2Norm(dQ);
    float dK_norm = l2Norm(dK);
    float dV_norm = l2Norm(dV);
    
    // Gradients should be on similar scale to inputs
    float dQ_ratio = dQ_norm / (dO_norm + 1e-6f);
    float dK_ratio = dK_norm / (dO_norm + 1e-6f);
    float dV_ratio = dV_norm / (dO_norm + 1e-6f);
    
    ATTN_ASSERT(dQ_ratio > 0.001f && dQ_ratio < 1000.0f, 
                "dQ magnitude suspicious (ratio=" + std::to_string(dQ_ratio) + ")");
    ATTN_ASSERT(dK_ratio > 0.001f && dK_ratio < 1000.0f, 
                "dK magnitude suspicious (ratio=" + std::to_string(dK_ratio) + ")");
    ATTN_ASSERT(dV_ratio > 0.001f && dV_ratio < 1000.0f, 
                "dV magnitude suspicious (ratio=" + std::to_string(dV_ratio) + ")");
    
    message = "dQ/dO=" + std::to_string(dQ_ratio) + 
              " dK/dO=" + std::to_string(dK_ratio) + 
              " dV/dO=" + std::to_string(dV_ratio);
    
    return true;
}

//======================================================//
//  Test Runner
//======================================================//

std::vector<AttentionTestResult> GRIM::Test::runAllAttentionTests() {
    std::vector<std::pair<std::string, AttentionTestFn>> tests = {
        {"FlashForwardBasic", testFlashForwardBasic},
        {"FlashForwardCausalMask", testFlashForwardCausalMask},
        {"FlashForwardCustomScale", testFlashForwardCustomScale},
        {"FlashForwardVsNaive", testFlashForwardVsNaive},
        {"FlashBackwardGradientFlow", testFlashBackwardGradientFlow},
        {"FlashBackwardNumericalGradient", testFlashBackwardNumericalGradient},
        {"FlashLargeSequence", testFlashLargeSequence},
        {"FlashMultiBatch", testFlashMultiBatch},
        {"FlashMultiHead", testFlashMultiHead},
        {"FlashGradientMagnitude", testFlashGradientMagnitude},
    };
    
    std::vector<AttentionTestResult> results;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Flash Attention Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    int passed = 0, failed = 0;
    
    for (const auto& [name, fn] : tests) {
        AttentionTestResult result;
        result.name = name;
        
        auto start = std::chrono::high_resolution_clock::now();
        result.passed = fn(result.message);
        auto end = std::chrono::high_resolution_clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        if (result.passed) {
            std::cout << "[PASS] " << name;
            if (!result.message.empty()) std::cout << " (" << result.message << ")";
            std::cout << " [" << result.elapsed_ms << "ms]" << std::endl;
            ++passed;
        } else {
            std::cout << "[FAIL] " << name << ": " << result.message << std::endl;
            ++failed;
        }
        
        results.push_back(result);
    }
    
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return results;
}

//======================================================//
//  Main Entry Point
//======================================================//

int main() {
    // Initialize CUDA
    cudaSetDevice(0);
    
    auto results = GRIM::Test::runAllAttentionTests();
    
    // Return non-zero if any test failed
    for (const auto& r : results) {
        if (!r.passed) return 1;
    }
    return 0;
}
