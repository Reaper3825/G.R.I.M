#define USE_CUDA

#include "Flash_attention_gpu.hpp"
#include "../../Shared/TensorConversion/TensorConversion.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace GRIM {

namespace {

size_t calcQElems(int batch, int heads, int seq, int head_dim) {
    return static_cast<size_t>(batch) * heads * seq * head_dim;
}

size_t calcLseElems(int batch, int heads, int seq) {
    return static_cast<size_t>(batch) * heads * seq;
}

void* allocOrThrow(size_t bytes, const char* name) {
    if (bytes == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FlashAttn] cudaMalloc failed for ") +
                                 name + ": " + cudaGetErrorString(err));
    }
    return ptr;
}

} // namespace

FlashAttentionLayer::FlashAttentionLayer() = default;

FlashAttentionLayer::~FlashAttentionLayer() {
    releaseScratch();
}

FlashAttentionLayer::FlashAttentionLayer(const FlashAttentionConfig& config)
    : config_(config) {}

FlashAttentionLayer::FlashAttentionLayer(const Dimensions& dims,
                                         const FlashAttentionConfig& config)
    : Layer(dims), config_(config) {}

void FlashAttentionLayer::ensureScratch(int batch, int heads, int kv_heads, int seq, int head_dim) {
    const size_t q_elems = calcQElems(batch, heads, seq, head_dim);
    const size_t kv_elems = calcQElems(batch, kv_heads, seq, head_dim);
    const size_t lse_elems = calcLseElems(batch, heads, seq);
    const size_t dq_accum_bytes = flash_attn_dq_accum_bytes(batch, seq, heads, head_dim);
    const size_t dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(batch, seq, heads);

    if (q_elems <= fa_q_elems_ &&
        kv_elems <= fa_kv_elems_ &&
        lse_elems <= fa_lse_elems_ &&
        dq_accum_bytes <= fa_dq_accum_bytes_ &&
        dsoftmax_sum_bytes <= fa_dsoftmax_sum_bytes_) {
        return;
    }

    releaseScratch();

    fa_q_bf16_ = allocOrThrow(q_elems * sizeof(__nv_bfloat16), "fa_q_bf16");
    fa_k_bf16_ = allocOrThrow(kv_elems * sizeof(__nv_bfloat16), "fa_k_bf16");
    fa_v_bf16_ = allocOrThrow(kv_elems * sizeof(__nv_bfloat16), "fa_v_bf16");
    fa_out_bf16_ = allocOrThrow(q_elems * sizeof(__nv_bfloat16), "fa_out_bf16");
    fa_dout_bf16_ = allocOrThrow(q_elems * sizeof(__nv_bfloat16), "fa_dout_bf16");
    fa_dq_bf16_ = allocOrThrow(q_elems * sizeof(__nv_bfloat16), "fa_dq_bf16");
    fa_dk_bf16_ = allocOrThrow(kv_elems * sizeof(__nv_bfloat16), "fa_dk_bf16");
    fa_dv_bf16_ = allocOrThrow(kv_elems * sizeof(__nv_bfloat16), "fa_dv_bf16");
    fa_softmax_lse_ = static_cast<float*>(
        allocOrThrow(lse_elems * sizeof(float), "fa_softmax_lse"));
    fa_dq_accum_ = allocOrThrow(dq_accum_bytes, "fa_dq_accum");
    fa_dsoftmax_sum_ = allocOrThrow(dsoftmax_sum_bytes, "fa_dsoftmax_sum");

    fa_q_elems_ = q_elems;
    fa_kv_elems_ = kv_elems;
    fa_lse_elems_ = lse_elems;
    fa_dq_accum_bytes_ = dq_accum_bytes;
    fa_dsoftmax_sum_bytes_ = dsoftmax_sum_bytes;
}

void FlashAttentionLayer::releaseScratch() {
    if (fa_q_bf16_) cudaFree(fa_q_bf16_);
    if (fa_k_bf16_) cudaFree(fa_k_bf16_);
    if (fa_v_bf16_) cudaFree(fa_v_bf16_);
    if (fa_out_bf16_) cudaFree(fa_out_bf16_);
    if (fa_dout_bf16_) cudaFree(fa_dout_bf16_);
    if (fa_dq_bf16_) cudaFree(fa_dq_bf16_);
    if (fa_dk_bf16_) cudaFree(fa_dk_bf16_);
    if (fa_dv_bf16_) cudaFree(fa_dv_bf16_);
    if (fa_softmax_lse_) cudaFree(fa_softmax_lse_);
    if (fa_dq_accum_) cudaFree(fa_dq_accum_);
    if (fa_dsoftmax_sum_) cudaFree(fa_dsoftmax_sum_);

    fa_q_bf16_ = nullptr;
    fa_k_bf16_ = nullptr;
    fa_v_bf16_ = nullptr;
    fa_out_bf16_ = nullptr;
    fa_dout_bf16_ = nullptr;
    fa_dq_bf16_ = nullptr;
    fa_dk_bf16_ = nullptr;
    fa_dv_bf16_ = nullptr;
    fa_softmax_lse_ = nullptr;
    fa_dq_accum_ = nullptr;
    fa_dsoftmax_sum_ = nullptr;
    fa_q_elems_ = 0;
    fa_kv_elems_ = 0;
    fa_lse_elems_ = 0;
    fa_dq_accum_bytes_ = 0;
    fa_dsoftmax_sum_bytes_ = 0;
    fa_fwd_valid_ = false;
}

void FlashAttentionLayer::forward(const FlashAttentionForwardArgs& args,
                                  LayerWorkspace<float>*) {
    auto cfg = makeConfig(args);

    if (!args.Q || !args.K || !args.V || !args.output) {
        throw std::runtime_error("[FlashAttn] forward requires Q/K/V and output buffers");
    }
    if (cfg.softmax_temperature != 1.0f) {
        throw std::runtime_error("[FlashAttn] softmax_temperature must be 1.0 for flash_attn_fwd_ex");
    }
    if (cfg.qk_norm_enabled) {
        throw std::runtime_error("[FlashAttn] qk_norm is not supported with flash_attn_fwd_ex");
    }

    ensureScratch(cfg.batch_size, cfg.num_heads, cfg.num_kv_heads, cfg.seq_len, cfg.head_dim);

    auto* fa_q = reinterpret_cast<__nv_bfloat16*>(fa_q_bf16_);
    auto* fa_k = reinterpret_cast<__nv_bfloat16*>(fa_k_bf16_);
    auto* fa_v = reinterpret_cast<__nv_bfloat16*>(fa_v_bf16_);
    auto* fa_out = reinterpret_cast<__nv_bfloat16*>(fa_out_bf16_);

    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.Q, fa_q, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.K, fa_k, cfg.batch_size, cfg.num_kv_heads, cfg.seq_len, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.V, fa_v, cfg.batch_size, cfg.num_kv_heads, cfg.seq_len, cfg.head_dim, cfg.stream);

    // ISSUE #67: Log Flash Attention FORWARD INPUTS
    {
        cudaStreamSynchronize(cfg.stream);
        const size_t q_elems = cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.head_dim;
        const size_t kv_elems = cfg.batch_size * cfg.num_kv_heads * cfg.seq_len * cfg.head_dim;
        std::vector<float> h_q(std::min(q_elems, size_t(100)));
        std::vector<float> h_k(std::min(kv_elems, size_t(100)));
        std::vector<float> h_v(std::min(kv_elems, size_t(100)));
        cudaMemcpy(h_q.data(), args.Q, h_q.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_k.data(), args.K, h_k.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v.data(), args.V, h_v.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int nan_q = 0, inf_q = 0, nan_k = 0, inf_k = 0, nan_v = 0, inf_v = 0;
        for (float val : h_q) { if (std::isnan(val)) nan_q++; if (std::isinf(val)) inf_q++; }
        for (float val : h_k) { if (std::isnan(val)) nan_k++; if (std::isinf(val)) inf_k++; }
        for (float val : h_v) { if (std::isnan(val)) nan_v++; if (std::isinf(val)) inf_v++; }
        fprintf(stderr, "[FA-FWD-IN] Q: nan=%d inf=%d first=%.6f | K: nan=%d inf=%d first=%.6f | V: nan=%d inf=%d first=%.6f\n",
                nan_q, inf_q, h_q.empty() ? 0.0f : h_q[0],
                nan_k, inf_k, h_k.empty() ? 0.0f : h_k[0],
                nan_v, inf_v, h_v.empty() ? 0.0f : h_v[0]);
    }

    flash_attn_fwd_ex(
        fa_q,
        fa_k,
        fa_v,
        fa_out,
        fa_softmax_lse_,
        nullptr,
        cfg.batch_size,
        cfg.seq_len,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.causal,
        true,
        0.0f,  // No attention dropout in inference path
        0,     // No dropout seed needed
        cfg.stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(
        fa_out, args.output, cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.head_dim, cfg.stream);

    // ISSUE #67: Log Flash Attention FORWARD OUTPUT
    {
        cudaStreamSynchronize(cfg.stream);
        const size_t out_elems = cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.head_dim;
        std::vector<float> h_out(std::min(out_elems, size_t(100)));
        cudaMemcpy(h_out.data(), args.output, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int nan_out = 0, inf_out = 0;
        if (DebugFlashFwd) {
        for (float val : h_out) { if (std::isnan(val)) nan_out++; if (std::isinf(val)) inf_out++; }
        fprintf(stderr, "[FA-FWD-OUT] output: nan=%d inf=%d first=%.6f\n",
                nan_out, inf_out, h_out.empty() ? 0.0f : h_out[0]);
        }
    }

    fa_fwd_valid_ = true;
    fa_last_batch_ = cfg.batch_size;
    fa_last_heads_ = cfg.num_heads;
    fa_last_kv_heads_ = cfg.num_kv_heads;
    fa_last_seq_ = cfg.seq_len;
    fa_last_head_dim_ = cfg.head_dim;
}

void FlashAttentionLayer::backward(const FlashAttentionBackwardArgs& args,
                                   LayerWorkspace<float>*) {
    auto cfg = makeConfig(args);

    if (!args.Q || !args.K || !args.V || !args.grad_output ||
        !args.grad_Q || !args.grad_K || !args.grad_V) {
        throw std::runtime_error("[FlashAttn] backward requires Q/K/V, grad_output, and grad_Q/K/V buffers");
    }
    if (!fa_fwd_valid_) {
        throw std::runtime_error("[FlashAttn] backward requires a matching forward pass");
    }
    if (cfg.batch_size != fa_last_batch_ || cfg.num_heads != fa_last_heads_ ||
        cfg.num_kv_heads != fa_last_kv_heads_ || cfg.seq_len != fa_last_seq_ ||
        cfg.head_dim != fa_last_head_dim_) {
        throw std::runtime_error("[FlashAttn] backward dims do not match cached forward");
    }
    if (cfg.softmax_temperature != 1.0f) {
        throw std::runtime_error("[FlashAttn] softmax_temperature must be 1.0 for flash_attn_bwd_ex");
    }
    if (cfg.qk_norm_enabled) {
        throw std::runtime_error("[FlashAttn] qk_norm is not supported with flash_attn_bwd_ex");
    }

    ensureScratch(cfg.batch_size, cfg.num_heads, cfg.num_kv_heads, cfg.seq_len, cfg.head_dim);

    auto* fa_q = reinterpret_cast<__nv_bfloat16*>(fa_q_bf16_);
    auto* fa_k = reinterpret_cast<__nv_bfloat16*>(fa_k_bf16_);
    auto* fa_v = reinterpret_cast<__nv_bfloat16*>(fa_v_bf16_);
    auto* fa_out = reinterpret_cast<__nv_bfloat16*>(fa_out_bf16_);
    auto* fa_dout = reinterpret_cast<__nv_bfloat16*>(fa_dout_bf16_);
    auto* fa_dq = reinterpret_cast<__nv_bfloat16*>(fa_dq_bf16_);
    auto* fa_dk = reinterpret_cast<__nv_bfloat16*>(fa_dk_bf16_);
    auto* fa_dv = reinterpret_cast<__nv_bfloat16*>(fa_dv_bf16_);

    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.Q, fa_q, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.K, fa_k, cfg.batch_size, cfg.num_kv_heads, cfg.seq_len, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.V, fa_v, cfg.batch_size, cfg.num_kv_heads, cfg.seq_len, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        args.grad_output, fa_dout, cfg.batch_size, cfg.num_heads, cfg.seq_len, cfg.head_dim, cfg.stream);

    // ISSUE #67: Log Flash Attention BACKWARD INPUT (grad_output)
    {
        cudaStreamSynchronize(cfg.stream);
        const size_t grad_elems = cfg.batch_size * cfg.num_heads * cfg.seq_len * cfg.head_dim;
        std::vector<float> h_grad_out(std::min(grad_elems, size_t(100)));
        cudaMemcpy(h_grad_out.data(), args.grad_output, h_grad_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        int nan_grad = 0, inf_grad = 0;
        for (float val : h_grad_out) { if (std::isnan(val)) nan_grad++; if (std::isinf(val)) inf_grad++; }
        fprintf(stderr, "[FA-BWD-IN] grad_output: nan=%d inf=%d first=%.10f\n",
                nan_grad, inf_grad, h_grad_out.empty() ? 0.0f : h_grad_out[0]);
    }

    flash_attn_bwd_ex(
        fa_q,
        fa_k,
        fa_v,
        fa_out,
        fa_dout,
        fa_softmax_lse_,
        nullptr,
        fa_dq,
        fa_dk,
        fa_dv,
        fa_dq_accum_,
        fa_dsoftmax_sum_,
        cfg.batch_size,
        cfg.seq_len,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.causal,
        true,
        0.0f,  // No attention dropout in inference path
        0,     // No dropout seed needed
        cfg.stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(
        fa_dq, args.grad_Q, cfg.batch_size, cfg.seq_len, cfg.num_heads, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(
        fa_dk, args.grad_K, cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim, cfg.stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(
        fa_dv, args.grad_V, cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_dim, cfg.stream);

    fa_fwd_valid_ = false;
}

FlashAttentionConfig FlashAttentionLayer::makeConfig(
    const FlashAttentionForwardArgs& args) const {
    if (args.batch_size == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] batch_size is 0 - caller MUST provide valid batch_size");
    }
    if (args.num_heads == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] num_heads is 0 - caller MUST provide valid num_heads");
    }
    if (args.seq_len == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] seq_len is 0 - caller MUST provide valid seq_len");
    }
    if (args.head_dim == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] head_dim is 0 - caller MUST provide valid head_dim");
    }
    if (args.stream == nullptr) {
        fprintf(stderr, "[FlashAttn makeConfig] FATAL: stream is NULL (default stream usage disallowed)\n");
        throw std::runtime_error("[FlashAttn makeConfig] stream is NULL - caller MUST provide valid CUDA stream");
    }

    const int kv_heads = (args.num_kv_heads > 0) ? args.num_kv_heads : args.num_heads;
    if (kv_heads <= 0) {
        throw std::runtime_error("[FlashAttn makeConfig] num_kv_heads is invalid");
    }
    if (args.num_heads % kv_heads != 0) {
        throw std::runtime_error("[FlashAttn makeConfig] num_heads must be divisible by num_kv_heads");
    }

    FlashAttentionConfig cfg = config_;
    cfg.batch_size = args.batch_size;
    cfg.num_heads = args.num_heads;
    cfg.num_kv_heads = kv_heads;
    cfg.seq_len = args.seq_len;
    cfg.head_dim = args.head_dim;
    cfg.causal = args.causal;
    cfg.stream = args.stream;
    return cfg;
}

FlashAttentionConfig FlashAttentionLayer::makeConfig(
    const FlashAttentionBackwardArgs& args) const {
    if (args.batch_size == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] batch_size is 0 - caller MUST provide valid batch_size");
    }
    if (args.num_heads == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] num_heads is 0 - caller MUST provide valid num_heads");
    }
    if (args.seq_len == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] seq_len is 0 - caller MUST provide valid seq_len");
    }
    if (args.head_dim == 0) {
        throw std::runtime_error("[FlashAttn makeConfig] head_dim is 0 - caller MUST provide valid head_dim");
    }
    if (args.stream == nullptr) {
        fprintf(stderr, "[FlashAttn makeConfig] FATAL: stream is NULL (default stream usage disallowed)\n");
        throw std::runtime_error("[FlashAttn makeConfig] stream is NULL - caller MUST provide valid CUDA stream");
    }

    const int kv_heads = (args.num_kv_heads > 0) ? args.num_kv_heads : args.num_heads;
    if (kv_heads <= 0) {
        throw std::runtime_error("[FlashAttn makeConfig] num_kv_heads is invalid");
    }
    if (args.num_heads % kv_heads != 0) {
        throw std::runtime_error("[FlashAttn makeConfig] num_heads must be divisible by num_kv_heads");
    }

    FlashAttentionConfig cfg = config_;
    cfg.batch_size = args.batch_size;
    cfg.num_heads = args.num_heads;
    cfg.num_kv_heads = kv_heads;
    cfg.seq_len = args.seq_len;
    cfg.head_dim = args.head_dim;
    cfg.causal = args.causal;
    cfg.stream = args.stream;
    return cfg;
}

} // namespace GRIM
