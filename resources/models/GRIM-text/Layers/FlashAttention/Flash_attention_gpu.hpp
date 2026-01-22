#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "../../Common/grim_layer_gpu.hpp"
#include "Flash_Attention_Kernal.hpp"

namespace GRIM {

struct FlashAttentionConfig {
    int batch_size = 1;
    int num_heads = 0;
    int num_kv_heads = 0;
    int seq_len = 0;
    int head_dim = 0;
    bool causal = true;
    float softmax_temperature = 1.0f;
    bool qk_norm_enabled = false;
    cudaStream_t stream = nullptr;
};

struct FlashAttentionForwardArgs {
    const float* Q = nullptr;
    const float* K = nullptr;
    const float* V = nullptr;
    float* output = nullptr;
    int batch_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int seq_len = 0;
    int head_dim = 0;
    bool causal = true;
    cudaStream_t stream = nullptr;
};

struct FlashAttentionBackwardArgs {
    const float* Q = nullptr;
    const float* K = nullptr;
    const float* V = nullptr;
    const float* output = nullptr;
    const float* grad_output = nullptr;
    float* grad_Q = nullptr;
    float* grad_K = nullptr;
    float* grad_V = nullptr;
    int batch_size = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int seq_len = 0;
    int head_dim = 0;
    bool causal = true;
    cudaStream_t stream = nullptr;
};

class FlashAttentionLayer final : public Layer<FlashAttentionLayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kAttention;

    FlashAttentionLayer();
    ~FlashAttentionLayer();
    explicit FlashAttentionLayer(const FlashAttentionConfig& config);
    FlashAttentionLayer(const Dimensions& dims, const FlashAttentionConfig& config);

    void setConfig(const FlashAttentionConfig& cfg) { config_ = cfg; }
    const FlashAttentionConfig& config() const noexcept { return config_; }

    void forward(const FlashAttentionForwardArgs& args,
                 LayerWorkspace<float>* workspace = nullptr);
    void backward(const FlashAttentionBackwardArgs& args,
                  LayerWorkspace<float>* workspace = nullptr);

private:
    FlashAttentionConfig config_{};
    void* fa_q_bf16_ = nullptr;
    void* fa_k_bf16_ = nullptr;
    void* fa_v_bf16_ = nullptr;
    void* fa_out_bf16_ = nullptr;
    void* fa_dout_bf16_ = nullptr;
    void* fa_dq_bf16_ = nullptr;
    void* fa_dk_bf16_ = nullptr;
    void* fa_dv_bf16_ = nullptr;
    float* fa_softmax_lse_ = nullptr;
    void* fa_dq_accum_ = nullptr;
    void* fa_dsoftmax_sum_ = nullptr;
    size_t fa_q_elems_ = 0;
    size_t fa_kv_elems_ = 0;
    size_t fa_lse_elems_ = 0;
    size_t fa_dq_accum_bytes_ = 0;
    size_t fa_dsoftmax_sum_bytes_ = 0;
    int fa_last_batch_ = 0;
    int fa_last_heads_ = 0;
    int fa_last_kv_heads_ = 0;
    int fa_last_seq_ = 0;
    int fa_last_head_dim_ = 0;
    bool fa_fwd_valid_ = false;

    FlashAttentionConfig makeConfig(const FlashAttentionForwardArgs& args) const;
    FlashAttentionConfig makeConfig(const FlashAttentionBackwardArgs& args) const;
    void ensureScratch(int batch, int heads, int kv_heads, int seq, int head_dim);
    void releaseScratch();
};

} // namespace GRIM
