#pragma once

#include <cuda_runtime.h>

extern "C" {
void launchIncrementalAttentionScoresFull(
    const float* Q,
    const float* K,
    float* scores,
    const float* alpha_q,
    const float* alpha_k,
    const float* alibi_slopes,
    const float* rope_inv_freq,
    int num_heads,
    int num_kv_heads,
    int d_head,
    int kv_len,
    int max_kv_len,
    int query_pos,
    int rotary_dim,
    float base_scale,
    float inv_temperature,
    bool qk_norm_enabled,
    cudaStream_t stream);

void launchIncrementalSoftmax(
    float* scores,
    int num_heads,
    int kv_len,
    cudaStream_t stream);

void launchIncrementalAttentionOutputGQA(
    const float* scores,
    const float* V,
    float* output,
    int num_heads,
    int num_kv_heads,
    int d_head,
    int kv_len,
    int max_kv_len,
    cudaStream_t stream);

void launchAppendKVCacheGQA(
    const float* new_kv,
    float* cache,
    int num_kv_heads,
    int d_head,
    int max_kv_len,
    int pos,
    cudaStream_t stream);

void launchSingleTokenRMSNorm(
    const float* input,
    const float* gamma,
    float* output,
    int d_model,
    float eps,
    cudaStream_t stream);

void launchSingleTokenResidual(
    const float* a,
    const float* b,
    float* output,
    int d_model,
    cudaStream_t stream);

void launchSingleTokenGELU(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream);
}
