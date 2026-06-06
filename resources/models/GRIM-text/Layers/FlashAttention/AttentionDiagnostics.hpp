#pragma once

#include <cuda_runtime.h>

#include <vector>

namespace GRIM::FlashAttentionDiagnostics {

struct TensorStrideTriplet {
    long long batch = 0;
    long long row = 0;
    long long head = 0;
};

struct BackwardStrideLayout {
    TensorStrideTriplet q;
    TensorStrideTriplet k;
    TensorStrideTriplet v;
    TensorStrideTriplet o;
    TensorStrideTriplet dO;
    TensorStrideTriplet dQ;
    TensorStrideTriplet dK;
    TensorStrideTriplet dV;
};

struct ForwardDiagnosticRequest {
    const void* q = nullptr;
    const void* k = nullptr;
    const void* v = nullptr;
    const void* out = nullptr;
    const void* softmax_lse = nullptr;
    const float* alibi_slopes = nullptr;
    int batch = 0;
    int seqlen = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    float softmax_scale = 0.0f;
    bool is_bf16 = false;
    cudaStream_t stream = nullptr;
};

struct BackwardDiagnosticRequest {
    const void* dout = nullptr;
    const void* softmax_lse = nullptr;
    const float* alibi_slopes = nullptr;
    const void* dq = nullptr;
    const void* dk = nullptr;
    const void* dv = nullptr;
    BackwardStrideLayout strides{};
    int batch = 0;
    int seqlen = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    bool is_bf16 = false;
    cudaStream_t stream = nullptr;
};

void emitForwardPreKernelDiagnostics(const ForwardDiagnosticRequest& request);
void emitForwardPostKernelDiagnostics(const ForwardDiagnosticRequest& request);
void emitBackwardPreKernelDiagnostics(const BackwardDiagnosticRequest& request);
void emitBackwardPostKernelDiagnostics(const BackwardDiagnosticRequest& request);
void emitAttentionBreadthEquation(const std::vector<float>& causal_scores_row,
                                  float lse_value,
                                  int query_index,
                                  int layer_idx,
                                  int head_idx);

}  // namespace GRIM::FlashAttentionDiagnostics

