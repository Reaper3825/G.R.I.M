//======================================================//
//  EncoderSelfAttention_GPU.cu
//  Attention-owned encoder self-attention forward facade.
//======================================================//

#include "EncoderSelfAttention_GPU.hpp"

#include "../../Shared/TensorContract/AutogradQKVDiagnostics.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    constexpr bool kEnableAttentionStepLogs = true;

    void validateWeights(const GRIM::Attention::EncoderSelfAttentionWeights& weights,
                         const GRIM::HyperParameters::EncoderSelfAttentionHP& hp) {
        if (!weights.W_qkv.data) {
            throw std::runtime_error("encoderSelfAttentionForward: W_qkv.data is NULL");
        }
        if (!weights.W_o.data) {
            throw std::runtime_error("encoderSelfAttentionForward: W_o.data is NULL");
        }
        weights.W_qkv.shape.require("encoderSelfAttentionForward W_qkv");
        weights.W_o.shape.require("encoderSelfAttentionForward W_o");
        if (!weights.W_qkv.shape.is_2d_layout()) {
            throw std::runtime_error("encoderSelfAttentionForward: W_qkv must be a 2D [qkv_dim,d_model] tensor");
        }
        if (!weights.W_o.shape.is_2d_layout()) {
            throw std::runtime_error("encoderSelfAttentionForward: W_o must be a 2D [d_model,d_model] tensor");
        }
        const auto wqkv_shape = weights.W_qkv.shape.as_2d();
        if (wqkv_shape.rows != hp.qkv_dim || wqkv_shape.cols != hp.d_model) {
            throw std::runtime_error("encoderSelfAttentionForward: W_qkv shape mismatch. expected=[" +
                                     std::to_string(hp.qkv_dim) + "," + std::to_string(hp.d_model) +
                                     "] got=[" + std::to_string(wqkv_shape.rows) + "," +
                                     std::to_string(wqkv_shape.cols) + "]");
        }
        const auto wo_shape = weights.W_o.shape.as_2d();
        if (wo_shape.rows != hp.d_model || wo_shape.cols != hp.d_model) {
            throw std::runtime_error("encoderSelfAttentionForward: W_o shape mismatch. expected=[" +
                                     std::to_string(hp.d_model) + "," + std::to_string(hp.d_model) +
                                     "] got=[" + std::to_string(wo_shape.rows) + "," +
                                     std::to_string(wo_shape.cols) + "]");
        }
        if (hp.use_bias) {
            if (!weights.b_qkv.data) {
                throw std::runtime_error("encoderSelfAttentionForward: hp.use_bias=true but b_qkv.data is NULL");
            }
            if (!weights.b_o.data) {
                throw std::runtime_error("encoderSelfAttentionForward: hp.use_bias=true but b_o.data is NULL");
            }
            if (static_cast<int>(weights.b_qkv.numel()) != hp.qkv_dim) {
                throw std::runtime_error("encoderSelfAttentionForward: b_qkv numel mismatch. expected=" +
                                         std::to_string(hp.qkv_dim) + " got=" +
                                         std::to_string(weights.b_qkv.numel()));
            }
            if (static_cast<int>(weights.b_o.numel()) != hp.d_model) {
                throw std::runtime_error("encoderSelfAttentionForward: b_o numel mismatch. expected=" +
                                         std::to_string(hp.d_model) + " got=" +
                                         std::to_string(weights.b_o.numel()));
            }
        }
    }

    void validatePBMSpec(const GRIM::PBM::PBMSpec& pbm,
                         const GRIM::HyperParameters::EncoderSelfAttentionHP& hp) {
        if (!pbm.valid) {
            throw std::runtime_error("encoderSelfAttentionForward: PBM spec is not valid - GRIM requires RoPE+ALiBi");
        }
        if (!pbm.rope_inv_freq) {
            throw std::runtime_error("encoderSelfAttentionForward: PBM rope_inv_freq is NULL");
        }
        if (hp.rotary_dim <= 0 || hp.rotary_dim > hp.head_dim) {
            throw std::runtime_error("encoderSelfAttentionForward: invalid attention rotary_dim=" +
                                     std::to_string(hp.rotary_dim) + " for head_dim=" +
                                     std::to_string(hp.head_dim));
        }
        if (!pbm.alibi_slopes) {
            throw std::runtime_error("encoderSelfAttentionForward: PBM alibi_slopes is NULL");
        }
        if (!pbm.upload_event) {
            throw std::runtime_error("encoderSelfAttentionForward: PBM upload_event is NULL");
        }
    }

    std::uint64_t attentionDropoutSeed(const GRIM::Attention::EncoderSelfAttentionForwardRequest& request) {
        const float attention_dropout_p = request.dropout_enabled ? request.hp.attention_dropout : 0.0f;
        if (attention_dropout_p <= 0.0f) {
            return 0;
        }
        return request.batch_idx * 2654435761ULL + 42 +
               1000 * static_cast<std::uint64_t>(request.layer_idx);
    }
}  // namespace

namespace GRIM::Attention {

void encoderSelfAttentionForward(const Tensor& norm_input,
                                 EncoderSelfAttentionWeights weights,
                                 EncoderSelfAttentionIntermediates intermediates,
                                 const EncoderSelfAttentionForwardRequest& request) {
    if (!request.stream) {
        throw std::runtime_error("encoderSelfAttentionForward: stream is NULL");
    }
    if (!request.cublas_handle) {
        throw std::runtime_error("encoderSelfAttentionForward: cublas_handle is NULL");
    }
    if (!norm_input.data) {
        throw std::runtime_error("encoderSelfAttentionForward: norm_input.data is NULL");
    }
    norm_input.shape.require("encoderSelfAttentionForward norm_input");
    if (!norm_input.shape.is_2d_layout()) {
        throw std::runtime_error("encoderSelfAttentionForward: norm_input must be a 2D [total_tokens,d_model] tensor");
    }
    const auto norm_shape = norm_input.shape.as_2d();
    if (norm_shape.rows != request.payload.total_tokens || norm_shape.cols != request.hp.d_model) {
        throw std::runtime_error("encoderSelfAttentionForward: norm_input shape mismatch. expected=[" +
                                 std::to_string(request.payload.total_tokens) + "," +
                                 std::to_string(request.hp.d_model) + "] got=[" +
                                 std::to_string(norm_shape.rows) + "," +
                                 std::to_string(norm_shape.cols) + "]");
    }

    validateWeights(weights, request.hp);
    validatePBMSpec(request.pbm, request.hp);
    cudaError_t wait_err = cudaStreamWaitEvent(request.stream, request.pbm.upload_event, 0);
    if (wait_err != cudaSuccess) {
        throw std::runtime_error(std::string("encoderSelfAttentionForward: cudaStreamWaitEvent(PBM upload_event) failed: ") +
                                 cudaGetErrorString(wait_err));
    }
    autograd::set_autograd_cublas_handle(request.cublas_handle);

    const int qkv_debug = autograd::qkvDebugLevel();
    if constexpr (kEnableAttentionStepLogs) {
        std::fprintf(stderr, "[EncoderSelfAttention] START layer=%d tokens=%d heads=%d kv_heads=%d head_dim=%d\n",
                     request.layer_idx, request.payload.total_tokens, request.hp.num_heads,
                     request.hp.num_kv_heads, request.hp.head_dim);
    }

    if (qkv_debug >= 3) {
        autograd::checkQKVTensorFinite("AutogradQKV:ln1_out", norm_input, request.stream);
        autograd::checkQKVTensorFinite("AutogradQKV:W_qkv", weights.W_qkv, request.stream);
        if (request.hp.use_bias) {
            autograd::checkQKVTensorFinite("AutogradQKV:b_qkv", weights.b_qkv, request.stream);
        }
    }

    if constexpr (kEnableAttentionStepLogs) {
        std::fprintf(stderr, "[EncoderSelfAttention] QKV projection...\n");
    }
    intermediates.qkv_out = autograd::matmul(norm_input, weights.W_qkv, request.stream,
                                             norm_input.data, nullptr, true);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradQKV:qkv_out_prebias", intermediates.qkv_out, request.stream);
    }

    if (request.hp.use_bias) {
        intermediates.qkv_out = autograd::broadcast_add(intermediates.qkv_out, weights.b_qkv, request.stream);
    }
    autograd::logQKVProjectionEquation(
        norm_input, weights.W_qkv, weights.b_qkv, intermediates.qkv_out,
        request.payload, request.hp, request.stream, request.layer_idx);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradQKV:qkv_out", intermediates.qkv_out, request.stream);
    }

    auto [Q_bhsd_tmp, K_bhsd_tmp, V_bhsd_tmp] = autograd::split_and_reshape_qkv(
        intermediates.qkv_out,
        request.payload, request.hp,
        request.stream);
    intermediates.Q_bhsd = std::move(Q_bhsd_tmp);
    intermediates.K_bhsd = std::move(K_bhsd_tmp);
    intermediates.V_bhsd = std::move(V_bhsd_tmp);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradQKV:Q_bhsd", intermediates.Q_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradQKV:K_bhsd", intermediates.K_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradQKV:V_bhsd", intermediates.V_bhsd, request.stream);
    }

    auto [Q_rot, K_rot] = autograd::rope_rotation(
        intermediates.Q_bhsd, intermediates.K_bhsd,
        request.pbm.rope_inv_freq,
        request.payload, request.hp,
        request.hp.rotary_dim, request.stream);
    intermediates.Q_bhsd = std::move(Q_rot);
    intermediates.K_bhsd = std::move(K_rot);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradSDPA:Q_rope", intermediates.Q_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradSDPA:K_rope", intermediates.K_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradSDPA:V_rope", intermediates.V_bhsd, request.stream);
    }

    const float attention_dropout_p = request.dropout_enabled ? request.hp.attention_dropout : 0.0f;
    const std::uint64_t dropout_seed = attentionDropoutSeed(request);
    intermediates.attn_out_bhsd = autograd::scaled_dot_product_attention(
        intermediates.Q_bhsd, intermediates.K_bhsd, intermediates.V_bhsd,
        request.pbm.alibi_slopes, request.flash_attention, 0.0f, request.stream,
        attention_dropout_p, dropout_seed);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradSDPA:attn_out_bhsd", intermediates.attn_out_bhsd, request.stream);
    }

    intermediates.attn_out = autograd::reshape_bhsd_to_flat(
        intermediates.attn_out_bhsd, request.payload, request.hp, request.stream);

    if (!intermediates.attn_out.data) {
        throw std::runtime_error("encoderSelfAttentionForward: attn_out.data is NULL before output projection matmul");
    }
    intermediates.proj_out = autograd::matmul(intermediates.attn_out, weights.W_o, request.stream,
                                              intermediates.attn_out.data, nullptr, true);
    if (request.hp.use_bias) {
        intermediates.proj_out = autograd::broadcast_add(intermediates.proj_out, weights.b_o, request.stream);
    }

    if constexpr (kEnableAttentionStepLogs) {
        std::fprintf(stderr, "[EncoderSelfAttention] DONE layer=%d\n", request.layer_idx);
    }
}

}  // namespace GRIM::Attention
