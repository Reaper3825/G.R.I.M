//======================================================//
//  EncoderSelfAttention_GPU.cu
//  Attention-owned encoder self-attention forward facade.
//======================================================//

#include "EncoderSelfAttention_GPU.hpp"

#include "../Encoding/AblationFlags.hpp"
#include "../../Shared/TensorContract/AutogradQKVDiagnostics.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    constexpr bool kEnableAttentionStepLogs = true;

    void validateWeights(const GRIM::Tensor& W_qkv,
                         const GRIM::Tensor& b_qkv,
                         const GRIM::Tensor& W_o,
                         const GRIM::Tensor& b_o,
                         const GRIM::HyperParameters::EncoderSelfAttentionHP& hp) {
        if (!W_qkv.data) {
            throw std::runtime_error("encoderSelfAttentionForward: W_qkv.data is NULL");
        }
        if (!W_o.data) {
            throw std::runtime_error("encoderSelfAttentionForward: W_o.data is NULL");
        }
        W_qkv.shape.require("encoderSelfAttentionForward W_qkv");
        W_o.shape.require("encoderSelfAttentionForward W_o");
        if (!W_qkv.shape.is_2d_layout()) {
            throw std::runtime_error("encoderSelfAttentionForward: W_qkv must be a 2D [qkv_dim,d_model] tensor");
        }
        if (!W_o.shape.is_2d_layout()) {
            throw std::runtime_error("encoderSelfAttentionForward: W_o must be a 2D [d_model,d_model] tensor");
        }
        const auto wqkv_shape = W_qkv.shape.as_2d();
        if (wqkv_shape.rows != hp.qkv_dim || wqkv_shape.cols != hp.d_model) {
            throw std::runtime_error("encoderSelfAttentionForward: W_qkv shape mismatch. expected=[" +
                                     std::to_string(hp.qkv_dim) + "," + std::to_string(hp.d_model) +
                                     "] got=[" + std::to_string(wqkv_shape.rows) + "," +
                                     std::to_string(wqkv_shape.cols) + "]");
        }
        const auto wo_shape = W_o.shape.as_2d();
        if (wo_shape.rows != hp.d_model || wo_shape.cols != hp.d_model) {
            throw std::runtime_error("encoderSelfAttentionForward: W_o shape mismatch. expected=[" +
                                     std::to_string(hp.d_model) + "," + std::to_string(hp.d_model) +
                                     "] got=[" + std::to_string(wo_shape.rows) + "," +
                                     std::to_string(wo_shape.cols) + "]");
        }
        if (hp.use_bias) {
            if (!b_qkv.data) {
                throw std::runtime_error("encoderSelfAttentionForward: hp.use_bias=true but b_qkv.data is NULL");
            }
            if (!b_o.data) {
                throw std::runtime_error("encoderSelfAttentionForward: hp.use_bias=true but b_o.data is NULL");
            }
            if (static_cast<int>(b_qkv.numel()) != hp.qkv_dim) {
                throw std::runtime_error("encoderSelfAttentionForward: b_qkv numel mismatch. expected=" +
                                         std::to_string(hp.qkv_dim) + " got=" +
                                         std::to_string(b_qkv.numel()));
            }
            if (static_cast<int>(b_o.numel()) != hp.d_model) {
                throw std::runtime_error("encoderSelfAttentionForward: b_o numel mismatch. expected=" +
                                         std::to_string(hp.d_model) + " got=" +
                                         std::to_string(b_o.numel()));
            }
        }
    }

    void validatePBMState(const GRIM::PBM::PBMState& pbm,
                         const GRIM::HyperParameters::EncoderSelfAttentionHP& hp) {
        if (!pbm.initialized) {
            throw std::runtime_error("encoderSelfAttentionForward: PBM state is not initialized - GRIM requires RoPE+ALiBi");
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
    }

    std::uint64_t attentionDropoutSeed(const GRIM::Attention::EncoderSelfAttentionForwardRequest& request) {
        const float attention_dropout_p = request.hp.dropout_enabled ? request.hp.attention_dropout : 0.0f;
        if (attention_dropout_p <= 0.0f) {
            return 0;
        }
        return request.batch_idx * 2654435761ULL + 42 +
               1000 * static_cast<std::uint64_t>(request.layer_idx);
    }
}  // namespace

namespace GRIM::Attention {

void encoderSelfAttentionForward(
    const Tensor& norm_input,
    const Tensor& W_qkv,
    const Tensor& b_qkv,
    const Tensor& W_o,
    const Tensor& b_o,
    const GRIM::PBM::PBMState& pbm,
    const EncoderSelfAttentionForwardRequest& request,
    Forward::ModelForwardOutputs& forward_outputs) {
    if (request.layer_idx < 0) {
        throw std::runtime_error("encoderSelfAttentionForward: layer_idx must be >= 0, got " +
                                 std::to_string(request.layer_idx));
    }
    const size_t layer_slot = static_cast<size_t>(request.layer_idx);
    forward_outputs.validateLayerIndex(layer_slot, "encoderSelfAttentionForward");
    Tensor& qkv_out = forward_outputs.qkv_out_per_layer[layer_slot];
    Tensor& Q_bhsd = forward_outputs.Q_bhsd_per_layer[layer_slot];
    Tensor& K_bhsd = forward_outputs.K_bhsd_per_layer[layer_slot];
    Tensor& V_bhsd = forward_outputs.V_bhsd_per_layer[layer_slot];
    Tensor& attn_out_bhsd = forward_outputs.attn_out_bhsd_per_layer[layer_slot];
    Tensor& attn_out = forward_outputs.attn_out_per_layer[layer_slot];
    Tensor& proj_out = forward_outputs.proj_out_per_layer[layer_slot];
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

    validateWeights(W_qkv, b_qkv, W_o, b_o, request.hp);
    validatePBMState(pbm, request.hp);
    autograd::set_autograd_cublas_handle(request.cublas_handle);

    const int qkv_debug = autograd::qkvDebugLevel();
    if constexpr (kEnableAttentionStepLogs) {
        std::fprintf(stderr, "[EncoderSelfAttention] START layer=%d tokens=%d heads=%d kv_heads=%d head_dim=%d\n",
                     request.layer_idx, request.payload.total_tokens, request.hp.num_heads,
                     request.hp.num_kv_heads, request.hp.head_dim);
    }

    if (qkv_debug >= 3) {
        autograd::checkQKVTensorFinite("AutogradQKV:ln1_out", norm_input, request.stream);
        autograd::checkQKVTensorFinite("AutogradQKV:W_qkv", W_qkv, request.stream);
        if (request.hp.use_bias) {
            autograd::checkQKVTensorFinite("AutogradQKV:b_qkv", b_qkv, request.stream);
        }
    }

    if constexpr (kEnableAttentionStepLogs) {
        std::fprintf(stderr, "[EncoderSelfAttention] QKV projection...\n");
    }
    qkv_out = autograd::matmul(norm_input, W_qkv, request.stream, true);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradQKV:qkv_out_prebias", qkv_out, request.stream);
    }

    if (request.hp.use_bias) {
        qkv_out = autograd::broadcast_add(qkv_out, b_qkv, request.stream);
    }
    autograd::logQKVProjectionEquation(
        norm_input, W_qkv, b_qkv, qkv_out,
        request.payload, request.hp, request.stream, request.layer_idx);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradQKV:qkv_out", qkv_out, request.stream);
    }

    auto [Q_bhsd_tmp, K_bhsd_tmp, V_bhsd_tmp] = autograd::split_and_reshape_qkv(
        qkv_out,
        request.payload, request.hp,
        request.stream);
    Q_bhsd = std::move(Q_bhsd_tmp);
    K_bhsd = std::move(K_bhsd_tmp);
    V_bhsd = std::move(V_bhsd_tmp);
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradQKV:Q_bhsd", Q_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradQKV:K_bhsd", K_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradQKV:V_bhsd", V_bhsd, request.stream);
    }

    // Experimental ablation (AblationFlags.hpp): kZeroRope skips the RoPE
    // rotation so Q/K reach attention unrotated (no rotary positional encoding).
    // Q/K stay as the split_and_reshape_qkv outputs, so the autograd graph
    // remains connected and W_qkv still receives full gradient.
    if constexpr (!GRIM::Ablation::kZeroRope) {
        auto [Q_rot, K_rot] = autograd::rope_rotation(
            Q_bhsd, K_bhsd,
            pbm.rope_inv_freq,
            request.payload, request.hp,
            request.hp.rotary_dim, request.stream);
        Q_bhsd = std::move(Q_rot);
        K_bhsd = std::move(K_rot);
    }
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradSDPA:Q_rope", Q_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradSDPA:K_rope", K_bhsd, request.stream);
        autograd::checkQKVTensorFinite("AutogradSDPA:V_rope", V_bhsd, request.stream);
    }

    // Experimental ablation (AblationFlags.hpp): fine-grained attention probes.
    // Each zeroes a specific internal signal while keeping tensor shapes and the
    // autograd graph intact (zeroed tensor's producers receive zero gradient).
    // These only matter when kZeroAttnResidual=false (attention still feeds the
    // residual); otherwise the whole branch is zeroed downstream in encoding.
    if constexpr (GRIM::Ablation::kZeroAttnQKScores) {
        // Zero Q (post-RoPE) so the QKᵀ content score is 0; attention weights
        // then come from the ALiBi positional bias only (content-independent).
        Q_bhsd = autograd::mul_scalar(Q_bhsd, 0.0f, request.stream);
    }
    if constexpr (GRIM::Ablation::kZeroAttnV) {
        // Zero the value vectors so the softmax-weighted sum is of zeros and
        // attn_out == 0, while QK/softmax routing is still computed.
        V_bhsd = autograd::mul_scalar(V_bhsd, 0.0f, request.stream);
    }

    const float attention_dropout_p = request.hp.dropout_enabled ? request.hp.attention_dropout : 0.0f;
    const float attention_softmax_scale = request.hp.attention_softmax_scale;
    if (!std::isfinite(attention_softmax_scale) || attention_softmax_scale <= 0.0f) {
        throw std::runtime_error("encoderSelfAttentionForward: attention_softmax_scale must be finite and > 0");
    }
    const std::uint64_t dropout_seed = attentionDropoutSeed(request);
    attn_out_bhsd = autograd::scaled_dot_product_attention(
        Q_bhsd, K_bhsd, V_bhsd,
        pbm.alibi_slopes, request.hp, attention_softmax_scale, request.stream,
        attention_dropout_p, dropout_seed);
    attn_out_bhsd.shape.require("encoderSelfAttentionForward attn_out_bhsd");
    if (attn_out_bhsd.shape.layout != TensorContract::Layout::BHSD) {
        throw std::runtime_error("encoderSelfAttentionForward: scaled_dot_product_attention must return BHSD. FlashAttention scratch/output buffers are BSHD internally, but SDPA must convert them back before reshape_bhsd_to_flat");
    }
    {
        const auto attn_shape = attn_out_bhsd.shape.as_4d();
        if (attn_shape.batch != request.payload.batch_size ||
            attn_shape.heads != request.hp.num_heads ||
            attn_shape.seq != request.payload.max_seq_len ||
            attn_shape.head_dim != request.hp.head_dim) {
            throw std::runtime_error("encoderSelfAttentionForward: attn_out_bhsd shape mismatch before flatten. expected=[" +
                                     std::to_string(request.payload.batch_size) + "," +
                                     std::to_string(request.hp.num_heads) + "," +
                                     std::to_string(request.payload.max_seq_len) + "," +
                                     std::to_string(request.hp.head_dim) + "] got=[" +
                                     std::to_string(attn_shape.batch) + "," +
                                     std::to_string(attn_shape.heads) + "," +
                                     std::to_string(attn_shape.seq) + "," +
                                     std::to_string(attn_shape.head_dim) + "]");
        }
    }
    if (qkv_debug > 0) {
        autograd::checkQKVTensorFinite("AutogradSDPA:attn_out_bhsd", attn_out_bhsd, request.stream);
    }

    attn_out = autograd::reshape_bhsd_to_flat(
        attn_out_bhsd, request.payload, request.hp, request.stream);

    if (!attn_out.data) {
        throw std::runtime_error("encoderSelfAttentionForward: attn_out.data is NULL before output projection matmul");
    }
    proj_out = autograd::matmul(attn_out, W_o, request.stream, true);
    if (request.hp.use_bias) {
        proj_out = autograd::broadcast_add(proj_out, b_o, request.stream);
    }

    // Experimental ablation (AblationFlags.hpp): zero the attention OUTPUT
    // PROJECTION. attn_out is still produced from V, but its projection into the
    // residual is suppressed (proj_out == 0) and W_o/b_o receive zero gradient.
    if constexpr (GRIM::Ablation::kZeroAttnOProj) {
        proj_out = autograd::mul_scalar(proj_out, 0.0f, request.stream);
    }

    if constexpr (kEnableAttentionStepLogs) {
        std::fprintf(stderr, "[EncoderSelfAttention] DONE layer=%d\n", request.layer_idx);
    }
}

}  // namespace GRIM::Attention
