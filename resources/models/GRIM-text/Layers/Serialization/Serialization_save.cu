#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Common/grim_model_serialization_version.hpp"
#include "Serialization_GPU.hpp"

namespace {

constexpr auto kLogModule = GRIM::Logging::ModuleId::Checkpoint;

template <typename... Args>
std::string Msg(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

void dumpSaveCudaState() {
    std::size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        GRIM::Logging::EmitModuleError(kLogModule,
            Msg("[save][DUMP] GPU memory: free=", free_mem / (1024*1024), "MB total=",
                 total_mem / (1024*1024), "MB used=", (total_mem - free_mem) / (1024*1024), "MB"));
    }
    cudaError_t sticky = cudaPeekAtLastError();
    if (sticky != cudaSuccess) {
        GRIM::Logging::EmitModuleError(kLogModule,
            Msg("[save][DUMP] CUDA sticky error: ", cudaGetErrorString(sticky)));
    }
}

} // namespace

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                             \
    do {                                                                                                             \
        cudaError_t err__ = (call);                                                                                  \
        if (err__ != cudaSuccess) {                                                                                  \
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[CUDA] ", __FILE__, ':', __LINE__, ": ",                 \
                                         cudaGetErrorString(err__)));                                                \
            throw std::runtime_error("CUDA failure");                                                                 \
        }                                                                                                            \
    } while (0)
#endif

namespace GRIM {

bool SerializationLayer::save(const SerializationSaveRequest& request) {
    const auto& cfg = request.sources.config;
    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        Logging::EmitModuleError(kLogModule, "[save] Invalid model dimensions");
        return false;
    }
    if (request.sources.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        Logging::EmitModuleError(kLogModule, Msg("[save] Encoder layer mismatch (expected ",
                                            cfg.num_layers, ", got ", request.sources.encoder_layers.size(), ")"));
        return false;
    }

    flatbuffers::FlatBufferBuilder builder(32ULL * 1024ULL * 1024ULL);

    GRIMTransformer::PositionalEncodingType fb_pos_enc;
    switch (cfg.positional_encoding) {
        case HyperParameters::PositionalEncodingType::NONE:
            // Rule 20: learned position embeddings were removed; NONE must never be serialized.
            throw std::runtime_error(
                "SerializationLayer::save: positional_encoding=NONE is no longer supported "
                "(learned position embeddings have been removed). Use ALIBI, ROPE, or ALIBI_ROPE.");
        case HyperParameters::PositionalEncodingType::ALIBI: fb_pos_enc = GRIMTransformer::PositionalEncodingType_ALIBI; break;
        case HyperParameters::PositionalEncodingType::ROPE: fb_pos_enc = GRIMTransformer::PositionalEncodingType_ROPE; break;
        case HyperParameters::PositionalEncodingType::ALIBI_ROPE: fb_pos_enc = GRIMTransformer::PositionalEncodingType_ALIBI_ROPE; break;
        default:
            throw std::runtime_error("SerializationLayer::save: unknown PositionalEncodingType " +
                                     std::to_string(static_cast<int>(cfg.positional_encoding)));
    }

    auto fb_config = GRIMTransformer::CreateModelConfig(
        builder,
        static_cast<uint32_t>(cfg.vocab_size),
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.num_layers),
        static_cast<uint32_t>(cfg.num_heads),
        static_cast<uint32_t>(cfg.num_kv_heads),
        static_cast<uint32_t>(cfg.d_ff),
        static_cast<uint32_t>(cfg.max_seq_len),
        cfg.dropout_rate,
        cfg.dropout_rate,
        fb_pos_enc,
        false,
        true,
        true,
        true,
        cfg.tie_embeddings,
        cfg.use_bias);

    auto download_device_vector = [](const DeviceReadView& view, const char* label) -> std::vector<float> {
        if (view.count == 0) return {};
        if (!view.ptr) {
            Logging::EmitModuleError(kLogModule, Msg("[save] DOWNLOAD FAIL — null source buffer for ", label));
            Logging::EmitModuleError(kLogModule, Msg("[save]   view.count=", view.count,
                " view.ptr=NULL — caller provided a ReadView with count>0 but no pointer"));
            dumpSaveCudaState();
            return {};
        }
        std::vector<float> host(view.count);
        cudaError_t err = cudaMemcpy(host.data(), view.ptr, view.count * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            Logging::EmitModuleError(kLogModule, Msg("[save] DOWNLOAD FAIL — cudaMemcpy D2H failed for ", label,
                ": ", cudaGetErrorString(err)));
            Logging::EmitModuleError(kLogModule, Msg("[save]   elements=", view.count,
                " bytes=", view.count * sizeof(float),
                " src_ptr=", reinterpret_cast<uintptr_t>(view.ptr)));
            dumpSaveCudaState();
            return {};
        }
        return host;
    };

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_token_embed = 0;

    Logging::EmitModuleInfo(kLogModule, Msg("[save] Downloading weights: embeddings + ", cfg.num_layers, " encoder layers (GPU->CPU)..."));

    if (!request.sources.gpu_embedding.token_embeddings.ptr) {
        Logging::EmitModuleError(kLogModule, "[save] FATAL: GPU token embedding buffer is NULL");
        return false;
    }
    auto token_embed_data = download_device_vector(request.sources.gpu_embedding.token_embeddings, "token embeddings");
    if (token_embed_data.empty() && request.sources.gpu_embedding.token_embeddings.count > 0) return false;
    fb_token_embed = builder.CreateVector(token_embed_data);

    auto fb_embeddings = GRIMTransformer::CreateEmbeddingWeights(
        builder, fb_token_embed, 0,
        static_cast<uint32_t>(cfg.vocab_size),
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.max_seq_len));

    std::vector<flatbuffers::Offset<GRIMTransformer::EncoderLayerWeights>> fb_layers;
    fb_layers.reserve(cfg.num_layers);

    auto download_into = [&download_device_vector](std::vector<float>& host, const DeviceReadView& view, const char* label) -> bool {
        host = download_device_vector(view, label);
        return !host.empty() || view.count == 0;
    };

    auto check_layer_scale_read_view = [&cfg](const DeviceReadView& view,
                                              const char* label,
                                              int layer_idx) -> bool {
        if (!view.ptr) return true;
        const std::size_t expected = static_cast<std::size_t>(cfg.d_model);
        if (view.count != expected) {
            Logging::EmitModuleError(kLogModule,
                Msg("[save] FATAL: ", label, " view count mismatch in layer ", layer_idx,
                    ": view=", view.count, " expected=d_model=", expected,
                    " — LayerScale checkpoint fields are per-channel gamma vectors"));
            return false;
        }
        return true;
    };

    for (int layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        if (layer_idx % 6 == 0 || layer_idx == cfg.num_layers - 1)
            Logging::EmitModuleInfo(kLogModule, Msg("[save] Downloading encoder layer ", layer_idx + 1, "/", cfg.num_layers));

        const auto& layer_view = request.sources.encoder_layers[layer_idx];

        std::vector<float> h_W_qkv, h_b_qkv, h_W_o, h_b_o;
        if (!download_into(h_W_qkv, layer_view.attn_w_qkv, "attn.W_qkv") ||
            !download_into(h_b_qkv, layer_view.attn_b_qkv, "attn.b_qkv") ||
            !download_into(h_W_o, layer_view.attn_w_o, "attn.W_o") ||
            !download_into(h_b_o, layer_view.attn_b_o, "attn.b_o"))
            return false;

        auto fb_attn = GRIMTransformer::CreateAttentionWeights(
            builder,
            builder.CreateVector(h_W_qkv), builder.CreateVector(h_b_qkv),
            0, 0, 0, 0, 0, 0,
            builder.CreateVector(h_W_o), builder.CreateVector(h_b_o),
            0, 0,  // alpha_q, alpha_k (not used)
            static_cast<uint32_t>(cfg.d_model),
            static_cast<uint32_t>(cfg.num_heads),
            static_cast<uint32_t>(cfg.num_kv_heads),
            true);

        std::vector<float> h_W_gate, h_W1, h_W2, h_b2;
        if (!download_into(h_W_gate, layer_view.ffn_w_gate, "ffn.W_gate") ||
            !download_into(h_W1, layer_view.ffn_w1, "ffn.W1") ||
            !download_into(h_W2, layer_view.ffn_w2, "ffn.W2") ||
            !download_into(h_b2, layer_view.ffn_b2, "ffn.b2"))
            return false;

        auto fb_ffn = GRIMTransformer::CreateFFNWeights(
            builder,
            builder.CreateVector(h_W1), 0,
            builder.CreateVector(h_W2), builder.CreateVector(h_b2),
            static_cast<uint32_t>(cfg.d_model),
            static_cast<uint32_t>(cfg.d_ff),
            builder.CreateVector(h_W_gate));

        std::vector<float> h_rms1_gamma, h_rms2_gamma;
        if (!download_into(h_rms1_gamma, layer_view.rms1_gamma, "rms1.gamma") ||
            !download_into(h_rms2_gamma, layer_view.rms2_gamma, "rms2.gamma"))
            return false;

        auto fb_rms1 = GRIMTransformer::CreateRMSNormWeights(builder, builder.CreateVector(h_rms1_gamma));
        auto fb_rms2 = GRIMTransformer::CreateRMSNormWeights(builder, builder.CreateVector(h_rms2_gamma));
        flatbuffers::Offset<GRIMTransformer::RMSNormWeights> fb_rms_post_attn = 0;
        flatbuffers::Offset<GRIMTransformer::RMSNormWeights> fb_rms_post_ffn = 0;

        flatbuffers::Offset<flatbuffers::Vector<float>> fb_ls1 = 0;
        flatbuffers::Offset<flatbuffers::Vector<float>> fb_ls2 = 0;
        if (layer_view.layer_scale1.ptr) {
            if (!check_layer_scale_read_view(layer_view.layer_scale1, "layer_scale1", layer_idx)) return false;
            std::vector<float> h_ls1;
            if (!download_into(h_ls1, layer_view.layer_scale1, "layer_scale1")) return false;
            fb_ls1 = builder.CreateVector(h_ls1);
        }
        if (layer_view.layer_scale2.ptr) {
            if (!check_layer_scale_read_view(layer_view.layer_scale2, "layer_scale2", layer_idx)) return false;
            std::vector<float> h_ls2;
            if (!download_into(h_ls2, layer_view.layer_scale2, "layer_scale2")) return false;
            fb_ls2 = builder.CreateVector(h_ls2);
        }

        fb_layers.push_back(GRIMTransformer::CreateEncoderLayerWeights(
            builder, fb_attn, fb_ffn, fb_rms1, fb_rms2,
            fb_ls1, fb_ls2,
            fb_rms_post_attn, fb_rms_post_ffn,
            static_cast<uint32_t>(layer_idx)));
    }

    auto fb_encoder_layers = builder.CreateVector(fb_layers);

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_proj = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_bias = 0;
    const auto& lm_head_view = request.sources.lm_head;
    if (lm_head_view.has_projection && lm_head_view.projection.ptr) {
        auto lm_proj = download_device_vector(lm_head_view.projection, "LM head projection");
        if (lm_proj.empty() && lm_head_view.projection.count > 0) return false;
        fb_lm_proj = builder.CreateVector(lm_proj);
    } else if (!cfg.tie_embeddings) {
        Logging::EmitModuleError(kLogModule, "[save] LM head projection missing (tie_embeddings=false)");
        return false;
    }

    // Save the LM-head bias whenever it is present on the model. Presence-driven
    // so it covers both use_bias and the dedicated unigram bias without threading
    // an extra config flag into the serialization view.
    if (lm_head_view.has_bias && lm_head_view.bias.ptr) {
        auto lm_bias = download_device_vector(lm_head_view.bias, "LM head bias");
        if (lm_bias.empty() && lm_head_view.bias.count > 0) return false;
        fb_lm_bias = builder.CreateVector(lm_bias);
    }

    // Head-side residual SwiGLU adapter — presence-driven like the bias.
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_mlp_gate = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_mlp_up = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_mlp_down = 0;
    const bool save_lm_mlp = lm_head_view.has_mlp
                          && lm_head_view.mlp_w_gate.ptr
                          && lm_head_view.mlp_w_up.ptr
                          && lm_head_view.mlp_w_down.ptr;
    if (save_lm_mlp) {
        auto mlp_gate = download_device_vector(lm_head_view.mlp_w_gate, "LM head mlp_W_gate");
        if (mlp_gate.empty() && lm_head_view.mlp_w_gate.count > 0) return false;
        fb_lm_mlp_gate = builder.CreateVector(mlp_gate);

        auto mlp_up = download_device_vector(lm_head_view.mlp_w_up, "LM head mlp_W_up");
        if (mlp_up.empty() && lm_head_view.mlp_w_up.count > 0) return false;
        fb_lm_mlp_up = builder.CreateVector(mlp_up);

        auto mlp_down = download_device_vector(lm_head_view.mlp_w_down, "LM head mlp_W_down");
        if (mlp_down.empty() && lm_head_view.mlp_w_down.count > 0) return false;
        fb_lm_mlp_down = builder.CreateVector(mlp_down);
        Logging::EmitModuleInfo(kLogModule, Msg("[save] LM head residual SwiGLU adapter serialized (mlp_d_ff=",
            lm_head_view.mlp_d_ff, ")"));
    }

    auto fb_lm_head = GRIMTransformer::CreateLMHeadWeights(
        builder, fb_lm_proj, fb_lm_bias,
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.vocab_size),
        cfg.tie_embeddings,
        lm_head_view.has_bias && lm_head_view.bias.ptr != nullptr,
        fb_lm_mlp_gate,
        fb_lm_mlp_up,
        fb_lm_mlp_down,
        static_cast<uint32_t>(save_lm_mlp ? lm_head_view.mlp_d_ff : 0),
        save_lm_mlp);

    flatbuffers::Offset<GRIMTransformer::NumberEncoderWeights> fb_number_encoder = 0;
    const auto& ne_view = request.sources.number_encoder;
    if (ne_view.enabled && ne_view.digit_emb.ptr) {
        auto dl = [&](const DeviceReadView& v, const char* n) { return download_device_vector(v, n); };
        // Hidden biases (b_c1/b_g1) are use_bias-gated: when absent the source view
        // has a null ptr, so emit a null (0) offset to leave the FlatBuffer field
        // unset rather than writing a spurious empty vector.
        using FloatVecOffset = flatbuffers::Offset<flatbuffers::Vector<float>>;
        FloatVecOffset b_c1_off = ne_view.b_c1.ptr ? builder.CreateVector(dl(ne_view.b_c1, "NE b_c1")) : 0;
        FloatVecOffset b_g1_off = ne_view.b_g1.ptr ? builder.CreateVector(dl(ne_view.b_g1, "NE b_g1")) : 0;
        fb_number_encoder = GRIMTransformer::CreateNumberEncoderWeights(
            builder,
            builder.CreateVector(dl(ne_view.digit_emb, "NE digit_emb")),
            builder.CreateVector(dl(ne_view.pow10_emb, "NE pow10_emb")),
            builder.CreateVector(dl(ne_view.W_c1, "NE W_c1")),
            b_c1_off,
            builder.CreateVector(dl(ne_view.W_c2, "NE W_c2")),
            builder.CreateVector(dl(ne_view.W_g1, "NE W_g1")),
            b_g1_off,
            builder.CreateVector(dl(ne_view.W_g2, "NE W_g2")));
        Logging::EmitModuleInfo(kLogModule, ne_view.b_c1.ptr
            ? "[save] NumberEncoder weights serialized (with hidden biases)"
            : "[save] NumberEncoder weights serialized (biases gated off)");
    }

    flatbuffers::Offset<GRIMTransformer::ArgSelectorWeights> fb_arg_selector = 0;
    const auto& sel_view = request.sources.arg_selector;
    if (sel_view.enabled && sel_view.W_q.ptr) {
        fb_arg_selector = GRIMTransformer::CreateArgSelectorWeights(
            builder,
            builder.CreateVector(download_device_vector(sel_view.W_q, "SEL W_q")));
        Logging::EmitModuleInfo(kLogModule, "[save] ArgSelector weights serialized");
    }

    flatbuffers::Offset<GRIMTransformer::ExecutionBlockWeights> fb_execution_block = 0;
    const auto& eb_view = request.sources.execution_block;
    if (eb_view.enabled && eb_view.w_decode_1.ptr) {
        auto dl = [&](const DeviceReadView& v, const char* n) { return download_device_vector(v, n); };
        fb_execution_block = GRIMTransformer::CreateExecutionBlockWeights(
            builder,
            builder.CreateVector(dl(eb_view.w_decode_1, "EB w_decode_1")),
            builder.CreateVector(dl(eb_view.b_decode_1, "EB b_decode_1")),
            builder.CreateVector(dl(eb_view.w_decode_2, "EB w_decode_2")),
            builder.CreateVector(dl(eb_view.w_arg1_select, "EB w_arg1_select")),
            builder.CreateVector(dl(eb_view.w_arg2_select, "EB w_arg2_select")),
            builder.CreateVector(dl(eb_view.W_op_select, "EB W_op_select")),
            builder.CreateVector(dl(eb_view.W_key_proj, "EB W_key_proj")),
            builder.CreateVector(dl(eb_view.W_write_query, "EB W_write_query")),
            builder.CreateVector(dl(eb_view.W_write_key, "EB W_write_key")),
            builder.CreateVector(dl(eb_view.alpha, "EB alpha")),
            builder.CreateVector(dl(eb_view.beta, "EB beta")),
            builder.CreateVector(dl(eb_view.step_embeddings, "EB step_embeddings")),
            builder.CreateVector(dl(eb_view.type_num_embed, "EB type_num_embed")),
            builder.CreateVector(dl(eb_view.W_value_to_emb, "EB W_value_to_emb")),
            builder.CreateVector(dl(eb_view.b_value_to_emb, "EB b_value_to_emb")),
            builder.CreateVector(dl(eb_view.w_inject_gate, "EB w_inject_gate")),
            builder.CreateVector(dl(eb_view.W_Q_read, "EB W_Q_read")),
            builder.CreateVector(dl(eb_view.W_K_read, "EB W_K_read")),
            builder.CreateVector(dl(eb_view.W_V_read, "EB W_V_read")),
            builder.CreateVector(dl(eb_view.W_O_read, "EB W_O_read")),
            builder.CreateVector(dl(eb_view.W_gate_read, "EB W_gate_read")),
            builder.CreateVector(dl(eb_view.tau, "EB tau")),
            builder.CreateVector(dl(eb_view.E_slot, "EB E_slot")),
            builder.CreateVector(dl(eb_view.E_op, "EB E_op")),
            builder.CreateVector(dl(eb_view.W_scal, "EB W_scal")),
            builder.CreateVector(dl(eb_view.b_scal, "EB b_scal")),
            builder.CreateVector(dl(eb_view.W_trace, "EB W_trace")),
            builder.CreateVector(dl(eb_view.b_trace, "EB b_trace")),
            builder.CreateVector(dl(eb_view.W_reason_gate, "EB W_reason_gate")),
            builder.CreateVector(dl(eb_view.W_trace_gate, "EB W_trace_gate")));
        Logging::EmitModuleInfo(kLogModule, "[save] ExecutionBlock v2 weights serialized");
    }

    flatbuffers::Offset<GRIMTransformer::LatentTrajectoryPresetWeights> fb_latent_trajectory_preset = 0;
    const auto& ltp_view = request.sources.latent_trajectory_preset;
    if (cfg.latent_trajectory_preset_enabled != ltp_view.enabled) {
        Logging::EmitModuleError(kLogModule,
            Msg("[save] FATAL: LatentTrajectoryPreset config/view mismatch: config_enabled=",
                cfg.latent_trajectory_preset_enabled, " view_enabled=", ltp_view.enabled));
        return false;
    }
    if (ltp_view.enabled) {
        if (cfg.latent_trajectory_preset_mtp_k <= 0 ||
            cfg.latent_trajectory_preset_fuse_dim <= 0 ||
            cfg.latent_trajectory_preset_dim <= 0 ||
            cfg.latent_trajectory_preset_gate_dim <= 0) {
            Logging::EmitModuleError(kLogModule,
                Msg("[save] FATAL: LatentTrajectoryPreset enabled with invalid derived dims: mtp_k=",
                    cfg.latent_trajectory_preset_mtp_k,
                    " fuse_dim=", cfg.latent_trajectory_preset_fuse_dim,
                    " preset_dim=", cfg.latent_trajectory_preset_dim,
                    " gate_dim=", cfg.latent_trajectory_preset_gate_dim));
            return false;
        }
        using FloatVecOffset = flatbuffers::Offset<flatbuffers::Vector<float>>;
        auto create_required_vector = [&](const DeviceReadView& v,
                                          const char* label,
                                          FloatVecOffset& out) -> bool {
            if (!v.ptr || v.count == 0) {
                Logging::EmitModuleError(kLogModule,
                    Msg("[save] FATAL: missing LatentTrajectoryPreset source tensor: ", label));
                return false;
            }
            auto data = download_device_vector(v, label);
            if (data.empty()) return false;
            out = builder.CreateVector(data);
            return true;
        };

        FloatVecOffset W_hidden_traj = 0;
        FloatVecOffset b_hidden_traj = 0;
        FloatVecOffset W_fuse = 0;
        FloatVecOffset b_fuse = 0;
        FloatVecOffset W_down = 0;
        FloatVecOffset b_down = 0;
        FloatVecOffset W_up = 0;
        FloatVecOffset b_up = 0;
        FloatVecOffset W_gate = 0;
        FloatVecOffset b_gate = 0;
        FloatVecOffset W_target = 0;
        FloatVecOffset b_target = 0;
        FloatVecOffset fuse_norm_gamma = 0;
        FloatVecOffset preset_norm_gamma = 0;
        if (!create_required_vector(ltp_view.W_hidden_traj, "LTP W_hidden_traj", W_hidden_traj) ||
            !create_required_vector(ltp_view.b_hidden_traj, "LTP b_hidden_traj", b_hidden_traj) ||
            !create_required_vector(ltp_view.W_fuse, "LTP W_fuse", W_fuse) ||
            !create_required_vector(ltp_view.b_fuse, "LTP b_fuse", b_fuse) ||
            !create_required_vector(ltp_view.W_down, "LTP W_down", W_down) ||
            !create_required_vector(ltp_view.b_down, "LTP b_down", b_down) ||
            !create_required_vector(ltp_view.W_up, "LTP W_up", W_up) ||
            !create_required_vector(ltp_view.b_up, "LTP b_up", b_up) ||
            !create_required_vector(ltp_view.W_gate, "LTP W_gate", W_gate) ||
            !create_required_vector(ltp_view.b_gate, "LTP b_gate", b_gate) ||
            !create_required_vector(ltp_view.W_target, "LTP W_target", W_target) ||
            !create_required_vector(ltp_view.b_target, "LTP b_target", b_target) ||
            !create_required_vector(ltp_view.fuse_norm_gamma, "LTP fuse_norm_gamma", fuse_norm_gamma) ||
            !create_required_vector(ltp_view.preset_norm_gamma, "LTP preset_norm_gamma", preset_norm_gamma)) {
            return false;
        }

        fb_latent_trajectory_preset = GRIMTransformer::CreateLatentTrajectoryPresetWeights(
            builder,
            W_fuse,
            b_fuse,
            W_down,
            b_down,
            W_up,
            b_up,
            W_gate,
            b_gate,
            W_target,
            b_target,
            fuse_norm_gamma,
            preset_norm_gamma,
            static_cast<uint32_t>(cfg.d_model),
            static_cast<uint32_t>(cfg.latent_trajectory_preset_mtp_k),
            static_cast<uint32_t>(cfg.latent_trajectory_preset_fuse_dim),
            static_cast<uint32_t>(cfg.latent_trajectory_preset_dim),
            static_cast<uint32_t>(cfg.latent_trajectory_preset_gate_dim),
            W_hidden_traj,
            b_hidden_traj);
        Logging::EmitModuleInfo(kLogModule, "[save] LatentTrajectoryPreset weights serialized");
    }

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_final_rms_gamma = 0;
    const auto& final_rms_view = request.sources.final_rms_gamma;
    if (final_rms_view.ptr && final_rms_view.count > 0) {
        auto final_rms_data = download_device_vector(final_rms_view, "final_rms_gamma");
        if (!final_rms_data.empty()) {
            fb_final_rms_gamma = builder.CreateVector(final_rms_data);
            Logging::EmitModuleInfo(kLogModule, Msg("[save] final_rms_gamma: size=", final_rms_data.size()));
        }
    }

    const auto now = std::chrono::system_clock::now();
    const auto timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());

    auto fb_optimizer = builder.CreateString("AdamW");
    auto fb_model_name = builder.CreateString("GRIM-text GPU model");
    auto fb_metadata = GRIMTransformer::CreateTrainingMetadata(
        builder, fb_optimizer,
        0.0f, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f,
        timestamp, 0, fb_model_name);

    const flatbuffers::Offset<GRIMTransformer::NumericHeadWeights> fb_numeric_head = 0;
    const flatbuffers::Offset<GRIMTransformer::LossWeightingWeights> fb_loss_weighting = 0;
    const flatbuffers::Offset<GRIMTransformer::ReasoningHeadWeights> fb_reasoning_head = 0;

    auto fb_model = GRIMTransformer::CreateTransformerModel(
        builder,
        static_cast<uint32_t>(request.model_version),
        fb_config,
        fb_embeddings,
        fb_encoder_layers,
        fb_lm_head,
        fb_numeric_head,
        fb_final_rms_gamma,
        fb_loss_weighting,
        fb_metadata,
        static_cast<uint32_t>(0),
        static_cast<uint64_t>(0),
        timestamp, timestamp,
        fb_reasoning_head,
        fb_execution_block,
        fb_number_encoder,
        fb_arg_selector,
        fb_latent_trajectory_preset);

    builder.Finish(fb_model, "GRMT");

    // In-memory verification: catch builder issues BEFORE writing to disk
    {
        const uint8_t* buf = builder.GetBufferPointer();
        const size_t buf_size = builder.GetSize();
        flatbuffers::Verifier pre_write_verifier(buf, buf_size);
        if (!GRIMTransformer::VerifyTransformerModelBuffer(pre_write_verifier)) {
            Logging::EmitModuleError(kLogModule, "[save] CRITICAL: in-memory FlatBuffer verification FAILED before writing to disk!");
            Logging::EmitModuleError(kLogModule, Msg("[save]   buf_size=", buf_size, " buf_ptr=", reinterpret_cast<uintptr_t>(buf)));

            // ── Component-level diagnostics ──
            const auto* raw = GRIMTransformer::GetTransformerModel(buf);
            if (!raw) {
                Logging::EmitModuleError(kLogModule, "[save] DIAG: GetTransformerModel returned NULL — root table broken");
            } else {
                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: root ptr OK, version=", raw->version()));

                // Step-by-step root table Verify decomposition
                {
                    flatbuffers::Verifier v(buf, buf_size);
                    bool s1 = v.VerifyTableStart(reinterpret_cast<const uint8_t*>(raw));
                    Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: VerifyTableStart=", s1 ? "OK" : "FAIL"));
                }

                // Individual components
                auto verify_component = [&](const char* name, const auto* tbl) {
                    if (!tbl) {
                        Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: ", name, " = NULL"));
                        return;
                    }
                    flatbuffers::Verifier v(buf, buf_size);
                    bool ok = tbl->Verify(v);
                    Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: ", name, " verify=", ok ? "OK" : "FAIL"));
                };

                verify_component("config", raw->config());
                verify_component("embeddings", raw->embeddings());

                if (raw->encoder_layers()) {
                    Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: encoder_layers count=", raw->encoder_layers()->size()));
                    for (uint32_t i = 0; i < raw->encoder_layers()->size(); i++) {
                        const auto* layer = raw->encoder_layers()->Get(i);
                        if (!layer) {
                            Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: encoder_layer[", i, "] = NULL"));
                            continue;
                        }
                        flatbuffers::Verifier vl(buf, buf_size);
                        bool layer_ok = layer->Verify(vl);
                        if (!layer_ok) {
                            Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: encoder_layer[", i, "] verify=FAIL"));
                            // Drill into sub-components
                            if (layer->attention()) {
                                flatbuffers::Verifier va(buf, buf_size);
                                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG:   .attention=", layer->attention()->Verify(va) ? "OK" : "FAIL"));
                            }
                            if (layer->ffn()) {
                                flatbuffers::Verifier vf(buf, buf_size);
                                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG:   .ffn=", layer->ffn()->Verify(vf) ? "OK" : "FAIL"));
                            }
                            if (layer->rms1()) {
                                flatbuffers::Verifier vr(buf, buf_size);
                                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG:   .rms1=", layer->rms1()->Verify(vr) ? "OK" : "FAIL"));
                            }
                            if (layer->rms2()) {
                                flatbuffers::Verifier vr(buf, buf_size);
                                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG:   .rms2=", layer->rms2()->Verify(vr) ? "OK" : "FAIL"));
                            }
                        } else {
                            Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: encoder_layer[", i, "] verify=OK"));
                        }
                    }
                } else {
                    Logging::EmitModuleError(kLogModule, "[save] DIAG: encoder_layers = NULL (required!)");
                }

                verify_component("lm_head", raw->lm_head());
                verify_component("training_metadata", raw->training_metadata());
                verify_component("number_encoder", raw->number_encoder());
                verify_component("arg_selector", raw->arg_selector());
                verify_component("execution_block", raw->execution_block());
                verify_component("latent_trajectory_preset", raw->latent_trajectory_preset());

                // Vector fields
                if (raw->final_rms_gamma()) {
                    flatbuffers::Verifier vr(buf, buf_size);
                    bool ok = vr.VerifyVector(raw->final_rms_gamma());
                    Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: final_rms_gamma vec verify=", ok ? "OK" : "FAIL",
                        " size=", raw->final_rms_gamma()->size()));
                } else {
                    Logging::EmitModuleError(kLogModule, "[save] DIAG: final_rms_gamma = NULL");
                }

                // Relaxed verifier
                flatbuffers::Verifier relaxed(buf, buf_size, 128, 10000000);
                bool relaxed_ok = GRIMTransformer::VerifyTransformerModelBuffer(relaxed);
                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: relaxed verifier (depth=128, tables=10M) = ", relaxed_ok ? "PASS" : "FAIL"));

                // Buffer header analysis
                if (buf_size >= 8) {
                    uint32_t root_off = flatbuffers::ReadScalar<uint32_t>(buf);
                    char id[5] = {0};
                    std::memcpy(id, buf + 4, 4);
                    Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: root_offset=", root_off, " identifier=\"", id, "\""));
                    if (buf_size > root_off) {
                        const uint8_t* root_table = buf + root_off;
                        int32_t vtable_soff = flatbuffers::ReadScalar<int32_t>(root_table);
                        const uint8_t* vtable = root_table - vtable_soff;
                        if (vtable >= buf && vtable < buf + buf_size) {
                            uint16_t vt_size = flatbuffers::ReadScalar<uint16_t>(vtable);
                            uint16_t obj_size = flatbuffers::ReadScalar<uint16_t>(vtable + 2);
                            Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: vtable_size=", vt_size,
                                " obj_inline_size=", obj_size, " num_fields=", (vt_size - 4) / 2));
                            // Dump all vtable slots
                            std::ostringstream slots;
                            slots << "[save] DIAG: vtable slots:";
                            for (uint16_t s = 4; s < vt_size; s += 2) {
                                uint16_t slot_val = flatbuffers::ReadScalar<uint16_t>(vtable + s);
                                slots << " [" << s << "]=" << slot_val;
                            }
                            Logging::EmitModuleError(kLogModule, slots.str());
                        }
                    }
                }

                // Alignment check
                Logging::EmitModuleError(kLogModule, Msg("[save] DIAG: buf alignment=", reinterpret_cast<uintptr_t>(buf) % 8));
            }

            return false;
        }
        Logging::EmitModuleInfo(kLogModule, "[save] In-memory FlatBuffer verification PASSED");
    }

    Logging::EmitModuleInfo(kLogModule, Msg("[save] Writing checkpoint to disk (", builder.GetSize() / (1024 * 1024), " MB)..."));

    const std::string final_path = request.path;
    const std::string temp_path = final_path + config_.temp_suffix;

    try {
        const auto parent = std::filesystem::path(final_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
    } catch (const std::exception& ex) {
        Logging::EmitModuleError(kLogModule, Msg("[save] Failed to create directories: ", ex.what()));
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);

    {
#ifdef _WIN32
        std::ofstream file(temp_path, std::ios::binary);
        if (!file) { Logging::EmitModuleError(kLogModule, Msg("[save] Failed to open: ", temp_path)); return false; }
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        if (!file) { Logging::EmitModuleError(kLogModule, "[save] Write failed"); return false; }
        file.flush();
        if (file) file.close();
#else
        const char* buf = reinterpret_cast<const char*>(builder.GetBufferPointer());
        const std::size_t buf_size = builder.GetSize();
        int fd = open(temp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { Logging::EmitModuleError(kLogModule, Msg("[save] Failed to open: ", temp_path)); return false; }
        std::size_t total_written = 0;
        while (total_written < buf_size) {
            ssize_t n = write(fd, buf + total_written, buf_size - total_written);
            if (n <= 0) break;
            total_written += static_cast<std::size_t>(n);
        }
        if (total_written != buf_size) {
            Logging::EmitModuleError(kLogModule, Msg("[save] WRITE FAIL — incomplete write to temp file"));
            Logging::EmitModuleError(kLogModule, Msg("[save]   written=", total_written, " expected=", buf_size,
                " shortfall=", buf_size - total_written, " bytes"));
            Logging::EmitModuleError(kLogModule, Msg("[save]   errno=", errno, " (", std::strerror(errno), ")"));
            Logging::EmitModuleError(kLogModule, Msg("[save]   temp_path=", temp_path));
            Logging::EmitModuleError(kLogModule, Msg("[save]   final_path=", final_path));
            close(fd);
            std::filesystem::remove(temp_path, ec);
            return false;
        }
        if (fsync(fd) != 0) {
            Logging::EmitModuleError(kLogModule, Msg("[save] FSYNC FAIL — refusing to rename (data may not be on disk)"));
            Logging::EmitModuleError(kLogModule, Msg("[save]   errno=", errno, " (", std::strerror(errno), ")"));
            Logging::EmitModuleError(kLogModule, Msg("[save]   file_size=", buf_size, " temp_path=", temp_path));
            close(fd);
            std::filesystem::remove(temp_path, ec);
            return false;
        }
        close(fd);

        // Post-write verification: re-read and verify FlatBuffer structure before rename.
        // Catches silent data corruption from filesystem (Lustre/NFS page cache issues).
        {
            int verify_fd = open(temp_path.c_str(), O_RDONLY);
            if (verify_fd < 0) {
                Logging::EmitModuleError(kLogModule, "[save] Post-write verify: failed to re-open temp file");
                std::filesystem::remove(temp_path, ec);
                return false;
            }
            struct stat st;
            if (fstat(verify_fd, &st) != 0 || static_cast<std::size_t>(st.st_size) != buf_size) {
                Logging::EmitModuleError(kLogModule, Msg("[save] POST-WRITE VERIFY FAIL — size mismatch"));
                Logging::EmitModuleError(kLogModule, Msg("[save]   expected=", buf_size,
                    " on_disk=", st.st_size, " diff=",
                    static_cast<int64_t>(buf_size) - static_cast<int64_t>(st.st_size)));
                Logging::EmitModuleError(kLogModule, Msg("[save]   temp_path=", temp_path));
                Logging::EmitModuleError(kLogModule, Msg("[save]   fstat errno=", errno, " (", std::strerror(errno), ")"));
                close(verify_fd);
                std::filesystem::remove(temp_path, ec);
                return false;
            }
            // Read back the header (first 64 bytes) and verify FlatBuffer file identifier
            uint8_t header[64];
            ssize_t hdr_read = read(verify_fd, header, std::min<std::size_t>(64, buf_size));
            close(verify_fd);
            if (hdr_read < 8) {
                Logging::EmitModuleError(kLogModule, Msg("[save] POST-WRITE VERIFY FAIL — could not read header"));
                Logging::EmitModuleError(kLogModule, Msg("[save]   bytes_read=", hdr_read,
                    " expected>=8 temp_path=", temp_path));
                std::filesystem::remove(temp_path, ec);
                return false;
            }
            // Verify file identifier at bytes 4-7
            if (std::memcmp(header + 4, "GRMT", 4) != 0) {
                char observed[5] = {static_cast<char>(header[4]), static_cast<char>(header[5]),
                                    static_cast<char>(header[6]), static_cast<char>(header[7]), '\0'};
                Logging::EmitModuleError(kLogModule, Msg("[save] POST-WRITE VERIFY FAIL — file identifier corrupted"));
                Logging::EmitModuleError(kLogModule, Msg("[save]   expected=\"GRMT\" observed=\"", observed, "\""));
                Logging::EmitModuleError(kLogModule, Msg("[save]   temp_path=", temp_path,
                    " — filesystem wrote wrong data (corruption between builder and disk)"));
                std::filesystem::remove(temp_path, ec);
                return false;
            }
            // Verify root offset is within bounds
            uint32_t root_offset = 0;
            std::memcpy(&root_offset, header, 4);
            if (root_offset >= buf_size) {
                Logging::EmitModuleError(kLogModule, Msg("[save] POST-WRITE VERIFY FAIL — root_offset out of bounds"));
                Logging::EmitModuleError(kLogModule, Msg("[save]   root_offset=", root_offset,
                    " file_size=", buf_size, " temp_path=", temp_path));
                std::filesystem::remove(temp_path, ec);
                return false;
            }
        }
#endif
    }

    if (!config_.atomic_write) {
        std::filesystem::rename(temp_path, final_path, ec);
        if (ec) { Logging::EmitModuleError(kLogModule, Msg("[save] Rename failed: ", ec.message())); return false; }
    } else {
#ifdef _WIN32
        if (!MoveFileExA(temp_path.c_str(), final_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            Logging::EmitModuleError(kLogModule, "[save] Rename failed"); return false;
        }
#else
        if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
            Logging::EmitModuleError(kLogModule, "[save] Rename failed"); return false;
        }
#endif
    }

    Logging::EmitModuleInfo(kLogModule, Msg("[save] Model saved: ", final_path));
    return true;
}

} // namespace GRIM
