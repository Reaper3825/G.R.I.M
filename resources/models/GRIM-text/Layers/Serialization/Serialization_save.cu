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
        case HyperParameters::PositionalEncodingType::NONE: fb_pos_enc = GRIMTransformer::PositionalEncodingType_NONE; break;
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
            Logging::EmitModuleError(kLogModule, Msg("[save] Missing ", label, " buffer"));
            return {};
        }
        std::vector<float> host(view.count);
        cudaError_t err = cudaMemcpy(host.data(), view.ptr, view.count * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            Logging::EmitModuleError(kLogModule, Msg("[save] Failed to download ", label, ": ", cudaGetErrorString(err)));
            return {};
        }
        return host;
    };

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_token_embed = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_rms_gamma = 0;
    bool use_rms = true;

    Logging::EmitModuleInfo(kLogModule, Msg("[save] Downloading weights: embeddings + ", cfg.num_layers, " encoder layers (GPU->CPU)..."));

    if (cfg.use_gpu && request.sources.gpu_embedding.token_embeddings.ptr) {
        auto token_embed_data = download_device_vector(request.sources.gpu_embedding.token_embeddings, "token embeddings");
        if (token_embed_data.empty() && request.sources.gpu_embedding.token_embeddings.count > 0) return false;
        fb_token_embed = builder.CreateVector(token_embed_data);

        if (request.sources.gpu_embedding.has_rms_norm) {
            auto rms_gamma = download_device_vector(request.sources.gpu_embedding.rms_gamma, "embedding rms gamma");
            if (rms_gamma.empty() && request.sources.gpu_embedding.rms_gamma.count > 0) return false;
            fb_rms_gamma = builder.CreateVector(rms_gamma);
            use_rms = true;
        }
    } else {
        const auto& cpu_embed = request.sources.cpu_embedding;
        if (cpu_embed.token_data.empty()) {
            Logging::EmitModuleError(kLogModule, "[save] CPU embedding data is missing");
            return false;
        }
        fb_token_embed = builder.CreateVector(cpu_embed.token_data);
        if (cpu_embed.has_rms_norm) {
            fb_rms_gamma = builder.CreateVector(cpu_embed.rms_gamma);
            use_rms = true;
        }
    }

    auto fb_embeddings = GRIMTransformer::CreateEmbeddingWeights(
        builder, fb_token_embed, 0, fb_rms_gamma,
        static_cast<uint32_t>(cfg.vocab_size),
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.max_seq_len),
        use_rms);

    std::vector<flatbuffers::Offset<GRIMTransformer::EncoderLayerWeights>> fb_layers;
    fb_layers.reserve(cfg.num_layers);

    auto download_into = [&download_device_vector](std::vector<float>& host, const DeviceReadView& view, const char* label) -> bool {
        host = download_device_vector(view, label);
        return !host.empty() || view.count == 0;
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
            std::vector<float> h_ls1;
            if (!download_into(h_ls1, layer_view.layer_scale1, "layer_scale1")) return false;
            fb_ls1 = builder.CreateVector(h_ls1);
        }
        if (layer_view.layer_scale2.ptr) {
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

    if (cfg.use_bias) {
        if (lm_head_view.has_bias && lm_head_view.bias.ptr) {
            auto lm_bias = download_device_vector(lm_head_view.bias, "LM head bias");
            if (lm_bias.empty() && lm_head_view.bias.count > 0) return false;
            fb_lm_bias = builder.CreateVector(lm_bias);
        } else {
            Logging::EmitModuleError(kLogModule, "[save] LM head bias missing (use_bias=true)");
            return false;
        }
    }

    auto fb_lm_head = GRIMTransformer::CreateLMHeadWeights(
        builder, fb_lm_proj, fb_lm_bias,
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.vocab_size),
        cfg.tie_embeddings,
        cfg.use_bias && lm_head_view.bias.ptr != nullptr);

    flatbuffers::Offset<GRIMTransformer::ScratchBlockWeights> fb_scratch_block = 0;
    const auto& sb_view = request.sources.scratch_block;
    if (sb_view.enabled && sb_view.atom_type_embeddings.ptr) {
        auto sb_atom_emb = download_device_vector(sb_view.atom_type_embeddings, "ScratchBlock atom_type_embeddings");
        auto sb_atom_proj = download_device_vector(sb_view.atom_projection, "ScratchBlock atom_projection");
        if (!sb_atom_emb.empty() || sb_view.atom_type_embeddings.count == 0) {
            fb_scratch_block = GRIMTransformer::CreateScratchBlockWeights(
                builder,
                builder.CreateVector(sb_atom_emb),
                builder.CreateVector(sb_atom_proj),
                0,
                static_cast<uint32_t>(sb_view.num_atom_types),
                static_cast<uint32_t>(sb_view.atom_embedding_dim),
                static_cast<uint32_t>(sb_view.d_model),
                sb_view.atom_scale,
                sb_view.enabled);
            Logging::EmitModuleInfo(kLogModule, Msg("[save] ScratchBlock: atom_emb=",
                sb_atom_emb.size(), " atom_proj=", sb_atom_proj.size()));
        }
    }

    flatbuffers::Offset<GRIMTransformer::NumericHeadWeights> fb_numeric_head = 0;
    const auto& nh_view = request.sources.numeric_head;
    if (nh_view.enabled && nh_view.weights.ptr) {
        auto nh_weights = download_device_vector(nh_view.weights, "NumericHead weights");
        auto nh_bias = download_device_vector(nh_view.bias, "NumericHead bias");
        fb_numeric_head = GRIMTransformer::CreateNumericHeadWeights(
            builder, builder.CreateVector(nh_weights), builder.CreateVector(nh_bias),
            static_cast<uint32_t>(nh_view.d_model), true);
        Logging::EmitModuleInfo(kLogModule, Msg("[save] NumericHead: weights=",
            nh_weights.size(), " bias=", nh_bias.size()));
    }

    flatbuffers::Offset<GRIMTransformer::ReasoningHeadWeights> fb_reasoning_head = 0;
    const auto& rh_view = request.sources.reasoning_head;
    if (rh_view.enabled && rh_view.w_op.ptr) {
        auto rh_w_op = download_device_vector(rh_view.w_op, "ReasoningHead W_op");
        auto rh_b_op = download_device_vector(rh_view.b_op, "ReasoningHead b_op");
        auto rh_w_arg1 = download_device_vector(rh_view.w_arg1, "ReasoningHead w_arg1");
        auto rh_w_arg2 = download_device_vector(rh_view.w_arg2, "ReasoningHead w_arg2");
        fb_reasoning_head = GRIMTransformer::CreateReasoningHeadWeights(
            builder,
            builder.CreateVector(rh_w_op), builder.CreateVector(rh_b_op),
            builder.CreateVector(rh_w_arg1), builder.CreateVector(rh_w_arg2),
            static_cast<uint32_t>(rh_view.num_ops),
            static_cast<uint32_t>(rh_view.d_total));
        Logging::EmitModuleInfo(kLogModule, Msg("[save] ReasoningHead: W_op=", rh_w_op.size(),
            " b_op=", rh_b_op.size(), " w_arg1=", rh_w_arg1.size(), " w_arg2=", rh_w_arg2.size()));
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
            builder.CreateVector(dl(eb_view.gamma, "EB gamma")),
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
            builder.CreateVector(dl(eb_view.tau, "EB tau")));
        Logging::EmitModuleInfo(kLogModule, "[save] ExecutionBlock v2 weights serialized");
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

    auto fb_model = GRIMTransformer::CreateTransformerModel(
        builder,
        request.model_version,
        fb_config,
        fb_embeddings,
        fb_encoder_layers,
        fb_lm_head,
        fb_numeric_head,
        fb_scratch_block,
        fb_final_rms_gamma,
        0,
        fb_metadata,
        0, 0,
        timestamp, timestamp,
        fb_reasoning_head,
        fb_execution_block);

    builder.Finish(fb_model, "GRMT");

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
            Logging::EmitModuleError(kLogModule, "[save] Write failed");
            close(fd);
            std::filesystem::remove(temp_path, ec);
            return false;
        }
        if (fsync(fd) != 0)
            Logging::EmitModuleError(kLogModule, "[save] fsync failed (checkpoint may be corrupt on NFS/Lustre)");
        close(fd);
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
