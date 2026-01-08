#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>
#include <vector>

#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda_runtime.h>

#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Common/grim_model_serialization_version.hpp"
#include "Serialization_GPU.hpp"

namespace {

constexpr auto kSerializationLogModule = GRIM::Logging::ModuleId::Checkpoint;

template <typename... Args>
std::string MakeLogMessage(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

inline void SerializationLogError(const std::string& message) {
    GRIM::Logging::EmitModuleError(kSerializationLogModule, message);
}

inline void SerializationLogWarn(const std::string& message) {
    GRIM::Logging::EmitModuleWarning(kSerializationLogModule, message);
}

inline void SerializationLogInfo(const std::string& message) {
    GRIM::Logging::EmitModuleInfo(kSerializationLogModule, message);
}

} // namespace

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                             \
    do {                                                                                                             \
        cudaError_t err__ = (call);                                                                                  \
        if (err__ != cudaSuccess) {                                                                                  \
            SerializationLogError(MakeLogMessage("[SerializationLayer] CUDA error ", __FILE__, ':', __LINE__, ": ", \
                                         cudaGetErrorString(err__)));                                                   \
            throw std::runtime_error("CUDA failure");                                                                 \
        }                                                                                                            \
    } while (0)
#endif

namespace GRIM {

SerializationLayer::SerializationLayer(SerializationConfig config) : config_(std::move(config)) {}

void SerializationLayer::setConfig(const SerializationConfig& config) {
    config_ = config;
}

bool SerializationLayer::load(const SerializationLoadRequest& request) {
    const auto& cfg = request.config;
    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        SerializationLogError("[SerializationLayer::load] Invalid model dimensions");
        return false;
    }
    if (request.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        SerializationLogError(MakeLogMessage("[SerializationLayer::load] Encoder layer view mismatch (expected ",
                                            cfg.num_layers,
                                            ", got ",
                                            request.encoder_layers.size(),
                                            ")"));
        return false;
    }

    std::ifstream file(request.path, std::ios::binary | std::ios::ate);
    if (!file) {
        SerializationLogError(MakeLogMessage("[SerializationLayer::load] Failed to open: ", request.path));
        return false;
    }
    const auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        SerializationLogError("[SerializationLayer::load] Read failed");
        return false;
    }

    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!GRIMTransformer::VerifyTransformerModelBuffer(verifier)) {
        SerializationLogError("[SerializationLayer::load] Invalid FlatBuffer");
        return false;
    }

    const auto* model_fb = GRIMTransformer::GetTransformerModel(buffer.data());
    if (!model_fb) {
        SerializationLogError("[SerializationLayer::load] Failed to parse FlatBuffer");
        return false;
    }
    if (model_fb->version() != GRIM_MODEL_VERSION) {
        SerializationLogError(MakeLogMessage("[SerializationLayer::load] Version mismatch (",
                                            model_fb->version(),
                                            " != ",
                                            GRIM_MODEL_VERSION,
                                            ")"));
        return false;
    }

    const auto* fb_config = model_fb->config();
    if (!fb_config || fb_config->vocab_size() != cfg.vocab_size || fb_config->d_model() != cfg.d_model ||
        fb_config->num_layers() != cfg.num_layers || fb_config->num_heads() != cfg.num_heads) {
        SerializationLogError("[SerializationLayer::load] Config mismatch");
        return false;
    }

    // GQA: Validate num_kv_heads match (no fallback - both must be properly set)
    const int checkpoint_kv_heads = static_cast<int>(fb_config->num_kv_heads());
    
    if (checkpoint_kv_heads != cfg.num_kv_heads) {
        SerializationLogError(MakeLogMessage("[SerializationLayer::load] GQA config mismatch: checkpoint has num_kv_heads=",
                                            checkpoint_kv_heads,
                                            " but model expects num_kv_heads=",
                                            cfg.num_kv_heads,
                                            ". Cannot load MHA checkpoint into GQA model or vice versa."));
        return false;
    }
    SerializationLogInfo(MakeLogMessage("[SerializationLayer::load] GQA config: num_heads=", 
                                        fb_config->num_heads(), ", num_kv_heads=", checkpoint_kv_heads));

    const auto* fb_embeddings = model_fb->embeddings();
    if (!fb_embeddings) {
        SerializationLogError("[SerializationLayer::load] Embedding block missing");
        return false;
    }

    const auto* fb_token_vec = fb_embeddings->token_embeddings();
    std::vector<float> token_host(fb_token_vec->begin(), fb_token_vec->end());
    const int vocab_size = fb_embeddings->vocab_size();
    const int d_model = fb_embeddings->d_model();

    auto upload_device_vector = [](const std::vector<float>& host,
                                   const DeviceWriteView& view,
                                   const char* label) -> bool {
        if (host.empty()) {
            SerializationLogInfo(MakeLogMessage("[SerializationLayer::load] Skipping empty ", label));
            return true;
        }
        if (!view.ptr) {
            SerializationLogError(MakeLogMessage("[SerializationLayer::load] Missing destination for ", label));
            return false;
        }
        if (view.count != host.size()) {
            SerializationLogError(MakeLogMessage("[SerializationLayer::load] Size mismatch for ",
                                                label,
                                                " (dest=",
                                                view.count,
                                                ", src=",
                                                host.size(),
                                                ")"));
            return false;
        }
        CUDA_CHECK(cudaMemcpy(view.ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        return true;
    };

    if (cfg.use_gpu) {
        if (!request.gpu_embedding.token_embeddings.ptr) {
            SerializationLogError("[SerializationLayer::load] GPU embedder not initialized");
            return false;
        }
        if (!upload_device_vector(token_host, request.gpu_embedding.token_embeddings, "token embeddings")) {
            return false;
        }
    }

    if (request.cpu_embedding.set_tokens) {
        request.cpu_embedding.set_tokens(token_host, vocab_size, d_model);
    }

    if (fb_embeddings->use_rms_norm() && fb_embeddings->rms_gamma()) {
        std::vector<float> rms_gamma_host(fb_embeddings->rms_gamma()->begin(), fb_embeddings->rms_gamma()->end());
        if (cfg.use_gpu && request.gpu_embedding.has_rms_norm) {
            if (!upload_device_vector(rms_gamma_host, request.gpu_embedding.rms_gamma, "embedding rms gamma")) {
                return false;
            }
        }
        if (request.cpu_embedding.set_rms_gamma) {
            request.cpu_embedding.set_rms_gamma(rms_gamma_host);
        }
    }

    const auto* fb_layers = model_fb->encoder_layers();
    if (!fb_layers || fb_layers->size() != static_cast<std::size_t>(cfg.num_layers)) {
        SerializationLogError("[SerializationLayer::load] Encoder layers missing or malformed");
        return false;
    }

    for (int layer_idx = 0; layer_idx < cfg.num_layers; ++layer_idx) {
        const auto* fb_layer = fb_layers->Get(layer_idx);
        const auto& layer_view = request.encoder_layers[layer_idx];

        const auto* fb_attn = fb_layer->attention();
        std::vector<float> h_W_qkv(fb_attn->w_qkv_data()->begin(), fb_attn->w_qkv_data()->end());
        std::vector<float> h_b_qkv(fb_attn->b_qkv_data()->begin(), fb_attn->b_qkv_data()->end());
        std::vector<float> h_W_o(fb_attn->w_o_data()->begin(), fb_attn->w_o_data()->end());
        std::vector<float> h_b_o(fb_attn->b_o_data()->begin(), fb_attn->b_o_data()->end());
        if (!upload_device_vector(h_W_qkv, layer_view.attn_w_qkv, "attn.W_qkv") ||
            !upload_device_vector(h_b_qkv, layer_view.attn_b_qkv, "attn.b_qkv") ||
            !upload_device_vector(h_W_o, layer_view.attn_w_o, "attn.W_o") ||
            !upload_device_vector(h_b_o, layer_view.attn_b_o, "attn.b_o")) {
            return false;
        }

        const auto* fb_ffn = fb_layer->ffn();
        std::vector<float> h_W1(fb_ffn->w1_data()->begin(), fb_ffn->w1_data()->end());
        std::vector<float> h_b1(fb_ffn->b1_data()->begin(), fb_ffn->b1_data()->end());
        std::vector<float> h_W2(fb_ffn->w2_data()->begin(), fb_ffn->w2_data()->end());
        std::vector<float> h_b2(fb_ffn->b2_data()->begin(), fb_ffn->b2_data()->end());
        if (!upload_device_vector(h_W1, layer_view.ffn_w1, "ffn.W1") ||
            !upload_device_vector(h_b1, layer_view.ffn_b1, "ffn.b1") ||
            !upload_device_vector(h_W2, layer_view.ffn_w2, "ffn.W2") ||
            !upload_device_vector(h_b2, layer_view.ffn_b2, "ffn.b2")) {
            return false;
        }

        const auto* fb_rms1 = fb_layer->rms1();
        const auto* fb_rms2 = fb_layer->rms2();
        std::vector<float> h_rms1_gamma(fb_rms1->gamma()->begin(), fb_rms1->gamma()->end());
        std::vector<float> h_rms2_gamma(fb_rms2->gamma()->begin(), fb_rms2->gamma()->end());
        if (!upload_device_vector(h_rms1_gamma, layer_view.rms1_gamma, "rms1.gamma") ||
            !upload_device_vector(h_rms2_gamma, layer_view.rms2_gamma, "rms2.gamma")) {
            return false;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    const auto* fb_lm_head = model_fb->lm_head();
    if (fb_lm_head) {
        if (request.lm_head.projection.ptr && fb_lm_head->projection_data()) {
            std::vector<float> lm_proj_host(fb_lm_head->projection_data()->begin(), fb_lm_head->projection_data()->end());
            if (!upload_device_vector(lm_proj_host, request.lm_head.projection, "LM head projection")) {
                return false;
            }
        } else if (cfg.tie_embeddings) {
            SerializationLogInfo("[SerializationLayer::load] LM head projection tied to embeddings");
        } else {
            SerializationLogWarn("[SerializationLayer::load] LM head projection missing");
        }

        if (cfg.use_bias && request.lm_head.bias.ptr && fb_lm_head->bias_data()) {
            std::vector<float> lm_bias_host(fb_lm_head->bias_data()->begin(), fb_lm_head->bias_data()->end());
            if (!upload_device_vector(lm_bias_host, request.lm_head.bias, "LM head bias")) {
                return false;
            }
        } else if (cfg.use_bias) {
            SerializationLogWarn("[SerializationLayer::load] LM head bias missing");
        }
    }

    const auto* fb_numeric_head = model_fb->numeric_head();
    if (fb_numeric_head && request.numeric_head.enabled) {
        if (request.numeric_head.projection.ptr && fb_numeric_head->projection_data()) {
            std::vector<float> num_proj_host(fb_numeric_head->projection_data()->begin(),
                                             fb_numeric_head->projection_data()->end());
            if (!upload_device_vector(num_proj_host, request.numeric_head.projection, "Numeric head projection")) {
                return false;
            }
        } else {
            SerializationLogWarn("[SerializationLayer::load] Numeric head projection missing");
        }

        if (request.numeric_head.expect_bias && request.numeric_head.bias.ptr && fb_numeric_head->bias_data()) {
            std::vector<float> num_bias_host(fb_numeric_head->bias_data()->begin(),
                                             fb_numeric_head->bias_data()->end());
            if (!upload_device_vector(num_bias_host, request.numeric_head.bias, "Numeric head bias")) {
                return false;
            }
        } else if (request.numeric_head.expect_bias) {
            SerializationLogWarn("[SerializationLayer::load] Numeric head bias missing");
        }
    } else if (request.numeric_head.enabled) {
        SerializationLogWarn("[SerializationLayer::load] Numeric head block missing");
    }

    // Load ScratchBlock weights (if present in file and destination is provided)
    const auto* fb_scratch_block = model_fb->scratch_block();
    if (fb_scratch_block && fb_scratch_block->enabled()) {
        if (request.scratch_block.atom_type_embeddings.ptr && fb_scratch_block->atom_type_embeddings()) {
            std::vector<float> sb_atom_emb(fb_scratch_block->atom_type_embeddings()->begin(),
                                           fb_scratch_block->atom_type_embeddings()->end());
            if (!upload_device_vector(sb_atom_emb, request.scratch_block.atom_type_embeddings, "ScratchBlock atom_type_embeddings")) {
                SerializationLogWarn("[SerializationLayer::load] Failed to load ScratchBlock atom_type_embeddings");
            }
        }
        
        if (request.scratch_block.atom_projection.ptr && fb_scratch_block->atom_projection()) {
            std::vector<float> sb_atom_proj(fb_scratch_block->atom_projection()->begin(),
                                            fb_scratch_block->atom_projection()->end());
            if (!upload_device_vector(sb_atom_proj, request.scratch_block.atom_projection, "ScratchBlock atom_projection")) {
                SerializationLogWarn("[SerializationLayer::load] Failed to load ScratchBlock atom_projection");
            }
        }
        
        // Load text feature projection (VALUE encoding path)
        if (request.scratch_block.text_feature_projection.ptr && fb_scratch_block->text_feature_projection()) {
            std::vector<float> sb_text_proj(fb_scratch_block->text_feature_projection()->begin(),
                                            fb_scratch_block->text_feature_projection()->end());
            if (!upload_device_vector(sb_text_proj, request.scratch_block.text_feature_projection, "ScratchBlock text_feature_projection")) {
                SerializationLogWarn("[SerializationLayer::load] Failed to load ScratchBlock text_feature_projection");
            }
        }
        
        SerializationLogInfo(MakeLogMessage("[SerializationLayer::load] ScratchBlock loaded: enabled=", 
                                            fb_scratch_block->enabled(),
                                            " atom_types=", fb_scratch_block->num_atom_types(),
                                            " atom_dim=", fb_scratch_block->atom_embedding_dim()));
    }

    SerializationLogInfo("[SerializationLayer::load] Model loaded successfully");
    return true;
}

bool SerializationLayer::save(const SerializationSaveRequest& request) {
    SerializationLogInfo("[SerializationLayer::save] ENTERED");
    const auto& cfg = request.sources.config;
    SerializationLogInfo(MakeLogMessage("[SerializationLayer::save] Config: vocab=", cfg.vocab_size,
                                        " d_model=", cfg.d_model,
                                        " layers=", cfg.num_layers,
                                        " encoder_layers_size=", request.sources.encoder_layers.size()));
    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        SerializationLogError("[SerializationLayer::save] Invalid model dimensions");
        return false;
    }
    if (request.sources.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        SerializationLogError(MakeLogMessage("[SerializationLayer::save] Encoder layer view mismatch (expected ",
                                            cfg.num_layers,
                                            ", got ",
                                            request.sources.encoder_layers.size(),
                                            ")"));
        return false;
    }

    flatbuffers::FlatBufferBuilder builder(32ULL * 1024ULL * 1024ULL);
    
    // Convert PositionalEncodingType to FlatBuffer enum
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
    
    // STRICT: No backward compatibility - use positional_encoding enum only
    // Old loaders must be updated to read the enum field
    
    auto fb_config = GRIMTransformer::CreateModelConfig(
        builder,
        static_cast<uint32_t>(cfg.vocab_size),
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.num_layers),
        static_cast<uint32_t>(cfg.num_heads),
        static_cast<uint32_t>(cfg.num_kv_heads),  // GQA: num_kv_heads (no fallback - must be set)
        static_cast<uint32_t>(cfg.d_ff),
        static_cast<uint32_t>(cfg.max_seq_len),
        cfg.dropout_rate,
        cfg.dropout_rate,
        fb_pos_enc,
        false,  // use_alibi deprecated - loaders must use positional_encoding field
        true,
        true,
        true,
        cfg.tie_embeddings,
        cfg.use_bias);

    auto download_device_vector = [](const DeviceReadView& view, const char* label) -> std::vector<float> {
        if (view.count == 0) {
            SerializationLogInfo(MakeLogMessage("[SerializationLayer::save] Skipping empty ", label));
            return {};
        }
        if (!view.ptr) {
            SerializationLogError(MakeLogMessage("[SerializationLayer::save] Missing ", label, " buffer"));
            return {};
        }
        SerializationLogInfo(MakeLogMessage("[SerializationLayer::save] Downloading ", label, " (", view.count, " floats)"));
        std::vector<float> host(view.count);
        cudaError_t err = cudaMemcpy(host.data(), view.ptr, view.count * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            SerializationLogError(MakeLogMessage("[SerializationLayer::save] Failed to download ",
                                                label,
                                                ": ",
                                                cudaGetErrorString(err)));
            return {};
        }
        return host;
    };

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_token_embed = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_rms_gamma = 0;
    bool use_rms = false;

    if (cfg.use_gpu && request.sources.gpu_embedding.token_embeddings.ptr) {
        auto token_embed_data = download_device_vector(request.sources.gpu_embedding.token_embeddings, "token embeddings");
        if (token_embed_data.empty() && request.sources.gpu_embedding.token_embeddings.count > 0) {
            return false;
        }
        fb_token_embed = builder.CreateVector(token_embed_data);

        if (request.sources.gpu_embedding.has_rms_norm) {
            auto rms_gamma = download_device_vector(request.sources.gpu_embedding.rms_gamma, "embedding rms gamma");
            if (rms_gamma.empty() && request.sources.gpu_embedding.rms_gamma.count > 0) {
                return false;
            }
            fb_rms_gamma = builder.CreateVector(rms_gamma);
            use_rms = true;
        }
    } else {
        const auto& cpu_embed = request.sources.cpu_embedding;
        if (cpu_embed.token_data.empty()) {
            SerializationLogError("[SerializationLayer::save] CPU embedding data is missing");
            return false;
        }
        fb_token_embed = builder.CreateVector(cpu_embed.token_data);
        if (cpu_embed.has_rms_norm) {
            fb_rms_gamma = builder.CreateVector(cpu_embed.rms_gamma);
            use_rms = true;
        }
    }

    auto fb_embeddings = GRIMTransformer::CreateEmbeddingWeights(
        builder,
        fb_token_embed,
        0,
        fb_rms_gamma,
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
        const auto& layer_view = request.sources.encoder_layers[layer_idx];

        std::vector<float> h_W_qkv;
        std::vector<float> h_b_qkv;
        std::vector<float> h_W_o;
        std::vector<float> h_b_o;
        if (!download_into(h_W_qkv, layer_view.attn_w_qkv, "attn.W_qkv") ||
            !download_into(h_b_qkv, layer_view.attn_b_qkv, "attn.b_qkv") ||
            !download_into(h_W_o, layer_view.attn_w_o, "attn.W_o") ||
            !download_into(h_b_o, layer_view.attn_b_o, "attn.b_o")) {
            return false;
        }

        auto fb_attn = GRIMTransformer::CreateAttentionWeights(
            builder,
            builder.CreateVector(h_W_qkv),
            builder.CreateVector(h_b_qkv),
            0,
            0,
            0,
            0,
            0,
            0,
            builder.CreateVector(h_W_o),
            builder.CreateVector(h_b_o),
            static_cast<uint32_t>(cfg.d_model),
            static_cast<uint32_t>(cfg.num_heads),
            static_cast<uint32_t>(cfg.num_kv_heads),  // GQA: num_kv_heads
            true);

        std::vector<float> h_W1;
        std::vector<float> h_b1;
        std::vector<float> h_W2;
        std::vector<float> h_b2;
        if (!download_into(h_W1, layer_view.ffn_w1, "ffn.W1") ||
            !download_into(h_b1, layer_view.ffn_b1, "ffn.b1") ||
            !download_into(h_W2, layer_view.ffn_w2, "ffn.W2") ||
            !download_into(h_b2, layer_view.ffn_b2, "ffn.b2")) {
            return false;
        }

        auto fb_ffn = GRIMTransformer::CreateFFNWeights(
            builder,
            builder.CreateVector(h_W1),
            builder.CreateVector(h_b1),
            builder.CreateVector(h_W2),
            builder.CreateVector(h_b2),
            static_cast<uint32_t>(cfg.d_model),
            static_cast<uint32_t>(cfg.d_ff));

        std::vector<float> h_rms1_gamma;
        std::vector<float> h_rms2_gamma;
        if (!download_into(h_rms1_gamma, layer_view.rms1_gamma, "rms1.gamma") ||
            !download_into(h_rms2_gamma, layer_view.rms2_gamma, "rms2.gamma")) {
            return false;
        }

        auto fb_rms1 = GRIMTransformer::CreateRMSNormWeights(
            builder,
            builder.CreateVector(h_rms1_gamma));
        auto fb_rms2 = GRIMTransformer::CreateRMSNormWeights(
            builder,
            builder.CreateVector(h_rms2_gamma));

        fb_layers.push_back(GRIMTransformer::CreateEncoderLayerWeights(builder, fb_attn, fb_ffn, fb_rms1, fb_rms2, static_cast<uint32_t>(layer_idx)));
    }

    auto fb_encoder_layers = builder.CreateVector(fb_layers);

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_proj = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_lm_bias = 0;
    const auto& lm_head_view = request.sources.lm_head;
    if (lm_head_view.has_projection && lm_head_view.projection.ptr) {
        auto lm_proj = download_device_vector(lm_head_view.projection, "LM head projection");
        if (lm_proj.empty() && lm_head_view.projection.count > 0) {
            return false;
        }
        fb_lm_proj = builder.CreateVector(lm_proj);
    } else if (cfg.tie_embeddings) {
        SerializationLogInfo("[SerializationLayer::save] LM head projection tied to embeddings");
    } else {
        SerializationLogWarn("[SerializationLayer::save] LM head weights not initialized");
    }

    if (cfg.use_bias) {
        if (lm_head_view.has_bias && lm_head_view.bias.ptr) {
            auto lm_bias = download_device_vector(lm_head_view.bias, "LM head bias");
            if (lm_bias.empty() && lm_head_view.bias.count > 0) {
                return false;
            }
            fb_lm_bias = builder.CreateVector(lm_bias);
        } else {
            SerializationLogWarn("[SerializationLayer::save] LM head bias not initialized");
        }
    }

    auto fb_lm_head = GRIMTransformer::CreateLMHeadWeights(
        builder,
        fb_lm_proj,
        fb_lm_bias,
        static_cast<uint32_t>(cfg.d_model),
        static_cast<uint32_t>(cfg.vocab_size),
        cfg.tie_embeddings,
        cfg.use_bias && lm_head_view.bias.ptr != nullptr);

    flatbuffers::Offset<GRIMTransformer::NumericHeadWeights> fb_numeric_head = 0;
    const auto& num_view = request.sources.numeric_head;
    if (num_view.enabled && num_view.projection.ptr) {
        auto num_proj = download_device_vector(num_view.projection, "Numeric head projection");
        if (num_proj.empty() && num_view.projection.count > 0) {
            return false;
        }
        flatbuffers::Offset<flatbuffers::Vector<float>> fb_num_proj = builder.CreateVector(num_proj);

        flatbuffers::Offset<flatbuffers::Vector<float>> fb_num_bias = 0;
        if (num_view.has_bias && num_view.bias.ptr) {
            auto num_bias = download_device_vector(num_view.bias, "Numeric head bias");
            if (num_bias.empty() && num_view.bias.count > 0) {
                return false;
            }
            fb_num_bias = builder.CreateVector(num_bias);
        }

        fb_numeric_head = GRIMTransformer::CreateNumericHeadWeights(
            builder,
            fb_num_proj,
            fb_num_bias,
            static_cast<uint32_t>(cfg.d_model),
            cfg.use_bias && num_view.bias.ptr != nullptr);
    } else if (num_view.enabled) {
        SerializationLogWarn("[SerializationLayer::save] Numeric head projection missing");
    }

    // Save ScratchBlock weights (if enabled)
    flatbuffers::Offset<GRIMTransformer::ScratchBlockWeights> fb_scratch_block = 0;
    const auto& sb_view = request.sources.scratch_block;
    if (sb_view.enabled && sb_view.atom_type_embeddings.ptr) {
        auto sb_atom_emb = download_device_vector(sb_view.atom_type_embeddings, "ScratchBlock atom_type_embeddings");
        auto sb_atom_proj = download_device_vector(sb_view.atom_projection, "ScratchBlock atom_projection");
        auto sb_text_proj = download_device_vector(sb_view.text_feature_projection, "ScratchBlock text_feature_projection");
        
        if (!sb_atom_emb.empty() || sb_view.atom_type_embeddings.count == 0) {
            fb_scratch_block = GRIMTransformer::CreateScratchBlockWeights(
                builder,
                builder.CreateVector(sb_atom_emb),
                builder.CreateVector(sb_atom_proj),
                builder.CreateVector(sb_text_proj),  // text feature projection
                static_cast<uint32_t>(sb_view.num_atom_types),
                static_cast<uint32_t>(sb_view.atom_embedding_dim),
                static_cast<uint32_t>(sb_view.d_model),
                sb_view.atom_scale,
                sb_view.enabled);
            SerializationLogInfo(MakeLogMessage("[SerializationLayer::save] ScratchBlock saved: atom_emb=",
                                                sb_atom_emb.size(), " atom_proj=", sb_atom_proj.size(),
                                                " text_proj=", sb_text_proj.size()));
        }
    }

    const auto now = std::chrono::system_clock::now();
    const auto timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());

    auto fb_optimizer = builder.CreateString("AdamW");
    auto fb_model_name = builder.CreateString("GRIM-text GPU model");
    auto fb_metadata = GRIMTransformer::CreateTrainingMetadata(
        builder,
        fb_optimizer,
        0.0f,
        0,
        0,
        0,
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        timestamp,
        0,
        fb_model_name);

    auto fb_model = GRIMTransformer::CreateTransformerModel(
        builder,
        request.model_version,
        fb_config,
        fb_embeddings,
        fb_encoder_layers,
        fb_lm_head,
        fb_numeric_head,
        fb_scratch_block,
        fb_metadata,
        0,
        0,
        timestamp,
        timestamp);

    builder.Finish(fb_model, "GRMT");

    const std::string final_path = request.path;
    const std::string temp_path = final_path + config_.temp_suffix;

    try {
        const auto parent = std::filesystem::path(final_path).parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
    } catch (const std::exception& ex) {
        SerializationLogError(MakeLogMessage("[SerializationLayer::save] Failed to create directories: ", ex.what()));
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);

    {
        std::ofstream file(temp_path, std::ios::binary);
        if (!file) {
            SerializationLogError(MakeLogMessage("[SerializationLayer::save] Failed to open: ", temp_path));
            return false;
        }
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        if (!file) {
            SerializationLogError("[SerializationLayer::save] Write failed");
            return false;
        }
        file.flush();
    }

    if (!config_.atomic_write) {
        std::filesystem::rename(temp_path, final_path, ec);
        if (ec) {
            SerializationLogError(MakeLogMessage("[SerializationLayer::save] Rename failed: ", ec.message()));
            return false;
        }
    } else {
    #ifdef _WIN32
        if (!MoveFileExA(temp_path.c_str(), final_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            SerializationLogError("[SerializationLayer::save] Rename failed");
            return false;
        }
    #else
        if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
            SerializationLogError("[SerializationLayer::save] Rename failed");
            return false;
        }
    #endif
    }

    SerializationLogInfo(MakeLogMessage("[SerializationLayer::save] Model saved: ", final_path));
    return true;
}

void SerializationLayer::onConfigure(const Dimensions& dims) {
    setDimensions(dims);
}

void SerializationLayer::forwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace) {
    (void)io;
    (void)workspace;
}

void SerializationLayer::backwardImpl(const LayerIO<float>& io, LayerWorkspace<float>* workspace) {
    (void)io;
    (void)workspace;
}

void SerializationLayer::applyGradientsImpl(value_type learning_rate) {
    (void)learning_rate;
}

} // namespace GRIM
