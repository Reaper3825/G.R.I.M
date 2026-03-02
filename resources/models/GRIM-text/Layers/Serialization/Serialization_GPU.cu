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

SerializationLayer::SerializationLayer(SerializationConfig config) : config_(std::move(config)) {}

void SerializationLayer::setConfig(const SerializationConfig& config) {
    config_ = config;
}

bool SerializationLayer::load(const SerializationLoadRequest& request) {
    const auto& cfg = request.config;
    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Invalid model dimensions");
        return false;
    }
    if (request.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        GRIM::Logging::EmitModuleError(kLogModule, Msg("[load] Encoder layer mismatch (expected ",
                                            cfg.num_layers, ", got ", request.encoder_layers.size(), ")"));
        return false;
    }

    std::ifstream file(request.path, std::ios::binary | std::ios::ate);
    if (!file) {
        GRIM::Logging::EmitModuleError(kLogModule, Msg("[load] Failed to open: ", request.path));
        return false;
    }
    const auto file_size = static_cast<std::size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), file_size)) {
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Read failed");
        return false;
    }

    flatbuffers::Verifier verifier(buffer.data(), buffer.size());
    if (!GRIMTransformer::VerifyTransformerModelBuffer(verifier)) {
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Invalid FlatBuffer");
        return false;
    }

    const auto* model_fb = GRIMTransformer::GetTransformerModel(buffer.data());
    if (!model_fb) {
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Failed to parse FlatBuffer");
        return false;
    }
    if (model_fb->version() != GRIM_MODEL_VERSION) {
        GRIM::Logging::EmitModuleError(kLogModule, Msg("[load] Version mismatch (",
                                            model_fb->version(), " != ", GRIM_MODEL_VERSION, ")"));
        return false;
    }

    const auto* fb_config = model_fb->config();
    if (!fb_config || fb_config->vocab_size() != cfg.vocab_size || fb_config->d_model() != cfg.d_model ||
        fb_config->num_layers() != cfg.num_layers || fb_config->num_heads() != cfg.num_heads) {
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Config mismatch");
        return false;
    }

    const int checkpoint_kv_heads = static_cast<int>(fb_config->num_kv_heads());
    if (checkpoint_kv_heads != cfg.num_kv_heads) {
        GRIM::Logging::EmitModuleError(kLogModule, Msg("[load] GQA mismatch: checkpoint num_kv_heads=",
                                            checkpoint_kv_heads, " but model expects ", cfg.num_kv_heads));
        return false;
    }
    GRIM::Logging::EmitModuleInfo(kLogModule, Msg("[load] GQA: num_heads=", 
                                        fb_config->num_heads(), " num_kv_heads=", checkpoint_kv_heads));

    const auto* fb_embeddings = model_fb->embeddings();
    if (!fb_embeddings) {
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Embedding block missing");
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
            return true;
        }
        if (!view.ptr) {
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[load] Missing destination for ", label));
            return false;
        }
        if (view.count != host.size()) {
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[load] Size mismatch for ", label,
                                                " (dest=", view.count, ", src=", host.size(), ")"));
            return false;
        }
        CUDA_CHECK(cudaMemcpy(view.ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
        return true;
    };

    if (cfg.use_gpu) {
        if (!request.gpu_embedding.token_embeddings.ptr) {
            GRIM::Logging::EmitModuleError(kLogModule, "[load] GPU embedder not initialized");
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
        GRIM::Logging::EmitModuleError(kLogModule, "[load] Encoder layers missing or malformed");
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
        std::vector<float> h_W_gate;
        if (fb_ffn->w_gate_data()) {
            h_W_gate.assign(fb_ffn->w_gate_data()->begin(), fb_ffn->w_gate_data()->end());
        }
        std::vector<float> h_W1(fb_ffn->w1_data()->begin(), fb_ffn->w1_data()->end());
        std::vector<float> h_W2(fb_ffn->w2_data()->begin(), fb_ffn->w2_data()->end());
        std::vector<float> h_b2(fb_ffn->b2_data()->begin(), fb_ffn->b2_data()->end());
        if ((!h_W_gate.empty() && !upload_device_vector(h_W_gate, layer_view.ffn_w_gate, "ffn.W_gate")) ||
            !upload_device_vector(h_W1, layer_view.ffn_w1, "ffn.W1") ||
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
        
        // Issue #148: Sandwich norm gammas REMOVED from model.
        // Old checkpoints may contain rms_post_attn/rms_post_ffn — silently skip them.
        // (layer_view.rms_post_attn_gamma.ptr will be null since model doesn't allocate them)

        // LayerScale (Issue #109) — single scalar per sublayer
        // Rule 20: If model expects LayerScale, checkpoint MUST have it. No silent fallback.
        if (layer_view.layer_scale1.ptr) {
            if (!fb_layer->layer_scale1() || fb_layer->layer_scale1()->size() == 0) {
                GRIM::Logging::EmitModuleError(kLogModule,
                    "[load] Checkpoint missing layer_scale1 for layer " + std::to_string(layer_idx)
                    + " but model requires use_layer_scale=true. Cannot load incompatible checkpoint.");
                return false;
            }
            std::vector<float> h_ls1(fb_layer->layer_scale1()->begin(), fb_layer->layer_scale1()->end());
            if (!upload_device_vector(h_ls1, layer_view.layer_scale1, "layer_scale1")) {
                return false;
            }
        }
        if (layer_view.layer_scale2.ptr) {
            if (!fb_layer->layer_scale2() || fb_layer->layer_scale2()->size() == 0) {
                GRIM::Logging::EmitModuleError(kLogModule,
                    "[load] Checkpoint missing layer_scale2 for layer " + std::to_string(layer_idx)
                    + " but model requires use_layer_scale=true. Cannot load incompatible checkpoint.");
                return false;
            }
            std::vector<float> h_ls2(fb_layer->layer_scale2()->begin(), fb_layer->layer_scale2()->end());
            if (!upload_device_vector(h_ls2, layer_view.layer_scale2, "layer_scale2")) {
                return false;
            }
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
        } else if (!cfg.tie_embeddings) {
            GRIM::Logging::EmitModuleError(kLogModule, "[load] LM head projection missing (tie_embeddings=false)");
            return false;
        }

        if (cfg.use_bias && request.lm_head.bias.ptr && fb_lm_head->bias_data()) {
            std::vector<float> lm_bias_host(fb_lm_head->bias_data()->begin(), fb_lm_head->bias_data()->end());
            if (!upload_device_vector(lm_bias_host, request.lm_head.bias, "LM head bias")) {
                return false;
            }
        } else if (cfg.use_bias) {
            GRIM::Logging::EmitModuleError(kLogModule, "[load] LM head bias missing (use_bias=true)");
            return false;
        }
    }

    const auto* fb_scratch_block = model_fb->scratch_block();
    if (fb_scratch_block && fb_scratch_block->enabled()) {
        if (request.scratch_block.atom_type_embeddings.ptr && fb_scratch_block->atom_type_embeddings()) {
            std::vector<float> sb_atom_emb(fb_scratch_block->atom_type_embeddings()->begin(),
                                           fb_scratch_block->atom_type_embeddings()->end());
            if (!upload_device_vector(sb_atom_emb, request.scratch_block.atom_type_embeddings, "ScratchBlock atom_type_embeddings")) {
                return false;
            }
        }
        
        if (request.scratch_block.atom_projection.ptr && fb_scratch_block->atom_projection()) {
            std::vector<float> sb_atom_proj(fb_scratch_block->atom_projection()->begin(),
                                            fb_scratch_block->atom_projection()->end());
            if (!upload_device_vector(sb_atom_proj, request.scratch_block.atom_projection, "ScratchBlock atom_projection")) {
                return false;
            }
        }
        
        // text_feature_projection ELIMINATED — old checkpoints may contain it, silently ignored
        // (same pattern as value_extraction_weight/bias below)

        // Old checkpoints may contain value_extraction_weight/bias — silently ignore them.
        // The extraction head has been removed from the architecture.
        
        GRIM::Logging::EmitModuleInfo(kLogModule, Msg("[load] ScratchBlock: atom_types=", 
                                            fb_scratch_block->num_atom_types(),
                                            " atom_dim=", fb_scratch_block->atom_embedding_dim()));
    }

    // Issue #33: Load final RMSNorm gamma (normalizes encoder output before LM head)
    const auto* fb_final_rms_gamma = model_fb->final_rms_gamma();
    if (fb_final_rms_gamma && request.final_rms_gamma.ptr) {
        std::vector<float> final_rms_data(fb_final_rms_gamma->begin(), fb_final_rms_gamma->end());
        if (!upload_device_vector(final_rms_data, request.final_rms_gamma, "final_rms_gamma")) {
            return false;
        }
        GRIM::Logging::EmitModuleInfo(kLogModule, Msg("[load] final_rms_gamma: size=", final_rms_data.size()));
    } else if (request.final_rms_gamma.ptr && !fb_final_rms_gamma) {
        // Checkpoint doesn't have final_rms_gamma but model expects it - initialize to 1.0
        GRIM::Logging::EmitModuleInfo(kLogModule, 
            "[load] final_rms_gamma not in checkpoint, using initialized values (gamma=1.0)");
    }

    GRIM::Logging::EmitModuleInfo(kLogModule, "[load] Model loaded successfully");
    return true;
}

bool SerializationLayer::save(const SerializationSaveRequest& request) {
    const auto& cfg = request.sources.config;
    if (cfg.vocab_size <= 0 || cfg.d_model <= 0 || cfg.num_layers <= 0) {
        GRIM::Logging::EmitModuleError(kLogModule, "[save] Invalid model dimensions");
        return false;
    }
    if (request.sources.encoder_layers.size() != static_cast<std::size_t>(cfg.num_layers)) {
        GRIM::Logging::EmitModuleError(kLogModule, Msg("[save] Encoder layer mismatch (expected ",
                                            cfg.num_layers, ", got ", request.sources.encoder_layers.size(), ")"));
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
            return {};
        }
        if (!view.ptr) {
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[save] Missing ", label, " buffer"));
            return {};
        }
        std::vector<float> host(view.count);
        cudaError_t err = cudaMemcpy(host.data(), view.ptr, view.count * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[save] Failed to download ", label, ": ", cudaGetErrorString(err)));
            return {};
        }
        return host;
    };

    flatbuffers::Offset<flatbuffers::Vector<float>> fb_token_embed = 0;
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_rms_gamma = 0;
    bool use_rms = true;

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
            GRIM::Logging::EmitModuleError(kLogModule, "[save] CPU embedding data is missing");
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

        std::vector<float> h_W_gate;
        std::vector<float> h_W1;
        std::vector<float> h_W2;
        std::vector<float> h_b2;
        if (!download_into(h_W_gate, layer_view.ffn_w_gate, "ffn.W_gate") ||
            !download_into(h_W1, layer_view.ffn_w1, "ffn.W1") ||
            !download_into(h_W2, layer_view.ffn_w2, "ffn.W2") ||
            !download_into(h_b2, layer_view.ffn_b2, "ffn.b2")) {
            return false;
        }

        auto fb_ffn = GRIMTransformer::CreateFFNWeights(
            builder,
            builder.CreateVector(h_W1),
            0,
            builder.CreateVector(h_W2),
            builder.CreateVector(h_b2),
            static_cast<uint32_t>(cfg.d_model),
            static_cast<uint32_t>(cfg.d_ff),
            builder.CreateVector(h_W_gate));

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

        // Issue #148: Sandwich norm gammas REMOVED from model.
        // Still write empty offsets (0) to maintain FlatBuffer schema compatibility.
        flatbuffers::Offset<GRIMTransformer::RMSNormWeights> fb_rms_post_attn = 0;
        flatbuffers::Offset<GRIMTransformer::RMSNormWeights> fb_rms_post_ffn = 0;
        // (layer_view.rms_post_attn_gamma.ptr and rms_post_ffn_gamma.ptr are null)

        // LayerScale serialization (Issue #109) — single scalar per sublayer
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
        if (lm_proj.empty() && lm_head_view.projection.count > 0) {
            return false;
        }
        fb_lm_proj = builder.CreateVector(lm_proj);
    } else if (!cfg.tie_embeddings) {
        GRIM::Logging::EmitModuleError(kLogModule, "[save] LM head projection missing (tie_embeddings=false)");
        return false;
    }

    if (cfg.use_bias) {
        if (lm_head_view.has_bias && lm_head_view.bias.ptr) {
            auto lm_bias = download_device_vector(lm_head_view.bias, "LM head bias");
            if (lm_bias.empty() && lm_head_view.bias.count > 0) {
                return false;
            }
            fb_lm_bias = builder.CreateVector(lm_bias);
        } else {
            GRIM::Logging::EmitModuleError(kLogModule, "[save] LM head bias missing (use_bias=true)");
            return false;
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

    // Save ScratchBlock weights (if enabled)
    flatbuffers::Offset<GRIMTransformer::ScratchBlockWeights> fb_scratch_block = 0;
    const auto& sb_view = request.sources.scratch_block;
    if (sb_view.enabled && sb_view.atom_type_embeddings.ptr) {
        auto sb_atom_emb = download_device_vector(sb_view.atom_type_embeddings, "ScratchBlock atom_type_embeddings");
        auto sb_atom_proj = download_device_vector(sb_view.atom_projection, "ScratchBlock atom_projection");
        // text_feature_projection ELIMINATED — write empty vector for schema compatibility
        
        if (!sb_atom_emb.empty() || sb_view.atom_type_embeddings.count == 0) {
            fb_scratch_block = GRIMTransformer::CreateScratchBlockWeights(
                builder,
                builder.CreateVector(sb_atom_emb),
                builder.CreateVector(sb_atom_proj),
                0,  // text_feature_projection — eliminated, empty for schema compat
                static_cast<uint32_t>(sb_view.num_atom_types),
                static_cast<uint32_t>(sb_view.atom_embedding_dim),
                static_cast<uint32_t>(sb_view.d_model),
                sb_view.atom_scale,
                sb_view.enabled);
            GRIM::Logging::EmitModuleInfo(kLogModule, Msg("[save] ScratchBlock: atom_emb=",
                                                sb_atom_emb.size(), " atom_proj=", sb_atom_proj.size()));
        }
    }

    // Issue #33: Save final RMSNorm gamma (normalizes encoder output before LM head)
    flatbuffers::Offset<flatbuffers::Vector<float>> fb_final_rms_gamma = 0;
    const auto& final_rms_view = request.sources.final_rms_gamma;
    if (final_rms_view.ptr && final_rms_view.count > 0) {
        auto final_rms_data = download_device_vector(final_rms_view, "final_rms_gamma");
        if (!final_rms_data.empty()) {
            fb_final_rms_gamma = builder.CreateVector(final_rms_data);
            GRIM::Logging::EmitModuleInfo(kLogModule, Msg("[save] final_rms_gamma: size=", final_rms_data.size()));
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
        0,                   // numeric_head removed - not used
        fb_scratch_block,
        fb_final_rms_gamma,  // Issue #33: Final RMSNorm gamma
        0,                   // loss_weighting (not used)
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
        GRIM::Logging::EmitModuleError(kLogModule, Msg("[save] Failed to create directories: ", ex.what()));
        return false;
    }

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);

    {
        std::ofstream file(temp_path, std::ios::binary);
        if (!file) {
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[save] Failed to open: ", temp_path));
            return false;
        }
        file.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
        if (!file) {
            GRIM::Logging::EmitModuleError(kLogModule, "[save] Write failed");
            return false;
        }
        file.flush();
    }

    if (!config_.atomic_write) {
        std::filesystem::rename(temp_path, final_path, ec);
        if (ec) {
            GRIM::Logging::EmitModuleError(kLogModule, Msg("[save] Rename failed: ", ec.message()));
            return false;
        }
    } else {
    #ifdef _WIN32
        if (!MoveFileExA(temp_path.c_str(), final_path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            GRIM::Logging::EmitModuleError(kLogModule, "[save] Rename failed");
            return false;
        }
    #else
        if (std::rename(temp_path.c_str(), final_path.c_str()) != 0) {
            GRIM::Logging::EmitModuleError(kLogModule, "[save] Rename failed");
            return false;
        }
    #endif
    }

    GRIM::Logging::EmitModuleInfo(kLogModule, Msg("[save] Model saved: ", final_path));
    return true;
}

} // namespace GRIM
