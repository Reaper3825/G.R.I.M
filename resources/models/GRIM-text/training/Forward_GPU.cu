#define USE_CUDA

#include <memory>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/AttentionDiagnostics.hpp"
#include "../Layers/ForwardOps/ForwardOps_Logging.hpp"

namespace GRIM {

#ifdef USE_CUDA

struct GPUGrimEncoder::Impl {
    EncoderConfig config_;
    std::vector<std::unique_ptr<GPUEncoderLayer>> gpu_layers_;
    
    explicit Impl(const EncoderConfig& config)
        : config_(config)
    {
        EncodingConfig enc_cfg{};
        enc_cfg.d_model = config.d_model;
        enc_cfg.num_heads = config.num_heads;
        enc_cfg.num_kv_heads = config.num_kv_heads;  // GQA support!
        enc_cfg.d_ff = config.d_ff;
        enc_cfg.rms_epsilon = config.rms_epsilon;
        enc_cfg.causal_mask = config.causal_mask;
        enc_cfg.use_flash_attention = config.use_flash_attention;
        enc_cfg.stream = config.stream;  // Use stream from encoder config
        enc_cfg.cublas_handle = config.cublas_handle;  // Centralized handle (Rule 22)
        enc_cfg.pos_encoding = config.pos_encoding;  // RoPE positional encoding

        // PBM DIAGNOSTIC: Log what positional encoding the encoder receives
        FWD_INFO("[GPUGrimEncoder] PBM config received:");
        FWD_INFO("  pos_encoding ptr: " << (void*)config.pos_encoding);
        if (config.pos_encoding) {
            FWD_INFO("  valid: " << (config.pos_encoding->valid ? "YES" : "NO"));
            FWD_INFO("  rope_inv_freq: " << (void*)config.pos_encoding->rope_inv_freq);
            FWD_INFO("  alibi_slopes: " << (void*)config.pos_encoding->alibi_slopes);
            FWD_INFO("  rotary_dim: " << config.pos_encoding->rotary_dim);
            FWD_INFO("  num_heads: " << config.pos_encoding->num_heads);
        } else {
            FWD_ERROR("[GPUGrimEncoder] FATAL: pos_encoding is NULL");
            FWD_ERROR("[GPUGrimEncoder] Attention requires positional encoding (PBM hybrid).");
            FWD_ERROR("[GPUGrimEncoder] PBM must be initialized BEFORE encoder construction.");
            FWD_ERROR("[GPUGrimEncoder] Check TrainingOps.cu initialization order.");
            std::abort();
        }

        for (int i = 0; i < config.num_layers; ++i) {
            gpu_layers_.emplace_back(std::make_unique<GPUEncoderLayer>(enc_cfg));
            gpu_layers_.back()->ensureWeightStorage();  // No args - uses config
        }
    }
    
    std::vector<Vector> forward(const std::vector<Vector>& input,
                               const ALiBiPositionalBias* alibi) {
        (void)alibi;
        // CPU path not implemented for EncodingLayer; return input unchanged.
        return input;
    }
    
    void forwardGPU(const float* d_embeddings, float* d_output,
                   int batch_size, int seq_len, float* d_workspace,
                   const ALiBiPositionalBias* /*alibi*/,
                   float** cache_Q_layers = nullptr,
                   float** cache_K_layers = nullptr,
                   float** cache_V_layers = nullptr,
                   EncoderLayerCache* layer_caches = nullptr,
                   const FlashAttentionBF16Scratch* fa_scratch = nullptr,
                   float* entropy_output = nullptr) {  // Per-layer entropy [num_layers * batch_size * num_heads]
        const float* d_layer_input = d_embeddings;
        const int total_tokens = batch_size * seq_len;

        auto encoder_start = std::chrono::high_resolution_clock::now();
        std::vector<double> layer_timings;

        for (size_t layer_idx = 0; layer_idx < gpu_layers_.size(); ++layer_idx) {
            auto layer_start = std::chrono::high_resolution_clock::now();
            
            // Update K-trace layer index
            auto& k_trace = getKTensorTrace();
            k_trace.current_layer = static_cast<int>(layer_idx);
            
            LayerWorkspace<float> ws{};
            ws.data = d_workspace;
            ws.bytes = gpu_layers_[layer_idx]->requiredWorkspaceBytes(total_tokens, seq_len);

            EncodingForwardArgs args{};
            args.input = d_layer_input;
            args.output = d_output;
            args.total_tokens = total_tokens;
            args.seq_len = seq_len;
            args.stream = config_.stream;  // Use stream from encoder config
            // Wire caches if provided
            if (cache_Q_layers && cache_K_layers && cache_V_layers) {
                args.cache_q = cache_Q_layers[layer_idx];
                args.cache_k = cache_K_layers[layer_idx];
                args.cache_v = cache_V_layers[layer_idx];
            }
            if (layer_caches) {
                args.cache_ln1_out = layer_caches[layer_idx].ln1_output;
                args.cache_attn_input = layer_caches[layer_idx].attn_input;
                args.cache_attn_bhsd = layer_caches[layer_idx].attn_bhsd;
                args.cache_softmax_lse = layer_caches[layer_idx].softmax_lse;
                args.cache_attn_output = layer_caches[layer_idx].attn_output;
                args.cache_residual1 = layer_caches[layer_idx].residual1;
                args.cache_ln2_out = layer_caches[layer_idx].ln2_output;
                args.cache_ffn_pre_gelu = layer_caches[layer_idx].ffn_pre_gelu;
                args.cache_ffn_output = layer_caches[layer_idx].ffn_output;
                args.cache_layer_output = layer_caches[layer_idx].layer_output;
            }
            if (fa_scratch) {
                args.fa_q_bf16 = fa_scratch->q;
                args.fa_k_bf16 = fa_scratch->k;
                args.fa_v_bf16 = fa_scratch->v;
                args.fa_out_bf16 = fa_scratch->out;
                args.fa_q_bf16_elems = fa_scratch->q_elems;
                args.fa_kv_bf16_elems = fa_scratch->kv_elems;
            }
            // Wire entropy output if provided (size: batch_size * num_heads per layer)
            if (entropy_output) {
                const int num_heads = config_.num_heads;
                args.entropy_output = entropy_output + (layer_idx * batch_size * num_heads);
            }

            // Set workspace before forward
            gpu_layers_[layer_idx]->setWorkspace(ws.data, ws.bytes);
            gpu_layers_[layer_idx]->forward(args);
            d_layer_input = d_output;
            
            auto layer_end = std::chrono::high_resolution_clock::now();
            double layer_ms = std::chrono::duration<double, std::milli>(layer_end - layer_start).count();
            layer_timings.push_back(layer_ms);
        }
        
        // Log all layer timings
        auto encoder_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(encoder_end - encoder_start).count();
        fprintf(stderr, "[ENCODER_TIMING] Total=%.2fms Layers=[", total_ms);
        for (size_t i = 0; i < layer_timings.size(); ++i) {
            fprintf(stderr, "L%zu:%.2fms", i, layer_timings[i]);
            if (i < layer_timings.size() - 1) fprintf(stderr, ", ");
        }
        fprintf(stderr, "]\n");
    }
};

GPUGrimEncoder::GPUGrimEncoder(const EncoderConfig& config)
    : pImpl(new Impl(config))
{
}

std::vector<Vector> GPUGrimEncoder::forward(
    const std::vector<Vector>& embeddings,
    const ALiBiPositionalBias* alibi)
{
    return pImpl->forward(embeddings, alibi);
}

void GPUGrimEncoder::forwardGPU(
    const float* d_embeddings, float* d_output,
    int batch_size, int seq_len, float* d_workspace,
    const ALiBiPositionalBias* alibi,
    float** cache_Q_layers,
    float** cache_K_layers,
    float** cache_V_layers,
    EncoderLayerCache* layer_caches,
    const FlashAttentionBF16Scratch* fa_scratch,
    float* entropy_output)
{
    pImpl->forwardGPU(d_embeddings, d_output, batch_size, seq_len, d_workspace, alibi,
                      cache_Q_layers, cache_K_layers, cache_V_layers, layer_caches, fa_scratch, entropy_output);
}

GPUEncoderLayer* GPUGrimEncoder::getLayer(int index) {
    if (index < 0 || index >= static_cast<int>(pImpl->gpu_layers_.size())) {
        return nullptr;
    }
    return pImpl->gpu_layers_[index].get();
}

const GPUEncoderLayer* GPUGrimEncoder::getLayer(int index) const {
    if (index < 0 || index >= static_cast<int>(pImpl->gpu_layers_.size())) {
        return nullptr;
    }
    return pImpl->gpu_layers_[index].get();
}

int GPUGrimEncoder::getNumLayers() const {
    return static_cast<int>(pImpl->gpu_layers_.size());
}

void GPUGrimEncoder::setFlashAttention(bool enable, int min_seq_len) {
    if (!pImpl) return;
    for (auto& layer : pImpl->gpu_layers_) {
        if (layer) {
            layer->setFlashAttention(enable, min_seq_len);
        }
    }
}

#endif // USE_CUDA

} // namespace GRIM
