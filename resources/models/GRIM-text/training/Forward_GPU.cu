#define USE_CUDA

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"

// Simple logging macros (legacy ForwardOps_Logging deleted per Rule 20)
#define FWD_ERROR(msg) do { std::cerr << msg << std::endl; } while(0)

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
// GPUGrimEncoder::Impl - Layer Container
// Owns the encoder layers; forward pass uses autograd in AutogradTraining.cu
//======================================================//
struct GPUGrimEncoder::Impl {
    EncoderConfig config_;
    std::vector<std::unique_ptr<GPUEncoderLayer>> gpu_layers_;
    
    explicit Impl(const EncoderConfig& config)
        : config_(config)
    {
        EncodingConfig enc_cfg{};
        enc_cfg.d_model = config.d_model;
        enc_cfg.num_heads = config.num_heads;
        enc_cfg.num_kv_heads = config.num_kv_heads;  // GQA support
        enc_cfg.d_ff = config.d_ff;
        enc_cfg.rms_epsilon = config.rms_epsilon;
        enc_cfg.causal_mask = config.causal_mask;
        enc_cfg.use_flash_attention = config.use_flash_attention;
        enc_cfg.stream = config.stream;
        enc_cfg.cublas_handle = config.cublas_handle;
        enc_cfg.pos_encoding = config.pos_encoding;  // RoPE/ALiBi positional encoding
        // Issue #109 FIX: Propagate LayerScale config (was relying on EncodingConfig defaults!)
        enc_cfg.use_layer_scale = config.use_layer_scale;
        enc_cfg.layer_scale_init = config.layer_scale_init;
        enc_cfg.center_encoder_residuals = config.center_encoder_residuals;
        enc_cfg.use_bias = config.use_bias;

        if (!config.pos_encoding) {
            FWD_ERROR("[GPUGrimEncoder] FATAL: pos_encoding is NULL");
            FWD_ERROR("[GPUGrimEncoder] PBM must be initialized BEFORE encoder construction.");
            std::abort();
        }

        for (int i = 0; i < config.num_layers; ++i) {
            gpu_layers_.emplace_back(std::make_unique<GPUEncoderLayer>(enc_cfg));
            // NOTE: FeedForwardLayer now requires weights in constructor (Option A)
            // EncodingLayer::useExternalWeights() handles FFN creation with external weights.
            // Rule 20: No backwards compatibility - single source of truth for weights.
        }
    }
};

// Constructor: Creates all encoder layers
GPUGrimEncoder::GPUGrimEncoder(const EncoderConfig& config)
    : pImpl(new Impl(config))
{
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
