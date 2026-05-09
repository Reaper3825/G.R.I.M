#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
// GPUGrimEncoder::Impl - Layer Container
// Owns the encoder layers; forward pass uses autograd in AutogradTraining.cu
//
// Hyperparameters come from EncoderLayerConstructionHP, the grouped read view
// owned by HyperparameterGroupings.hpp. Runtime device handles (PBM, stream,
// cuBLAS, init seed) come from EncoderRuntimeBindings; config and runtime are
// still deliberately separate ownership lanes.
//======================================================//
struct GPUGrimEncoder::Impl {
    HyperParameters::EncoderLayerConstructionHP config_;
    EncoderRuntimeBindings bindings_;
    std::vector<std::unique_ptr<GPUEncoderLayer>> gpu_layers_;

    Impl(const HyperParameters::EncoderLayerConstructionHP& config,
         const EncoderRuntimeBindings& bindings)
        : config_(config), bindings_(bindings)
    {
        if (!bindings.pos_encoding) {
            throw std::runtime_error("[GPUGrimEncoder] pos_encoding is NULL — "
                                     "PBM must be initialized BEFORE encoder construction");
        }

        EncodingConfig enc_cfg{};
        enc_cfg.d_model = config.d_model;
        enc_cfg.num_heads = config.num_heads;
        enc_cfg.num_kv_heads = config.num_kv_heads;  // GQA support
        enc_cfg.d_ff = config.d_ff;
        enc_cfg.rms_epsilon = config.rms_epsilon;
        enc_cfg.causal_mask = config.causal_mask;
        enc_cfg.use_flash_attention = config.use_flash_attention;
        enc_cfg.min_seq_len_for_flash = config.min_seq_len_for_flash;
        enc_cfg.stream = bindings.stream;
        enc_cfg.cublas_handle = bindings.cublas_handle;
        enc_cfg.pos_encoding = bindings.pos_encoding;
        enc_cfg.use_layer_scale = config.use_layer_scale;
        enc_cfg.layer_scale_init = config.layer_scale_init;
        enc_cfg.center_encoder_residuals = config.center_encoder_residuals;
        enc_cfg.use_bias = config.use_bias;
        enc_cfg.dropout_rate = config.dropout_rate;
        enc_cfg.attention_dropout = config.attention_dropout;
        enc_cfg.qk_norm_enabled = config.qk_norm_enabled;

        for (int i = 0; i < config.num_layers; ++i) {
            // Pattern B: Layer self-allocates and Xavier-inits its own weights.
            // Seed offsets per layer: base + 2 + layer*10
            const uint64_t layer_seed = bindings.weight_seed + 2 + i * 10;
            gpu_layers_.emplace_back(std::make_unique<GPUEncoderLayer>(
                enc_cfg, layer_seed, bindings.residual_scale, config.layer_scale_init));
        }
    }
};

GPUGrimEncoder::GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& config,
                               const EncoderRuntimeBindings& bindings)
    : pImpl(new Impl(config, bindings))
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
    if (!pImpl) {
        throw std::runtime_error("GPUGrimEncoder::setFlashAttention called before initialization");
    }
    if (min_seq_len <= 0) {
        throw std::runtime_error("GPUGrimEncoder::setFlashAttention: min_seq_len must be > 0 (configured from hyperparameters)");
    }
    for (auto& layer : pImpl->gpu_layers_) {
        if (layer) {
            layer->setFlashAttention(enable, min_seq_len);
        }
    }
}

#endif // USE_CUDA

} // namespace GRIM
