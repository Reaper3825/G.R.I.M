#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <cmath>
#include <memory>
#include <stdexcept>
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
// owned by HyperparameterGroupings.hpp. Startup model-assembly resource inputs
// (PBM, init stream) come from EncoderConstructionBindings; the Phase1 RNG seed
// is an explicit constructor input. Forward-time stream/cuBLAS handles come from
// ModelForwardRequest/AutogradContext.
//======================================================//
struct GPUGrimEncoder::Impl {
    HyperParameters::EncoderLayerConstructionHP config_;
    std::vector<std::unique_ptr<GPUEncoderLayer>> gpu_layers_;

    Impl(const HyperParameters::EncoderLayerConstructionHP& config,
         const EncoderConstructionBindings& bindings,
         uint64_t weight_seed)
        : config_(config)
    {
        if (!bindings.pos_encoding) {
            throw std::runtime_error("[GPUGrimEncoder] pos_encoding is NULL — "
                                     "PBM must be initialized BEFORE encoder construction");
        }
        if (!std::isfinite(config.residual_scale) || config.residual_scale <= 0.0f) {
            throw std::runtime_error("[GPUGrimEncoder] residual_scale must be a positive finite value from encoderLayerConstructionHP");
        }

        EncodingConfig enc_cfg{};
        enc_cfg.hp = config;
        enc_cfg.pos_encoding = bindings.pos_encoding;

        for (int i = 0; i < config.num_layers; ++i) {
            // Pattern B: Layer self-allocates and Xavier-inits its own weights.
            // Seed offsets per layer: base + 2 + layer*10
            const uint64_t layer_seed = weight_seed + 2 + i * 10;
            gpu_layers_.emplace_back(std::make_unique<GPUEncoderLayer>(
                enc_cfg, layer_seed, bindings.init_stream));
        }
    }
};

GPUGrimEncoder::GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& config,
                               const EncoderConstructionBindings& bindings,
                               uint64_t weight_seed)
    : pImpl(new Impl(config, bindings, weight_seed))
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

#endif // USE_CUDA

} // namespace GRIM
