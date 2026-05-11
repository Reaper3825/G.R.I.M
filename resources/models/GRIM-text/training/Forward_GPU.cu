#ifndef USE_CUDA
#define USE_CUDA
#endif
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
    std::vector<std::unique_ptr<GPUEncoderLayer>> gpu_layers_;

    Impl(const HyperParameters::EncoderLayerConstructionHP& hp,
         const EncoderConstructionBindings& bindings,
         uint64_t weight_seed)
    {
        HyperParameters::validateEncoderLayerConstructionHP(hp, "GPUGrimEncoder::Impl");
        if (!bindings.pos_encoding) {
            throw std::runtime_error("[GPUGrimEncoder] pos_encoding is NULL — "
                                     "PBM must be initialized BEFORE encoder construction");
        }

        for (int i = 0; i < hp.num_layers; ++i) {
            // Pattern B: Layer self-allocates and Xavier-inits its own weights.
            // Seed offsets per layer: base + 2 + layer*10
            const uint64_t layer_seed = weight_seed + 2 + i * 10;
            gpu_layers_.emplace_back(std::make_unique<GPUEncoderLayer>(
                hp, *bindings.pos_encoding, layer_seed, bindings.init_stream));
        }
    }
};

GPUGrimEncoder::GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& hp,
                               const EncoderConstructionBindings& bindings,
                               uint64_t weight_seed)
    : pImpl(new Impl(hp, bindings, weight_seed))
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
