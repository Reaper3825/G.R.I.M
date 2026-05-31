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
// (PBM, init stream) are passed explicitly; the Phase1 RNG seed is an explicit
// constructor input. Forward-time stream/cuBLAS handles come from
// ModelForwardRequest/AutogradContext.
//======================================================//
struct GPUGrimEncoder::Impl {
    std::vector<std::unique_ptr<EncodingLayer>> gpu_layers_;

    Impl(const HyperParameters::EncoderLayerConstructionHP& hp,
         cudaStream_t init_stream,
         uint64_t weight_seed)
    {
        for (int i = 0; i < hp.num_layers; ++i) {
            // Layer owns encoder attention/RMS tensors. FFN tensors are supplied
            // explicitly at shared-forward time from ParameterRegistry.
            // Seed offsets per layer for encoder-owned tensors: base + 2 + layer*10
            const uint64_t layer_seed = weight_seed + 2 + i * 10;
            gpu_layers_.emplace_back(std::make_unique<EncodingLayer>(
                hp, layer_seed, init_stream));
        }
    }
};

GPUGrimEncoder::GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& hp,
                               cudaStream_t init_stream,
                               uint64_t weight_seed)
    : pImpl(new Impl(hp, init_stream, weight_seed))
{
}

EncodingLayer* GPUGrimEncoder::getLayer(int index) {
    if (index < 0 || index >= static_cast<int>(pImpl->gpu_layers_.size())) {
        return nullptr;
    }
    return pImpl->gpu_layers_[index].get();
}

const EncodingLayer* GPUGrimEncoder::getLayer(int index) const {
    if (index < 0 || index >= static_cast<int>(pImpl->gpu_layers_.size())) {
        return nullptr;
    }
    return pImpl->gpu_layers_[index].get();
}

#endif // USE_CUDA

} // namespace GRIM
