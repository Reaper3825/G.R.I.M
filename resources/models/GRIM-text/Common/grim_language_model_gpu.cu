//======================================================//
//  grim_language_model_gpu.cu
//  GPU implementations for LanguageModel class
//  Compiled with nvcc and linked with C++ code
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif
// Standard library includes - BEFORE namespace
#include <iostream>
#include <vector>
#include <set>
#include <memory>
#include <string>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <fstream>
#include <sstream>
#include <limits>
#include <functional>
#include <utility>
#include <unordered_set>
#include <cstdint>
// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"    // For Flash Attention kernels
#include "../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"   // For ScratchBlock reasoning layer
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/PBM/PBMStateOwner.hpp"                        // Unified PBM RAII owner (ALiBi + RoPE)

namespace GRIM {

//======================================================//
//  GPU Runtime Accessors (StreamController pattern)
//======================================================//

GPUGrimEncoder& LanguageModel::getGpuEncoder() {
    if (!gpu_encoder_) {
        throw std::runtime_error("GPU encoder not initialized - complete Startup::assembleGpuModel() first");
    }
    return *gpu_encoder_;
}

const GPUGrimEncoder& LanguageModel::getGpuEncoder() const {
    if (!gpu_encoder_) {
        throw std::runtime_error("GPU encoder not initialized - complete Startup::assembleGpuModel() first");
    }
    return *gpu_encoder_;
}

LanguageModel::MTPHead* LanguageModel::getMtpHead(int k) {
    if (k < 0 || k >= static_cast<int>(mtp_heads_.size())) {
        return nullptr;
    }
    return &mtp_heads_[k];
}

const LanguageModel::MTPHead* LanguageModel::getMtpHead(int k) const {
    if (k < 0 || k >= static_cast<int>(mtp_heads_.size())) {
        return nullptr;
    }
    return &mtp_heads_[k];
}

//======================================================//
//  Constructor - moved from header to avoid duplicates
//======================================================//

LanguageModel::LanguageModel(const HyperParameters::LanguageModelConfig& config)
    : config_(config)
{
    // Positional-bias device state is initialized by Phase1 startup PBM bootstrap after
    // StreamController exists. The constructor only creates CPU-side model
    // topology and must not allocate CUDA PBM resources.
    
    // NOTE: Startup::assembleGpuModel() is deliberately NOT called here anymore!
    // The initialization order MUST be:
    //   1. CUDA device context (cudaSetDevice)
    //   2. LanguageModel constructor (this)
    //   3. StreamController initialization
    //   4. Phase1 startup PBM initialization with the StreamController primary stream
    //   5. Startup::assembleGpuModel(*model, weight_init_seed) explicitly called by Phase1
    // This ensures StreamController exists before GPU encoder tries to use it.
#ifdef USE_CUDA
    if (!config_.use_gpu) {
        throw std::runtime_error("grim_language_model_gpu.cu requires config.use_gpu=true with CUDA available");
    }
#endif
}

LanguageModel::~LanguageModel() = default;

} // namespace GRIM

