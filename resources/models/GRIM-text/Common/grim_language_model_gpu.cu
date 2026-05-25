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
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <fstream>
#include <sstream>
#include <random>
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
#include "../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../Shared/Forward/ModelForward_GPU.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/PBM/PBMStateOwner.hpp"                        // Unified PBM RAII owner (ALiBi + RoPE)

namespace GRIM {

//======================================================//
//  GPU Runtime Accessors (StreamController pattern)
//======================================================//

GPUGrimEncoder& LanguageModel::getGpuEncoder() {
    if (!gpu_encoder_) {
        throw std::runtime_error("GPU encoder not initialized - call initGPU() first");
    }
    return *gpu_encoder_;
}

const GPUGrimEncoder& LanguageModel::getGpuEncoder() const {
    if (!gpu_encoder_) {
        throw std::runtime_error("GPU encoder not initialized - call initGPU() first");
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

namespace {

// Production constants - using centralized HyperParameters (Rule 20)
[[maybe_unused]] constexpr float kNegInf = HyperParameters::NEG_INF_ATTENTION;
[[maybe_unused]] constexpr float kProbabilityFloor = HyperParameters::PROBABILITY_FLOOR;
[[maybe_unused]] constexpr float kSoftmaxClipThreshold = HyperParameters::SOFTMAX_CLIP_THRESHOLD;
[[maybe_unused]] constexpr float kTemperatureEpsilon = HyperParameters::EPSILON_TEMPERATURE;

}  // namespace

//======================================================//
//  Basic Type Implementations (from grim_text_embedding.cpp)
//======================================================//

Vector::Vector(size_t n, float v) {
    data.resize(n, v);
}

size_t Vector::size() const { return data.size(); }

float& Vector::operator[](size_t idx) { return data[idx]; }

const float& Vector::operator[](size_t idx) const { return data[idx]; }

Vector& Vector::operator+=(const Vector& other) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Vector& Vector::operator*=(float s) {
    for (auto& v : data) v *= s;
    return *this;
}

Vector Vector::operator+(const Vector& other) const {
    Vector result = *this;
    result += other;
    return result;
}

Vector Vector::operator*(float scalar) const {
    Vector result = *this;
    result *= scalar;
    return result;
}

Matrix::Matrix(int rows, int cols, float init_val, bool random) 
    : num_rows(rows), num_cols(cols)
{
    this->rows.resize(rows, Vector(cols, init_val));
    if (random) {
        std::mt19937 gen(42);
        float scale = std::sqrt(2.f / (rows + cols));
        std::normal_distribution<float> dist(0.f, scale);
        for (auto& row : this->rows) {
            for (auto& v : row.data) {
                v = dist(gen);
            }
        }
    }
}

Vector& Matrix::operator[](size_t idx) { return rows[idx]; }

const Vector& Matrix::operator[](size_t idx) const { return rows[idx]; }

//======================================================//
//  Component Implementations with CUDA Kernels
//======================================================//

// GrimEmbeddingStack implementation - uses public members directly
GrimEmbeddingStack::GrimEmbeddingStack(int vocab_size, int d_model, int max_seq_len)
    : vocab_size_(vocab_size),
      d_model_(d_model),
      max_seq_len_(max_seq_len)
{
    token_embed = Matrix(vocab_size, d_model, 0.0f, true);
    // NOTE: Durable GPU embedding tensors are assembled by the Startup/Model allocation module.
}

const Matrix& GrimEmbeddingStack::getTokenEmbeddings() const {
    return token_embed;
}



//======================================================//
//  Constructor - moved from header to avoid duplicates
//======================================================//

LanguageModel::LanguageModel(const HyperParameters::LanguageModelConfig& config)
    : config_(config)
{
    // 1. Create embedding layer
    embedder_ = std::make_unique<GrimEmbeddingStack>(
        config_.vocab_size,
        config_.d_model,
        config_.max_seq_len
    );
    
    // Positional-bias device state is initialized by initPBM() after
    // StreamController exists. The constructor only creates CPU-side model
    // topology and must not allocate CUDA PBM resources.
    
    // NOTE: initGPU() is deliberately NOT called here anymore!
    // The initialization order MUST be:
    //   1. CUDA device context (cudaSetDevice)
    //   2. LanguageModel constructor (this)
    //   3. StreamController initialization
    //   4. initPBM() with the StreamController primary stream
    //   5. initGPU() explicitly called by Phase1
    // This ensures StreamController exists before GPU encoder tries to use it.
#ifdef USE_CUDA
    if (!config_.use_gpu) {
        throw std::runtime_error("grim_language_model_gpu.cu requires config.use_gpu=true with CUDA available");
    }
#endif
}

LanguageModel::~LanguageModel() = default;

//======================================================//
//  BatchPayload-only inference scoring APIs
//======================================================//

Vector LanguageModel::getNextTokenLogits(const Batching::BatchPayload& context_payload) {
    context_payload.validate("LanguageModel::getNextTokenLogits(BatchPayload)");
    if (!context_payload.isInferencePrefill()) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): payload must be InferencePrefill");
    }
    if (context_payload.batch_size != 1) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): batch_size must be 1");
    }
    if (!config_.use_gpu || !gpu_encoder_) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload) requires initialized GPU encoder");
    }
    if (!training_state_.initialized) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): training state not initialized");
    }
    if (context_payload.max_seq_len <= 0 || context_payload.max_seq_len > config_.max_seq_len) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): payload max_seq_len=" +
                                 std::to_string(context_payload.max_seq_len) + " out of range [1, " +
                                 std::to_string(config_.max_seq_len) + "]");
    }
    if (context_payload.vocab_size != config_.vocab_size) {
        throw std::runtime_error(
            "LanguageModel::getNextTokenLogits(BatchPayload): payload.vocab_size=" +
            std::to_string(context_payload.vocab_size) + " != config_.vocab_size=" +
            std::to_string(config_.vocab_size));
    }

    const auto bindings = uploadBatchToDevice(context_payload);

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    Forward::ModelForwardRuntimePayload runtime_payload{};
    runtime_payload.autograd_intermediates = &training_state_.autograd_intermediates;
    runtime_payload.execution_trace_by_row = &generation_state_.execution_trace_by_row;
    runtime_payload.trace_state_by_row = &generation_state_.trace_state_by_row;
    runtime_payload.read_gate_accum_tensor = nullptr;

    Forward::ModelForwardRequest request{};
    request.config = &config_;
    request.gpu_encoder = &getGpuEncoder();
    request.embedding_layer = getEmbeddingLayer();
    request.lm_head = getLmHeadLayer();
    request.scratch_block = getScratchBlockLayer();
    request.reasoning_head = getReasoningHeadLayer();
    request.execution_block = getExecutionBlockLayer();
    request.cublas_handle = training_state_.cublas_handle.get();
    request.stream = stream;
    request.payload = &context_payload;
    request.bindings = &bindings;
    request.batch_idx = 0;
    request.graph = Forward::ModelForwardGraphPolicy{
        /*connect_parameter_graph=*/false,
        /*retain_backward_graph=*/false,
        /*enable_dropout=*/false};

    Forward::executeModelForward(request, runtime_payload);

    const auto& live_logits = training_state_.autograd_intermediates.logits_tensor;
    if (!live_logits.data) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): AutogradIntermediates.logits_tensor.data is NULL after shared forward");
    }
    const size_t expected_logits = static_cast<size_t>(context_payload.total_tokens) * static_cast<size_t>(config_.vocab_size);
    if (static_cast<size_t>(live_logits.numel()) < expected_logits) {
        throw std::runtime_error(
            "LanguageModel::getNextTokenLogits(BatchPayload): live logits numel=" +
            std::to_string(live_logits.numel()) +
            " < payload.total_tokens * config.vocab_size=" + std::to_string(expected_logits));
    }

    Vector logits(config_.vocab_size);
    const size_t last_token_offset =
        static_cast<size_t>(context_payload.max_seq_len - 1) * static_cast<size_t>(config_.vocab_size);
    cudaError_t copy_err = cudaMemcpyAsync(
        logits.data.data(),
        live_logits.data + last_token_offset,
        static_cast<size_t>(config_.vocab_size) * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream);
    if (copy_err != cudaSuccess) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): cudaMemcpyAsync logits failed: " +
                                 std::string(cudaGetErrorString(copy_err)));
    }
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error("LanguageModel::getNextTokenLogits(BatchPayload): cudaStreamSynchronize failed: " +
                                 std::string(cudaGetErrorString(sync_err)));
    }

    training_state_.autograd_intermediates.clear();
    return logits;
}

//======================================================//
//  ScratchBlock Helper Methods
//======================================================//

void LanguageModel::setScratchBlockEnabled(bool enabled) {
    if (!scratch_block_layer_) {
        if (enabled) {
            throw std::runtime_error("setScratchBlockEnabled: ScratchBlock layer is not initialized");
        }
        return;
    }
    scratch_block_layer_->setEnabled(enabled);
}

bool LanguageModel::isScratchBlockEnabled() const {
    return scratch_block_layer_ && scratch_block_layer_->isEnabled();
}

} // namespace GRIM

