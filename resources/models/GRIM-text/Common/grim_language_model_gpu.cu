//======================================================//
//  grim_language_model_gpu.cu
//  GPU implementations for LanguageModel class
//  Compiled with nvcc and linked with C++ code
//======================================================//

#define USE_CUDA

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
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"                 // Unified PBM (ALiBi + RoPE)
#include "../Shared/Sampling/Sampling.hpp"                        // SamplingPipeline
#include "../Shared/Execution/DecodeTimeNumPolicy.hpp"            // SlotSelectionStatus

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
constexpr uint32_t kMaxReasonableVocabSize = HyperParameters::MAX_REASONABLE_VOCAB_SIZE;

int detectVocabSizeFromBinary(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return -1;
    }

    uint32_t magic = 0;
    uint16_t version = 0;
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
        return -1;
    }
    if (!file.read(reinterpret_cast<char*>(&version), sizeof(version))) {
        return -1;
    }
    // Only accept 'GMTK' magic (serialization format)
    if (magic != 0x474D544B) { // 'GMTK'
        fprintf(stderr, "[LanguageModel] Invalid magic 0x%08X, expected 0x474D544B ('GMTK')\n", magic);
        return -1;
    }

    if (version < 2) {
        fprintf(stderr, "[LanguageModel] Unsupported version %d, minimum required is 2\n", version);
        return -1;
    }

    uint32_t checksum = 0;
    if (!file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum))) {
        return -1;
    }

    int config_vocab_size = 0;
    if (!file.read(reinterpret_cast<char*>(&config_vocab_size), sizeof(config_vocab_size))) {
        return -1;
    }

    int max_length = 0;
    if (!file.read(reinterpret_cast<char*>(&max_length), sizeof(max_length))) {
        return -1;
    }

    bool nfkc = false;
    bool lower = false;
    bool byte_fallback = false;
    if (!file.read(reinterpret_cast<char*>(&nfkc), sizeof(nfkc))) {
        return -1;
    }
    if (!file.read(reinterpret_cast<char*>(&lower), sizeof(lower))) {
        return -1;
    }
    if (!file.read(reinterpret_cast<char*>(&byte_fallback), sizeof(byte_fallback))) {
        return -1;
    }
    (void)max_length;
    (void)nfkc;
    (void)lower;
    (void)byte_fallback;

    uint32_t vocab_size = 0;
    if (!file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size))) {
        return -1;
    }

    // Defensive sanity: avoid using absurd header values.
    if (vocab_size > kMaxReasonableVocabSize) {
        return -1;
    }

    if (vocab_size == 0) {
        // Rule 20: no backward-compat fallback to legacy config_vocab_size.
        fprintf(stderr,
                "[LanguageModel] Invalid vocab header in %s: vocab_size=0 (legacy config_vocab_size=%d)\n",
                path.c_str(),
                config_vocab_size);
        return -1;
    }

    return static_cast<int>(vocab_size);
}

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

ContextState::ContextState() 
    : sentiment(0.0f), depth(0) 
{}

float GeneratedSequence::getNormalizedScore(float length_penalty) const {
    if (token_ids.empty()) {
        return score;
    }
    return score / std::pow(static_cast<float>(token_ids.size()), length_penalty);
}

//======================================================//
//======================================================//
//  ALiBiPositionalBias Implementation (Unified PBM)
//======================================================//

ALiBiPositionalBias::ALiBiPositionalBias()
    : num_heads(0),
      initialized(false),
      type(PositionalEncodingType::NONE) {}

ALiBiPositionalBias::ALiBiPositionalBias(ALiBiPositionalBias&& other) noexcept {
    *this = std::move(other);
}

ALiBiPositionalBias& ALiBiPositionalBias::operator=(ALiBiPositionalBias&& other) noexcept {
    if (this != &other) {
        cleanup();
#ifdef USE_CUDA
        // Move unified PBM state
        pbm_state_.alibi_slopes = other.pbm_state_.alibi_slopes;
        pbm_state_.alibi_slopes_host = std::move(other.pbm_state_.alibi_slopes_host);
        pbm_state_.num_heads = other.pbm_state_.num_heads;
        pbm_state_.rope_inv_freq = other.pbm_state_.rope_inv_freq;
        pbm_state_.rope_inv_freq_host = std::move(other.pbm_state_.rope_inv_freq_host);
        pbm_state_.head_dim = other.pbm_state_.head_dim;
        pbm_state_.rotary_dim = other.pbm_state_.rotary_dim;
        pbm_state_.num_kv_heads = other.pbm_state_.num_kv_heads;
        pbm_state_.initialized = other.pbm_state_.initialized;
        
        // Clear other
        other.pbm_state_.alibi_slopes = nullptr;
        other.pbm_state_.rope_inv_freq = nullptr;
        other.pbm_state_.initialized = false;
        other.pbm_state_.alibi_slopes_host.clear();
        other.pbm_state_.rope_inv_freq_host.clear();
#endif
        num_heads = other.num_heads;
        initialized = other.initialized;
        type = other.type;
    }
    return *this;
}

ALiBiPositionalBias::~ALiBiPositionalBias() {
    cleanup();
}

void ALiBiPositionalBias::computeSlopes(int num_heads_, int num_kv_heads_, int d_head_, int max_seq_len_, PositionalEncodingType type_) {
    if (num_heads_ <= 0) {
        throw std::runtime_error("ALiBiPositionalBias::computeSlopes: num_heads must be > 0");
    }
    if (num_kv_heads_ <= 0 || num_kv_heads_ > num_heads_) {
        throw std::runtime_error("ALiBiPositionalBias::computeSlopes: num_kv_heads must be in (0, num_heads]");
    }
    if (d_head_ <= 0) {
        throw std::runtime_error("ALiBiPositionalBias::computeSlopes: d_head must be > 0");
    }

    num_heads = num_heads_;
    type = type_;
#ifdef USE_CUDA
    // Use unified PBM - always allocates both ALiBi + RoPE
    PBM::PBMConfig config{};
    config.num_heads = num_heads_;
    config.head_dim = d_head_;
    config.rotary_dim = config.head_dim;
    config.num_kv_heads = num_kv_heads_;
    config.max_seq_len = max_seq_len_; 
    config.rope_theta = 10000.0f;
    config.verbose = false;
    
    if (!PBM::ensurePBM(config, pbm_state_)) {
        throw std::runtime_error("ALiBiPositionalBias::computeSlopes: PBM initialization failed");
    }
    if (!pbm_state_.initialized) {
        throw std::runtime_error("ALiBiPositionalBias::computeSlopes: PBM state not initialized");
    }

    initialized = true;
#else
    throw std::runtime_error("ALiBiPositionalBias::computeSlopes requires CUDA");
#endif
}

float* ALiBiPositionalBias::getSlopes() const {
#ifdef USE_CUDA
    if (!initialized) {
        throw std::runtime_error("ALiBiPositionalBias::getSlopes: PBM not initialized");
    }
    if (!pbm_state_.alibi_slopes) {
        throw std::runtime_error("ALiBiPositionalBias::getSlopes: initialized=true but alibi_slopes is NULL — PBM state corrupted at "
                                 + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    return pbm_state_.alibi_slopes;
#else
    throw std::runtime_error("ALiBiPositionalBias::getSlopes requires CUDA build");
#endif
}

float* ALiBiPositionalBias::getRoPEFreqs() const {
#ifdef USE_CUDA
    if (!initialized) {
        throw std::runtime_error("ALiBiPositionalBias::getRoPEFreqs: PBM not initialized");
    }
    if (!pbm_state_.rope_inv_freq) {
        throw std::runtime_error("ALiBiPositionalBias::getRoPEFreqs: initialized=true but rope_inv_freq is NULL — PBM state corrupted at "
                                 + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    return pbm_state_.rope_inv_freq;
#else
    throw std::runtime_error("ALiBiPositionalBias::getRoPEFreqs requires CUDA build");
#endif
}

bool ALiBiPositionalBias::isInitialized() const {
#ifdef USE_CUDA
    return initialized && pbm_state_.initialized;
#else
    return false;
#endif
}

void ALiBiPositionalBias::cleanup() {
#ifdef USE_CUDA
    PBM::releasePBM(pbm_state_);
#endif
    initialized = false;
    type = PositionalEncodingType::NONE;
    num_heads = 0;
}

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
    rms_gamma = Vector(d_model, 1.0f);  // RMSNorm gamma
    // NOTE: Position embeddings initialized directly on GPU in TrainingOps.cu
}

void GrimEmbeddingStack::enableALiBi(int num_heads, int num_kv_heads, int max_seq_len) {
    if (num_heads <= 0) {
        throw std::runtime_error("GrimEmbeddingStack::enableALiBi: num_heads must be > 0");
    }
    if (num_kv_heads <= 0 || num_kv_heads > num_heads) {
        throw std::runtime_error("GrimEmbeddingStack::enableALiBi: num_kv_heads must be in (0, num_heads]");
    }
    if (d_model_ % num_heads != 0) {
        throw std::runtime_error("GrimEmbeddingStack::enableALiBi: d_model not divisible by num_heads");
    }
    const int d_head = d_model_ / num_heads;
    alibi_ = std::make_unique<ALiBiPositionalBias>();
    alibi_->computeSlopes(num_heads, num_kv_heads, d_head, max_seq_len, PositionalEncodingType::ALIBI);
}

void GrimEmbeddingStack::enableHybridPositionalEncoding(int num_heads, int d_head, int num_kv_heads, int max_seq_len) {
    if (num_heads <= 0) {
        throw std::runtime_error("GrimEmbeddingStack::enableHybridPositionalEncoding: num_heads must be > 0");
    }
    if (num_kv_heads <= 0 || num_kv_heads > num_heads) {
        throw std::runtime_error("GrimEmbeddingStack::enableHybridPositionalEncoding: num_kv_heads must be in (0, num_heads]");
    }
    if (d_model_ % num_heads != 0) {
        throw std::runtime_error("GrimEmbeddingStack::enableHybridPositionalEncoding: d_model not divisible by num_heads");
    }
    if (d_head != d_model_ / num_heads) {
        throw std::runtime_error("GrimEmbeddingStack::enableHybridPositionalEncoding: d_head must match d_model/num_heads");
    }
    alibi_ = std::make_unique<ALiBiPositionalBias>();
    // Pass through num_kv_heads so PBM and encoder GQA settings match
    alibi_->computeSlopes(num_heads, num_kv_heads, d_head, max_seq_len, PositionalEncodingType::ALIBI_ROPE);
}

const ALiBiPositionalBias* GrimEmbeddingStack::getALiBiBias() const {
    return alibi_.get();
}

const Matrix& GrimEmbeddingStack::getTokenEmbeddings() const {
    return token_embed;
}



//======================================================//
//  Constructor - moved from header to avoid duplicates
//======================================================//

LanguageModel::LanguageModel(const LanguageModelConfig& config)
    : config_(config)
{
    if (config_.infer_vocab_from_file) {
        if (config_.vocab_path.empty()) {
            fprintf(stderr, "[LanguageModel] FATAL: infer_vocab_from_file enabled but vocab_path is empty\n");
            throw std::runtime_error("LanguageModel: vocab_path required when infer_vocab_from_file is true");
        }
        int detected_vocab = detectVocabSizeFromBinary(config_.vocab_path);
        if (detected_vocab > 0) {
            config_.vocab_size = detected_vocab;
        } else {
            fprintf(stderr, "[LanguageModel] FATAL: Failed to detect vocab size from %s\n",
                    config_.vocab_path.c_str());
            throw std::runtime_error("LanguageModel: failed to detect vocab size from vocab file");
        }
    }
    
    // 1. Create embedding layer
    embedder_ = std::make_unique<GrimEmbeddingStack>(
        config_.vocab_size,
        config_.d_model,
        config_.max_seq_len
    );
    
    // 2. Enable positional encoding if requested
    if (HyperParameters::usesALiBi(config_.positional_encoding) || 
        HyperParameters::usesRoPE(config_.positional_encoding)) {
        if (config_.num_heads <= 0) {
            throw std::runtime_error("LanguageModel: num_heads must be > 0 before positional initialization");
        }
        if (config_.d_model % config_.num_heads != 0) {
            throw std::runtime_error("LanguageModel: d_model must be divisible by num_heads before positional initialization");
        }
        int d_head = config_.d_model / config_.num_heads;
        
        // Use appropriate initialization based on encoding type
        if (config_.positional_encoding == PositionalEncodingType::ALIBI_ROPE) {
            embedder_->enableHybridPositionalEncoding(config_.num_heads, d_head, config_.num_kv_heads, config_.max_seq_len);
        } else if (config_.positional_encoding == PositionalEncodingType::ALIBI) {
            embedder_->enableALiBi(config_.num_heads, config_.num_kv_heads, config_.max_seq_len);
        }
        // Note: Pure RoPE handled differently (integrated into attention)
    }
    
    // NOTE: initGPU() is deliberately NOT called here anymore!
    // The initialization order MUST be:
    //   1. CUDA device context (cudaSetDevice)
    //   2. LanguageModel constructor (this)
    //   3. StreamController initialization
    //   4. initGPU() explicitly called by Phase1
    // This ensures StreamController exists before GPU encoder tries to use it.
#ifdef USE_CUDA
    if (!config_.use_gpu) {
        throw std::runtime_error("grim_language_model_gpu.cu requires config.use_gpu=true with CUDA available");
    }
#endif
}

LanguageModel::~LanguageModel() = default;

//======================================================//
//  Helper methods - moved from header
//======================================================//

Vector LanguageModel::forward(const std::vector<int>& token_ids,
                              const std::vector<float>& token_numeric_values,
                              const std::vector<uint8_t>& token_atom_mask,
                              const std::vector<int32_t>& token_to_slot_map) {
#ifdef USE_CUDA
    // Use GPU if available
    if (config_.use_gpu && gpu_encoder_) {
        return forwardGPU(token_ids, token_numeric_values, token_atom_mask, token_to_slot_map);
    }
#endif
    throw std::runtime_error("LanguageModel::forward requires GPU initialization");
}

Vector LanguageModel::getNextTokenLogits(const std::vector<int>& context_tokens,
                                         const std::vector<float>& context_numeric_values,
                                         const std::vector<uint8_t>& context_atom_mask,
                                         const std::vector<int32_t>& token_to_slot_map) {
#ifdef USE_CUDA
    // Use GPU if available
    if (config_.use_gpu && gpu_encoder_) {
        return getNextTokenLogitsGPU(context_tokens,
                                     context_numeric_values,
                                     context_atom_mask,
                                     token_to_slot_map);
    }
#endif
    throw std::runtime_error("LanguageModel::getNextTokenLogits requires GPU initialization");
}


namespace {

// Copy per-token slot assignment map from host to device (inference path).
//
//   prompt_map semantics:
//     - Empty vector → all tokens mapped to -1 (non-state-bearing).
//     - Entry == -1  → this token is non-state-bearing (no slot selected).
//     - Entry in [0, num_slots) → token is bound to that execution slot.
//
//   At decode time, <NUM> can only be generated when the selector
//   resolves to a single live slot (Selected). Otherwise <NUM> is
//   masked out of the vocabulary and cannot be sampled.
void copyTokenSlotMapH2D_Inference(TrainingState& ts, cudaStream_t stream, int seq_len,
                                    const std::vector<int32_t>& prompt_map,
                                    int num_slots) {
    if (!ts.cached_token_to_slot_map.data || seq_len <= 0)
        return;
    auto* dst = reinterpret_cast<int32_t*>(ts.cached_token_to_slot_map.data);
    if (prompt_map.empty()) {
        // Empty prompt map → all tokens are non-state-bearing (slot_id = -1)
        std::vector<int32_t> neg(static_cast<size_t>(seq_len), -1);
        cudaError_t err = cudaMemcpyAsync(dst, neg.data(),
            static_cast<size_t>(seq_len) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("copyTokenSlotMapH2D_Inference (sentinel): ") +
                                     cudaGetErrorString(err));
        }
        return;
    }
    if (static_cast<int>(prompt_map.size()) != seq_len) {
        throw std::runtime_error(
            "token_to_slot_map size must match sequence length (or pass empty for all -1)");
    }
    // Validate slot-range: each entry must be -1 (non-state-bearing) or in [0, num_slots)
    for (int i = 0; i < seq_len; ++i) {
        int32_t sid = prompt_map[i];
        if (sid != -1 && (sid < 0 || sid >= num_slots)) {
            throw std::runtime_error(
                "copyTokenSlotMapH2D_Inference: slot_id=" + std::to_string(sid) +
                " at position " + std::to_string(i) + " out of range [0, " +
                std::to_string(num_slots) + ") — must be -1 or valid slot index");
        }
    }
    cudaError_t err = cudaMemcpyAsync(dst, prompt_map.data(),
        static_cast<size_t>(seq_len) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("copyTokenSlotMapH2D_Inference: ") + cudaGetErrorString(err));
    }
}

}  // namespace

Vector LanguageModel::getNextTokenLogitsGPU(const std::vector<int>& context_tokens,
                                            const std::vector<float>& context_numeric_values,
                                            const std::vector<uint8_t>& context_atom_mask,
                                            const std::vector<int32_t>& token_to_slot_map) {
    if (!config_.use_gpu || !gpu_encoder_) {
        throw std::runtime_error("getNextTokenLogitsGPU requires initialized GPU encoder");
    }
    if (context_tokens.empty()) {
        throw std::runtime_error("getNextTokenLogitsGPU requires non-empty context_tokens");
    }
    if (context_numeric_values.size() != context_tokens.size() ||
        context_atom_mask.size() != context_tokens.size()) {
        throw std::runtime_error("getNextTokenLogitsGPU: side-channel length mismatch");
    }
    const int seq_len = static_cast<int>(context_tokens.size());
    if (seq_len > config_.max_seq_len) {
        throw std::runtime_error("getNextTokenLogitsGPU: context length " +
                                 std::to_string(seq_len) + " exceeds max_seq_len " +
                                 std::to_string(config_.max_seq_len));
    }
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
        if (!training_state_.initialized) {
            throw std::runtime_error("getNextTokenLogitsGPU: state initialization failed");
        }
    }
    if (training_state_.max_cached_seq_len > 0 &&
        seq_len > training_state_.max_cached_seq_len) {
        throw std::runtime_error("getNextTokenLogitsGPU: context length " +
                                 std::to_string(seq_len) + " exceeds max_cached_seq_len " +
                                 std::to_string(training_state_.max_cached_seq_len));
    }
    if (!training_state_.stream_ctrl.isInitialized()) {
        throw std::runtime_error("getNextTokenLogitsGPU: StreamController not initialized");
    }
    
    // Copy host data to cached GPU tensors
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    cudaMemcpyAsync(reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data),
                    context_tokens.data(),
                    seq_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(training_state_.cached_token_numeric_values.data,
                    context_numeric_values.data(),
                    seq_len * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data),
                    context_atom_mask.data(),
                    seq_len * sizeof(uint8_t),
                    cudaMemcpyHostToDevice, stream);

    copyTokenSlotMapH2D_Inference(training_state_, stream, seq_len, token_to_slot_map,
                                   config_.execution_block_num_slots);
    
    // Single inference forward path
    return executeInferenceForward_(seq_len);
}

Vector LanguageModel::forwardGPU(const std::vector<int>& token_ids,
                                 const std::vector<float>& token_numeric_values,
                                 const std::vector<uint8_t>& token_atom_mask,
                                 const std::vector<int32_t>& token_to_slot_map) {
    return getNextTokenLogitsGPU(token_ids, token_numeric_values, token_atom_mask, token_to_slot_map);
}

TokenBufferView LanguageModel::getTokenBufferView() {
    TokenBufferView view{};
    if (!config_.use_gpu) {
        throw std::runtime_error("getTokenBufferView: config.use_gpu=false in GPU-only build");
    }
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
        if (!training_state_.initialized) {
            throw std::runtime_error("getTokenBufferView: state initialization failed");
        }
    }
    if (!training_state_.cached_token_ids_tensor.data ||
        !training_state_.cached_token_numeric_values.data ||
        !training_state_.cached_token_atom_mask.data) {
        throw std::runtime_error("getTokenBufferView: token buffers are not allocated");
    }
    view.device_token_ids = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
    view.device_token_numeric_values = training_state_.cached_token_numeric_values.data;
    view.device_token_atom_mask = reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data);
    view.device_token_to_slot_map = reinterpret_cast<int32_t*>(training_state_.cached_token_to_slot_map.data);
    view.max_tokens = config_.max_seq_len;
    view.stream = training_state_.stream_ctrl.getPrimaryStream();
    return view;
}

void LanguageModel::markDevicePromptReady(int token_count) {
    if (!config_.use_gpu) {
        throw std::runtime_error("markDevicePromptReady: config.use_gpu=false in GPU-only build");
    }
    if (token_count < 0) {
        throw std::runtime_error("markDevicePromptReady: token_count must be >= 0");
    }
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
        if (!training_state_.initialized) {
            throw std::runtime_error("markDevicePromptReady: state initialization failed");
        }
    }
    staged_prompt_ready_ = true;
    staged_prompt_len_ = std::min(token_count, config_.max_seq_len);
}

std::vector<GeneratedSequence> LanguageModel::generate(
    const std::vector<int>& prompt_tokens,
    const std::vector<float>& prompt_numeric_values,
    const std::vector<uint8_t>& prompt_atom_mask,
    const GenerationConfig* gen_config,
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table,
    const std::vector<uint32_t>& prompt_atom_entry_ids,
    const std::vector<int32_t>& prompt_token_to_slot_map)
{
#ifdef USE_CUDA
    if (config_.use_gpu && gpu_encoder_) {
        GenerationConfig cfg = gen_config ? *gen_config : config_.generation;
        if (cfg.num_return_sequences <= 0) {
            throw std::runtime_error("LanguageModel::generate: num_return_sequences must be > 0");
        }
        const int sequences = cfg.num_return_sequences;
        std::vector<GeneratedSequence> outputs;
        outputs.reserve(sequences);
        for (int i = 0; i < sequences; ++i) {
            GenerationConfig seq_cfg = cfg;
            if (seq_cfg.seed != 0) seq_cfg.seed += i;
            outputs.push_back(generateSequenceGPU(prompt_tokens,
                                                  prompt_numeric_values,
                                                  prompt_atom_mask,
                                                  seq_cfg,
                                                  nullptr,
                                                  prompt_atom_table,
                                                  prompt_atom_entry_ids,
                                                  prompt_token_to_slot_map));
        }
        return outputs;
    }
#endif
    throw std::runtime_error("LanguageModel::generate requires GPU initialization");
}

GeneratedSequence LanguageModel::generateStream(
    const std::vector<int>& prompt_tokens,
    const std::vector<float>& prompt_numeric_values,
    const std::vector<uint8_t>& prompt_atom_mask,
    GenerationStreamCallback callback,
    const GenerationConfig* gen_config,
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table,
    const std::vector<uint32_t>& prompt_atom_entry_ids,
    const std::vector<int32_t>& prompt_token_to_slot_map)
{
#ifdef USE_CUDA
    if (config_.use_gpu && gpu_encoder_) {
        GenerationConfig cfg = gen_config ? *gen_config : config_.generation;
        return generateSequenceGPU(prompt_tokens,
                                   prompt_numeric_values,
                                   prompt_atom_mask,
                                   cfg,
                                   &callback,
                                   prompt_atom_table,
                                   prompt_atom_entry_ids,
                                   prompt_token_to_slot_map);
    }
#endif
    throw std::runtime_error("LanguageModel::generateStream requires GPU initialization");
}


// Generate a token sequence from prompt, applying autoregressive decoding.
//
//   prompt_token_to_slot_map semantics:
//     - Empty vector → all prompt tokens treated as non-state-bearing (-1).
//     - Entry == -1  → non-state-bearing token (no execution slot selected).
//     - Entry in [0, num_slots) → this token is bound to that execution slot.
//
//   During generation, the decode-time <NUM> token is only bindable when
//   the slot selector resolves status == Selected for exactly one live slot.
//   Otherwise <NUM> is masked and cannot be generated at that step.
GeneratedSequence LanguageModel::generateSequenceGPU(const std::vector<int>& prompt_tokens,
                                                     const std::vector<float>& prompt_numeric_values,
                                                     const std::vector<uint8_t>& prompt_atom_mask,
                                                     const GenerationConfig& cfg,
                                                     GenerationStreamCallback* stream_callback,
                                                     std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table,
                                                     const std::vector<uint32_t>& prompt_atom_entry_ids,
                                                     const std::vector<int32_t>& prompt_token_to_slot_map) {
    GeneratedSequence sequence;
    sequence.token_ids = prompt_tokens;
    sequence.token_numeric_values = prompt_numeric_values;
    sequence.token_atom_mask = prompt_atom_mask;
    sequence.context_atom_table = prompt_atom_table;
    if (!prompt_token_to_slot_map.empty()) {
        if (prompt_token_to_slot_map.size() != prompt_tokens.size()) {
            throw std::runtime_error("generateSequenceGPU: prompt_token_to_slot_map length mismatch");
        }
        sequence.token_to_slot_map = prompt_token_to_slot_map;
    } else {
        sequence.token_to_slot_map.assign(prompt_tokens.size(), -1);
    }
    if (!prompt_atom_entry_ids.empty() && prompt_atom_entry_ids.size() == prompt_tokens.size()) {
        sequence.atom_entry_ids = prompt_atom_entry_ids;
    } else {
        sequence.atom_entry_ids.assign(prompt_tokens.size(), GRIM::Tokenizer::kAtomEntryNone);
    }
    
    if (!config_.use_gpu || !gpu_encoder_) {
        throw std::runtime_error("generateSequenceGPU requires initialized GPU encoder");
    }
    
    if (prompt_tokens.size() >= static_cast<size_t>(config_.max_seq_len)) {
        throw std::runtime_error("generateSequenceGPU: prompt length " +
                                 std::to_string(prompt_tokens.size()) + " exceeds max_seq_len " +
                                 std::to_string(config_.max_seq_len));
    }
    if (prompt_numeric_values.size() != prompt_tokens.size() ||
        prompt_atom_mask.size() != prompt_tokens.size()) {
        throw std::runtime_error("generateSequenceGPU: side-channel length mismatch");
    }
    
    if (cfg.max_new_tokens < 0) {
        throw std::runtime_error("generateSequenceGPU: max_new_tokens must be non-negative");
    }
    const int max_steps = cfg.max_new_tokens;
    if (config_.vocab_size <= 0) {
        throw std::runtime_error("generateSequenceGPU: invalid vocab_size");
    }
    const int vocab_size = config_.vocab_size;
    if (cfg.strategy == SamplingStrategy::BEAM_SEARCH) {
        throw std::runtime_error("generateSequenceGPU: BEAM_SEARCH is not supported");
    }
    
    // =========================================================================
    // Build SamplingPipeline from GenerationConfig (single sampling path)
    // =========================================================================
    Sampling::SamplingConfig sampling_cfg = Sampling::buildFromGenerationConfig(
        static_cast<int>(cfg.strategy),
        cfg.do_sample,
        cfg.temperature,
        cfg.top_k,
        cfg.top_p,
        cfg.min_p,
        cfg.typical_p,
        cfg.repetition_penalty,
        cfg.repetition_penalty_window,
        cfg.frequency_penalty,
        cfg.presence_penalty,
        cfg.no_repeat_ngram_size,
        cfg.eos_token_id,
        cfg.bos_token_id,
        cfg.pad_token_id,
        cfg.unk_token_id,
        cfg.bad_words_ids,
        cfg.seed);

    // Execution-first (spec step 10): mask raw ASCII digit byte tokens so the LM cannot
    // decode numeric magnitudes as literal strings; `<NUM>` stays allowed (atom range).
    if (config_.execution_block_enabled) {
        std::vector<int> numeric_mask = cfg.masked_numeric_literal_ids;
        if (numeric_mask.empty()) {
            numeric_mask.reserve(10);
            for (int b = 0x30; b <= 0x39; ++b) {
                numeric_mask.push_back(Tokenizer::BYTE_TOKEN_OFFSET + b);
            }
        }
        sampling_cfg.bad_token_ids.insert(sampling_cfg.bad_token_ids.end(),
                                            numeric_mask.begin(), numeric_mask.end());
        std::sort(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end());
        sampling_cfg.bad_token_ids.erase(
            std::unique(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end()),
            sampling_cfg.bad_token_ids.end());
    }

    // Decode-time <NUM> admissibility: when selector is active, <NUM> is allowed
    // only when the selector has resolved a slot (Selected status). When the
    // selector is inactive or unavailable, <NUM> remains masked as before.
    const bool selector_active = config_.selector_enabled
        && getDecodeTimeSlotSelectorLayer() != nullptr
        && getDecodeTimeNumPolicy() != nullptr
        && config_.execution_block_enabled
        && getScratchBlockLayer() != nullptr
        && isScratchBlockEnabled()
        && training_state_.has_inference_exec_memory;
    if (!selector_active) {
        // No selector → hard-mask <NUM> when scratchblock generation is active
        const bool scratchblock_generation_active = cfg.enable_scratchblock_reasoning &&
                                                    config_.use_scratch_block &&
                                                    isScratchBlockEnabled();
        if (scratchblock_generation_active) {
            const int num_tid = Tokenizer::atomTypeToTokenId(Tokenizer::AtomType::ATOM_NUM);
            sampling_cfg.bad_token_ids.push_back(num_tid);
            std::sort(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end());
            sampling_cfg.bad_token_ids.erase(
                std::unique(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end()),
                sampling_cfg.bad_token_ids.end());
        }
    }
    // When selector IS active, <NUM> masking is decided per-step below
    
    Sampling::SamplingPipeline pipeline(sampling_cfg);
    
    // ScratchBlock reasoning: determine if we should classify generated atom tokens
    const bool scratchblock_active = cfg.enable_scratchblock_reasoning &&
                                     config_.use_scratch_block &&
                                     isScratchBlockEnabled();
    
    // =========================================================================
    // INCREMENTAL GENERATION WITH KV CACHE
    // Step 0: Process full prompt with forwardInit() - O(n)
    // Step 1+: Process single token with forwardStep() - O(1) per step
    // Total: O(n) instead of O(n²)
    // =========================================================================
    
    // Ensure KV cache buffers exist (training path skips allocation)
    ensureKVCacheAllocated();

    // Reset KV cache for new generation
    resetKVCache();
    
    // Process prompt and get logits for first new token
    Vector logits_vec = forwardInit(prompt_tokens,
                                    prompt_numeric_values,
                                    prompt_atom_mask,
                                    sequence.token_to_slot_map);
    if (logits_vec.data.empty()) {
        throw std::runtime_error("generateSequenceGPU: forwardInit returned empty logits");
    }

    for (int step = 0; step < max_steps; ++step) {
        // Check max sequence length
        const int current_len = getKVCacheLength();
        if (current_len >= config_.max_seq_len) {
            sequence.finished = true;
            break;
        }
        
        // =====================================================================
        // Selector-aware <NUM> masking (per step)
        // When selector is active, both forwardInit (prefill) and
        // forwardStep (decode) evaluate the selector, so
        // decode_selector_valid MUST be true by this point.
        // Allow <NUM> only when status == Selected; otherwise mask it.
        // =====================================================================
        if (selector_active) {
            if (!training_state_.decode_selector_valid) {
                std::fprintf(stderr,
                    "[Selector Debug] step=%d  selector_enabled=%d  selectorLayer=%d  numPolicy=%d"
                    "  exec_block_enabled=%d  scratchLayer=%d  scratchEnabled=%d  has_exec_mem=%d"
                    "  decode_selector_valid=%d\n",
                    step,
                    (int)config_.selector_enabled,
                    (int)(getDecodeTimeSlotSelectorLayer() != nullptr),
                    (int)(getDecodeTimeNumPolicy() != nullptr),
                    (int)config_.execution_block_enabled,
                    (int)(getScratchBlockLayer() != nullptr),
                    (int)isScratchBlockEnabled(),
                    (int)training_state_.has_inference_exec_memory,
                    (int)training_state_.decode_selector_valid);
                throw std::runtime_error(
                    "generateSequenceGPU: selector_active but decode_selector_valid is false at step "
                    + std::to_string(step) + " — selector was not evaluated after "
                    + (step == 0 ? "forwardInit (prefill)" : "forwardStep (decode)"));
            }
            const int num_tid = Tokenizer::atomTypeToTokenId(Tokenizer::AtomType::ATOM_NUM);
            if (training_state_.decode_selector_status
                   != static_cast<uint8_t>(SlotSelectionStatus::Selected)) {
                logits_vec.data[num_tid] = -1e30f;
            }
        }

        // =====================================================================
        // Sample next token using SamplingPipeline
        // Pipeline handles: penalties → n-gram blocking → token masking →
        //                   temperature+softmax → filter → renorm → sample
        // =====================================================================
        Sampling::SampleResult sample = pipeline.sample(
            logits_vec.data, sequence.token_ids, vocab_size);
        
        // Validate sampled token (Rule 20: crash, no fallback)
        if (sample.token_id < 0 || sample.token_id >= vocab_size) {
            throw std::runtime_error("generateSequenceGPU: sampled token out of range (token_id=" +
                                     std::to_string(sample.token_id) + ", vocab=" +
                                     std::to_string(vocab_size) + ")");
        }
        
        float token_numeric_value = 0.0f;
        uint8_t token_atom_mask_val = 0;
        
        int32_t new_token_slot_id = -1;

        if (scratchblock_active && GRIM::Tokenizer::isAtomToken(sample.token_id)) {
            token_atom_mask_val = 1;
            if (GRIM::Tokenizer::tokenIdToAtomType(sample.token_id)
                == GRIM::Tokenizer::AtomType::ATOM_NUM) {
                // <NUM> was sampled — selector MUST have resolved a slot
                if (!selector_active || !training_state_.decode_selector_valid
                    || training_state_.decode_selector_status
                       != static_cast<uint8_t>(SlotSelectionStatus::Selected)) {
                    throw std::runtime_error(
                        "generateSequenceGPU: sampled <NUM> but selector did not resolve a slot "
                        "(status=" + std::to_string(training_state_.decode_selector_status) + ")");
                }
                new_token_slot_id = training_state_.decode_selected_slot;
                token_numeric_value = training_state_.decode_selected_value;
            }
        }
        
        // Check for EOS BEFORE processing next step
        // EOS (token 3) can never be an atom token (atoms are 260+), so no numeric extraction needed.
        if (sample.token_id == cfg.eos_token_id &&
            step + 1 >= cfg.min_new_tokens) {
            sequence.token_ids.push_back(sample.token_id);
            sequence.token_scores.push_back(sample.log_probability);
            sequence.token_numeric_values.push_back(0.0f);
            sequence.token_atom_mask.push_back(0);
            sequence.token_to_slot_map.push_back(-1);
            sequence.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
            sequence.score += sample.log_probability;
            sequence.finished = true;
            if (stream_callback) {
                (*stream_callback)(sample.token_id, sample.probability);
            }
            break;
        }
        
        // Run forward pass for this token WITH ScratchBlock metadata
        logits_vec = forwardStep(sample.token_id, token_numeric_value, token_atom_mask_val,
                                 new_token_slot_id);
        if (logits_vec.data.empty()) {
            throw std::runtime_error("generateSequenceGPU: forwardStep returned empty logits");
        }
        
        // Add token to sequence
        sequence.token_ids.push_back(sample.token_id);
        sequence.token_scores.push_back(sample.log_probability);
        sequence.token_numeric_values.push_back(token_numeric_value);
        sequence.token_atom_mask.push_back(token_atom_mask_val);
        sequence.token_to_slot_map.push_back(new_token_slot_id);
        sequence.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
        sequence.score += sample.log_probability;
        
        if (stream_callback) {
            (*stream_callback)(sample.token_id, sample.probability);
        }
    }
    
    if (!sequence.finished) {
        sequence.finished = true;
    }
    
    return sequence;
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

void LanguageModel::configureScratchPool(bool enabled) {
    if (!training_state_.scratch_pool) {
        if (enabled) {
            throw std::runtime_error("configureScratchPool: scratch_pool is not initialized");
        }
        training_state_.scratch_enabled = false;
        return;
    }
    training_state_.scratch_enabled = enabled;
    training_state_.scratch_pool->setEnabled(enabled);
}

bool LanguageModel::isScratchPoolInitialized() const {
    return training_state_.scratch_pool != nullptr && training_state_.scratch_pool->isInitialized();
}




} // namespace GRIM

