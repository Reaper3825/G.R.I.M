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

#include "grim_scale_buffer.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Shared/Loss/ComputeLoss/ComputeLoss_GPU.hpp"
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"    // For Flash Attention kernels
#include "../Layers/ScratchBlock/ScratchBlock_GPU.hpp"            // For ScratchBlock reasoning layer
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/UnigramByte/AtomTable.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"                 // Unified PBM (ALiBi + RoPE)
#include "../Layers/LayernNorm/RMSNorm_Kernel_GPU.hpp"            // For RMSNorm kernels

namespace GRIM {

//======================================================//
//  GPU Runtime Accessors (StreamController pattern)
//======================================================//

EmbeddingRuntime& LanguageModel::getGpuEmbedder() {
    if (!gpu_embedder_) {
        throw std::runtime_error("GPU embedder not initialized - call initGPU() first");
    }
    return *gpu_embedder_;
}

const EmbeddingRuntime& LanguageModel::getGpuEmbedder() const {
    if (!gpu_embedder_) {
        throw std::runtime_error("GPU embedder not initialized - call initGPU() first");
    }
    return *gpu_embedder_;
}

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

//======================================================//

namespace {

// Production constants - using centralized HyperParameters (Rule 20)
constexpr float kNegInf = HyperParameters::NEG_INF_ATTENTION;
constexpr float kProbabilityFloor = HyperParameters::PROBABILITY_FLOOR;
constexpr float kSoftmaxClipThreshold = HyperParameters::SOFTMAX_CLIP_THRESHOLD;
constexpr float kTemperatureEpsilon = HyperParameters::EPSILON_TEMPERATURE;
constexpr uint32_t kMaxReasonableVocabSize = HyperParameters::MAX_REASONABLE_VOCAB_SIZE;

bool isNumericAtomToken(int token_id) {
    using GRIM::Tokenizer::AtomType;
    const AtomType type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
    return GRIM::Tokenizer::isNumericAtom(type);
}

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

    // Version 2+ required (Rule 20: no backwards compatibility with v1)
    if (version < 2) {
        fprintf(stderr, "[LanguageModel] Unsupported version %d, minimum required is 2\n", version);
        return -1;
    }
    const bool has_v2_fields = true;

    if (has_v2_fields) {
        uint32_t checksum = 0;
        if (!file.read(reinterpret_cast<char*>(&checksum), sizeof(checksum))) {
            return -1;
        }
    }

    int config_vocab_size = 0;
    if (!file.read(reinterpret_cast<char*>(&config_vocab_size), sizeof(config_vocab_size))) {
        return -1;
    }

    int max_length = 0;
    if (!file.read(reinterpret_cast<char*>(&max_length), sizeof(max_length))) {
        return -1;
    }

    if (has_v2_fields) {
        bool nfkc = false;
        bool lower = false;
        bool fallback = false;
        if (!file.read(reinterpret_cast<char*>(&nfkc), sizeof(nfkc))) {
            return -1;
        }
        if (!file.read(reinterpret_cast<char*>(&lower), sizeof(lower))) {
            return -1;
        }
        if (!file.read(reinterpret_cast<char*>(&fallback), sizeof(fallback))) {
            return -1;
        }
    }

    uint32_t vocab_size = 0;
    if (!file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size))) {
        return -1;
    }

    // Defensive sanity: avoid using absurd header values.
    if (vocab_size > kMaxReasonableVocabSize) {
        return -1;
    }

    if (vocab_size == 0) {
        return config_vocab_size > 0 ? config_vocab_size : -1;
    }

    return static_cast<int>(vocab_size);
}

struct SampleResult {
    int token_id = -1;
    float probability = 0.0f;
    float log_probability = -std::numeric_limits<float>::infinity();
};

std::mt19937 makeGenerator(unsigned int seed, int sequence_offset = 0) {
    if (seed == 0) {
        std::random_device rd;
        return std::mt19937(rd());
    }
    return std::mt19937(seed + sequence_offset);
}

void applyBadWordMask(std::vector<float>& logits, const std::vector<int>& bad_words) {
    if (bad_words.empty()) return;
    const int vocab = static_cast<int>(logits.size());
    for (int token_id : bad_words) {
        if (token_id >= 0 && token_id < vocab) {
            logits[token_id] = kNegInf;
        }
    }
}

void applyRepetitionPenalty(std::vector<float>& logits,
                            const std::vector<int>& history,
                            float penalty,
                            int window = 128,
                            float decay = 0.98f)
{
    if (penalty <= 1.0f || history.empty()) return;

    const int vocab = static_cast<int>(logits.size());
    const int start = std::max(0, static_cast<int>(history.size()) - window);

    // Track which tokens we've already penalized
    std::unordered_set<int> seen;
    seen.reserve(window);

    float weight = 1.0f;

    // Walk backward = most recent tokens first
    for (int i = static_cast<int>(history.size()) - 1; i >= start; --i) {
        int token_id = history[i];
        if (token_id < 0 || token_id >= vocab) continue;

        // Only penalize each token once
        if (!seen.insert(token_id).second) continue;

        float& logit = logits[token_id];

        // Scaled penalty (bounded, sign-aware)
        float scaled_penalty = 1.0f + (penalty - 1.0f) * weight;

        if (logit > 0.0f) {
            logit /= scaled_penalty;
        } else {
            logit *= scaled_penalty;
        }

        weight *= decay;
        if (weight < 0.01f) break;
    }
}


std::vector<float> softmaxWithTemperature(const std::vector<float>& logits, // plateau debug change
                                          float temperature)
{
    const size_t n = logits.size();
    std::vector<float> probs(n);

    if (n == 0) return probs;

    // Temperature scaling (safe)
    float inv_temp = 1.0f;
    if (temperature > 0.0f && std::fabs(temperature - 1.0f) > kTemperatureEpsilon) {
        inv_temp = 1.0f / temperature;
    }

    // Find max logit for numerical stability
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float v : logits) {
        max_logit = std::max(max_logit, v * inv_temp);
    }

    float sum = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float shifted = logits[i] * inv_temp - max_logit;

        // TRUE clipping: prevent exp underflow, not probability mass deletion
        shifted = std::max(shifted, kSoftmaxClipThreshold);

        float v = std::exp(shifted);
        probs[i] = v;
        sum += v;
    }

    // Normalize (or fail loudly)
    if (sum <= 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error("softmaxWithTemperature: invalid probability sum");
    }

    float inv_sum = 1.0f / sum;
    for (float& p : probs) {
        p *= inv_sum;
    }

    return probs;
}


void normalizeProbabilities(std::vector<float>& probs) {
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f) 
    return;
    for (float& p : probs) {
        p /= sum;
    }
}

void applyTopKFilter(std::vector<float>& probs, int top_k) { 
    if (top_k <= 0 || top_k >= static_cast<int>(probs.size())) return;
    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });
    for (size_t i = top_k; i < indices.size(); ++i) {
        probs[indices[i]] = 0.0f;
    }
}

void applyTopPFilter(std::vector<float>& probs, float top_p) { // plateau debug change
    if (top_p <= 0.0f || top_p >= 1.0f) return;

    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    float cumulative = 0.0f;
    bool kept_any = false;

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        float p = probs[idx];

        // Always keep at least 1 token (the max-prob token)
        if (!kept_any) {
            cumulative += p;
            kept_any = true;
            continue;
        }

        // Once we have enough mass, cut the rest
        if (cumulative >= top_p) {
            probs[idx] = 0.0f;
        } else {
            cumulative += p;
        }
    }
}


SampleResult sampleFromLogits(const std::vector<float>& logits,
                              const GenerationConfig& cfg,
                              bool allow_sampling,
                              std::mt19937& rng,
                              int vocab_size) {
    SampleResult result;
    if (logits.empty()) {
        throw std::runtime_error("sampleFromLogits: empty logits vector");
    }
    
    if (vocab_size <= 0) {
        throw std::runtime_error("sampleFromLogits: vocab_size must be > 0");
    }
    // STRICT: vocab_size must match logits size exactly
    if (static_cast<int>(logits.size()) != vocab_size) {
        throw std::runtime_error("sampleFromLogits: vocab_size=" + std::to_string(vocab_size) + 
                                 " but logits.size()=" + std::to_string(logits.size()));
    }
    const int valid_vocab_size = vocab_size;
    if (cfg.strategy == SamplingStrategy::BEAM_SEARCH) {
        throw std::runtime_error("sampleFromLogits: BEAM_SEARCH is not supported");
    }
    if (!allow_sampling && cfg.strategy != SamplingStrategy::GREEDY) {
        throw std::runtime_error("sampleFromLogits: non-greedy strategy requires sampling");
    }
    auto probabilities = softmaxWithTemperature(logits, cfg.temperature);
    const bool use_sampling = allow_sampling && cfg.strategy != SamplingStrategy::GREEDY;

    if (use_sampling) {
        if (cfg.strategy == SamplingStrategy::TOP_K) {
            if (cfg.top_k <= 0 || cfg.top_k > valid_vocab_size) {
                throw std::runtime_error("sampleFromLogits: top_k out of range");
            }
            applyTopKFilter(probabilities, cfg.top_k);
            normalizeProbabilities(probabilities);   // keep distribution valid
        } else if (cfg.strategy == SamplingStrategy::TOP_P) {
            if (cfg.top_p <= 0.0f || cfg.top_p >= 1.0f) {
                throw std::runtime_error("sampleFromLogits: top_p out of range");
            }
            // probabilities are already normalized coming out of softmaxWithTemperature
            applyTopPFilter(probabilities, cfg.top_p);
            normalizeProbabilities(probabilities);   // renormalize nucleus
        } else {
            throw std::runtime_error("sampleFromLogits: unsupported sampling strategy");
        }
    }

    float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    if (sum <= 0.0f) {
        // STRICT: Zero probability sum indicates broken model/logits
        throw std::runtime_error("sampleFromLogits: probability sum is zero (temperature=" + 
                                 std::to_string(cfg.temperature) + "). Model outputs are degenerate.");
    }
    if (!use_sampling) {
        // Greedy decoding
        int idx = static_cast<int>(std::distance(
            probabilities.begin(),
            std::max_element(probabilities.begin(), probabilities.end())));
        if (idx < 0 || idx >= valid_vocab_size) {
            throw std::runtime_error("sampleFromLogits: greedy index out of range: idx=" +
                                     std::to_string(idx) + " vocab_size=" + std::to_string(valid_vocab_size));
        }
        float prob = std::max(probabilities[idx], kProbabilityFloor);
        result.token_id = idx;
        result.probability = prob;
        result.log_probability = std::log(prob);
        return result;
    }
    
    // Sampling with temperature
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    int sampled = dist(rng);
    
    // STRICT: Sampled token must be valid
    if (sampled < 0 || sampled >= valid_vocab_size) {
        throw std::runtime_error("sampleFromLogits: sampled token out of range: sampled=" + 
                                 std::to_string(sampled) + " vocab_size=" + std::to_string(valid_vocab_size));
    }
    
    float prob = std::max(probabilities[sampled], kProbabilityFloor);
    result.token_id = sampled;
    result.probability = prob;
    result.log_probability = std::log(prob);
    return result;
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
//  PositionalEncoding Implementation
//======================================================//

PositionalEncoding::PositionalEncoding(int max_len, int dim)
    : max_seq_len(max_len), d_model(dim)
{
    encodings.resize(max_len);
    
    // Generate sinusoidal positional encodings
    for (int pos = 0; pos < max_len; ++pos) {
        Vector encoding(dim);
        for (int i = 0; i < dim; ++i) {
            float angle = pos / std::pow(10000.0f, (2.0f * (i / 2)) / dim);
            encoding[i] = (i % 2 == 0) ? std::sin(angle) : std::cos(angle);
        }
        encodings[pos] = encoding;
    }
}

Vector PositionalEncoding::getEncoding(int position) const {
    if (position >= 0 && position < max_seq_len) {
        return encodings[position];
    }
    return Vector(d_model, 0.0f);
}

//======================================================//
//  ALiBiPositionalBias Implementation (Unified PBM)
//======================================================//

ALiBiPositionalBias::ALiBiPositionalBias()
    : num_heads(0),
      d_head(0),
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
        d_head = other.d_head;
        initialized = other.initialized;
        type = other.type;
    }
    return *this;
}

ALiBiPositionalBias::~ALiBiPositionalBias() {
    cleanup();
}

void ALiBiPositionalBias::computeSlopes(int num_heads_, int num_kv_heads_, int d_head_, PositionalEncodingType type_) {
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
    d_head = d_head_;
#ifdef USE_CUDA
    // Use unified PBM - always allocates both ALiBi + RoPE
    PBM::PBMConfig config{};
    config.num_heads = num_heads_;
    config.head_dim = d_head_;
    config.rotary_dim = config.head_dim;
    config.num_kv_heads = num_kv_heads_;
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
    return (initialized && pbm_state_.alibi_slopes) ? pbm_state_.alibi_slopes : nullptr;
#else
    return nullptr;
#endif
}

float* ALiBiPositionalBias::getRoPEFreqs() const {
#ifdef USE_CUDA
    return (initialized && pbm_state_.rope_inv_freq) ? pbm_state_.rope_inv_freq : nullptr;
#else
    return nullptr;
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
    d_head = 0;
}

// Legacy extern declarations removed - pure GPU training uses:
// - Flash Attention with GQA (Layers/FlashAttention/Flash_Attention_Kernal.cu)
// - GPUEncoderLayer for FFN (Layers/Encoder/GPUEncoderLayer.cu)
// - RMSNorm kernels (Layers/LayernNorm/RMSNorm_Kernel_GPU.cu)

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
    pos_encoding = PositionalEncoding(max_seq_len, d_model);
    rms_gamma = Vector(d_model, 1.0f);  // RMSNorm gamma
}

void GrimEmbeddingStack::enableALiBi(int num_heads, int num_kv_heads) {
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
    alibi_->computeSlopes(num_heads, num_kv_heads, d_head, PositionalEncodingType::ALIBI);
}

void GrimEmbeddingStack::enableHybridPositionalEncoding(int num_heads, int d_head, int num_kv_heads) {
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
    alibi_->d_head = d_head;
    // Pass through num_kv_heads so PBM and encoder GQA settings match
    alibi_->computeSlopes(num_heads, num_kv_heads, d_head, PositionalEncodingType::ALIBI_ROPE);
}

const ALiBiPositionalBias* GrimEmbeddingStack::getALiBiBias() const {
    return alibi_.get();
}

const Matrix& GrimEmbeddingStack::getTokenEmbeddings() const {
    return token_embed;
}

// getBatchEmbeddings removed - pure GPU training uses EmbeddingRuntime directly

// MultiHeadAttention removed - pure GPU training uses Flash Attention with GQA
// See: Layers/FlashAttention/Flash_Attention_Kernal.cu

// FeedForwardNetwork removed - pure GPU training uses GPUEncoderLayer FFN directly
// See: Layers/Encoder/GPUEncoderLayer.cu
    


// RMSNorm, MultiHeadAttention, FeedForwardNetwork CPU wrappers removed
// Pure GPU training uses:
// - Layers/LayernNorm/RMSNorm_Kernel_GPU.cu for normalization
// - Layers/FlashAttention/Flash_Attention_Kernal.cu for GQA attention
// - Layers/Encoder/GPUEncoderLayer.cu for FFN

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
        int d_head = config_.d_model / config_.num_heads;
        
        // Use appropriate initialization based on encoding type
        if (config_.positional_encoding == PositionalEncodingType::ALIBI_ROPE) {
            embedder_->enableHybridPositionalEncoding(config_.num_heads, d_head, config_.num_kv_heads);
        } else if (config_.positional_encoding == PositionalEncodingType::ALIBI) {
            embedder_->enableALiBi(config_.num_heads, config_.num_kv_heads);
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
                              const std::vector<uint8_t>& token_numeric_mask) {
#ifdef USE_CUDA
    // Use GPU if available
    if (config_.use_gpu && gpu_encoder_) {
        return forwardGPU(token_ids, token_numeric_values, token_numeric_mask);
    }
#endif
    throw std::runtime_error("LanguageModel::forward requires GPU initialization");
}

Vector LanguageModel::getNextTokenLogits(const std::vector<int>& context_tokens,
                                         const std::vector<float>& context_numeric_values,
                                         const std::vector<uint8_t>& context_numeric_mask) {
#ifdef USE_CUDA
    // Use GPU if available
    if (config_.use_gpu && gpu_encoder_) {
        return getNextTokenLogitsGPU(context_tokens,
                                     context_numeric_values,
                                     context_numeric_mask);
    }
#endif
    throw std::runtime_error("LanguageModel::getNextTokenLogits requires GPU initialization");
}


Vector LanguageModel::getNextTokenLogitsGPU(const std::vector<int>& context_tokens,
                                            const std::vector<float>& context_numeric_values,
                                            const std::vector<uint8_t>& context_numeric_mask) {
    if (!config_.use_gpu || !gpu_encoder_) {
        throw std::runtime_error("getNextTokenLogitsGPU requires initialized GPU encoder");
    }
    if (context_tokens.empty()) {
        throw std::runtime_error("getNextTokenLogitsGPU requires non-empty context_tokens");
    }
    if (context_numeric_values.size() != context_tokens.size() ||
        context_numeric_mask.size() != context_tokens.size()) {
        throw std::runtime_error("getNextTokenLogitsGPU: numeric side-channel length mismatch");
    }
    const size_t context_len = context_tokens.size();
    if (context_len > static_cast<size_t>(config_.max_seq_len)) {
        throw std::runtime_error("getNextTokenLogitsGPU: context length " +
                                 std::to_string(context_len) + " exceeds max_seq_len " +
                                 std::to_string(config_.max_seq_len));
    }
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
    }
    if (training_state_.max_cached_seq_len > 0 &&
        context_len > static_cast<size_t>(training_state_.max_cached_seq_len)) {
        throw std::runtime_error("getNextTokenLogitsGPU: context length " +
                                 std::to_string(context_len) + " exceeds max_cached_seq_len " +
                                 std::to_string(training_state_.max_cached_seq_len));
    }
    if (!training_state_.stream_ctrl.isInitialized()) {
        throw std::runtime_error("getNextTokenLogitsGPU: StreamController not initialized");
    }
    
    forwardWithCache(context_tokens, context_numeric_values, context_numeric_mask);
    
    const auto& cfg = getConfig();
    if (cfg.vocab_size <= 0) {
        throw std::runtime_error("getNextTokenLogitsGPU: invalid vocab_size");
    }
    Vector logits(cfg.vocab_size);
    if (training_state_.cached_seq_len != static_cast<int>(context_len)) {
        throw std::runtime_error("getNextTokenLogitsGPU: cached_seq_len mismatch after forwardWithCache");
    }
    if (!training_state_.cached_logits) {
        throw std::runtime_error("getNextTokenLogitsGPU: cached_logits not initialized");
    }
    const size_t seq_len = static_cast<size_t>(training_state_.cached_seq_len);
    const size_t column_offset = (seq_len - 1) * static_cast<size_t>(cfg.vocab_size);
    
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    if (!stream) {
        std::cerr << "[FATAL] getNextTokenLogitsGPU: primary stream is null (default stream usage disallowed)"
                  << std::endl;
        throw std::runtime_error("getNextTokenLogitsGPU: primary stream is null");
    }
    CUDA_CHECK(cudaMemcpyAsync(logits.data.data(),
                               training_state_.cached_logits + column_offset,
                               static_cast<size_t>(cfg.vocab_size) * sizeof(float),
                               cudaMemcpyDeviceToHost,
                               stream));
    if (!training_state_.stream_ctrl.syncPrimaryStream()) {
        throw std::runtime_error("getNextTokenLogitsGPU: failed to sync primary stream");
    }
    return logits;
}

Vector LanguageModel::forwardGPU(const std::vector<int>& token_ids,
                                 const std::vector<float>& token_numeric_values,
                                 const std::vector<uint8_t>& token_numeric_mask) {
    return forwardWithCache(token_ids, token_numeric_values, token_numeric_mask);
}

TokenBufferView LanguageModel::getTokenBufferView() {
    TokenBufferView view{};
    if (!config_.use_gpu) {
        return view;
    }
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
    }
    view.device_token_ids = training_state_.cached_token_ids;
    view.device_token_numeric_values = training_state_.cached_token_numeric_values;
    view.device_token_numeric_mask = training_state_.cached_token_numeric_mask;
    view.max_tokens = config_.max_seq_len;
    view.stream = training_state_.stream_ctrl.getPrimaryStream();
    return view;
}

void LanguageModel::markDevicePromptReady(int token_count) {
    if (!config_.use_gpu) {
        return;
    }
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            throw std::runtime_error("markDevicePromptReady: Inference state not initialized unless in inference mode");
            initInferenceState();
        }
    }
    staged_prompt_ready_ = true;
    staged_prompt_len_ = std::min(token_count, config_.max_seq_len);
}

std::vector<GeneratedSequence> LanguageModel::generate(
    const std::vector<int>& prompt_tokens,
    const std::vector<float>& prompt_numeric_values,
    const std::vector<uint8_t>& prompt_numeric_mask,
    const GenerationConfig* gen_config)
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
            std::mt19937 rng = makeGenerator(cfg.seed, i);
            outputs.push_back(generateSequenceGPU(prompt_tokens,
                                                  prompt_numeric_values,
                                                  prompt_numeric_mask,
                                                  cfg,
                                                  nullptr,
                                                  rng));
        }
        return outputs;
    }
#endif
    throw std::runtime_error("LanguageModel::generate requires GPU initialization");
}

GeneratedSequence LanguageModel::generateStream(
    const std::vector<int>& prompt_tokens,
    const std::vector<float>& prompt_numeric_values,
    const std::vector<uint8_t>& prompt_numeric_mask,
    GenerationStreamCallback callback,
    const GenerationConfig* gen_config)
{
#ifdef USE_CUDA
    if (config_.use_gpu && gpu_encoder_) {
        GenerationConfig cfg = gen_config ? *gen_config : config_.generation;
        std::mt19937 rng = makeGenerator(cfg.seed);
        return generateSequenceGPU(prompt_tokens,
                                   prompt_numeric_values,
                                   prompt_numeric_mask,
                                   cfg,
                                   &callback,
                                   rng);
    }
#endif
    throw std::runtime_error("LanguageModel::generateStream requires GPU initialization");
}


GeneratedSequence LanguageModel::generateSequenceGPU(const std::vector<int>& prompt_tokens,
                                                     const std::vector<float>& prompt_numeric_values,
                                                     const std::vector<uint8_t>& prompt_numeric_mask,
                                                     const GenerationConfig& cfg,
                                                     GenerationStreamCallback* stream_callback,
                                                     std::mt19937& rng) {
    GeneratedSequence sequence;
    sequence.token_ids = prompt_tokens;
    sequence.token_numeric_values = prompt_numeric_values;
    sequence.token_numeric_mask = prompt_numeric_mask;
    
    if (!config_.use_gpu || !gpu_encoder_) {
        throw std::runtime_error("generateSequenceGPU requires initialized GPU encoder");
    }
    
    if (prompt_tokens.size() >= static_cast<size_t>(config_.max_seq_len)) {
        throw std::runtime_error("generateSequenceGPU: prompt length " +
                                 std::to_string(prompt_tokens.size()) + " exceeds max_seq_len " +
                                 std::to_string(config_.max_seq_len));
    }
    if (prompt_numeric_values.size() != prompt_tokens.size() ||
        prompt_numeric_mask.size() != prompt_tokens.size()) {
        throw std::runtime_error("generateSequenceGPU: numeric side-channel length mismatch");
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
    if (!cfg.do_sample && cfg.strategy != SamplingStrategy::GREEDY) {
        throw std::runtime_error("generateSequenceGPU: non-greedy strategy requires sampling");
    }
    if (cfg.strategy == SamplingStrategy::TOP_K &&
        (cfg.top_k <= 0 || cfg.top_k > vocab_size)) {
        throw std::runtime_error("generateSequenceGPU: top_k out of range");
    }
    if (cfg.strategy == SamplingStrategy::TOP_P &&
        (cfg.top_p <= 0.0f || cfg.top_p >= 1.0f)) {
        throw std::runtime_error("generateSequenceGPU: top_p out of range");
    }
    if (cfg.strategy != SamplingStrategy::GREEDY && cfg.temperature <= 0.0f) {
        throw std::runtime_error("generateSequenceGPU: temperature must be > 0 for sampling");
    }
    
    // =========================================================================
    // INCREMENTAL GENERATION WITH KV CACHE
    // Step 0: Process full prompt with forwardInit() - O(n)
    // Step 1+: Process single token with forwardStep() - O(1) per step
    // Total: O(n) instead of O(n²)
    // =========================================================================
    
    // Reset KV cache for new generation
    resetKVCache();
    
    // Process prompt and get logits for first new token
    Vector logits_vec = forwardInit(prompt_tokens,
                                    prompt_numeric_values,
                                    prompt_numeric_mask);
    if (logits_vec.data.empty()) {
        throw std::runtime_error("generateSequenceGPU: forwardInit returned empty logits");
    }

    const auto decodeNumericPrediction = [&](float pred) -> float {
        if (!std::isfinite(pred)) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (!config_.numeric_head_log_scale) {
            return pred;
        }
        constexpr float kLogMax = 20.0f;
        float sign = pred < 0.0f ? -1.0f : 1.0f;
        float abs_pred = std::fabs(pred);
        if (abs_pred > kLogMax) {
            abs_pred = kLogMax;
        }
        return sign * (std::exp(abs_pred) - 1.0f);
    };

    auto fetchNumericPrediction = [&](int logits_pos) -> float {
        if (!config_.numeric_head_enabled || !training_state_.cached_numeric_predictions) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (logits_pos < 0) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        float pred = std::numeric_limits<float>::quiet_NaN();
        cudaMemcpyAsync(&pred,
                        training_state_.cached_numeric_predictions + logits_pos,
                        sizeof(float),
                        cudaMemcpyDeviceToHost,
                        training_state_.stream_ctrl.getPrimaryStream());
        training_state_.stream_ctrl.syncPrimaryStream();
        return decodeNumericPrediction(pred);
    };

    int logits_pos = static_cast<int>(prompt_tokens.size()) - 1;
    
    for (int step = 0; step < max_steps; ++step) {
        // Check max sequence length
        const int current_len = getKVCacheLength();
        if (current_len >= config_.max_seq_len) {
            sequence.finished = true;
            break;
        }
        
        std::vector<float> logits = logits_vec.data;
        
        if (cfg.repetition_penalty > 1.0f) {
            applyRepetitionPenalty(logits, sequence.token_ids, cfg.repetition_penalty);
        }
        if (!cfg.bad_words_ids.empty()) {
            applyBadWordMask(logits, cfg.bad_words_ids);
        }
        
        if (static_cast<int>(logits.size()) > vocab_size) {
            // Zero out logits for invalid token positions
            std::fill(logits.begin() + vocab_size, logits.end(), kNegInf);
        }
        
        const bool allow_sampling = cfg.do_sample &&
                                    cfg.strategy != SamplingStrategy::GREEDY &&
                                    cfg.strategy != SamplingStrategy::BEAM_SEARCH &&
                                    cfg.temperature > 0.0f;
        
        // Pass vocab_size for validation
        SampleResult sample = sampleFromLogits(logits, cfg, allow_sampling, rng, vocab_size);
        
        // Validate sampled token (no fallback)
        if (sample.token_id < 0 || sample.token_id >= vocab_size) {
            fprintf(stderr, "[LanguageModel] FATAL: sampled token out of range (token_id=%d, vocab=%d)\n",
                    sample.token_id, vocab_size);
            throw std::runtime_error("LanguageModel: sampled token out of range");
        }
        
        float step_numeric_value = 0.0f;
        uint8_t step_numeric_mask = 0;
        float output_numeric_value = 0.0f;
        uint8_t output_numeric_mask = 0;
        if (isNumericAtomToken(sample.token_id)) {
            // Emit numeric prediction side-channel; keep generation inputs neutral.
            output_numeric_mask = 1;
            output_numeric_value = fetchNumericPrediction(logits_pos);
        }

        // Check for EOS BEFORE processing next step
        if (sample.token_id == cfg.eos_token_id &&
            step + 1 >= cfg.min_new_tokens) {
            sequence.token_ids.push_back(sample.token_id);
            sequence.token_scores.push_back(sample.log_probability);
            sequence.token_numeric_values.push_back(output_numeric_value);
            sequence.token_numeric_mask.push_back(output_numeric_mask);
            sequence.score += sample.log_probability;
            sequence.finished = true;
            if (stream_callback) {
                (*stream_callback)(sample.token_id, sample.probability);
            }
            break;
        }
        
        // Add token to sequence
        sequence.token_ids.push_back(sample.token_id);
        sequence.token_scores.push_back(sample.log_probability);
        sequence.token_numeric_values.push_back(output_numeric_value);
        sequence.token_numeric_mask.push_back(output_numeric_mask);
        sequence.score += sample.log_probability;
        
        if (stream_callback) {
            (*stream_callback)(sample.token_id, sample.probability);
        }
        
        // Get logits for NEXT step using incremental forward
        // This only processes the new token, reusing cached K,V
        logits_vec = forwardStep(sample.token_id, step_numeric_value, step_numeric_mask);
        if (logits_vec.data.empty()) {
            throw std::runtime_error("generateSequenceGPU: forwardStep returned empty logits");
        }
        logits_pos = training_state_.kv_cache_len - 1;
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
    if (scratch_block_layer_) {
        scratch_block_layer_->setEnabled(enabled);
    }
}

bool LanguageModel::isScratchBlockEnabled() const {
    return scratch_block_layer_ && scratch_block_layer_->isEnabled();
}

void LanguageModel::configureScratchPool(bool enabled) {
    if (training_state_.scratch_pool) {
        training_state_.scratch_enabled = enabled;
        training_state_.scratch_pool->setEnabled(enabled);
    }
}

bool LanguageModel::isScratchPoolInitialized() const {
    return training_state_.scratch_pool != nullptr && training_state_.scratch_pool->isInitialized();
}




} // namespace GRIM

