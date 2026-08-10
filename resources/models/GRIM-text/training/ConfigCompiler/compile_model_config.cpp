#include "grim_compiled_hyperparameters_generated.h"

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr std::uint32_t kSchemaVersion = 1;
constexpr std::uint32_t kSemanticVersion = 1;
constexpr std::uint32_t kFfnMultiplier = 4;
constexpr std::uint32_t kSpecialTokenCount = 4;
constexpr std::uint32_t kByteTokenCount = 256;
constexpr std::uint32_t kAtomTokenCount = 2;
constexpr std::uint32_t kUnigramOffset =
    kSpecialTokenCount + kByteTokenCount + kAtomTokenCount;
constexpr std::uint32_t kMaxPieceLength = 32;

struct Cli {
    fs::path input;
    fs::path vocab;
    fs::path output;
};

struct VocabFacts {
    std::vector<std::uint8_t> bytes;
    std::array<std::uint8_t, 32> sha256{};
    std::uint32_t token_space_size = 0;
};

struct EffectiveConfig {
    std::uint32_t d_model = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t num_kv_heads = 0;
    std::uint32_t d_ff = 0;
    std::uint32_t max_seq_len = 0;
    std::uint32_t vocab_size = 0;
    bool tie_embeddings = false;
    float embedding_scale = 0.0f;

    std::uint32_t head_dim = 0;
    std::uint32_t heads_per_kv_group = 0;
    std::uint32_t kv_dim = 0;
    std::uint32_t qkv_dim = 0;
    std::uint32_t rotary_dim = 0;
    bool is_gqa = false;
    float attention_softmax_scale = 0.0f;
    std::vector<float> alibi_slopes;
    std::vector<float> rope_inv_freq;

    bool attention_qkv_bias = false;
    bool attention_output_bias = false;
    bool ffn_output_bias = false;
    bool lm_head_bias = false;
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool qk_norm = false;
    bool attention_off_by_one = false;
    bool attention_residual_gate = false;

    bool use_rope = false;
    bool use_alibi = false;
    GRIMConfig::PositionalEncodingType positional_kind =
        GRIMConfig::PositionalEncodingType_Unspecified;
    std::uint32_t rope_base_seq_len = 0;
    std::uint32_t alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = 0.0f;
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;

    float rms_epsilon = 1.0e-5f;
    bool use_layer_scale = false;
    bool center_residuals = false;

    bool lm_head_unigram_bias = false;
    bool lm_head_center_hidden_states = false;
    bool center_logits = false;
    bool project_out_pc1 = false;
    std::uint32_t pc1_power_iters = 0;
    bool lm_head_mlp_enabled = false;
    std::uint32_t lm_head_mlp_d_ff = 0;
    float lm_head_mlp_alpha = 0.0f;

    bool use_atom_data = false;
    std::uint32_t atom_embedding_dim = 0;
    bool execution_block_enabled = false;
    std::int32_t execution_block_layer = -1;
    std::uint32_t execution_block_num_ops = 0;
    std::uint32_t execution_block_num_slots = 0;
    std::uint32_t execution_block_num_scratch_slots = 0;
    std::uint32_t execution_block_num_steps = 0;
    std::uint32_t execution_block_value_decode_input_dim = 0;
    std::uint32_t execution_block_value_decode_hidden_dim = 0;
    std::uint32_t execution_block_d_type = 0;
    std::uint32_t execution_block_cross_attn_topk = 0;
    float execution_block_usage_decay = 0.0f;
    float execution_block_inject_gate_temp = 0.0f;
    std::uint32_t execution_block_result_slot_mode = 0;
    std::int32_t execution_block_result_slot_index = -1;
    float execution_block_magnitude_limit = 0.0f;
    float execution_block_causal_w1_transition = 0.0f;
    bool execution_block_decode_bias = false;
    bool execution_block_value_embedding_bias = false;
    bool execution_block_scalar_bias = false;
    bool execution_block_trace_bias = false;

    bool number_encoder_enabled = false;
    std::uint32_t number_encoder_max_digit_slots = 0;
    std::uint32_t number_encoder_d_hidden = 0;
    std::uint32_t number_encoder_max_abs_pow10 = 0;
    bool number_encoder_contribution_bias = false;
    bool number_encoder_global_bias = false;

    bool arg_selector_enabled = false;
    bool slot_seed_encoder_enabled = false;
    std::uint32_t slot_seed_encoder_d_hidden = 0;
    bool slot_seed_encoder_bias = false;
    bool slot_seed_encoder_type_embedding = false;

    std::string tokenizer_model_type;
    std::vector<std::string> tokenizer_special_tokens;
    bool tokenizer_add_bos = false;
    bool tokenizer_add_eos = false;
    std::string tokenizer_unk_token;
    std::string tokenizer_pad_token;
    std::string tokenizer_bos_token;
    std::string tokenizer_eos_token;
    bool tokenizer_enable_nfkc = false;
    bool tokenizer_enable_lowercasing = false;
    bool tokenizer_enable_byte_fallback = false;
    bool tokenizer_enable_atom_reasoning = false;
    bool tokenizer_detect_numbers = false;

    std::vector<GRIMConfig::ModelCapability> capabilities;
};

// Small, dependency-free SHA-256 implementation. The compiler must remain a
// host-only tool and cannot acquire a CUDA/runtime dependency through hashing.
class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, take);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bit_count = static_cast<std::uint64_t>(total_bytes_) * 8u;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<std::uint8_t>(bit_count >> (i * 8));
        }
        transform(block_.data());

        std::array<std::uint8_t, 32> out{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            out[i * 4 + 0] = static_cast<std::uint8_t>(state_[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return out;
    }

private:
    static std::uint32_t rotateRight(std::uint32_t value, unsigned count) {
        return (value >> count) | (value << (32u - count));
    }

    static std::uint32_t readBe32(const std::uint8_t* p) {
        return (static_cast<std::uint32_t>(p[0]) << 24) |
               (static_cast<std::uint32_t>(p[1]) << 16) |
               (static_cast<std::uint32_t>(p[2]) << 8) |
               static_cast<std::uint32_t>(p[3]);
    }

    void transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> k = {{
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
        }};

        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) w[i] = readBe32(block + i * 4);
        for (std::size_t i = 16; i < w.size(); ++i) {
            const std::uint32_t s0 = rotateRight(w[i - 15], 7) ^
                                     rotateRight(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotateRight(w[i - 2], 17) ^
                                     rotateRight(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            const std::uint32_t choice = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choice + k[i] + w[i];
            const std::uint32_t s0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    }};
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_ = 0;
    std::size_t total_bytes_ = 0;
};

std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t size) {
    Sha256 hasher;
    hasher.update(data, size);
    return hasher.finish();
}

std::uint64_t rotateLeft64(std::uint64_t value, unsigned count) {
    return (value << count) | (value >> (64u - count));
}

std::uint32_t readLe32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t readLe64(const std::uint8_t* p) {
    return static_cast<std::uint64_t>(readLe32(p)) |
           (static_cast<std::uint64_t>(readLe32(p + 4)) << 32);
}

std::uint64_t xxhash64(const std::uint8_t* input, std::size_t length, std::uint64_t seed = 0) {
    constexpr std::uint64_t p1 = 11400714785074694791ull;
    constexpr std::uint64_t p2 = 14029467366897019727ull;
    constexpr std::uint64_t p3 = 1609587929392839161ull;
    constexpr std::uint64_t p4 = 9650029242287828579ull;
    constexpr std::uint64_t p5 = 2870177450012600261ull;
    const auto round = [=](std::uint64_t acc, std::uint64_t lane) {
        acc += lane * p2;
        acc = rotateLeft64(acc, 31);
        return acc * p1;
    };
    const auto merge = [&](std::uint64_t acc, std::uint64_t value) {
        acc ^= round(0, value);
        return acc * p1 + p4;
    };

    const std::uint8_t* p = input;
    const std::uint8_t* const end = input + length;
    std::uint64_t h = 0;
    if (length >= 32) {
        std::uint64_t v1 = seed + p1 + p2;
        std::uint64_t v2 = seed + p2;
        std::uint64_t v3 = seed;
        std::uint64_t v4 = seed - p1;
        const std::uint8_t* const limit = end - 32;
        do {
            v1 = round(v1, readLe64(p)); p += 8;
            v2 = round(v2, readLe64(p)); p += 8;
            v3 = round(v3, readLe64(p)); p += 8;
            v4 = round(v4, readLe64(p)); p += 8;
        } while (p <= limit);
        h = rotateLeft64(v1, 1) + rotateLeft64(v2, 7) +
            rotateLeft64(v3, 12) + rotateLeft64(v4, 18);
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed + p5;
    }
    h += length;
    while (p + 8 <= end) {
        const std::uint64_t lane = round(0, readLe64(p));
        h ^= lane;
        h = rotateLeft64(h, 27) * p1 + p4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<std::uint64_t>(readLe32(p)) * p1;
        h = rotateLeft64(h, 23) * p2 + p3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<std::uint64_t>(*p++) * p5;
        h = rotateLeft64(h, 11) * p1;
    }
    h ^= h >> 33; h *= p2; h ^= h >> 29; h *= p3; h ^= h >> 32;
    return h;
}

std::vector<std::uint8_t> readFile(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open file: " + path.string());
    const auto end = stream.tellg();
    if (end < 0) throw std::runtime_error("cannot determine file size: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    stream.seekg(0);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream) throw std::runtime_error("failed reading file: " + path.string());
    return bytes;
}

class ByteCursor {
public:
    ByteCursor(const std::vector<std::uint8_t>& bytes, const fs::path& source)
        : bytes_(bytes), source_(source.string()) {}

    std::uint16_t u16() {
        require(2);
        const auto value = static_cast<std::uint16_t>(bytes_[at_]) |
                           static_cast<std::uint16_t>(bytes_[at_ + 1] << 8);
        at_ += 2;
        return value;
    }
    std::uint32_t u32() {
        require(4);
        const auto value = readLe32(bytes_.data() + at_);
        at_ += 4;
        return value;
    }
    std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    float f32() {
        const std::uint32_t bits = u32();
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    std::string text(std::size_t size) {
        require(size);
        std::string value(reinterpret_cast<const char*>(bytes_.data() + at_), size);
        at_ += size;
        return value;
    }
    void skip(std::size_t size) { require(size); at_ += size; }
    bool done() const { return at_ == bytes_.size(); }

private:
    void require(std::size_t size) const {
        if (size > bytes_.size() - at_) {
            throw std::runtime_error("truncated KTMG vocabulary artifact: " + source_);
        }
    }
    const std::vector<std::uint8_t>& bytes_;
    std::string source_;
    std::size_t at_ = 0;
};

VocabFacts inspectVocab(const fs::path& path) {
    VocabFacts facts;
    facts.bytes = readFile(path);
    facts.sha256 = sha256(facts.bytes.data(), facts.bytes.size());
    ByteCursor cursor(facts.bytes, path);
    if (cursor.text(4) != "KTMG") throw std::runtime_error("vocabulary is not a KTMG artifact: " + path.string());
    const std::uint16_t version = cursor.u16();
    if (version != 4) throw std::runtime_error("unsupported KTMG vocabulary version " + std::to_string(version));
    cursor.skip(4); // Historical checksum placeholder; exact bytes are covered by SHA-256.
    const std::uint32_t record_count = cursor.u32();
    const std::uint32_t max_length = cursor.u32();
    if (max_length != kMaxPieceLength) {
        throw std::runtime_error("KTMG max piece length must be " + std::to_string(kMaxPieceLength));
    }
    cursor.skip(3); // Reserved flags.
    facts.token_space_size = cursor.u32();
    if (record_count < kSpecialTokenCount) throw std::runtime_error("KTMG vocabulary omits special-token records");

    static constexpr std::array<const char*, 4> special_text{{"<unk>", "<pad>", "<s>", "</s>"}};
    std::array<bool, 4> special_seen{};
    std::set<std::string> learned_tokens;
    std::uint32_t learned_count = 0;
    for (std::uint32_t record = 0; record < record_count; ++record) {
        const std::uint32_t length = cursor.u32();
        if (length == 0 || length > kMaxPieceLength) {
            throw std::runtime_error("invalid KTMG piece length at record " + std::to_string(record));
        }
        const std::string token = cursor.text(length);
        const float score = cursor.f32();
        const std::int32_t token_id = cursor.i32();
        if (!std::isfinite(score)) throw std::runtime_error("non-finite KTMG score at record " + std::to_string(record));
        if (token_id >= 0 && token_id < static_cast<std::int32_t>(kSpecialTokenCount)) {
            const auto index = static_cast<std::size_t>(token_id);
            if (special_seen[index] || token != special_text[index]) {
                throw std::runtime_error("invalid KTMG special-token metadata at record " + std::to_string(record));
            }
            special_seen[index] = true;
        } else {
            const std::int32_t expected = static_cast<std::int32_t>(kUnigramOffset + learned_count);
            if (token_id != expected) {
                throw std::runtime_error("non-contiguous KTMG learned token ID at record " + std::to_string(record));
            }
            if (!learned_tokens.insert(token).second) {
                throw std::runtime_error("duplicate KTMG learned token at record " + std::to_string(record));
            }
            ++learned_count;
        }
    }
    if (!cursor.done()) throw std::runtime_error("KTMG vocabulary has trailing bytes: " + path.string());
    if (std::find(special_seen.begin(), special_seen.end(), false) != special_seen.end()) {
        throw std::runtime_error("KTMG vocabulary does not define all four special tokens");
    }
    if (record_count != learned_count + kSpecialTokenCount) {
        throw std::runtime_error("KTMG vocabulary record accounting mismatch");
    }
    const std::uint32_t computed_size = kUnigramOffset + learned_count;
    if (facts.token_space_size != computed_size) {
        throw std::runtime_error("KTMG token-space header " + std::to_string(facts.token_space_size) +
                                 " does not match computed size " + std::to_string(computed_size));
    }
    return facts;
}

template <typename T>
T required(const json& config, const char* name) {
    try {
        return config.at(name).get<T>();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("invalid required training.config.") + name + ": " + e.what());
    }
}

std::uint32_t requiredU32(const json& config, const char* name, bool allow_zero = false) {
    const auto value = required<std::int64_t>(config, name);
    if (value < (allow_zero ? 0 : 1) ||
        value > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error(std::string("training.config.") + name + " is out of uint32 range");
    }
    return static_cast<std::uint32_t>(value);
}

std::int32_t requiredI32(const json& config, const char* name) {
    const auto value = required<std::int64_t>(config, name);
    if (value < std::numeric_limits<std::int32_t>::min() || value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(std::string("training.config.") + name + " is out of int32 range");
    }
    return static_cast<std::int32_t>(value);
}

float requiredFinite(const json& config, const char* name) {
    const float value = required<float>(config, name);
    if (!std::isfinite(value)) throw std::runtime_error(std::string("training.config.") + name + " must be finite");
    return value;
}

void requirePositive(float value, const char* name) {
    if (!(value > 0.0f) || !std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
}

void requireBiasParent(bool use_bias, bool child, const char* name) {
    if (child && !use_bias) throw std::runtime_error(std::string(name) + "=true requires use_bias=true");
}

std::vector<float> computeAlibiSlopes(const EffectiveConfig& c) {
    const std::uint32_t d_max = c.max_seq_len;
    const std::uint32_t locality_floor = std::min(c.alibi_min_locality_distance, d_max);
    const std::uint32_t d_min = std::max(locality_floor, std::min(c.rotary_dim / 2, d_max));
    const float target_bias = std::abs(c.alibi_slope_exponent);
    if (d_min == 0 || target_bias == 0.0f) throw std::runtime_error("ALiBi derivation has a zero distance or slope exponent");
    if (c.alibi_max_bias > 0.0f) throw std::runtime_error("alibi_max_bias must be <= 0");
    const float m_max = target_bias / static_cast<float>(d_min);
    const float m_min = target_bias / static_cast<float>(d_max);
    const float cap = c.alibi_max_bias != 0.0f
        ? std::abs(c.alibi_max_bias) / static_cast<float>(c.max_seq_len) : 0.0f;
    std::vector<float> slopes(c.num_heads);
    if (c.num_heads == 1) {
        slopes[0] = cap > 0.0f ? std::min(m_max, cap) : m_max;
        return slopes;
    }
    const float log_max = std::log(m_max);
    const float log_min = std::log(m_min);
    for (std::uint32_t h = 0; h < c.num_heads; ++h) {
        const float t = static_cast<float>(h) / static_cast<float>(c.num_heads - 1);
        float slope = std::exp(log_max + t * (log_min - log_max));
        if (cap > 0.0f) slope = std::min(slope, cap);
        slopes[h] = slope;
    }
    return slopes;
}

std::vector<float> computeRopeInvFreq(const EffectiveConfig& c) {
    float theta = c.rope_theta;
    if (c.max_seq_len > c.rope_base_seq_len && c.rotary_dim > 2) {
        const float ratio = static_cast<float>(c.max_seq_len) / static_cast<float>(c.rope_base_seq_len);
        const float exponent = static_cast<float>(c.rotary_dim) / static_cast<float>(c.rotary_dim - 2);
        theta *= std::pow(ratio, exponent);
    }
    requirePositive(theta, "effective RoPE theta");
    std::vector<float> frequencies(c.rotary_dim / 2);
    for (std::uint32_t i = 0; i < c.rotary_dim / 2; ++i) {
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(c.rotary_dim);
        frequencies[i] = c.rope_scaling / std::pow(theta, exponent);
    }
    return frequencies;
}

EffectiveConfig compileEffectiveConfig(const json& document, const VocabFacts& vocab) {
    const json* config_ptr = nullptr;
    try {
        config_ptr = &document.at("training").at("config");
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("input must contain object training.config: ") + e.what());
    }
    const json& j = *config_ptr;
    if (!j.is_object()) throw std::runtime_error("training.config must be an object");
    EffectiveConfig c;
    c.d_model = requiredU32(j, "d_model");
    c.num_layers = requiredU32(j, "num_layers");
    c.num_heads = requiredU32(j, "num_heads");
    c.num_kv_heads = requiredU32(j, "num_kv_heads");
    c.max_seq_len = requiredU32(j, "max_seq_len");
    if (c.d_model % c.num_heads != 0) throw std::runtime_error("d_model must be divisible by num_heads");
    if (c.num_heads % c.num_kv_heads != 0 || c.num_kv_heads > c.num_heads) {
        throw std::runtime_error("num_kv_heads must divide num_heads and cannot exceed it");
    }
    if (c.d_model > std::numeric_limits<std::uint32_t>::max() / kFfnMultiplier) {
        throw std::runtime_error("derived d_ff overflows uint32");
    }
    c.d_ff = c.d_model * kFfnMultiplier;
    c.vocab_size = vocab.token_space_size;
    c.tie_embeddings = required<bool>(j, "tie_embeddings");
    c.embedding_scale = requiredFinite(j, "embedding_scale");
    requirePositive(c.embedding_scale, "embedding_scale");

    c.head_dim = c.d_model / c.num_heads;
    if ((c.head_dim & 1u) != 0) throw std::runtime_error("derived head_dim/rotary_dim must be even");
    c.heads_per_kv_group = c.num_heads / c.num_kv_heads;
    if (c.num_kv_heads > std::numeric_limits<std::uint32_t>::max() / c.head_dim) {
        throw std::runtime_error("derived kv_dim overflows uint32");
    }
    c.kv_dim = c.num_kv_heads * c.head_dim;
    if (c.kv_dim > (std::numeric_limits<std::uint32_t>::max() - c.d_model) / 2u) {
        throw std::runtime_error("derived qkv_dim overflows uint32");
    }
    c.qkv_dim = c.d_model + 2u * c.kv_dim;
    c.rotary_dim = c.head_dim;
    c.is_gqa = c.num_kv_heads < c.num_heads;
    c.attention_softmax_scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));

    const bool use_bias = required<bool>(j, "use_bias");
    c.attention_qkv_bias = required<bool>(j, "attention_qkv_bias_enabled");
    c.attention_output_bias = required<bool>(j, "attention_output_bias_enabled");
    c.ffn_output_bias = required<bool>(j, "ffn_output_bias_enabled");
    c.lm_head_bias = required<bool>(j, "lm_head_bias_enabled");
    requireBiasParent(use_bias, c.attention_qkv_bias, "attention_qkv_bias_enabled");
    requireBiasParent(use_bias, c.attention_output_bias, "attention_output_bias_enabled");
    requireBiasParent(use_bias, c.ffn_output_bias, "ffn_output_bias_enabled");
    requireBiasParent(use_bias, c.lm_head_bias, "lm_head_bias_enabled");

    c.causal_mask = j.value("causal_mask", true);
    c.use_pre_norm = j.value("use_pre_norm", true);
    c.fuse_qkv = j.value("fuse_qkv", true);
    c.qk_norm = required<bool>(j, "qk_norm_enabled");
    c.attention_off_by_one = required<bool>(j, "attention_off_by_one");
    c.attention_residual_gate = required<bool>(j, "attention_residual_gate_enabled");

    c.use_rope = required<bool>(j, "use_rope");
    c.use_alibi = required<bool>(j, "use_alibi");
    if (c.use_rope && c.use_alibi) c.positional_kind = GRIMConfig::PositionalEncodingType_AlibiRope;
    else if (c.use_rope) c.positional_kind = GRIMConfig::PositionalEncodingType_Rope;
    else if (c.use_alibi) c.positional_kind = GRIMConfig::PositionalEncodingType_Alibi;
    else c.positional_kind = GRIMConfig::PositionalEncodingType_None;
    c.rope_base_seq_len = requiredU32(j, "rope_base_seq_len");
    c.alibi_min_locality_distance = requiredU32(j, "alibi_min_locality_distance");
    c.alibi_slope_exponent = requiredFinite(j, "alibi_slope_exponent");
    c.alibi_max_bias = requiredFinite(j, "alibi_max_bias");
    c.rope_theta = requiredFinite(j, "rope_theta");
    c.rope_scaling = requiredFinite(j, "rope_scaling");
    requirePositive(c.rope_theta, "rope_theta");
    requirePositive(c.rope_scaling, "rope_scaling");
    c.alibi_slopes = computeAlibiSlopes(c);
    c.rope_inv_freq = computeRopeInvFreq(c);

    c.rms_epsilon = j.value("rms_epsilon", 1.0e-5f);
    requirePositive(c.rms_epsilon, "rms_epsilon");
    c.use_layer_scale = required<bool>(j, "use_layer_scale");
    c.center_residuals = required<bool>(j, "center_encoder_residuals");

    c.lm_head_unigram_bias = required<bool>(j, "lm_head_unigram_bias");
    if (c.lm_head_unigram_bias && !c.lm_head_bias) {
        throw std::runtime_error("lm_head_unigram_bias=true requires lm_head_bias_enabled=true");
    }
    c.lm_head_center_hidden_states = required<bool>(j, "lm_head_center_hidden_states");
    c.center_logits = required<bool>(j, "center_logits");
    c.project_out_pc1 = required<bool>(j, "project_out_pc1");
    c.pc1_power_iters = requiredU32(j, "pc1_power_iters", true);
    if (c.project_out_pc1 && c.pc1_power_iters == 0) throw std::runtime_error("project_out_pc1 requires pc1_power_iters > 0");
    c.lm_head_mlp_enabled = required<bool>(j, "lm_head_mlp_enabled");
    c.lm_head_mlp_d_ff = requiredU32(j, "lm_head_mlp_d_ff", !c.lm_head_mlp_enabled);
    c.lm_head_mlp_alpha = requiredFinite(j, "lm_head_mlp_alpha");
    if (c.lm_head_mlp_enabled) requirePositive(c.lm_head_mlp_alpha, "lm_head_mlp_alpha");

    c.use_atom_data = required<bool>(j, "use_atom_data");
    c.atom_embedding_dim = requiredU32(j, "atom_embedding_dim", !c.use_atom_data);
    c.execution_block_enabled = required<bool>(j, "execution_block_enabled");
    if (c.execution_block_enabled) {
        if (!c.use_atom_data) throw std::runtime_error("execution_block_enabled requires use_atom_data");
        c.execution_block_layer = requiredI32(j, "execution_block_layer");
        c.execution_block_num_ops = requiredU32(j, "execution_block_num_ops");
        c.execution_block_num_slots = requiredU32(j, "execution_block_num_slots");
        c.execution_block_num_scratch_slots = requiredU32(j, "execution_block_num_scratch_slots", true);
        c.execution_block_num_steps = requiredU32(j, "execution_block_num_steps");
        c.execution_block_value_decode_input_dim = requiredU32(j, "execution_block_value_decode_input_dim");
        c.execution_block_value_decode_hidden_dim = requiredU32(j, "execution_block_value_decode_hidden_dim");
        c.execution_block_d_type = requiredU32(j, "execution_block_d_type");
        c.execution_block_cross_attn_topk = requiredU32(j, "execution_block_cross_attn_topk");
        c.execution_block_usage_decay = requiredFinite(j, "execution_block_usage_decay");
        c.execution_block_inject_gate_temp = requiredFinite(j, "execution_block_inject_gate_temp");
        c.execution_block_result_slot_mode = requiredU32(j, "execution_block_result_slot_mode", true);
        c.execution_block_result_slot_index = requiredI32(j, "execution_block_result_slot_index");
        c.execution_block_magnitude_limit = requiredFinite(j, "execution_block_magnitude_limit");
        c.execution_block_causal_w1_transition = requiredFinite(j, "execution_block_causal_w1_transition");
        c.execution_block_decode_bias = required<bool>(j, "execution_block_decode_bias_enabled");
        c.execution_block_value_embedding_bias = required<bool>(j, "execution_block_value_embedding_bias_enabled");
        c.execution_block_scalar_bias = required<bool>(j, "execution_block_scalar_bias_enabled");
        c.execution_block_trace_bias = required<bool>(j, "execution_block_trace_bias_enabled");
        requireBiasParent(use_bias, c.execution_block_decode_bias, "execution_block_decode_bias_enabled");
        requireBiasParent(use_bias, c.execution_block_value_embedding_bias, "execution_block_value_embedding_bias_enabled");
        requireBiasParent(use_bias, c.execution_block_scalar_bias, "execution_block_scalar_bias_enabled");
        requireBiasParent(use_bias, c.execution_block_trace_bias, "execution_block_trace_bias_enabled");
        if (c.execution_block_layer < -1 || c.execution_block_layer >= static_cast<std::int32_t>(c.num_layers)) {
            throw std::runtime_error("execution_block_layer is outside [-1, num_layers)");
        }
        if (c.execution_block_num_scratch_slots > c.execution_block_num_slots) {
            throw std::runtime_error("execution_block_num_scratch_slots exceeds num_slots");
        }
        if (c.execution_block_result_slot_index < -1 ||
            c.execution_block_result_slot_index >= static_cast<std::int32_t>(c.execution_block_num_slots)) {
            throw std::runtime_error("execution_block_result_slot_index is outside the slot range");
        }
        if (c.execution_block_cross_attn_topk > c.execution_block_num_slots) {
            throw std::runtime_error("execution_block_cross_attn_topk exceeds num_slots");
        }
        if (c.execution_block_value_decode_input_dim + 16u > c.atom_embedding_dim) {
            throw std::runtime_error("execution block value decode input plus type width exceeds atom_embedding_dim");
        }
        if (!(c.execution_block_usage_decay > 0.0f && c.execution_block_usage_decay <= 1.0f)) {
            throw std::runtime_error("execution_block_usage_decay must be in (0, 1]");
        }
        requirePositive(c.execution_block_inject_gate_temp, "execution_block_inject_gate_temp");
        requirePositive(c.execution_block_magnitude_limit, "execution_block_magnitude_limit");
    }

    c.number_encoder_enabled = required<bool>(j, "number_encoder_enabled");
    if (c.number_encoder_enabled) {
        if (!c.use_atom_data) throw std::runtime_error("number_encoder_enabled requires use_atom_data");
        c.number_encoder_max_digit_slots = requiredU32(j, "number_encoder_max_digit_slots");
        c.number_encoder_d_hidden = requiredU32(j, "number_encoder_d_hidden");
        c.number_encoder_max_abs_pow10 = requiredU32(j, "number_encoder_max_abs_pow10");
        if (c.number_encoder_max_abs_pow10 > 32766u) throw std::runtime_error("number_encoder_max_abs_pow10 exceeds 32766");
        c.number_encoder_contribution_bias = required<bool>(j, "number_encoder_contribution_bias_enabled");
        c.number_encoder_global_bias = required<bool>(j, "number_encoder_global_bias_enabled");
        requireBiasParent(use_bias, c.number_encoder_contribution_bias, "number_encoder_contribution_bias_enabled");
        requireBiasParent(use_bias, c.number_encoder_global_bias, "number_encoder_global_bias_enabled");
    }

    c.arg_selector_enabled = required<bool>(j, "selector_enabled");
    if (c.arg_selector_enabled && !c.use_atom_data) throw std::runtime_error("selector_enabled requires use_atom_data");
    c.slot_seed_encoder_enabled = required<bool>(j, "slot_seed_encoder_enabled");
    if (c.slot_seed_encoder_enabled) {
        if (!c.use_atom_data || !c.execution_block_enabled) {
            throw std::runtime_error("slot_seed_encoder_enabled requires atom data and execution block");
        }
        c.slot_seed_encoder_d_hidden = requiredU32(j, "slot_seed_encoder_d_hidden");
        c.slot_seed_encoder_bias = required<bool>(j, "slot_seed_encoder_bias_enabled");
        c.slot_seed_encoder_type_embedding = required<bool>(j, "slot_seed_encoder_type_embedding_enabled");
        requireBiasParent(use_bias, c.slot_seed_encoder_bias, "slot_seed_encoder_bias_enabled");
    }

    c.tokenizer_model_type = required<std::string>(j, "tokenizer_model_type");
    c.tokenizer_special_tokens = required<std::vector<std::string>>(j, "tokenizer_special_tokens");
    c.tokenizer_add_bos = required<bool>(j, "tokenizer_add_bos");
    c.tokenizer_add_eos = required<bool>(j, "tokenizer_add_eos");
    c.tokenizer_unk_token = required<std::string>(j, "tokenizer_unk_token");
    c.tokenizer_pad_token = required<std::string>(j, "tokenizer_pad_token");
    c.tokenizer_bos_token = required<std::string>(j, "tokenizer_bos_token");
    c.tokenizer_eos_token = required<std::string>(j, "tokenizer_eos_token");
    c.tokenizer_enable_nfkc = required<bool>(j, "tokenizer_enable_nfkc_normalization");
    c.tokenizer_enable_lowercasing = required<bool>(j, "tokenizer_enable_lowercasing");
    c.tokenizer_enable_byte_fallback = required<bool>(j, "tokenizer_enable_byte_fallback");
    c.tokenizer_enable_atom_reasoning = required<bool>(j, "tokenizer_enable_atom_reasoning");
    c.tokenizer_detect_numbers = required<bool>(j, "tokenizer_detect_numbers");
    if (c.tokenizer_model_type.empty()) throw std::runtime_error("tokenizer_model_type must not be empty");
    if (c.tokenizer_unk_token != "<unk>" || c.tokenizer_pad_token != "<pad>" ||
        c.tokenizer_bos_token != "<s>" || c.tokenizer_eos_token != "</s>") {
        throw std::runtime_error("configured special-token strings do not match KTMG v4 token layout");
    }
    std::set<std::string> specials(c.tokenizer_special_tokens.begin(), c.tokenizer_special_tokens.end());
    if (specials.size() != c.tokenizer_special_tokens.size() || specials.size() != 4 ||
        specials.count("<unk>") == 0 || specials.count("<pad>") == 0 ||
        specials.count("<s>") == 0 || specials.count("</s>") == 0) {
        throw std::runtime_error("tokenizer_special_tokens must contain each KTMG v4 special token exactly once");
    }
    if (c.tokenizer_enable_atom_reasoning && !c.use_atom_data) {
        throw std::runtime_error("tokenizer_enable_atom_reasoning requires use_atom_data");
    }
    if (c.tokenizer_detect_numbers && !c.tokenizer_enable_atom_reasoning) {
        throw std::runtime_error("tokenizer_detect_numbers requires tokenizer_enable_atom_reasoning");
    }

    auto addCapability = [&](bool enabled, GRIMConfig::ModelCapability capability) {
        if (enabled) c.capabilities.push_back(capability);
    };
    addCapability(c.use_alibi, GRIMConfig::ModelCapability_Alibi);
    addCapability(c.use_rope, GRIMConfig::ModelCapability_Rope);
    addCapability(c.is_gqa, GRIMConfig::ModelCapability_GroupedQueryAttention);
    addCapability(c.use_layer_scale, GRIMConfig::ModelCapability_LayerScale);
    addCapability(c.qk_norm, GRIMConfig::ModelCapability_QkNorm);
    addCapability(c.attention_off_by_one, GRIMConfig::ModelCapability_AttentionOffByOne);
    addCapability(c.attention_residual_gate, GRIMConfig::ModelCapability_AttentionResidualGate);
    addCapability(c.execution_block_enabled, GRIMConfig::ModelCapability_ExecutionBlock);
    addCapability(c.number_encoder_enabled, GRIMConfig::ModelCapability_NumberEncoder);
    addCapability(c.arg_selector_enabled, GRIMConfig::ModelCapability_ArgSelector);
    addCapability(c.slot_seed_encoder_enabled, GRIMConfig::ModelCapability_SlotSeedEncoder);
    addCapability(c.lm_head_mlp_enabled, GRIMConfig::ModelCapability_LmHeadMlp);
    addCapability(c.use_atom_data, GRIMConfig::ModelCapability_AtomData);
    std::sort(c.capabilities.begin(), c.capabilities.end(), [](auto a, auto b) {
        return static_cast<std::uint16_t>(a) < static_cast<std::uint16_t>(b);
    });
    return c;
}

std::uint64_t capabilityHash(const std::vector<GRIMConfig::ModelCapability>& capabilities) {
    std::vector<std::uint8_t> canonical;
    canonical.reserve(capabilities.size() * 2);
    for (const auto capability : capabilities) {
        const auto value = static_cast<std::uint16_t>(capability);
        canonical.push_back(static_cast<std::uint8_t>(value));
        canonical.push_back(static_cast<std::uint8_t>(value >> 8));
    }
    return xxhash64(canonical.data(), canonical.size());
}

std::vector<std::uint8_t> buildArtifact(
    const EffectiveConfig& c,
    const VocabFacts& vocab,
    const std::array<std::uint8_t, 32>& semantic_hash,
    std::uint64_t model_hash,
    std::uint64_t capability_hash) {
    flatbuffers::FlatBufferBuilder builder(4096);
    builder.ForceDefaults(true);
    const auto semantic_vector = builder.CreateVector(semantic_hash.data(), semantic_hash.size());
    const auto integrity = GRIMConfig::CreateConfigIntegrity(
        builder, semantic_vector, model_hash, capability_hash);

    std::vector<std::uint16_t> capability_values;
    capability_values.reserve(c.capabilities.size());
    for (const auto value : c.capabilities) capability_values.push_back(static_cast<std::uint16_t>(value));
    const auto capabilities = builder.CreateVector(capability_values);

    const auto architecture = GRIMConfig::CreateArchitectureConfig(
        builder, c.d_model, c.num_layers, c.num_heads, c.num_kv_heads, c.d_ff,
        c.max_seq_len, c.vocab_size, c.tie_embeddings, c.embedding_scale);
    const auto alibi = builder.CreateVector(c.alibi_slopes);
    const auto rope = builder.CreateVector(c.rope_inv_freq);
    const auto derived = GRIMConfig::CreateDerivedArchitecture(
        builder, c.head_dim, c.heads_per_kv_group, c.kv_dim, c.qkv_dim,
        c.rotary_dim, c.is_gqa, c.attention_softmax_scale, alibi, rope);

    const auto bias = GRIMConfig::CreateBiasPolicy(
        builder, c.attention_qkv_bias, c.attention_output_bias, c.ffn_output_bias, c.lm_head_bias);
    const auto attention = GRIMConfig::CreateAttentionConfig(
        builder, c.causal_mask, c.use_pre_norm, c.fuse_qkv, c.qk_norm,
        c.attention_off_by_one, c.attention_residual_gate);
    const auto positional = GRIMConfig::CreatePositionalEncodingConfig(
        builder, c.positional_kind, c.rope_base_seq_len, c.alibi_min_locality_distance,
        c.alibi_slope_exponent, c.alibi_max_bias, c.rope_theta, c.rope_scaling);
    const auto encoder = GRIMConfig::CreateEncoderConfig(
        builder, c.rms_epsilon, c.use_layer_scale, c.center_residuals);
    const auto lm_head = GRIMConfig::CreateLmHeadConfig(
        builder, c.lm_head_unigram_bias, c.lm_head_center_hidden_states,
        c.center_logits, c.project_out_pc1, c.pc1_power_iters,
        c.lm_head_mlp_enabled, c.lm_head_mlp_d_ff, c.lm_head_mlp_alpha);

    flatbuffers::Offset<GRIMConfig::ExecutionBlockConfig> execution;
    if (c.execution_block_enabled) {
        execution = GRIMConfig::CreateExecutionBlockConfig(
            builder, c.execution_block_layer, c.execution_block_num_ops,
            c.execution_block_num_slots, c.execution_block_num_scratch_slots,
            c.execution_block_num_steps, c.execution_block_value_decode_input_dim,
            c.execution_block_value_decode_hidden_dim, c.head_dim,
            c.execution_block_d_type, c.head_dim, c.execution_block_cross_attn_topk,
            c.execution_block_usage_decay, c.execution_block_inject_gate_temp,
            c.execution_block_result_slot_mode, c.execution_block_result_slot_index,
            c.execution_block_magnitude_limit, c.execution_block_causal_w1_transition,
            c.execution_block_decode_bias, c.execution_block_value_embedding_bias,
            c.execution_block_scalar_bias, c.execution_block_trace_bias);
    }
    flatbuffers::Offset<GRIMConfig::NumberEncoderConfig> number_encoder;
    if (c.number_encoder_enabled) {
        number_encoder = GRIMConfig::CreateNumberEncoderConfig(
            builder, c.number_encoder_max_digit_slots, c.number_encoder_d_hidden,
            c.number_encoder_max_abs_pow10, c.number_encoder_max_abs_pow10 * 2u + 1u,
            c.number_encoder_contribution_bias, c.number_encoder_global_bias);
    }
    flatbuffers::Offset<GRIMConfig::SlotSeedEncoderConfig> slot_seed;
    if (c.slot_seed_encoder_enabled) {
        slot_seed = GRIMConfig::CreateSlotSeedEncoderConfig(
            builder, c.slot_seed_encoder_d_hidden, c.slot_seed_encoder_bias,
            c.slot_seed_encoder_type_embedding);
    }
    const auto features = GRIMConfig::CreateModelFeatures(
        builder, c.use_atom_data, c.atom_embedding_dim, bias, attention, positional,
        encoder, lm_head, execution, number_encoder, c.arg_selector_enabled, slot_seed);

    const auto model_type = builder.CreateString(c.tokenizer_model_type);
    std::vector<flatbuffers::Offset<flatbuffers::String>> special_strings;
    special_strings.reserve(c.tokenizer_special_tokens.size());
    for (const auto& value : c.tokenizer_special_tokens) special_strings.push_back(builder.CreateString(value));
    const auto special_tokens = builder.CreateVector(special_strings);
    const auto vocab_hash = builder.CreateVector(vocab.sha256.data(), vocab.sha256.size());
    const auto tokenizer = GRIMConfig::CreateTokenizerConfig(
        builder, model_type, vocab_hash, special_tokens, c.tokenizer_add_bos,
        c.tokenizer_add_eos, builder.CreateString(c.tokenizer_unk_token),
        builder.CreateString(c.tokenizer_pad_token), builder.CreateString(c.tokenizer_bos_token),
        builder.CreateString(c.tokenizer_eos_token), 0, 1, 2, 3,
        c.tokenizer_enable_nfkc, c.tokenizer_enable_lowercasing,
        c.tokenizer_enable_byte_fallback, c.tokenizer_enable_atom_reasoning,
        c.tokenizer_detect_numbers);

    const auto root = GRIMConfig::CreateCompiledModelHyperparameters(
        builder, kSchemaVersion, kSemanticVersion, integrity, capabilities,
        architecture, derived, features, tokenizer);
    GRIMConfig::FinishCompiledModelHyperparametersBuffer(builder, root);
    return std::vector<std::uint8_t>(builder.GetBufferPointer(),
                                     builder.GetBufferPointer() + builder.GetSize());
}

void verifyFlatBuffer(const std::vector<std::uint8_t>& bytes) {
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!GRIMConfig::VerifyCompiledModelHyperparametersBuffer(verifier)) {
        throw std::runtime_error("compiler produced an invalid FlatBuffer");
    }
    if (!GRIMConfig::CompiledModelHyperparametersBufferHasIdentifier(bytes.data())) {
        throw std::runtime_error("compiler produced an artifact without GCFG identifier");
    }
}

std::vector<std::uint8_t> normalizedArtifact(std::vector<std::uint8_t> bytes) {
    auto* root = flatbuffers::GetMutableRoot<GRIMConfig::CompiledModelHyperparameters>(bytes.data());
    if (!root || !root->integrity() || !root->integrity()->semantic_sha256() ||
        root->integrity()->semantic_sha256()->size() != 32) {
        throw std::runtime_error("artifact integrity table is malformed");
    }
    auto* integrity = root->mutable_integrity();
    auto* digest = integrity->mutable_semantic_sha256();
    for (flatbuffers::uoffset_t i = 0; i < digest->size(); ++i) digest->Mutate(i, 0);
    if (!integrity->mutate_model_compatibility_xxhash64(0) ||
        !integrity->mutate_capability_xxhash64(0)) {
        throw std::runtime_error("artifact integrity scalars were not serialized");
    }
    return bytes;
}

std::string hex(const std::uint8_t* data, std::size_t size) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < size; ++i) out << std::setw(2) << static_cast<unsigned>(data[i]);
    return out.str();
}

void writeArtifact(const fs::path& output, const std::vector<std::uint8_t>& bytes) {
    if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
    fs::path temporary = output;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("cannot create output: " + temporary.string());
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        stream.flush();
        if (!stream) throw std::runtime_error("failed writing output: " + temporary.string());
    }
    std::error_code ec;
    fs::rename(temporary, output, ec);
    if (ec) {
        fs::remove(output, ec);
        ec.clear();
        fs::rename(temporary, output, ec);
    }
    if (ec) throw std::runtime_error("cannot install output artifact: " + ec.message());
}

Cli parseCli(int argc, char** argv) {
    Cli cli;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> fs::path {
            if (++i >= argc) throw std::runtime_error(arg + " requires a path");
            return fs::path(argv[i]);
        };
        if (arg == "--input") cli.input = value();
        else if (arg == "--vocab") cli.vocab = value();
        else if (arg == "--output") cli.output = value();
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: compile_model_config --input ai_config.json --vocab vocab.bin --output model.grimcfg\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (cli.input.empty() || cli.vocab.empty() || cli.output.empty()) {
        throw std::runtime_error("--input, --vocab, and --output are required");
    }
    return cli;
}

json readJson(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open JSON input: " + path.string());
    json document;
    try {
        stream >> document;
    } catch (const std::exception& e) {
        throw std::runtime_error("cannot parse JSON input " + path.string() + ": " + e.what());
    }
    return document;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Cli cli = parseCli(argc, argv);
        const json source = readJson(cli.input);
        const VocabFacts vocab = inspectVocab(cli.vocab);
        const EffectiveConfig config = compileEffectiveConfig(source, vocab);
        const std::uint64_t capability_hash = capabilityHash(config.capabilities);

        const std::array<std::uint8_t, 32> zero_hash{};
        const auto provisional = buildArtifact(config, vocab, zero_hash, 0, 0);
        const auto normalized = normalizedArtifact(provisional);
        const auto semantic_hash = sha256(normalized.data(), normalized.size());
        const std::uint64_t model_hash = xxhash64(normalized.data(), normalized.size());
        const auto artifact = buildArtifact(config, vocab, semantic_hash, model_hash, capability_hash);
        verifyFlatBuffer(artifact);

        const auto verification_image = normalizedArtifact(artifact);
        if (sha256(verification_image.data(), verification_image.size()) != semantic_hash ||
            xxhash64(verification_image.data(), verification_image.size()) != model_hash) {
            throw std::runtime_error("artifact integrity self-verification failed");
        }
        writeArtifact(cli.output, artifact);

        std::cout << "compiled " << cli.output.string() << "\n"
                  << "  schema_version: " << kSchemaVersion << "\n"
                  << "  semantic_version: " << kSemanticVersion << "\n"
                  << "  vocab_size: " << config.vocab_size << "\n"
                  << "  vocab_sha256: " << hex(vocab.sha256.data(), vocab.sha256.size()) << "\n"
                  << "  semantic_sha256: " << hex(semantic_hash.data(), semantic_hash.size()) << "\n"
                  << "  model_compatibility_xxhash64: 0x" << std::hex << model_hash << "\n"
                  << "  capability_xxhash64: 0x" << capability_hash << std::dec << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "compile_model_config: " << e.what() << '\n';
        return 1;
    }
}
