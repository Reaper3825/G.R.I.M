#include "CompiledModelConfig.hpp"

#include "grim_compiled_hyperparameters_generated.h"

#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/verifier.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace GRIM::Config {
namespace {

namespace fs = std::filesystem;

constexpr std::uint32_t kSupportedSchemaVersion = 2;
constexpr std::uint32_t kSupportedSemanticVersion = 2;
constexpr std::uintmax_t kMaximumArtifactBytes = 16u * 1024u * 1024u;

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

std::array<std::uint8_t, 32> sha256(const std::vector<std::uint8_t>& bytes) {
    Sha256 hasher;
    hasher.update(bytes.data(), bytes.size());
    return hasher.finish();
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

std::uint64_t rotateLeft64(std::uint64_t value, unsigned count) {
    return (value << count) | (value >> (64u - count));
}

std::uint64_t xxhash64(const std::uint8_t* input, std::size_t length) {
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
    if (length == 0) {
        std::uint64_t h = p5;
        h ^= h >> 33; h *= p2; h ^= h >> 29; h *= p3; h ^= h >> 32;
        return h;
    }
    const std::uint8_t* p = input;
    const std::uint8_t* const end = input + length;
    std::uint64_t h = 0;
    if (length >= 32) {
        std::uint64_t v1 = p1 + p2, v2 = p2, v3 = 0, v4 = 0 - p1;
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
        h = p5;
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

std::string normalizedArtifactName(std::string_view value) {
    std::string normalized;
    bool separator = false;
    for (const unsigned char ch : value) {
        if (std::isalnum(ch)) {
            if (separator && !normalized.empty()) normalized.push_back('_');
            normalized.push_back(static_cast<char>(std::tolower(ch)));
            separator = false;
        } else {
            separator = !normalized.empty();
        }
    }
    return normalized;
}

fs::path resolveAgainstConfig(const fs::path& path, const fs::path& ai_config_path) {
    if (path.is_absolute()) return path.lexically_normal();
    return (ai_config_path.parent_path() / path).lexically_normal();
}

std::vector<fs::path> discoverArtifacts(const fs::path& model_store) {
    std::vector<fs::path> artifacts;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(model_store, ec)) {
        if (ec) throw std::runtime_error("cannot enumerate model store " + model_store.string() + ": " + ec.message());
        if (!entry.is_directory(ec) || ec) {
            ec.clear();
            continue;
        }
        const fs::path candidate = entry.path() / "model.grimcfg";
        if (fs::is_regular_file(candidate, ec) && !ec) artifacts.push_back(candidate.lexically_normal());
        ec.clear();
    }
    std::sort(artifacts.begin(), artifacts.end());
    return artifacts;
}

std::vector<std::uint8_t> readArtifact(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    if (ec) throw std::runtime_error("cannot determine model config size: " + path.string() + ": " + ec.message());
    if (size == 0 || size > kMaximumArtifactBytes) {
        throw std::runtime_error("model config size is outside 1..16 MiB: " + path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open model config: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) throw std::runtime_error("failed reading model config: " + path.string());
    return bytes;
}

template <typename FlatVector>
std::vector<float> copyFloats(const FlatVector* values) {
    if (!values) throw std::runtime_error("required float vector is missing");
    return std::vector<float>(values->begin(), values->end());
}

std::string copyString(const flatbuffers::String* value, const char* name) {
    if (!value) throw std::runtime_error(std::string("required string is missing: ") + name);
    return value->str();
}

void requireFinite(float value, const char* name) {
    if (!std::isfinite(value)) throw std::runtime_error(std::string("non-finite model config field: ") + name);
}

void validateDecoded(const CompiledModelConfigSnapshot& c) {
    const auto& a = c.architecture;
    const auto& d = c.derived_architecture;
    const auto& f = c.features;
    if (a.d_model == 0 || a.num_layers == 0 || a.num_heads == 0 ||
        a.num_kv_heads == 0 || a.max_seq_len == 0) {
        throw std::runtime_error("compiled architecture contains a zero required dimension");
    }
    if (a.d_model % a.num_heads != 0 || a.num_heads % a.num_kv_heads != 0 ||
        a.num_kv_heads > a.num_heads) {
        throw std::runtime_error("compiled architecture has invalid attention divisibility");
    }
    const std::uint32_t expected_head = a.d_model / a.num_heads;
    const std::uint32_t expected_kv = a.num_kv_heads * expected_head;
    if (a.d_ff != a.d_model * 4u || d.head_dim != expected_head ||
        d.heads_per_kv_group != a.num_heads / a.num_kv_heads ||
        d.kv_dim != expected_kv || d.qkv_dim != a.d_model + 2u * expected_kv ||
        d.rotary_dim != expected_head || d.is_gqa != (a.num_kv_heads < a.num_heads)) {
        throw std::runtime_error("compiled derived architecture does not match architecture geometry");
    }
    if (d.pbm_alibi_slopes.size() != a.num_heads ||
        d.pbm_rope_inv_freq.size() != d.rotary_dim / 2u) {
        throw std::runtime_error("compiled PBM table dimensions do not match architecture");
    }
    requireFinite(a.embedding_scale, "architecture.embedding_scale");
    requireFinite(d.attention_softmax_scale, "derived_architecture.attention_softmax_scale");
    requireFinite(f.encoder.rms_epsilon, "features.encoder.rms_epsilon");
    if (a.embedding_scale <= 0.0f || d.attention_softmax_scale <= 0.0f || f.encoder.rms_epsilon <= 0.0f) {
        throw std::runtime_error("compiled model contains a non-positive scale or epsilon");
    }
    if (f.positional_encoding.kind <= CompiledPositionalEncoding::Unspecified ||
        f.positional_encoding.kind > CompiledPositionalEncoding::AlibiRope) {
        throw std::runtime_error("compiled positional encoding is invalid");
    }
    if (f.execution_block) {
        const auto& e = *f.execution_block;
        if (!f.use_atom_data || e.num_ops == 0 || e.num_slots == 0 || e.num_steps == 0 ||
            e.d_key != d.head_dim || e.cross_attention_head_dim != d.head_dim ||
            e.num_scratch_slots > e.num_slots || e.cross_attention_top_k > e.num_slots ||
            e.layer < -1 || e.layer >= static_cast<std::int32_t>(a.num_layers)) {
            throw std::runtime_error("compiled execution-block contract is invalid");
        }
    }
    if (f.number_encoder) {
        const auto& n = *f.number_encoder;
        if (!f.use_atom_data || n.max_digit_slots == 0 || n.d_hidden == 0 ||
            n.pow10_buckets != n.max_abs_pow10 * 2u + 1u) {
            throw std::runtime_error("compiled number-encoder contract is invalid");
        }
    }
    if (f.arg_selector_enabled && !f.use_atom_data) {
        throw std::runtime_error("compiled arg selector requires atom data");
    }
    if (f.slot_seed_encoder && (!f.use_atom_data || !f.execution_block || f.slot_seed_encoder->d_hidden == 0)) {
        throw std::runtime_error("compiled slot-seed encoder contract is invalid");
    }
    if (c.tokenizer.model_type.empty() || c.tokenizer.special_tokens.size() != 4 ||
        c.tokenizer.unk_token_id != 0 || c.tokenizer.pad_token_id != 1 ||
        c.tokenizer.bos_token_id != 2 || c.tokenizer.eos_token_id != 3) {
        throw std::runtime_error("compiled tokenizer contract is invalid");
    }

    std::vector<CompiledModelCapability> expected;
    const auto add = [&](bool enabled, CompiledModelCapability capability) {
        if (enabled) expected.push_back(capability);
    };
    const auto positional = f.positional_encoding.kind;
    add(positional == CompiledPositionalEncoding::Alibi || positional == CompiledPositionalEncoding::AlibiRope,
        CompiledModelCapability::Alibi);
    add(positional == CompiledPositionalEncoding::Rope || positional == CompiledPositionalEncoding::AlibiRope,
        CompiledModelCapability::Rope);
    add(d.is_gqa, CompiledModelCapability::GroupedQueryAttention);
    add(f.encoder.use_layer_scale, CompiledModelCapability::LayerScale);
    add(f.attention.qk_norm_enabled, CompiledModelCapability::QkNorm);
    add(f.attention.off_by_one_enabled, CompiledModelCapability::AttentionOffByOne);
    add(f.attention.residual_gate_enabled, CompiledModelCapability::AttentionResidualGate);
    add(f.execution_block.has_value(), CompiledModelCapability::ExecutionBlock);
    add(f.number_encoder.has_value(), CompiledModelCapability::NumberEncoder);
    add(f.arg_selector_enabled, CompiledModelCapability::ArgSelector);
    add(f.slot_seed_encoder.has_value(), CompiledModelCapability::SlotSeedEncoder);
    add(f.lm_head.mlp_enabled, CompiledModelCapability::LmHeadMlp);
    add(f.use_atom_data, CompiledModelCapability::AtomData);
    if (expected != c.required_capabilities) {
        throw std::runtime_error("compiled required-capability vector does not match model features");
    }
}

} // namespace

std::optional<fs::path> resolveCompiledModelConfigPath(
    const nlohmann::json& document,
    const fs::path& ai_config_path)
{
    const auto& training = document.at("training").at("config");
    std::string selection;
    if (const auto it = training.find("current_model_training");
        it != training.end() && it->is_string()) {
        selection = it->get<std::string>();
    }

    std::string raw_store;
    try {
        raw_store = document.at("paths").at("grim_text").at("model_store").get<std::string>();
    } catch (const std::exception&) {
        raw_store = training.at("grim_text_model_store").get<std::string>();
    }
    if (raw_store.empty()) throw std::runtime_error("configured model store is empty");
    const fs::path model_store = resolveAgainstConfig(raw_store, ai_config_path);
    if (!fs::is_directory(model_store)) {
        throw std::runtime_error("configured model store is not a directory: " + model_store.string());
    }

    const auto artifacts = discoverArtifacts(model_store);
    if (selection.empty()) {
        if (artifacts.empty()) return std::nullopt;
        if (artifacts.size() == 1) return artifacts.front();
        throw std::runtime_error(
            "multiple model.grimcfg artifacts were discovered; set training.config.current_model_training");
    }
    if (fs::path(selection).has_parent_path() || selection == "." || selection == "..") {
        throw std::runtime_error("current_model_training must be a model ID or display name, not a path");
    }

    const fs::path exact = model_store / selection / "model.grimcfg";
    if (fs::is_regular_file(exact)) return exact.lexically_normal();

    const std::string normalized_selection = normalizedArtifactName(selection);
    std::vector<fs::path> matches;
    for (const auto& artifact : artifacts) {
        if (normalizedArtifactName(artifact.parent_path().filename().string()) == normalized_selection) {
            matches.push_back(artifact);
        }
    }
    if (matches.size() == 1) return matches.front();
    if (matches.size() > 1) {
        throw std::runtime_error("current_model_training matches multiple model directories: " + selection);
    }
    throw std::runtime_error(
        "model config not found for current_model_training='" + selection +
        "' under " + model_store.string());
}

CompiledModelConfigSnapshot loadCompiledModelConfig(const fs::path& artifact_path) {
    std::vector<std::uint8_t> bytes = readArtifact(artifact_path);
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    if (!GRIMConfig::VerifyCompiledModelHyperparametersBuffer(verifier) ||
        !GRIMConfig::CompiledModelHyperparametersBufferHasIdentifier(bytes.data())) {
        throw std::runtime_error("invalid GCFG FlatBuffer: " + artifact_path.string());
    }

    const auto* root = GRIMConfig::GetCompiledModelHyperparameters(bytes.data());
    if (!root || root->schema_version() != kSupportedSchemaVersion ||
        root->semantic_version() != kSupportedSemanticVersion) {
        throw std::runtime_error("unsupported model config schema or semantic version: " + artifact_path.string());
    }
    const auto* integrity = root->integrity();
    if (!integrity || !integrity->semantic_sha256() || integrity->semantic_sha256()->size() != 32) {
        throw std::runtime_error("model config integrity table is malformed: " + artifact_path.string());
    }
    std::array<std::uint8_t, 32> stored_sha{};
    std::copy(integrity->semantic_sha256()->begin(), integrity->semantic_sha256()->end(), stored_sha.begin());
    const std::uint64_t stored_model_hash = integrity->model_compatibility_xxhash64();
    const std::uint64_t stored_capability_hash = integrity->capability_xxhash64();

    std::vector<std::uint8_t> normalized = bytes;
    auto* mutable_root = flatbuffers::GetMutableRoot<GRIMConfig::CompiledModelHyperparameters>(normalized.data());
    auto* mutable_integrity = mutable_root ? mutable_root->mutable_integrity() : nullptr;
    auto* mutable_digest = mutable_integrity ? mutable_integrity->mutable_semantic_sha256() : nullptr;
    if (!mutable_digest || mutable_digest->size() != 32) {
        throw std::runtime_error("model config integrity fields are not mutable/serialized");
    }
    for (flatbuffers::uoffset_t i = 0; i < mutable_digest->size(); ++i) mutable_digest->Mutate(i, 0);
    if (!mutable_integrity->mutate_model_compatibility_xxhash64(0) ||
        !mutable_integrity->mutate_capability_xxhash64(0)) {
        throw std::runtime_error("model config integrity scalars are not serialized");
    }
    if (sha256(normalized) != stored_sha ||
        xxhash64(normalized.data(), normalized.size()) != stored_model_hash) {
        throw std::runtime_error("model config semantic integrity verification failed: " + artifact_path.string());
    }

    CompiledModelConfigSnapshot result;
    result.source_path = fs::absolute(artifact_path).lexically_normal();
    result.schema_version = root->schema_version();
    result.semantic_version = root->semantic_version();
    result.integrity.semantic_sha256 = stored_sha;
    result.integrity.model_compatibility_xxhash64 = stored_model_hash;
    result.integrity.capability_xxhash64 = stored_capability_hash;

    const auto* capabilities = root->required_capabilities();
    if (!capabilities) throw std::runtime_error("model config required_capabilities is missing");
    std::vector<std::uint8_t> capability_bytes;
    capability_bytes.reserve(capabilities->size() * 2u);
    for (const auto raw : *capabilities) {
        if (raw == 0 || raw > static_cast<std::uint16_t>(CompiledModelCapability::AtomData)) {
            throw std::runtime_error("model config contains an unknown required capability");
        }
        result.required_capabilities.push_back(static_cast<CompiledModelCapability>(raw));
        capability_bytes.push_back(static_cast<std::uint8_t>(raw));
        capability_bytes.push_back(static_cast<std::uint8_t>(raw >> 8));
    }
    if (!std::is_sorted(result.required_capabilities.begin(), result.required_capabilities.end()) ||
        std::adjacent_find(result.required_capabilities.begin(), result.required_capabilities.end()) !=
            result.required_capabilities.end() ||
        xxhash64(capability_bytes.data(), capability_bytes.size()) != stored_capability_hash) {
        throw std::runtime_error("model config capability integrity verification failed");
    }

    const auto* a = root->architecture();
    const auto* d = root->derived_architecture();
    const auto* f = root->features();
    const auto* t = root->tokenizer();
    if (!a || !d || !f || !t || !f->bias() || !f->attention() ||
        !f->positional_encoding() || !f->encoder() || !f->lm_head()) {
        throw std::runtime_error("model config is missing a required table");
    }

    result.architecture = {a->d_model(), a->num_layers(), a->num_heads(), a->num_kv_heads(),
        a->d_ff(), a->max_seq_len(), a->tie_embeddings(), a->embedding_scale()};
    result.derived_architecture = {d->head_dim(), d->heads_per_kv_group(), d->kv_dim(),
        d->qkv_dim(), d->rotary_dim(), d->is_gqa(), d->attention_softmax_scale(),
        copyFloats(d->pbm_alibi_slopes()), copyFloats(d->pbm_rope_inv_freq())};

    result.features.use_atom_data = f->use_atom_data();
    result.features.atom_embedding_dim = f->atom_embedding_dim();
    result.features.bias = {f->bias()->attention_qkv(), f->bias()->attention_output(),
        f->bias()->ffn_output(), f->bias()->lm_head()};
    result.features.attention = {f->attention()->causal_mask(), f->attention()->use_pre_norm(),
        f->attention()->fuse_qkv(), f->attention()->qk_norm_enabled(),
        f->attention()->off_by_one_enabled(), f->attention()->residual_gate_enabled()};
    result.features.positional_encoding = {
        static_cast<CompiledPositionalEncoding>(f->positional_encoding()->kind()),
        f->positional_encoding()->rope_base_seq_len(),
        f->positional_encoding()->alibi_min_locality_distance(),
        f->positional_encoding()->alibi_slope_exponent(),
        f->positional_encoding()->alibi_max_bias(), f->positional_encoding()->rope_theta(),
        f->positional_encoding()->rope_scaling()};
    result.features.encoder = {f->encoder()->rms_epsilon(), f->encoder()->use_layer_scale(),
        f->encoder()->center_residuals()};
    result.features.lm_head = {f->lm_head()->unigram_bias_enabled(),
        f->lm_head()->center_hidden_states(), f->lm_head()->center_logits(),
        f->lm_head()->project_out_pc1(), f->lm_head()->pc1_power_iters(),
        f->lm_head()->mlp_enabled(), f->lm_head()->mlp_d_ff(), f->lm_head()->mlp_alpha()};
    if (const auto* e = f->execution_block()) {
        result.features.execution_block = CompiledExecutionBlockConfig{
            e->layer(), e->num_ops(), e->num_slots(), e->num_scratch_slots(), e->num_steps(),
            e->value_decode_input_dim(), e->value_decode_hidden_dim(), e->d_key(), e->d_type(),
            e->cross_attention_head_dim(), e->cross_attention_top_k(), e->usage_decay(),
            e->inject_gate_temperature(), e->result_slot_mode(), e->result_slot_index(),
            e->magnitude_limit(), e->causal_w1_transition(), e->decode_bias_enabled(),
            e->value_embedding_bias_enabled(), e->scalar_bias_enabled(), e->trace_bias_enabled()};
    }
    if (const auto* n = f->number_encoder()) {
        result.features.number_encoder = CompiledNumberEncoderConfig{
            n->max_digit_slots(), n->d_hidden(), n->max_abs_pow10(), n->pow10_buckets(),
            n->contribution_bias_enabled(), n->global_bias_enabled()};
    }
    result.features.arg_selector_enabled = f->arg_selector_enabled();
    if (const auto* s = f->slot_seed_encoder()) {
        result.features.slot_seed_encoder = CompiledSlotSeedEncoderConfig{
            s->d_hidden(), s->bias_enabled(), s->type_embedding_enabled()};
    }

    result.tokenizer.model_type = copyString(t->model_type(), "tokenizer.model_type");
    if (!t->special_tokens()) throw std::runtime_error("model config tokenizer special_tokens is missing");
    for (const auto* token : *t->special_tokens()) {
        result.tokenizer.special_tokens.push_back(copyString(token, "tokenizer.special_tokens[]"));
    }
    result.tokenizer.add_bos = t->add_bos();
    result.tokenizer.add_eos = t->add_eos();
    result.tokenizer.unk_token = copyString(t->unk_token(), "tokenizer.unk_token");
    result.tokenizer.pad_token = copyString(t->pad_token(), "tokenizer.pad_token");
    result.tokenizer.bos_token = copyString(t->bos_token(), "tokenizer.bos_token");
    result.tokenizer.eos_token = copyString(t->eos_token(), "tokenizer.eos_token");
    result.tokenizer.unk_token_id = t->unk_token_id();
    result.tokenizer.pad_token_id = t->pad_token_id();
    result.tokenizer.bos_token_id = t->bos_token_id();
    result.tokenizer.eos_token_id = t->eos_token_id();
    result.tokenizer.enable_nfkc_normalization = t->enable_nfkc_normalization();
    result.tokenizer.enable_lowercasing = t->enable_lowercasing();
    result.tokenizer.enable_byte_fallback = t->enable_byte_fallback();
    result.tokenizer.enable_atom_reasoning = t->enable_atom_reasoning();
    result.tokenizer.detect_numbers = t->detect_numbers();

    validateDecoded(result);
    return result;
}

} // namespace GRIM::Config
