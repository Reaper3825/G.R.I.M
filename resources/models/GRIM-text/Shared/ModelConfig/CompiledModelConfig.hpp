#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace GRIM::Config {

enum class CompiledPositionalEncoding : std::uint8_t {
    Unspecified = 0,
    None = 1,
    Alibi = 2,
    Rope = 3,
    AlibiRope = 4,
};

enum class CompiledModelCapability : std::uint16_t {
    Unknown = 0,
    Alibi = 1,
    Rope = 2,
    GroupedQueryAttention = 3,
    LayerScale = 4,
    QkNorm = 5,
    AttentionOffByOne = 6,
    AttentionResidualGate = 7,
    ExecutionBlock = 8,
    NumberEncoder = 9,
    ArgSelector = 10,
    SlotSeedEncoder = 11,
    LmHeadMlp = 12,
    AtomData = 13,
};

struct CompiledConfigIntegrity {
    std::array<std::uint8_t, 32> semantic_sha256{};
    std::uint64_t model_compatibility_xxhash64 = 0;
    std::uint64_t capability_xxhash64 = 0;
};

struct CompiledArchitectureConfig {
    std::uint32_t d_model = 0;
    std::uint32_t num_layers = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t num_kv_heads = 0;
    std::uint32_t d_ff = 0;
    std::uint32_t max_seq_len = 0;
    bool tie_embeddings = false;
    float embedding_scale = 0.0f;
};

struct CompiledDerivedArchitecture {
    std::uint32_t head_dim = 0;
    std::uint32_t heads_per_kv_group = 0;
    std::uint32_t kv_dim = 0;
    std::uint32_t qkv_dim = 0;
    std::uint32_t rotary_dim = 0;
    bool is_gqa = false;
    float attention_softmax_scale = 0.0f;
    float residual_projection_init_gain = 0.0f;
    std::vector<float> pbm_alibi_slopes;
    std::vector<float> pbm_rope_inv_freq;
};

struct CompiledBiasPolicy {
    bool use_bias = false;
    bool attention_qkv = false;
    bool attention_output = false;
    bool ffn_output = false;
    bool lm_head = false;
};

struct CompiledAttentionConfig {
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool qk_norm_enabled = false;
    bool off_by_one_enabled = false;
    bool residual_gate_enabled = false;
};

struct CompiledPositionalEncodingConfig {
    CompiledPositionalEncoding kind = CompiledPositionalEncoding::Unspecified;
    std::uint32_t rope_base_seq_len = 0;
    std::uint32_t alibi_min_locality_distance = 0;
    float alibi_slope_exponent = 0.0f;
    float alibi_max_bias = 0.0f;
    float rope_theta = 0.0f;
    float rope_scaling = 0.0f;
};

struct CompiledEncoderConfig {
    float rms_epsilon = 0.0f;
    bool use_layer_scale = false;
    float layer_scale_init = 0.0f;
    bool center_residuals = false;
};
  
struct CompiledLmHeadConfig {
    bool unigram_bias_enabled = false;
    bool center_hidden_states = false;
    bool center_logits = false;
    bool project_out_pc1 = false;
    std::uint32_t pc1_power_iters = 0;
    bool mlp_enabled = false;
    std::uint32_t mlp_d_ff = 0;
    float mlp_alpha = 0.0f;
};

struct CompiledExecutionBlockConfig {
    std::int32_t layer = -1;
    std::uint32_t num_ops = 0;
    std::uint32_t num_slots = 0;
    std::uint32_t num_scratch_slots = 0;
    std::uint32_t num_steps = 0;
    std::uint32_t value_decode_input_dim = 0;
    std::uint32_t value_decode_hidden_dim = 0;
    std::uint32_t d_key = 0;
    std::uint32_t d_type = 0;
    std::uint32_t cross_attention_head_dim = 0;
    std::uint32_t cross_attention_top_k = 0;
    float usage_decay = 0.0f;
    float inject_gate_temperature = 0.0f;
    std::uint32_t result_slot_mode = 0;
    std::int32_t result_slot_index = -1;
    float magnitude_limit = 0.0f;
    float causal_w1_transition = 0.0f;
    bool decode_bias_enabled = false;
    bool value_embedding_bias_enabled = false;
    bool scalar_bias_enabled = false;
    bool trace_bias_enabled = false;
};

struct CompiledNumberEncoderConfig {
    std::uint32_t max_digit_slots = 0;
    std::uint32_t d_hidden = 0;
    std::uint32_t max_abs_pow10 = 0;
    std::uint32_t pow10_buckets = 0;
    bool contribution_bias_enabled = false;
    bool global_bias_enabled = false;
};

struct CompiledSlotSeedEncoderConfig {
    std::uint32_t d_hidden = 0;
    bool bias_enabled = false;
    bool type_embedding_enabled = false;
};

struct CompiledModelFeatures {
    bool use_atom_data = false;
    std::uint32_t atom_embedding_dim = 0;
    CompiledBiasPolicy bias;
    CompiledAttentionConfig attention;
    CompiledPositionalEncodingConfig positional_encoding;
    CompiledEncoderConfig encoder;
    CompiledLmHeadConfig lm_head;
    std::optional<CompiledExecutionBlockConfig> execution_block;
    std::optional<CompiledNumberEncoderConfig> number_encoder;
    bool arg_selector_enabled = false;
    std::optional<CompiledSlotSeedEncoderConfig> slot_seed_encoder;
};

struct CompiledTokenizerConfig {
    std::string model_type;
    std::vector<std::string> special_tokens;
    bool add_bos = false;
    bool add_eos = false;
    std::string unk_token;
    std::string pad_token;
    std::string bos_token;
    std::string eos_token;
    std::int32_t unk_token_id = -1;
    std::int32_t pad_token_id = -1;
    std::int32_t bos_token_id = -1;
    std::int32_t eos_token_id = -1;
    bool enable_nfkc_normalization = false;
    bool enable_lowercasing = false;
    bool enable_byte_fallback = false;
    bool enable_atom_reasoning = false;
    bool detect_numbers = false;
};

struct CompiledModelConfigSnapshot {
    std::filesystem::path source_path;
    std::uint32_t schema_version = 0;
    std::uint32_t semantic_version = 0;
    CompiledConfigIntegrity integrity;
    std::vector<CompiledModelCapability> required_capabilities;
    CompiledArchitectureConfig architecture;
    CompiledDerivedArchitecture derived_architecture;
    CompiledModelFeatures features;
    CompiledTokenizerConfig tokenizer;
};

// Resolve model.grimcfg from the configured model store. An explicit model
// selection is strict. With no selection, discovery succeeds only when exactly
// one immediate model directory contains an artifact.
std::optional<std::filesystem::path> resolveCompiledModelConfigPath(
    const nlohmann::json& document,
    const std::filesystem::path& ai_config_path);

// Read, structurally verify, integrity-check, validate, and decode one GCFG.
CompiledModelConfigSnapshot loadCompiledModelConfig(
    const std::filesystem::path& artifact_path);

} // namespace GRIM::Config
