#pragma once
//======================================================//
// HyperparameterRegistry — exposes only the authored fields that feed the
// grim_compiled_hyperparameters.fbs model artifact. Runtime training policy
// (optimizer, batching, logging, telemetry, and similar settings) is excluded.
//
// This file intentionally does NOT include GRIM-text
// headers. GRIM.exe owns runtime ai_config.json access
// through resources.hpp / settings/runtime_ai_config.*.
// GRIM-text remains responsible for converting authored
// JSON leaves into typed training config inside its own
// standalone build.
//======================================================//

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GRIM {
namespace Config {

enum class HyperparamType : uint8_t {
    Bool   = 0,
    Int    = 1,
    Int64  = 2,
    Float  = 3,
    String = 4,
    SizeT  = 5,
    Json   = 6
};

struct HyperparamEntry {
    std::string key;
    std::string display_name;
    std::string category;
    HyperparamType type = HyperparamType::Json;
    nlohmann::json value;

    std::string valueAsString() const {
        std::ostringstream oss;
        switch (type) {
            case HyperparamType::Bool:
                return value.get<bool>() ? "true" : "false";
            case HyperparamType::Int:
                return std::to_string(value.get<int>());
            case HyperparamType::Int64:
                return std::to_string(value.get<int64_t>());
            case HyperparamType::Float:
                oss << std::setprecision(6) << value.get<double>();
                return oss.str();
            case HyperparamType::String:
                return value.get<std::string>();
            case HyperparamType::SizeT:
                return std::to_string(value.get<uint64_t>());
            case HyperparamType::Json:
                return value.dump();
        }
        throw std::runtime_error("HyperparamEntry::valueAsString: unknown type for key " + key);
    }

    nlohmann::json parseEditedValue(const std::string& text) const {
        switch (type) {
            case HyperparamType::Bool:
                if (text == "true" || text == "1") return true;
                if (text == "false" || text == "0") return false;
                throw std::runtime_error("boolean field requires true/false: " + key);
            case HyperparamType::Int:
                return std::stoi(text);
            case HyperparamType::Int64:
                return static_cast<int64_t>(std::stoll(text));
            case HyperparamType::Float:
                return std::stod(text);
            case HyperparamType::String:
                return text;
            case HyperparamType::SizeT:
                return static_cast<uint64_t>(std::stoull(text));
            case HyperparamType::Json:
                return nlohmann::json::parse(text);
        }
        throw std::runtime_error("HyperparamEntry::parseEditedValue: unknown type for key " + key);
    }
};

class HyperparameterRegistry {
public:
    void populateModelConfigSchema(const nlohmann::json& sourceConfig) {
        if (!sourceConfig.is_object()) {
            throw std::runtime_error(
                "HyperparameterRegistry::populateModelConfigSchema: source config must be an object");
        }

        struct SchemaField {
            const char* key;
            const char* table;
        };

        // These are the human-authored compiler inputs for the corresponding
        // FlatBuffer tables. Compiler-derived fields (hashes, dimensions,
        // capabilities, vocab facts, and lookup vectors) are intentionally not
        // editable here.
        static constexpr SchemaField fields[] = {
            {"d_model", "ArchitectureConfig"},
            {"num_layers", "ArchitectureConfig"},
            {"num_heads", "ArchitectureConfig"},
            {"num_kv_heads", "ArchitectureConfig"},
            {"max_seq_len", "ArchitectureConfig"},
            {"tie_embeddings", "ArchitectureConfig"},
            {"embedding_scale", "ArchitectureConfig"},

            {"use_bias", "BiasPolicy"},
            {"attention_qkv_bias_enabled", "BiasPolicy"},
            {"attention_output_bias_enabled", "BiasPolicy"},
            {"ffn_output_bias_enabled", "BiasPolicy"},
            {"lm_head_bias_enabled", "BiasPolicy"},

            {"causal_mask", "AttentionConfig"},
            {"use_pre_norm", "AttentionConfig"},
            {"fuse_qkv", "AttentionConfig"},
            {"qk_norm_enabled", "AttentionConfig"},
            {"attention_off_by_one", "AttentionConfig"},
            {"attention_residual_gate_enabled", "AttentionConfig"},

            {"use_rope", "PositionalEncodingConfig"},
            {"use_alibi", "PositionalEncodingConfig"},
            {"rope_base_seq_len", "PositionalEncodingConfig"},
            {"alibi_min_locality_distance", "PositionalEncodingConfig"},
            {"alibi_slope_exponent", "PositionalEncodingConfig"},
            {"alibi_max_bias", "PositionalEncodingConfig"},
            {"rope_theta", "PositionalEncodingConfig"},
            {"rope_scaling", "PositionalEncodingConfig"},

            {"rms_epsilon", "EncoderConfig"},
            {"use_layer_scale", "EncoderConfig"},
            {"layer_scale_init", "EncoderConfig"},
            {"center_encoder_residuals", "EncoderConfig"},

            {"lm_head_unigram_bias", "LmHeadConfig"},
            {"lm_head_center_hidden_states", "LmHeadConfig"},
            {"center_logits", "LmHeadConfig"},
            {"project_out_pc1", "LmHeadConfig"},
            {"pc1_power_iters", "LmHeadConfig"},
            {"lm_head_mlp_enabled", "LmHeadConfig"},
            {"lm_head_mlp_d_ff", "LmHeadConfig"},
            {"lm_head_mlp_alpha", "LmHeadConfig"},

            {"use_atom_data", "ModelFeatures"},
            {"atom_embedding_dim", "ModelFeatures"},
            {"selector_enabled", "ModelFeatures"},

            {"execution_block_enabled", "ExecutionBlockConfig"},
            {"execution_block_layer", "ExecutionBlockConfig"},
            {"execution_block_num_ops", "ExecutionBlockConfig"},
            {"execution_block_num_slots", "ExecutionBlockConfig"},
            {"execution_block_num_scratch_slots", "ExecutionBlockConfig"},
            {"execution_block_num_steps", "ExecutionBlockConfig"},
            {"execution_block_value_decode_input_dim", "ExecutionBlockConfig"},
            {"execution_block_value_decode_hidden_dim", "ExecutionBlockConfig"},
            {"execution_block_d_type", "ExecutionBlockConfig"},
            {"execution_block_cross_attn_topk", "ExecutionBlockConfig"},
            {"execution_block_usage_decay", "ExecutionBlockConfig"},
            {"execution_block_inject_gate_temp", "ExecutionBlockConfig"},
            {"execution_block_result_slot_mode", "ExecutionBlockConfig"},
            {"execution_block_result_slot_index", "ExecutionBlockConfig"},
            {"execution_block_magnitude_limit", "ExecutionBlockConfig"},
            {"execution_block_causal_w1_transition", "ExecutionBlockConfig"},
            {"execution_block_decode_bias_enabled", "ExecutionBlockConfig"},
            {"execution_block_value_embedding_bias_enabled", "ExecutionBlockConfig"},
            {"execution_block_scalar_bias_enabled", "ExecutionBlockConfig"},
            {"execution_block_trace_bias_enabled", "ExecutionBlockConfig"},

            {"number_encoder_enabled", "NumberEncoderConfig"},
            {"number_encoder_max_digit_slots", "NumberEncoderConfig"},
            {"number_encoder_d_hidden", "NumberEncoderConfig"},
            {"number_encoder_max_abs_pow10", "NumberEncoderConfig"},
            {"number_encoder_contribution_bias_enabled", "NumberEncoderConfig"},
            {"number_encoder_global_bias_enabled", "NumberEncoderConfig"},

            {"slot_seed_encoder_enabled", "SlotSeedEncoderConfig"},
            {"slot_seed_encoder_d_hidden", "SlotSeedEncoderConfig"},
            {"slot_seed_encoder_bias_enabled", "SlotSeedEncoderConfig"},
            {"slot_seed_encoder_type_embedding_enabled", "SlotSeedEncoderConfig"},

            {"tokenizer_model_type", "TokenizerConfig"},
            {"tokenizer_special_tokens", "TokenizerConfig"},
            {"tokenizer_add_bos", "TokenizerConfig"},
            {"tokenizer_add_eos", "TokenizerConfig"},
            {"tokenizer_unk_token", "TokenizerConfig"},
            {"tokenizer_pad_token", "TokenizerConfig"},
            {"tokenizer_bos_token", "TokenizerConfig"},
            {"tokenizer_eos_token", "TokenizerConfig"},
            {"tokenizer_enable_nfkc_normalization", "TokenizerConfig"},
            {"tokenizer_enable_lowercasing", "TokenizerConfig"},
            {"tokenizer_enable_byte_fallback", "TokenizerConfig"},
            {"tokenizer_enable_atom_reasoning", "TokenizerConfig"},
            {"tokenizer_detect_numbers", "TokenizerConfig"},
        };

        entries_.clear();
        categories_.clear();
        for (const SchemaField& field : fields) {
            const auto value = sourceConfig.find(field.key);
            if (value == sourceConfig.end()) continue;
            addEntry(field.key, field.table, *value);
            if (std::find(categories_.begin(), categories_.end(), field.table) == categories_.end()) {
                categories_.push_back(field.table);
            }
        }
    }

    const std::vector<HyperparamEntry>& entries() const { return entries_; }
    const std::vector<std::string>& categories() const { return categories_; }

    std::vector<const HyperparamEntry*> filtered(const std::string& category) const {
        std::vector<const HyperparamEntry*> result;
        for (const auto& e : entries_) {
            if (category.empty() || e.category == category) {
                result.push_back(&e);
            }
        }
        return result;
    }

    bool empty() const { return entries_.empty(); }

private:
    std::vector<HyperparamEntry> entries_;
    std::vector<std::string> categories_;

    static HyperparamType inferType(const nlohmann::json& value) {
        if (value.is_boolean()) return HyperparamType::Bool;
        if (value.is_number_unsigned()) return HyperparamType::SizeT;
        if (value.is_number_integer()) {
            const int64_t v = value.get<int64_t>();
            if (v >= static_cast<int64_t>(std::numeric_limits<int>::min()) &&
                v <= static_cast<int64_t>(std::numeric_limits<int>::max())) {
                return HyperparamType::Int;
            }
            return HyperparamType::Int64;
        }
        if (value.is_number_float()) return HyperparamType::Float;
        if (value.is_string()) return HyperparamType::String;
        return HyperparamType::Json;
    }

    static std::string makeDisplayName(const std::string& key) {
        std::string out;
        out.reserve(key.size());
        bool capitalizeNext = true;
        for (char ch : key) {
            if (ch == '_') {
                out.push_back(' ');
                capitalizeNext = true;
                continue;
            }
            const unsigned char uch = static_cast<unsigned char>(ch);
            out.push_back(capitalizeNext ? static_cast<char>(std::toupper(uch)) : ch);
            capitalizeNext = false;
        }
        return out;
    }

    void addEntry(const std::string& key,
                  const std::string& schemaTable,
                  const nlohmann::json& value) {
        HyperparamEntry e;
        e.key = key;
        e.display_name = makeDisplayName(key);
        e.category = schemaTable;
        e.type = inferType(value);
        e.value = value;
        entries_.push_back(std::move(e));
    }
};

} // namespace Config
} // namespace GRIM
