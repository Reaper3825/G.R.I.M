#pragma once
//======================================================//
// HyperparameterRegistry — enumerates GRIM runtime
// training.config leaves with display metadata for UI
// browsing.
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
    void populate(const nlohmann::json& trainingConfig) {
        if (!trainingConfig.is_object()) {
            throw std::runtime_error("HyperparameterRegistry::populate: training.config must be an object");
        }

        entries_.clear();
        categories_.clear();

        for (const auto& item : trainingConfig.items()) {
            addEntry(item.key(), item.value());
        }

        std::set<std::string> cats;
        for (const auto& e : entries_) cats.insert(e.category);
        categories_.assign(cats.begin(), cats.end());
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

    static bool startsWith(const std::string& text, const std::string& prefix) {
        return text.rfind(prefix, 0) == 0;
    }

    static bool contains(const std::string& text, const std::string& needle) {
        return text.find(needle) != std::string::npos;
    }

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

    static std::string categoryForKey(const std::string& key) {
        if (startsWith(key, "optimizer_") || key == "learning_rate" || key == "weight_decay" ||
            key == "grad_clip_norm" || startsWith(key, "cosine_") || key == "warmup_fraction") {
            return "Optimizer";
        }
        if (startsWith(key, "soft_restart_")) return "Soft Restart";
        if (startsWith(key, "auto_stop_")) return "Auto Stop";
        if (startsWith(key, "shuffle_")) return "Shuffle";
        if (startsWith(key, "telemetry_lattice_")) return "Telemetry Lattice";
        if (startsWith(key, "telemetry_")) return "Telemetry";
        if (startsWith(key, "logging_") || startsWith(key, "log_recorder_")) return "Logging";
        if (startsWith(key, "loss_")) return "Loss";
        if (startsWith(key, "lm_head_") || key == "freeze_learned_rms_gammas" ||
            key == "center_logits" || key == "center_encoder_residuals" ||
            key == "project_out_pc1" || key == "pc1_power_iters") {
            return "LM Head";
        }
        if (contains(key, "attention") || startsWith(key, "qk_norm") || startsWith(key, "rope_") ||
            startsWith(key, "alibi_") || key == "use_flash_attention") {
            return "Attention";
        }
        if (startsWith(key, "layer_scale") || key == "use_layer_scale") return "Layer Scale";
        if (startsWith(key, "scratch") || key == "use_scratch_block") return "Scratch Block";
        if (startsWith(key, "execution_block_") || startsWith(key, "selector_") ||
            startsWith(key, "step_") || startsWith(key, "entropy_") ||
            startsWith(key, "value_match_") || startsWith(key, "final_slot_") ||
            startsWith(key, "div_") ||
            startsWith(key, "structured_ce_") || startsWith(key, "decode_time_slot_")) {
            return "Execution Block";
        }
        if (startsWith(key, "single_stream") || startsWith(key, "disable_async") ||
            startsWith(key, "synchronize_after")) {
            return "CUDA";
        }
        if (startsWith(key, "embedding_")) return "Embedding";
        if (startsWith(key, "stability_")) return "Stability";
        if (startsWith(key, "prediction_comparison_") || startsWith(key, "logit_update_trace_") ||
            startsWith(key, "hardcoded_")) {
            return "Diagnostics";
        }
        if (startsWith(key, "generation_")) return "Generation";
        if (startsWith(key, "tokenizer_") || startsWith(key, "subprocess_")) return "Tokenizer";
        if (startsWith(key, "parameter_precision_")) return "Precision";
        return "Core";
    }

    void addEntry(const std::string& key, const nlohmann::json& value) {
        HyperparamEntry e;
        e.key = key;
        e.display_name = makeDisplayName(key);
        e.category = categoryForKey(key);
        e.type = inferType(value);
        e.value = value;
        entries_.push_back(std::move(e));
    }
};

} // namespace Config
} // namespace GRIM
