// Multi-Model Orchestration (MMO) - Model Registry
// See ModelRegistry.hpp for interface documentation.
//======================================================//
#include "ModelRegistry.hpp"

#include <nlohmann/json.hpp>  // full definition (fwd-only in header)

#include <sstream>
#include <stdexcept>

namespace GRIM::MMO {

// =========================================================
// Singleton
// =========================================================
ModelRegistry& ModelRegistry::instance() {
    static ModelRegistry s_instance;
    return s_instance;
}

// =========================================================
// Load from ai_config.json
//
// Expected JSON shape:
//   {
//     "mmo": {
//       "enabled": true,
//       "mode": "shadow",
//       "router": { ... ModelInfo fields ... },
//       "sub_models": [ { ... }, { ... } ]
//     }
//   }
// =========================================================
void ModelRegistry::loadFromConfig(const nlohmann::json& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    models_.clear();
    router_id_.clear();
    config_ = MMOConfig{};

    // ---- mmo section must exist ----
    if (!config.contains("mmo") || !config["mmo"].is_object()) {
        throw std::runtime_error(
            "ModelRegistry: ai_config.json is missing required 'mmo' object");
    }

    const auto& mmo = config["mmo"];

    // ---- top-level MMO fields ----
    config_.enabled = mmo.value("enabled", false);
    config_.mode    = mmo.value("mode", "shadow");

    if (config_.mode != "shadow" && config_.mode != "enforced") {
        throw std::runtime_error(
            "ModelRegistry: mmo.mode must be 'shadow' or 'enforced', got '"
            + config_.mode + "'");
    }

    // ---- router (required) ----
    if (!mmo.contains("router") || !mmo["router"].is_object()) {
        throw std::runtime_error(
            "ModelRegistry: mmo.router object is required");
    }

    ModelInfo router = parseModelInfo(mmo["router"]);
    validateModel(router, /*is_router=*/true);
    router_id_     = router.id;
    config_.router_id = router.id;
    models_.emplace(router.id, std::move(router));

    // ---- sub_models (optional array) ----
    if (mmo.contains("sub_models")) {
        if (!mmo["sub_models"].is_array()) {
            throw std::runtime_error(
                "ModelRegistry: mmo.sub_models must be an array");
        }

        for (size_t i = 0; i < mmo["sub_models"].size(); ++i) {
            const auto& entry = mmo["sub_models"][i];
            if (!entry.is_object()) {
                throw std::runtime_error(
                    "ModelRegistry: mmo.sub_models[" + std::to_string(i)
                    + "] is not an object");
            }

            ModelInfo sub = parseModelInfo(entry);
            validateModel(sub, /*is_router=*/false);

            if (models_.count(sub.id)) {
                throw std::runtime_error(
                    "ModelRegistry: duplicate model id '" + sub.id
                    + "' (sub_models[" + std::to_string(i) + "])");
            }

            models_.emplace(sub.id, std::move(sub));
        }
    }
}

// =========================================================
// Queries
// =========================================================

const ModelInfo* ModelRegistry::getRouter() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(router_id_);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

const ModelInfo* ModelRegistry::getModelById(const std::string& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(id);
    if (it == models_.end()) return nullptr;
    return &it->second;
}

std::vector<const ModelInfo*> ModelRegistry::getSubModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    for (const auto& [id, model] : models_) {
        if (id != router_id_) {
            result.push_back(&model);
        }
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::getModelsBySubjectTag(
    const std::string& tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    for (const auto& [id, model] : models_) {
        if (id == router_id_) continue;
        for (const auto& t : model.subject_tags) {
            if (t == tag) {
                result.push_back(&model);
                break;
            }
        }
    }
    return result;
}

std::vector<const ModelInfo*> ModelRegistry::getAllModels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<const ModelInfo*> result;
    result.reserve(models_.size());
    for (const auto& [id, model] : models_) {
        result.push_back(&model);
    }
    return result;
}

size_t ModelRegistry::modelCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return models_.size();
}

bool ModelRegistry::isEnabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.enabled;
}

const std::string& ModelRegistry::mode() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.mode;
}

// =========================================================
// Lifecycle
// =========================================================

void ModelRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    models_.clear();
    router_id_.clear();
    config_ = MMOConfig{};
}

// =========================================================
// Parsing helpers
// =========================================================

ModelInfo ModelRegistry::parseModelInfo(const nlohmann::json& entry) {
    ModelInfo m;

    m.id          = entry.value("id", "");
    m.name        = entry.value("name", "");
    m.version     = entry.value("version", "");
    m.subject     = entry.value("subject", "");
    m.description = entry.value("description", "");
    m.model_path  = entry.value("model_path", "");
    m.url         = entry.value("url", "");
    m.usage_weight = entry.value("usage_weight", 0.0f);

    if (entry.contains("backend_type") && entry["backend_type"].is_string()) {
        m.backend_type = parseBackendType(entry["backend_type"].get<std::string>());
    }

    if (entry.contains("subject_tags") && entry["subject_tags"].is_array()) {
        for (const auto& tag : entry["subject_tags"]) {
            if (tag.is_string()) {
                m.subject_tags.push_back(tag.get<std::string>());
            }
        }
    }

    // Router-only fields
    m.lora_path      = entry.value("lora_path", "");
    m.hard_copy_path = entry.value("hard_copy_path", "");

    // Resource estimates
    m.estimated_ram_mb  = entry.value("estimated_ram_mb", 0L);
    m.estimated_vram_mb = entry.value("estimated_vram_mb", 0L);

    return m;
}

BackendType ModelRegistry::parseBackendType(const std::string& str) {
    if (str == "grim_text_server") return BackendType::GrimTextServer;
    if (str == "llama_cpp")        return BackendType::LlamaCpp;
    if (str == "ollama")           return BackendType::Ollama;
    if (str == "external")         return BackendType::External;

    throw std::runtime_error(
        "ModelRegistry: unknown backend_type '" + str
        + "' (valid: grim_text_server, llama_cpp, ollama, external)");
}

void ModelRegistry::validateModel(const ModelInfo& model, bool is_router) {
    // Required fields
    if (model.id.empty()) {
        throw std::runtime_error(
            "ModelRegistry: model is missing required 'id' field");
    }
    if (model.name.empty()) {
        throw std::runtime_error(
            "ModelRegistry: model '" + model.id + "' is missing required 'name' field");
    }
    if (model.url.empty()) {
        throw std::runtime_error(
            "ModelRegistry: model '" + model.id + "' is missing required 'url' field");
    }

    // Sub-model invariant: no LoRA, no hard copy
    if (!is_router) {
        if (!model.lora_path.empty()) {
            throw std::runtime_error(
                "ModelRegistry: sub-model '" + model.id
                + "' has lora_path set — only the router may have LoRA. "
                  "Sub-models are frozen information bricks.");
        }
        if (!model.hard_copy_path.empty()) {
            throw std::runtime_error(
                "ModelRegistry: sub-model '" + model.id
                + "' has hard_copy_path set — only the router may have a hard copy. "
                  "Sub-models are frozen information bricks.");
        }
    }
}

} // namespace GRIM::MMO
