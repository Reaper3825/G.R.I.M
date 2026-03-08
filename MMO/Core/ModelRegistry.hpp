// Multi-Model Orchestration (MMO) - Model Registry
// Canonical registry of all known models (router + sub-models).
// Loaded from ai_config.json at startup. Validates invariants:
//   - Exactly one router must exist
//   - Sub-models must NOT have lora_path or hard_copy_path
//   - All model IDs must be unique
//   - All required fields must be populated
//======================================================//
#pragma once

#include "../Shared/MMD.hpp"

#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// Registry configuration (from ai_config.json → mmo section)
// =========================================================
struct MMOConfig {
    bool        enabled        = false;      // master MMO toggle
    std::string mode           = "shadow";   // "shadow" | "enforced"
    std::string router_id;                   // must match one ModelInfo::id
};

// =========================================================
// ModelRegistry
//
// Usage:
//   auto& reg = ModelRegistry::instance();
//   reg.loadFromConfig(aiConfig);
//   const ModelInfo* router = reg.getRouter();
//   const ModelInfo* sub    = reg.getModelById("medical-7b");
// =========================================================
class ModelRegistry {
public:
    static ModelRegistry& instance();

    // Load and validate models from ai_config.json root object.
    // Throws std::runtime_error on any invariant violation.
    void loadFromConfig(const nlohmann::json& config);

    // --- Queries --------------------------------------------------------

    // The router model (never null after successful load).
    const ModelInfo* getRouter() const;

    // Lookup by exact model id. Returns nullptr if not found.
    const ModelInfo* getModelById(const std::string& id) const;

    // All sub-models (excludes router).
    std::vector<const ModelInfo*> getSubModels() const;

    // Sub-models whose subject_tags contain the given tag.
    std::vector<const ModelInfo*> getModelsBySubjectTag(const std::string& tag) const;

    // All registered models (router + sub-models).
    std::vector<const ModelInfo*> getAllModels() const;

    // Number of registered models (router + sub-models).
    size_t modelCount() const;

    // Whether MMO is enabled in config.
    bool isEnabled() const;

    // Current MMO mode ("shadow" or "enforced").
    const std::string& mode() const;

    // --- Lifecycle ------------------------------------------------------

    // Clear all state (for tests or shutdown).
    void clear();

private:
    ModelRegistry() = default;
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    // Parse one model entry from JSON into ModelInfo.
    static ModelInfo parseModelInfo(const nlohmann::json& entry);

    // Parse BackendType from string. Throws on unknown value.
    static BackendType parseBackendType(const std::string& str);

    // Validate a single ModelInfo. Throws on violation.
    static void validateModel(const ModelInfo& model, bool is_router);

    mutable std::mutex mutex_;
    MMOConfig config_;
    std::string router_id_;
    std::unordered_map<std::string, ModelInfo> models_;  // keyed by ModelInfo::id
};

} // namespace GRIM::MMO
