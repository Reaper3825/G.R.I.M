#pragma once
#include "core/host_exports.hpp"
#include <filesystem>
#include <nlohmann/json.hpp>

namespace Settings {

// Resolve the canonical path to ai_config.json using getGrimRootDir().
// Throws if the file cannot be found.
std::filesystem::path resolveAiConfigPath();

// Resolve the path to ai_config.local.json (next to the main config).
std::filesystem::path resolveAiConfigLocalPath();

// Deep-merge `overlay` into `base` (recursively merges objects, overwrites scalars/arrays).
void deepMerge(nlohmann::json& base, const nlohmann::json& overlay);

// Copy the effective in-memory configuration on the UI thread.
GRIM_HOST_API nlohmann::json snapshotRuntimeAiConfig();

// Load ai_config.json from disk, deep-merge ai_config.local.json, and
// update the global `aiConfig`. Returns the merged document.
GRIM_HOST_API nlohmann::json loadRuntimeAiConfig();

// Deep-merge `pending` edits into the full on-disk document and write back.
// Updates global `aiConfig`. Returns the saved document.
GRIM_HOST_API nlohmann::json saveRuntimeAiConfig(const nlohmann::json& pending);

} // namespace Settings
