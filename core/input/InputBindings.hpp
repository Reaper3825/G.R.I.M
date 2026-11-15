#pragma once
#include "core/host_exports.hpp"

#include "helpers/key.hpp"
#include <atomic>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct InputState;

namespace GRIM::InputBindings {

struct Binding {
    KeyCode key = KeyCode::Unknown;
    bool ctrl = false;
    bool shift = false;
    bool alt = false;

    bool empty() const { return key == KeyCode::Unknown; }
    bool operator==(const Binding& other) const {
        return key == other.key && ctrl == other.ctrl &&
               shift == other.shift && alt == other.alt;
    }
};

struct ActionDefinition {
    std::string id;
    std::string label;
    std::string category;
    Binding defaultBinding;
};

GRIM_HOST_API const std::vector<ActionDefinition>& actions();
GRIM_HOST_API std::optional<ActionDefinition> action(const std::string& actionId);

std::string toConfigString(const Binding& binding);
std::string toDisplayString(const Binding& binding);
std::optional<Binding> fromConfigString(const std::string& value);
std::optional<Binding> capturePressedBinding(const InputState& input);

GRIM_HOST_API Binding bindingFromConfig(const nlohmann::json& config, const std::string& actionId);
GRIM_HOST_API void setBindingInConfig(nlohmann::json& config,
                        const std::string& actionId,
                        const std::optional<Binding>& binding);
GRIM_HOST_API void resetBindingsInConfig(nlohmann::json& config);
GRIM_HOST_API std::optional<std::string> findConflict(const nlohmann::json& config,
                                        const std::string& actionId,
                                        const Binding& binding);

GRIM_HOST_API void loadRuntimeConfig(const nlohmann::json& config);
bool wasPressed(const std::string& actionId);
bool isDown(const std::string& actionId);

void beginCapture();
GRIM_HOST_API void endCapture();
bool isCaptureActive();

} // namespace GRIM::InputBindings
