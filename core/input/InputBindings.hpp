#pragma once

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

const std::vector<ActionDefinition>& actions();
std::optional<ActionDefinition> action(const std::string& actionId);

std::string toConfigString(const Binding& binding);
std::string toDisplayString(const Binding& binding);
std::optional<Binding> fromConfigString(const std::string& value);
std::optional<Binding> capturePressedBinding(const InputState& input);

Binding bindingFromConfig(const nlohmann::json& config, const std::string& actionId);
void setBindingInConfig(nlohmann::json& config,
                        const std::string& actionId,
                        const std::optional<Binding>& binding);
void resetBindingsInConfig(nlohmann::json& config);
std::optional<std::string> findConflict(const nlohmann::json& config,
                                        const std::string& actionId,
                                        const Binding& binding);

void loadRuntimeConfig(const nlohmann::json& config);
bool wasPressed(const std::string& actionId);
bool isDown(const std::string& actionId);

void beginCapture();
void endCapture();
bool isCaptureActive();

} // namespace GRIM::InputBindings
