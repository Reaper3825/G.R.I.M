#include "InputBindings.hpp"
#include "core/input_parser.hpp"
#include "logger.hpp"
#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace GRIM::InputBindings {
namespace {

std::unordered_map<std::string, Binding> s_runtimeBindings;
std::atomic<bool> s_captureActive{false};

bool isCtrlKey(KeyCode key)
{
    return key == KeyCode::LCtrl || key == KeyCode::RCtrl;
}

bool isShiftKey(KeyCode key)
{
    return key == KeyCode::LShift || key == KeyCode::RShift;
}

bool isAltKey(KeyCode key)
{
    return key == KeyCode::LAlt || key == KeyCode::RAlt;
}

bool modifiersMatch(const Binding& binding)
{
    const bool ctrlDown = Key::isDown(KeyCode::LCtrl) || Key::isDown(KeyCode::RCtrl);
    const bool shiftDown = Key::isDown(KeyCode::LShift) || Key::isDown(KeyCode::RShift);
    const bool altDown = Key::isDown(KeyCode::LAlt) || Key::isDown(KeyCode::RAlt);
    return (binding.ctrl || isCtrlKey(binding.key)) == ctrlDown &&
           (binding.shift || isShiftKey(binding.key)) == shiftDown &&
           (binding.alt || isAltKey(binding.key)) == altDown;
}

std::string trim(std::string value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::string lower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

} // namespace

const std::vector<ActionDefinition>& actions()
{
    static const std::vector<ActionDefinition> definitions = {
        {"wake_voice", "Wake Voice", "General", {KeyCode::RCtrl}},
        {"toggle_console", "Toggle Console", "Interface", {KeyCode::Grave, true, true}},
        {"toggle_settings", "Toggle Settings", "Interface", {KeyCode::F10, true, true}},
        {"toggle_training", "Toggle Training", "Interface", {KeyCode::F9, true, true}},
        {"toggle_data_hub", "Toggle Data Hub", "Interface", {KeyCode::F8, true, true}},
        {"toggle_storage", "Toggle Storage", "Interface", {KeyCode::F7, true, true}},
        {"toggle_geospatial", "Toggle Geospatial", "Interface", {KeyCode::F6, true, true}},
        {"toggle_physical_environment", "Toggle Physical Environment", "Interface", {KeyCode::F5, true, true}},
        {"toggle_digital_environment", "Toggle Digital Environment", "Interface", {KeyCode::F4, true, true}},
    };
    return definitions;
}

std::optional<ActionDefinition> action(const std::string& actionId)
{
    const auto& definitions = actions();
    const auto it = std::find_if(definitions.begin(), definitions.end(), [&](const auto& definition) {
        return definition.id == actionId;
    });
    if (it == definitions.end()) return std::nullopt;
    return *it;
}

std::string toConfigString(const Binding& binding)
{
    if (binding.empty()) return {};
    std::string result;
    if (binding.ctrl) result += "Ctrl+";
    if (binding.shift) result += "Shift+";
    if (binding.alt) result += "Alt+";
    result += Key::name(binding.key);
    return result;
}

std::string toDisplayString(const Binding& binding)
{
    if (binding.empty()) return "Unbound";
    std::string result;
    if (binding.ctrl) {
#if defined(__APPLE__)
        result += "Cmd+";
#else
        result += "Ctrl+";
#endif
    }
    if (binding.shift) result += "Shift+";
    if (binding.alt) result += "Alt+";
    result += Key::name(binding.key);
    return result;
}

std::optional<Binding> fromConfigString(const std::string& value)
{
    Binding result;
    std::string remaining = trim(value);
    if (remaining.empty()) return result;

    size_t start = 0;
    while (start <= remaining.size())
    {
        const size_t delimiter = remaining.find('+', start);
        const std::string token = trim(remaining.substr(
            start, delimiter == std::string::npos ? std::string::npos : delimiter - start));
        const std::string normalized = lower(token);
        if (normalized == "ctrl" || normalized == "control" || normalized == "cmd" || normalized == "command")
            result.ctrl = true;
        else if (normalized == "shift")
            result.shift = true;
        else if (normalized == "alt" || normalized == "option")
            result.alt = true;
        else {
            if (result.key != KeyCode::Unknown) return std::nullopt;
            const auto key = Key::fromName(token);
            if (!key) return std::nullopt;
            result.key = *key;
        }

        if (delimiter == std::string::npos) break;
        start = delimiter + 1;
    }

    if (result.key == KeyCode::Unknown) return std::nullopt;
    return result;
}

std::optional<Binding> capturePressedBinding(const InputState& input)
{
    for (int code = 0; code < static_cast<int>(input.keyPressed.size()); ++code)
    {
        if (!input.keyPressed[code]) continue;
        const KeyCode key = Key::fromPlatformCode(code);
        if (key == KeyCode::Unknown) continue;
        return Binding{
            key,
            input.ctrl && !isCtrlKey(key),
            input.shift && !isShiftKey(key),
            input.alt && !isAltKey(key)};
    }
    return std::nullopt;
}

Binding bindingFromConfig(const nlohmann::json& config, const std::string& actionId)
{
    const auto definition = action(actionId);
    if (!definition) return {};

    if (!config.contains("controls") || !config["controls"].is_object() ||
        !config["controls"].contains("bindings") || !config["controls"]["bindings"].is_object() ||
        !config["controls"]["bindings"].contains(actionId))
        return definition->defaultBinding;

    const auto& value = config["controls"]["bindings"][actionId];
    if (value.is_null()) return {};
    if (!value.is_string()) return definition->defaultBinding;
    const auto parsed = fromConfigString(value.get<std::string>());
    return parsed.value_or(definition->defaultBinding);
}

void setBindingInConfig(nlohmann::json& config,
                        const std::string& actionId,
                        const std::optional<Binding>& binding)
{
    config["controls"]["bindings"][actionId] = binding && !binding->empty()
        ? nlohmann::json(toConfigString(*binding))
        : nlohmann::json(nullptr);
}

void resetBindingsInConfig(nlohmann::json& config)
{
    for (const auto& definition : actions())
        setBindingInConfig(config, definition.id, definition.defaultBinding);
}

std::optional<std::string> findConflict(const nlohmann::json& config,
                                        const std::string& actionId,
                                        const Binding& binding)
{
    if (binding.empty()) return std::nullopt;
    for (const auto& definition : actions())
    {
        if (definition.id != actionId && bindingFromConfig(config, definition.id) == binding)
            return definition.id;
    }
    return std::nullopt;
}

void loadRuntimeConfig(const nlohmann::json& config)
{
    s_runtimeBindings.clear();
    for (const auto& definition : actions())
        s_runtimeBindings[definition.id] = bindingFromConfig(config, definition.id);
    LOG_DEBUG("InputBindings", "Loaded " + std::to_string(s_runtimeBindings.size()) + " runtime bindings");
}

bool wasPressed(const std::string& actionId)
{
    if (s_captureActive.load(std::memory_order_relaxed)) return false;
    const auto it = s_runtimeBindings.find(actionId);
    return it != s_runtimeBindings.end() && !it->second.empty() &&
           modifiersMatch(it->second) && Key::wasPressed(it->second.key);
}

bool isDown(const std::string& actionId)
{
    if (s_captureActive.load(std::memory_order_relaxed)) return false;
    const auto it = s_runtimeBindings.find(actionId);
    return it != s_runtimeBindings.end() && !it->second.empty() &&
           modifiersMatch(it->second) && Key::isDown(it->second.key);
}

void beginCapture() { s_captureActive.store(true, std::memory_order_relaxed); }
void endCapture() { s_captureActive.store(false, std::memory_order_relaxed); }
bool isCaptureActive() { return s_captureActive.load(std::memory_order_relaxed); }

} // namespace GRIM::InputBindings
