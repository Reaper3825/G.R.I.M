// Native Settings DLL boundary tests: no real configuration files or OS effects.
#include "settings/runtime_ai_config.hpp"
#include "settings/settings_apply.hpp"
#include "core/input/InputBindings.hpp"
#include "core/platform_window.hpp"
#include "ui/overlay_renderer.hpp"
#include <stdexcept>

nlohmann::json settingsTestConfig = {{"backend", "auto"},
    {"geospatial", {{"imagery", {{"layers", nlohmann::json::array()}}}}}};
int settingsTestSaves = 0, settingsTestApplies = 0;
bool settingsTestLoadFailure = false;
namespace Settings {
nlohmann::json loadRuntimeAiConfig() {
    if (settingsTestLoadFailure) throw std::runtime_error("fixture load failure");
    return settingsTestConfig;
}
nlohmann::json saveRuntimeAiConfig(const nlohmann::json& config) { ++settingsTestSaves; return settingsTestConfig = config; }
void applyRuntimeSideEffects(const nlohmann::json&, const std::unordered_map<std::string,std::string>&, SetFontFn, SetBlurFn) { ++settingsTestApplies; }
std::vector<std::string> scanSpeakerEmbeddings() { return {"default"}; }
std::vector<std::string> scanFonts(std::unordered_map<std::string,std::string>&) { return {}; }
}
namespace GRIM::InputBindings {
bool capturing = false;
const std::vector<ActionDefinition>& actions() { static std::vector<ActionDefinition> a{{"fixture", "Fixture action", "test", {}}}; return a; }
std::optional<ActionDefinition> action(const std::string&) { return actions()[0]; }
Binding bindingFromConfig(const nlohmann::json&, const std::string&) { return {}; }
void setBindingInConfig(nlohmann::json&, const std::string&, const std::optional<Binding>&) {}
void resetBindingsInConfig(nlohmann::json&) {}
std::optional<std::string> findConflict(const nlohmann::json&, const std::string&, const Binding&) { return {}; }
void loadRuntimeConfig(const nlohmann::json&) {}
void beginCapture() { capturing = true; }
void endCapture() { capturing = false; }
bool isCaptureActive() { return capturing; }
std::optional<Binding> capturePressedBinding(const InputState&) { return {}; }
std::string toDisplayString(const Binding&) { return "Unbound"; }
}
void PlatformWindow::setOverlayBlurStyle(void*, bool, float, int) {}
void OverlayRenderer::setFont(const std::string&, int) {}
