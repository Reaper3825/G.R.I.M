// No model loader, server, dataset service, or native viewport is started.
#include "ui/model_workspace_services.hpp"
#include "ui/observatory/ui_observatory_view.hpp"
#include "MMO/Core/ModelLoader.hpp"
#include "MMO/Core/ResourceSignal.hpp"
#include "resources.hpp"
#include <stdexcept>

std::filesystem::path modelWorkspaceTestRoot;
bool modelWorkspaceTestFail = false;
namespace GRIM::UI {
MMO::ModelLoader* modelWorkspaceLoader() { return nullptr; }
MMO::ResourceSignal* modelWorkspaceResources() { return nullptr; }
nlohmann::json modelWorkspaceConfig() { return {{"paths", {{"grim_text", {{"model_store", modelWorkspaceTestRoot.string()}}}}}}; }
std::string modelWorkspaceServerURL() { throw std::runtime_error("Model server access is forbidden in this fixture"); }
WorldLab::GoalAccess modelWorkspaceGoalAccess() {
    if (modelWorkspaceTestFail) throw std::runtime_error("fixture initialization failure");
    return {};
}
}
std::string getGrimRootDir() { return modelWorkspaceTestRoot.string(); }
std::string getResourcePath() { return modelWorkspaceTestRoot.string(); }
nlohmann::json loadGrimRuntimeAiConfig() { return GRIM::UI::modelWorkspaceConfig(); }
nlohmann::json saveGrimRuntimeAiConfig(const nlohmann::json&) { throw std::runtime_error("Config writes forbidden in fixture"); }
namespace GRIM::MMO {
LoadResult ModelLoader::ensureLoaded(const std::string&) { throw std::runtime_error("Model load forbidden"); }
void ModelLoader::unload(const std::string&) { throw std::runtime_error("Model unload forbidden"); }
ResidencyState ModelLoader::getState(const std::string&) const { return ResidencyState::Unloaded; }
const char* loadResultToString(LoadResult) { return "fixture"; }
const char* residencyStateToString(ResidencyState) { return "fixture"; }
ResourceSnapshot ResourceSignal::latest() const { return {}; }
}
struct UIObservatoryView::Impl {};
UIObservatoryView::UIObservatoryView() : impl_(std::make_unique<Impl>()) {}
UIObservatoryView::~UIObservatoryView() = default;
void UIObservatoryView::update(const InputState&, float, const PanelRect&, uint64_t) {}
void UIObservatoryView::draw(OverlayRenderer&, const PanelRect&) {}
void UIObservatoryView::hide() {}
void UIObservatoryView::refreshModels() {}
bool UIObservatoryView::shouldPassThroughAt(float,float) const { return false; }
void UIObservatoryView::collectPassThroughRects(std::vector<PanelRect>&) const {}
