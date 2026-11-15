#include "model_workspace_services.hpp"
#include "resources.hpp"
#include "ai/grim_text_server_manager.hpp"
#include "bootstrap/dataset_bootstrap.hpp"
#include "control/tokenizer_client.hpp"
#include "resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp"

extern GRIM::MMO::ModelLoader* g_modelLoader;
extern GRIM::MMO::ResourceSignal* g_resourceSignal;

namespace GRIM::UI {
MMO::ModelLoader* modelWorkspaceLoader() { return g_modelLoader; }
MMO::ResourceSignal* modelWorkspaceResources() { return g_resourceSignal; }
nlohmann::json modelWorkspaceConfig() { return aiConfig; }
std::string modelWorkspaceServerURL() { return GRIMTextServerManager::getInstance().getServerURL(); }
WorldLab::GoalAccess modelWorkspaceGoalAccess() {
    const auto catalog = DataCollection::runtimeDatasetCatalog();
    return {[catalog](const std::string& query){return catalog->goalSelections(query);},
            [catalog](const nlohmann::json& value){catalog->importGoal(value);},
            [catalog]{return catalog->curriculumRegistry();},
            [catalog](const std::string& id){return catalog->conceptBlock(id);},
            [](const nlohmann::json& input){
                return SpanText::renderInput(input, aiConfig.at("training").at("config")
                    .at("concept_spans").get<NamedConceptSpanDefinitions>());
            },
            [](std::string text) {
                const auto url = modelWorkspaceServerURL();
                return std::async(std::launch::async, [url, text = std::move(text)]() -> WorldLab::PreviewTokenResult {
                    const auto result = GRIMText::TokenizerClient(url).encodeText(text);
                    if (!result.success) return {std::nullopt, result.error};
                    return {result.token_count, {}};
                });
            }};
}
}
