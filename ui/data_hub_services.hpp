#pragma once
#include "core/host_exports.hpp"
#include "DataCollection/dataset_catalog.hpp"
#include "DataCollection/dataset_creation_service.hpp"
#include "DataCollection/pipeline/pipeline_orchestrator.hpp"
#include "ui/world_lab/environment_registry.hpp"

namespace GRIM::UI::DataHub {
struct Statistics {
    bool datasetExists = false;
    uintmax_t datasetBytes = 0;
    int checkpoints = 0;
};
GRIM_HOST_API Statistics statistics();
GRIM_HOST_API std::shared_ptr<DataCollection::DatasetCatalog> catalog();
GRIM_HOST_API nlohmann::json config();
GRIM_HOST_API std::string token();
GRIM_HOST_API nlohmann::json loadSettings();
GRIM_HOST_API void saveSettings(const nlohmann::json& settings);
GRIM_HOST_API nlohmann::json sources();
GRIM_HOST_API nlohmann::json saveSources(const nlohmann::json& rows, const nlohmann::json& baseline);
void addDownloadedSource(const std::string& datasetId, const std::string& displayName);
GRIM_HOST_API Pipeline::PipelineOrchestrator::Status pipelineStatus();
GRIM_HOST_API void startPipeline(Pipeline::PipelineMode mode);
GRIM_HOST_API void stopPipeline();
GRIM_HOST_API std::shared_ptr<const DataCollection::ConceptGenerator> generatorAssets();
GRIM_HOST_API std::string generatorError();
GRIM_HOST_API DataCollection::DatasetCreationService::Status generatorStatus();
GRIM_HOST_API bool startGenerator(const DataCollection::DatasetCreationService::Request& request);
GRIM_HOST_API void pauseGenerator();
GRIM_HOST_API bool resumeGenerator();
GRIM_HOST_API size_t saveGeneratorQueue();
GRIM_HOST_API std::vector<std::filesystem::path> files(const std::filesystem::path& path, std::error_code& error);
GRIM_HOST_API void collectFile(const std::filesystem::path& path, bool deleteAfter);
GRIM_HOST_API std::vector<std::string> modelDirectories();
GRIM_HOST_API std::vector<WorldLab::EnvironmentEntry> environments(bool discover);
GRIM_HOST_API void requireEnvironment(const std::string& id);
GRIM_HOST_API std::string trainingPreview(const GRIM::ConceptBlock& block, bool conceptMode);
}
