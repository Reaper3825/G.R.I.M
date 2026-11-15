#include "ui/data_hub_services.hpp"
#include "DataCollection/data_hub_jobs.hpp"
#include "bootstrap/dataset_bootstrap.hpp"
#include "bootstrap/dataset_services_bootstrap.hpp"
#include "resources.hpp"
#include "DataCollection/collection_state.hpp"

#include <map>
#include <fstream>

nlohmann::json aiConfig;
std::filesystem::path dataHubTestRoot;
bool dataHubTestFail = false;
namespace {
std::shared_ptr<GRIM::DataCollection::DatasetStream> stream;
std::shared_ptr<GRIM::DataCollection::DatasetServices> services;
std::map<uint64_t, GRIM::DataCollection::DataHubJobs::QueryResult> queries;
uint64_t nextQuery = 1;
}
#ifndef GRIM_MODEL_WORKSPACE_TESTS
std::string getGrimRootDir() { return dataHubTestRoot.string(); }
std::string getResourcePath() { return dataHubTestRoot.string(); }
#endif
std::string resolveHuggingFaceApiToken() { return {}; }
std::vector<std::filesystem::path> listFiles(const std::filesystem::path& path, std::error_code& error) {
    std::vector<std::filesystem::path> result;
    for (const auto& entry : std::filesystem::directory_iterator(path, error))
        if (entry.is_regular_file()) result.push_back(entry.path());
    return result;
}
namespace GRIM::Pipeline {
PipelineOrchestrator::PipelineOrchestrator() = default;
PipelineOrchestrator::~PipelineOrchestrator() = default;
PipelineOrchestrator::Status PipelineOrchestrator::getStatus() const { return {}; }
void PipelineOrchestrator::startPipeline(PipelineMode) {}
void PipelineOrchestrator::stopPipeline() {}
}
namespace GRIM::DataCollection {
std::shared_ptr<DatasetStream> runtimeDatasetStream() {
    if (dataHubTestFail || !stream) throw std::runtime_error("DataHub fixture unavailable");
    return stream;
}
std::shared_ptr<DatasetCatalog> runtimeDatasetCatalog() {
    return std::make_shared<DatasetCatalog>(runtimeDatasetStream());
}
std::shared_ptr<DatasetServices> runtimeDatasetServices() {
    if (dataHubTestFail || !services) throw std::runtime_error("DataHub fixture unavailable");
    return services;
}
namespace DataHubJobs {
uint64_t submit(QueryRequest) {
    const auto id = nextQuery++;
    queries.emplace(id, QueryResult{});
    return id;
}
QueryResult poll(uint64_t id) { return queries.at(id); }
void release(uint64_t id) {
    if (queries.erase(id) != 1) throw std::runtime_error("Unknown DataHub fixture request");
}
Downloads downloads() {
    if (dataHubTestFail) throw std::runtime_error("DataHub fixture unavailable");
    Downloads value; value.revision = 1;
    return value;
}
void enqueueDownload(const std::string&, const std::string&) {}
void processDownloads(const std::string&) {}
void removeDownload(const std::string&) {}
void retryDownload(const std::string&) {}
void clearDownloads() {}
void clearCompletedDownloads() {}
void startBatch() {}
BatchStatus batchStatus() { return {}; }
}
}
void startDataHubFixture(const std::filesystem::path& root) {
    dataHubTestRoot = root;
    aiConfig = {{"paths", {{"grim_text", {
        {"model_store", (root / "models").string()}, {"training_data", (root / "data").string()},
        {"checkpoints", (root / "checkpoints").string()}, {"source_config", (root / "sources.json").string()},
        {"directory_collection", (root / "imports").string()}}}}}};
    std::ifstream spans(std::filesystem::path(__FILE__).parent_path() / "fixtures" / "concept_spans.json");
    if (!spans) throw std::runtime_error("DataHub fixture span definitions are missing");
    aiConfig["training"]["config"]["concept_spans"] = nlohmann::json::parse(spans);
    std::filesystem::create_directories(root / "models");
    std::filesystem::create_directories(root / "imports");
    stream = std::make_shared<GRIM::DataCollection::DatasetStream>(root / "models", root / "data" / "mass_dataset.jsonl");
    if (!stream->refresh()) throw std::runtime_error("DataHub fixture catalog load failed");
    services = std::make_shared<GRIM::DataCollection::DatasetServices>();
    services->pipeline = std::make_shared<GRIM::Pipeline::PipelineOrchestrator>();
    services->creation = std::make_shared<GRIM::DataCollection::DatasetCreationService>(
        GRIM::DataCollection::runtimeDatasetCatalog(), nullptr,
        GRIM::DataCollection::GeneratorAdapterOptions{}, root / "data");
}
void stopDataHubFixture() {
    services->creation->shutdown();
    services.reset(); stream.reset(); queries.clear();
}
size_t pendingDataHubFixtureQueries() { return queries.size(); }
void completeDataHubFixtureQueries() {
    for (auto& [id, result] : queries) {
        result.ready = true;
        GRIM::DataCollection::HFDatasetInfo dataset;
        dataset.id = "fixture/dataset"; dataset.name = "Fixture dataset";
        result.datasets = {dataset};
    }
}
