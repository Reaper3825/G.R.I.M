#include "dataset_services_bootstrap.hpp"
#include "dataset_bootstrap.hpp"
#include "resources.hpp"
#include "logger.hpp"

#include <nlohmann/json.hpp>

namespace {
std::mutex servicesMutex;
std::shared_ptr<GRIM::DataCollection::DatasetServices> services;
}

void bootstrapDatasetServices() {
    using namespace GRIM::DataCollection;
    std::lock_guard<std::mutex> lock(servicesMutex);
    if (services) return;
    auto candidate = std::make_shared<DatasetServices>();
    candidate->pipeline = std::make_shared<GRIM::Pipeline::PipelineOrchestrator>();
    candidate->pipeline->setDatasetWriter([stream = runtimeDatasetStream()](
        const std::filesystem::path& path, const std::vector<GRIM::Pipeline::TaggedEntry>& entries, bool replace) {
        return stream->writeMassDataset(path, entries, replace);
    });
    candidate->huggingFace = std::make_shared<HuggingFaceWebhook>();
    candidate->huggingFace->setApiToken(resolveHuggingFaceApiToken());
    DataStructuringConfig config;
    if (aiConfig.contains("data_collection") && aiConfig["data_collection"].contains("data_structuring"))
        config = DataStructuringConfig::fromJson(aiConfig["data_collection"]["data_structuring"]);
    if (config.ollama_model.empty()) config.ollama_model = "llama3.1:8b";
    candidate->structurer = std::make_shared<DataStructurer>(config);

    OllamaConceptBindingConfig adapterConfig;
    adapterConfig.url = aiConfig.value("ollama_url", std::string("http://127.0.0.1:11434"));
    adapterConfig.model = config.ollama_model;
    adapterConfig.maxInputChars = config.max_input_chars;
    std::shared_ptr<IConceptBindingAdapter> adapter;
    try { adapter = std::make_shared<OllamaConceptBindingAdapter>(std::move(adapterConfig)); }
    catch (const std::exception& error) { LOG_ERROR("Dataset", error.what()); }
    GeneratorAdapterOptions options;
    if (aiConfig.contains("ollama_options")) {
        const auto& values = aiConfig["ollama_options"];
        options.temperature = std::min(values.value("temperature", 0.2f), 0.3f);
        options.topP = values.value("top_p", 0.9f);
        options.topK = values.value("top_k", 40);
        options.maxTokens = std::max(values.value("num_predict", 2048), 1024);
    }
    if (aiConfig.contains("data_collection") && aiConfig["data_collection"].contains("data_structuring"))
        options.timeoutMs = aiConfig["data_collection"]["data_structuring"].value("timeout_ms", 60000);
    std::filesystem::path directory(aiConfig.at("paths").at("grim_text").at("training_data").get<std::string>());
    if (!directory.is_absolute()) directory = std::filesystem::path(getGrimRootDir()) / directory;
    candidate->creation = std::make_shared<DatasetCreationService>(
        runtimeDatasetCatalog(), std::move(adapter), options, directory);
    services = std::move(candidate);
    LOG_PHASE("Dataset creation services initialized", true);
}

void shutdownDatasetServices() {
    std::shared_ptr<GRIM::DataCollection::DatasetServices> previous;
    {
        std::lock_guard<std::mutex> lock(servicesMutex);
        previous = std::move(services);
    }
    if (!previous) return;
    previous->creation->shutdown();
    previous->pipeline->shutdown();
}

std::shared_ptr<GRIM::DataCollection::DatasetServices>
GRIM::DataCollection::runtimeDatasetServices() {
    std::lock_guard<std::mutex> lock(servicesMutex);
    if (!services) throw std::runtime_error("Dataset services are not initialized");
    return services;
}
