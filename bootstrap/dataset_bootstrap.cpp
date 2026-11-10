#include "dataset_bootstrap.hpp"

#include "logger.hpp"
#include "resources.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {
std::mutex serviceMutex;
std::shared_ptr<GRIM::DataCollection::DatasetStream> service;
std::shared_ptr<GRIM::DataCollection::DatasetCatalog> catalog;

std::filesystem::path resolveDatasetPath(const std::string& value) {
    if (value.empty()) throw std::runtime_error("Dataset path must not be empty");
    const std::filesystem::path path(value);
    return path.is_absolute() ? path : std::filesystem::path(getGrimRootDir()) / path;
}
}

void bootstrapDatasetSubsystem() {
    std::lock_guard<std::mutex> lock(serviceMutex);
    if (service) return;
    const auto& paths = aiConfig.at("paths").at("grim_text");
    auto candidate = std::make_shared<GRIM::DataCollection::DatasetStream>(
        resolveDatasetPath(paths.at("model_store").get<std::string>()),
        resolveDatasetPath(paths.at("training_data").get<std::string>()) / "mass_dataset.jsonl");
    candidate->start();
    const auto error = candidate->lastError();
    service = std::move(candidate);
    catalog = std::make_shared<GRIM::DataCollection::DatasetCatalog>(service);
    if (!error.empty()) LOG_ERROR("Dataset", "Initial catalog load failed; refresh remains active: " + error);
    LOG_PHASE("Dataset stream initialized", true);
}

void shutdownDatasetSubsystem() {
    std::shared_ptr<GRIM::DataCollection::DatasetStream> previous;
    {
        std::lock_guard<std::mutex> lock(serviceMutex);
        previous = std::move(service);
        catalog.reset();
    }
    if (previous) previous->stop();
}

std::shared_ptr<GRIM::DataCollection::DatasetStream>
GRIM::DataCollection::runtimeDatasetStream() {
    std::lock_guard<std::mutex> lock(serviceMutex);
    if (!service) throw std::runtime_error("Dataset subsystem is not initialized");
    return service;
}

std::shared_ptr<GRIM::DataCollection::DatasetCatalog>
GRIM::DataCollection::runtimeDatasetCatalog() {
    std::lock_guard<std::mutex> lock(serviceMutex);
    if (!catalog) throw std::runtime_error("Dataset subsystem is not initialized");
    return catalog;
}
