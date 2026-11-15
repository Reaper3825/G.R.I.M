#include "data_hub_services.hpp"
#include "bootstrap/dataset_bootstrap.hpp"
#include "bootstrap/dataset_services_bootstrap.hpp"
#include "DataCollection/concept_block_canonical.hpp"
#include "memory/atomic_writer.hpp"
#include "resources.hpp"

#include <fstream>
#include <mutex>

namespace GRIM::UI::DataHub {
namespace {
std::mutex persistenceMutex;
std::filesystem::path path(const char* key) {
    auto value = std::filesystem::path(aiConfig.at("paths").at("grim_text").at(key).get<std::string>());
    if (value.is_relative()) value = std::filesystem::path(getGrimRootDir()) / value;
    return value;
}
nlohmann::json readJson(const std::filesystem::path& file, const nlohmann::json& initial) {
    if (!std::filesystem::exists(file)) return initial;
    std::ifstream input(file);
    if (!input) throw std::runtime_error("DataHub cannot read " + file.string());
    return nlohmann::json::parse(input);
}
nlohmann::json sourceDocument() {
    auto document = readJson(path("source_config"), {{"data_sources", nlohmann::json::array()}});
    if (!document.at("data_sources").is_array())
        throw std::runtime_error("DataHub source configuration requires data_sources array");
    return document;
}
}
std::shared_ptr<DataCollection::DatasetCatalog> catalog() {
    return std::make_shared<DataCollection::DatasetCatalog>(DataCollection::runtimeDatasetStream(), true);
}
Statistics statistics() {
    Statistics value;
    const auto data = path("training_data");
    value.datasetExists = std::filesystem::exists(data);
    if (value.datasetExists)
        value.datasetBytes = std::filesystem::file_size(data);
    const auto checkpoints = path("checkpoints");
    if (std::filesystem::exists(checkpoints))
        for (const auto& entry : std::filesystem::directory_iterator(checkpoints))
            if (entry.path().extension() == ".ckpt") ++value.checkpoints;
    return value;
}
nlohmann::json config() {
    return {{"paths", {{"grim_text", aiConfig.at("paths").at("grim_text")}}}};
}
std::string token() { return resolveHuggingFaceApiToken(); }
nlohmann::json loadSettings() {
    std::lock_guard lock(persistenceMutex);
    return readJson(path("checkpoints") / "collection_state" / "ui_config.json", nlohmann::json::object());
}
void saveSettings(const nlohmann::json& settings) {
    if (!settings.is_object()) throw std::invalid_argument("DataHub settings must be an object");
    std::lock_guard lock(persistenceMutex);
    AtomicWriter::writeString((path("checkpoints") / "collection_state" / "ui_config.json").string(), settings.dump(2));
}
nlohmann::json sources() {
    std::lock_guard lock(persistenceMutex);
    return sourceDocument().at("data_sources");
}
nlohmann::json saveSources(const nlohmann::json& rows, const nlohmann::json& baseline) {
    if (!rows.is_array() || !baseline.is_array())
        throw std::invalid_argument("DataHub sources must be arrays");
    std::lock_guard lock(persistenceMutex);
    auto document = sourceDocument();
    const auto& current = document.at("data_sources");
    if (current.size() < baseline.size() ||
        !std::equal(baseline.begin(), baseline.end(), current.begin()))
        throw std::runtime_error("DataHub sources changed externally; reload Sources before saving");
    auto merged = rows;
    for (size_t i = baseline.size(); i < current.size(); ++i) {
        const auto& added = current[i];
        const auto found = std::find_if(merged.begin(), merged.end(), [&](const auto& row) {
            return row.at("url") == added.at("url");
        });
        if (found == merged.end()) merged.push_back(added);
    }
    document["data_sources"] = merged;
    AtomicWriter::writeString(path("source_config").string(), document.dump(2));
    return merged;
}
void addDownloadedSource(const std::string& datasetId, const std::string& displayName) {
    std::lock_guard lock(persistenceMutex);
    auto document = sourceDocument();
    auto& rows = document["data_sources"];
    const std::string url = "huggingface://" + datasetId;
    for (const auto& row : rows) if (row.at("url") == url) return;
    rows.push_back({{"name", displayName}, {"url", url}, {"source_type", "huggingface"},
        {"priority", 5}, {"enabled", true}, {"crawl_depth", 2}, {"fetch_limit", 100}, {"requires_auth", false}});
    AtomicWriter::writeString(path("source_config").string(), document.dump(2));
}
Pipeline::PipelineOrchestrator::Status pipelineStatus() {
    return DataCollection::runtimeDatasetServices()->pipeline->getStatus();
}
void startPipeline(Pipeline::PipelineMode mode) { DataCollection::runtimeDatasetServices()->pipeline->startPipeline(mode); }
void stopPipeline() { DataCollection::runtimeDatasetServices()->pipeline->stopPipeline(); }
std::shared_ptr<const DataCollection::ConceptGenerator> generatorAssets() {
    return DataCollection::runtimeDatasetServices()->creation->assets();
}
std::string generatorError() { return DataCollection::runtimeDatasetServices()->creation->assetError(); }
DataCollection::DatasetCreationService::Status generatorStatus() {
    return DataCollection::runtimeDatasetServices()->creation->status();
}
bool startGenerator(const DataCollection::DatasetCreationService::Request& request) {
    return DataCollection::runtimeDatasetServices()->creation->start(request);
}
void pauseGenerator() { DataCollection::runtimeDatasetServices()->creation->pause(); }
bool resumeGenerator() { return DataCollection::runtimeDatasetServices()->creation->resume(); }
size_t saveGeneratorQueue() {
    size_t added = 0;
    if (!DataCollection::runtimeDatasetServices()->creation->saveQueue(added))
        throw std::runtime_error("DataHub generator queue save failed: " + DataCollection::runtimeDatasetCatalog()->lastError());
    return added;
}
std::vector<std::filesystem::path> files(const std::filesystem::path& value, std::error_code& error) {
    return listFiles(value, error);
}
void collectFile(const std::filesystem::path& value, bool deleteAfter) {
    std::ifstream input(value, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read collection file: " + value.string());
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    if (input.bad() || text.empty()) throw std::runtime_error("Empty or unreadable collection file: " + value.string());
    const auto extension = value.extension().string();
    std::string type = "file";
    for (const auto* name : {"txt", "json", "jsonl", "md", "csv", "html", "htm"}) {
        if (extension != std::string(".") + name) continue;
        type = std::string(name) + "_file";
        if (extension == ".txt") type = "text_file";
        if (extension == ".md") type = "markdown_file";
        if (extension == ".htm") type = "html_file";
    }
    const auto target = DataCollection::runtimeDatasetCatalog();
    if (!target->appendStructuredEntry(text, "", type, value.string()))
        throw std::runtime_error("Collection append failed: " + target->lastError());
    input.close();
    if (deleteAfter && !std::filesystem::remove(value))
        throw std::runtime_error("Collected file could not be removed: " + value.string());
}
std::vector<std::string> modelDirectories() {
    std::vector<std::string> names;
    const auto root = path("model_store");
    if (std::filesystem::exists(root))
        for (const auto& entry : std::filesystem::directory_iterator(root))
            if (entry.is_directory()) names.push_back(entry.path().filename().string());
    return names;
}
std::vector<WorldLab::EnvironmentEntry> environments(bool discover) {
    const WorldLab::EnvironmentRegistry registry(std::filesystem::path(getResourcePath()) / "models" / "games");
    if (discover) registry.indexDirectory();
    return registry.entries();
}
void requireEnvironment(const std::string& id) {
    WorldLab::EnvironmentRegistry(std::filesystem::path(getResourcePath()) / "models" / "games").resolve(id);
}
std::string trainingPreview(const GRIM::ConceptBlock& block, bool conceptMode) {
    if (!conceptMode) return ConceptCanonical::renderPlainText(block);
    return ConceptCanonical::render(block, aiConfig.at("training").at("config").at("concept_spans")
        .get<NamedConceptSpanDefinitions>()).text;
}
}
