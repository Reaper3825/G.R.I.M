#include "DataCollection/dataset_stream.hpp"
#include "DataCollection/io/concept_block_io_flatbuffer.hpp"
#include "DataCollection/dataset_creation_service.hpp"
#include "DataCollection/pipeline/pipeline_context.hpp"
#include "bootstrap/dataset_bootstrap.hpp"
#include "logger.hpp"
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace fs = std::filesystem;
using GRIM::DataCollection::DatasetStream;
static_assert(!std::is_invocable_v<decltype(&DatasetTarget::saveConceptBlocks), const DatasetTarget*>);
static_assert(!std::is_invocable_v<decltype(&DatasetTarget::saveAssignments), const DatasetTarget*>);

// Host-only bootstrap fixture: configuration/root and logging, no app runtime.
nlohmann::json aiConfig;
std::string getGrimRootDir() { return fs::temp_directory_path().string(); }
void logPhaseInternal(const std::string&, const std::string&, bool) {}
void logError(const std::string&, const std::string&) {}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    output.close();
    if (!output) throw std::runtime_error("Fixture write failed");
}

class FixtureAdapter : public GRIM::DataCollection::IConceptBindingAdapter {
public:
    GRIM::DataCollection::GeneratorBindingSet fill(
        const GRIM::DataCollection::GeneratorDocument& document,
        const GRIM::DataCollection::GeneratorFrameDefinition&,
        const GRIM::DataCollection::GeneratorMemoryContext&,
        const GRIM::DataCollection::GeneratorAdapterOptions&) override {
        GRIM::DataCollection::GeneratorBindingSet result;
        result.documentId = document.id;
        result.question = "Fixture question";
        result.answer = document.text;
        return result;
    }
    bool isAvailable() const override { return true; }
    std::string adapterId() const override { return "fixture"; }
};

int main() {
    const auto root = fs::temp_directory_path() /
        ("grim-dataset-stream-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    try {
        DatasetStream stream(root / "models", root / "mass_dataset.jsonl",
                             std::chrono::milliseconds(15));
        require(stream.refresh(), "Empty catalog should load without a UI");
        auto empty = stream.latest();
        require(empty.revision == 1 && empty.catalog, "Initial revision missing");
        require(stream.refresh() && stream.latest().revision == 1, "Unchanged files republished");

        write(root / "mass_dataset.jsonl", "{\"id\":\"first\",\"content\":\"one\"}\n");
        require(stream.refresh(), "Initial dataset failed");
        auto pinned = stream.latest();
        require(pinned.catalog->massDatasetSize() == 1, "Missing sequence");

        write(root / "mass_dataset.jsonl", "{\"id\":\"second\",\"content\":\"new longer text\"}\n");
        require(stream.refresh(), "Hot reload failed");
        require(stream.latest().catalog->getSequence(0).id == "second", "Reload not visible");
        require(pinned.catalog->getSequence(0).id == "first", "Pinned snapshot mutated");

        auto valid = stream.latest();
        write(root / "mass_dataset.jsonl", "{\"id\":\"valid\"}\n{broken\n");
        require(!stream.refresh(), "Malformed dataset accepted");
        require(stream.latest().revision == valid.revision, "Bad update replaced catalog");
        require(!stream.lastError().empty(), "Failure has no diagnostic");
        write(root / "mass_dataset.jsonl", "{\"id\":42}\n");
        require(!stream.refresh(), "Invalid typed row silently dropped");

        write(root / "mass_dataset.jsonl", "{\"id\":\"recovered\",\"content\":\"fixed\"}\n");
        require(stream.refresh() && stream.lastError().empty(), "Stream did not recover");
        const auto goodRevision = stream.latest().revision;
        write(root / "curriculum_registry.json", "{broken registry");
        require(!stream.refresh(), "Invalid curriculum registry accepted");
        require(stream.latest().revision == goodRevision, "Registry failure replaced catalog");
        write(root / "curriculum_registry.json", "{\"curriculums\":[],\"courses\":[]}");
        require(stream.refresh(), "Registry recovery failed");

        uint64_t cursor = 0;
        DatasetStream::Snapshot next;
        require(stream.pullLatest(next, cursor), "First pull failed");
        require(!stream.pullLatest(next, cursor), "Same revision delivered twice");

        stream.start();
        write(root / "mass_dataset.jsonl", "{\"id\":\"background\",\"content\":\"worker reload\"}\n");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        bool observed = false;
        while (std::chrono::steady_clock::now() < deadline) {
            auto snapshot = stream.latest();
            if (snapshot.catalog->getSequence(0).id == "background") {
                observed = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        stream.stop();
        stream.stop();
        require(observed, "Worker did not reload without process restart");
        require(pinned.catalog->getSequence(0).id == "first", "Old snapshot invalid after stop");

        write(root / "concept_blocks.fb", "invalid flatbuffer");
        require(!stream.refresh(), "Malformed ConceptBlock catalog accepted");
        require(!fs::exists(root / "concept_blocks.jsonl"), "Reader wrote legacy data");

        fs::remove(root / "concept_blocks.fb");
        write(root / "concept_blocks.jsonl", "{\"id\":\"legacy\"}\n");
        require(stream.refresh(), "Read-only legacy load failed");
        require(!fs::exists(root / "concept_blocks.fb"), "Stream migrated legacy data");
        require(stream.latest().catalog->getConceptBlockById("legacy").id == "legacy",
                "Legacy catalog not available");
        fs::remove(root / "concept_blocks.jsonl");
        GRIM::ConceptBlock block;
        block.id = "stored";
        require(GRIM::ConceptBlockIO::saveFlatBuffer(root / "concept_blocks.fb", {block}),
                "FlatBuffer fixture failed");
        require(stream.refresh(), "Valid FlatBuffer load failed");
        require(stream.latest().catalog->getConceptBlockById("stored").id == "stored",
                "FlatBuffer catalog not available");
        write(root / "curriculum_registry.json",
              "{\"courses\":[{\"id\":\"c\",\"name\":\"course\",\"concept_block_ids\":[\"stored\",\"missing\"]}],"
              "\"curriculums\":[{\"id\":\"curriculum\",\"name\":\"Fixture curriculum\",\"course_ids\":[\"c\"]}]}");
        require(stream.refresh(), "A stale block reference hid the entire catalog");
        auto coldStream = std::make_shared<DatasetStream>(root / "models", root / "mass_dataset.jsonl");
        require(coldStream->refresh(), "Cold startup rejected browsable registry membership");
        GRIM::DataCollection::DatasetCatalog dropdownClient(coldStream, true);
        require(dropdownClient.getCurriculums().size() == 1 && dropdownClient.getCourses().size() == 1,
                "DataHub dropdown data disappeared because of a missing block");
        const auto registry = dropdownClient.curriculumRegistry();
        require(registry.at("curriculums").size() == 1 && registry.at("courses").size() == 1,
                "World Lab dropdown data disappeared because of a missing block");
        require(registry.at("courses").at(0).at("concept_block_ids").size() == 2,
                "Loading silently removed a stale reference");
        require(dropdownClient.conceptBlock("stored").at("source_id") == "stored",
                "An unrelated missing block hid a valid block");
        bool missingRejected = false;
        try { dropdownClient.conceptBlock("missing"); }
        catch (const std::runtime_error& error) {
            missingRejected = std::string(error.what()).find("missing") != std::string::npos;
        }
        require(missingRejected, "Exact lookup did not report the missing block");
        write(root / "curriculum_registry.json", "{\"courses\":[],\"curriculums\":[]}");

        aiConfig = {{"paths", {{"grim_text", {
            {"model_store", (root / "models").string()},
            {"training_data", root.string()}}}}}};
        bootstrapDatasetSubsystem();
        auto service = GRIM::DataCollection::runtimeDatasetStream();
        require(service->latest().catalog->getConceptBlockById("stored").id == "stored",
                "Bootstrap did not publish catalog before UI");
        bootstrapDatasetSubsystem();
        require(service == GRIM::DataCollection::runtimeDatasetStream(), "Bootstrap duplicated service");
        auto catalog = GRIM::DataCollection::runtimeDatasetCatalog();
        GRIM::ConceptBlockGoal importedGoal;
        importedGoal.target_state = "Fixture target";
        const nlohmann::json goalSelection = {{"source_id", "fixture-goal"},
            {"name", "Fixture Goal"}, {"goal", GRIM::goalToJson(importedGoal)}};
        catalog->importGoal(goalSelection);
        require(!catalog->goalSelections("Fixture Goal").empty(), "Goal selection required a panel");
        require(catalog->conceptBlock("fixture-goal").at("goal") == goalSelection.at("goal"),
                "Goal lookup changed payload");
        require(catalog->curriculumRegistry().contains("courses"), "Registry access required a panel");
        catalog->importGoal(goalSelection);
        auto conflicting = goalSelection;
        conflicting["goal"]["target_state"] = "Conflicting target";
        bool conflictRejected = false;
        try { catalog->importGoal(conflicting); }
        catch (const std::runtime_error&) { conflictRejected = true; }
        require(conflictRejected, "Conflicting Goal import overwrote existing content");
        const auto beforeEdit = service->latest();
        GRIM::ConceptBlock authored;
        authored.id = "headless-authoring";
        require(catalog->addConceptBlock(authored), "Headless catalog creation failed");
        require(service->latest().revision > beforeEdit.revision, "Edit not immediately published");
        require(beforeEdit.catalog->getConceptBlockById(authored.id).id.empty(), "Edit mutated pinned catalog");
        bool staleInvoked = false;
        require(!service->edit(beforeEdit.revision, [&](DatasetTarget&) {
            staleInvoked = true; return true;
        }) && !staleInvoked, "Stale editor was allowed to write");
        const auto current = service->latest();
        require(!service->edit(current.revision, [&](DatasetTarget& candidate) {
            candidate.setActiveModel("unpublished"); return false;
        }), "Failed mutation succeeded");
        require(service->latest().revision == current.revision, "Failed edit published a revision");
        require(catalog->appendStructuredEntry("service sequence", "first"), "Headless sequence append failed");
        const auto row = catalog->massDatasetSize() - 1;
        require(catalog->writeStructuredOutput(row, "second"), "Replacing existing mass dataset failed");
        require(catalog->getSequence(row).structured == "second", "Updated sequence not published");
        catalog->setActiveModel("fixture-model", "fixture-model");
        require(catalog->assignSequenceToModel(row), "First assignment save failed");
        require(catalog->removeSequenceFromModel(row), "Replacing existing assignments failed");
        const auto edited = service->latest();
        write(root / "mass_dataset.jsonl", "{bad update");
        require(!catalog->appendStructuredEntry("must not overwrite", ""), "Authoring overwrote invalid external update");
        require(service->latest().revision == edited.revision, "Invalid external data replaced catalog");
        write(root / "mass_dataset.jsonl", "{\"id\":\"external\"}\n");
        require(service->refresh(), "External repair failed");
        require(catalog->activeModelId() == "fixture-model", "Reload lost active model");
        GRIM::Pipeline::TaggedEntry pipelineEntry;
        pipelineEntry.id = "pipeline-appended";
        pipelineEntry.content = "pipeline content";
        const auto beforePipeline = service->latest();
        require(service->writeMassDataset(root / "mass_dataset.jsonl", {pipelineEntry}, false),
                "Service pipeline append failed");
        require(service->latest().revision > beforePipeline.revision && catalog->massDatasetSize() == 2,
                "Pipeline append lost data or was not published immediately");
        require(!service->writeMassDataset(root / "wrong-output.jsonl", {pipelineEntry}, false),
                "Mismatched pipeline output accepted");
        require(!fs::exists(root / "wrong-output.jsonl"), "Pipeline wrote outside configured catalog");
        require(service->writeMassDataset(root / "mass_dataset.jsonl", {pipelineEntry}, true)
            && catalog->massDatasetSize() == 1, "Pipeline replacement failed");
        GRIM::DataCollection::DatasetCatalog editor(service, true);
        const auto editorRevision = editor.viewRevision();
        pipelineEntry.id = "replacement-row";
        require(service->writeMassDataset(root / "mass_dataset.jsonl", {pipelineEntry}, true),
                "Concurrent editor fixture update failed");
        require(editor.viewRevision() == editorRevision && editor.getSequence(0).id == "pipeline-appended",
                "Editor view drifted during an operation");
        require(!editor.writeStructuredOutput(0, "stale edit"), "Stale editor changed a replacement row");
        editor.pinLatest();
        require(editor.getSequence(0).id == "replacement-row", "Editor could not advance its view");

        // The worker advances and retains its queue without any panel/tick calls.
        write(root / "LexiconPieces.json", R"({"groups":[{"id":"fixture","values":["word"]}],"frames":[{"id":"fixture","label":"Fixture"}]})");
        write(root / "GeneratorDataSource.json", R"([{"id":"fixture","text":"source text"}])");
        auto creator = std::make_shared<GRIM::DataCollection::DatasetCreationService>(
            catalog, std::make_shared<FixtureAdapter>(), GRIM::DataCollection::GeneratorAdapterOptions{}, root);
        GRIM::DataCollection::DatasetCreationService::Request request;
        request.attempts = 3;
        require(creator->start(request), "Headless generator did not start");
        const auto generationDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (creator->status().attempts < 3 && std::chrono::steady_clock::now() < generationDeadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        require(creator->status().generated == 3 && creator->status().queued == 3,
                "Generation depended on UI polling");
        size_t added = 0;
        require(creator->saveQueue(added) && added == 3, "Headless queue save failed");
        require(creator->status().queued == 0, "Saved queue was not cleared");
        const auto assets = creator->assets();
        write(root / "LexiconPieces.json", "{invalid");
        require(!creator->reloadAssets() && creator->assets() == assets, "Bad assets replaced good version");
        creator->shutdown();
        creator->shutdown();
        shutdownDatasetSubsystem();
        shutdownDatasetSubsystem();
        require(service->latest().catalog != nullptr, "Retained service invalid after shutdown");
        bool unavailable = false;
        try { GRIM::DataCollection::runtimeDatasetStream(); }
        catch (const std::runtime_error&) { unavailable = true; }
        require(unavailable, "Shutdown left service registered");
        fs::remove_all(root);
        std::cout << "Dataset stream tests passed\n";
    } catch (...) {
        shutdownDatasetSubsystem();
        fs::remove_all(root);
        throw;
    }
}
