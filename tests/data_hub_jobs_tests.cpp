#include "DataCollection/data_hub_jobs.hpp"
#include "DataCollection/data_structurer.hpp"
#include "DataCollection/dataset_stream.hpp"

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace fs = std::filesystem;
namespace Jobs = GRIM::DataCollection::DataHubJobs;
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Operation>
void rejects(Operation operation, const char* message) {
    bool rejected = false;
    try { operation(); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, message);
}
template<class Predicate>
void await(Predicate predicate, const char* message) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline, message);
        std::this_thread::sleep_for(2ms);
    }
}
void write(const fs::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << text;
    output.close();
    require(static_cast<bool>(output), "Fixture write failed");
}
struct Gate {
    std::mutex mutex;
    std::condition_variable wake;
    std::string key;
    bool closed = false, entered = false;
    void arm(const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex);
        key = value;
        closed = true;
        entered = false;
    }
    void hit(const std::string& value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (value != key || !closed) return;
        entered = true;
        wake.notify_all();
        require(wake.wait_for(lock, 8s, [&] { return !closed; }), "Backend gate timed out");
    }
    void waitEntered() {
        std::unique_lock<std::mutex> lock(mutex);
        require(wake.wait_for(lock, 5s, [&] { return entered; }), "Backend never entered gate");
    }
    void open() {
        std::lock_guard<std::mutex> lock(mutex);
        closed = false;
        wake.notify_all();
    }
} hfGate, structureGate;

std::atomic<int> hfActive{0}, structureActive{0};
std::atomic<bool> hfOverlap{false}, structureOverlap{false}, failDownload{false};
struct Active {
    std::atomic<int>& count;
    Active(std::atomic<int>& value, std::atomic<bool>& overlap) : count(value) {
        if (++count != 1) overlap = true;
    }
    ~Active() { --count; }
};
std::mutex sourceMutex;
std::vector<std::string> sources, downloadPaths;

// Stub only transport-facing methods. The service, workers, persistence,
// DatasetStream revision checks, and DatasetTarget writes are production code.
namespace GRIM::DataCollection {
struct HuggingFaceWebhook::Impl { std::string token, error; };
HuggingFaceWebhook::HuggingFaceWebhook() : pImpl(std::make_unique<Impl>()) {}
HuggingFaceWebhook::~HuggingFaceWebhook() = default;
void HuggingFaceWebhook::setApiToken(const std::string& token) {
    if (hfActive != 0) hfOverlap = true;
    pImpl->token = token;
}
bool HuggingFaceWebhook::hasError() const { return !pImpl->error.empty(); }
std::string HuggingFaceWebhook::getLastError() const { return pImpl->error; }
std::vector<HFDatasetInfo> HuggingFaceWebhook::searchDatasets(
    const std::string& query, int, const std::string& filter) {
    Active active(hfActive, hfOverlap);
    const auto token = pImpl->token;
    pImpl->error.clear();
    hfGate.hit(query);
    require(pImpl->token == token, "Search auth changed during request");
    if (query == "error") { pImpl->error = "Search transport error"; return {}; }
    HFDatasetInfo info;
    info.id = query;
    info.description = token + ":" + filter;
    return {info};
}
std::string HuggingFaceWebhook::getDatasetPreviewSample(const std::string& id, int, size_t) {
    Active active(hfActive, hfOverlap);
    pImpl->error.clear();
    hfGate.hit(id);
    return "preview:" + id + ":" + pImpl->token;
}
bool HuggingFaceWebhook::downloadDataset(const std::string& id, const std::string& directory,
    const std::string&, const std::string&, ProgressCallback progress) {
    Active active(hfActive, hfOverlap);
    const auto token = pImpl->token;
    pImpl->error.clear();
    progress(1, 4, "Started");
    hfGate.hit(id);
    require(token == pImpl->token, "Download auth changed during transfer");
    progress(3, 4, "Progress after queue mutation");
    {
        std::lock_guard<std::mutex> lock(sourceMutex);
        downloadPaths.push_back(directory);
    }
    if (failDownload) { pImpl->error = "Download transport error"; return false; }
    return true;
}
namespace DataHubJobs::TestBackend {
std::vector<std::string> structure(DataStructurer&, const std::string& text,
    const std::string& mode, const std::string& prompt, bool batch) {
    Active active(structureActive, structureOverlap);
    structureGate.hit(text);
    if (text == "fail") return {};
    if (text == "throw") throw std::runtime_error("Structurer backend failed");
    if (!batch) require(mode == "qa" && prompt == "fixture prompt", "Query parameters not copied");
    return {"Q: " + text, "A: structured"};
}
}
}
namespace GRIM::UI::DataHub {
void addDownloadedSource(const std::string& id, const std::string& name) {
    std::lock_guard<std::mutex> lock(sourceMutex);
    sources.push_back(id + ":" + name);
}
}

Jobs::QueryRequest search(std::string value) {
    return {Jobs::QueryKind::Search, std::move(value), "filter", "search-token", "", ""};
}
Jobs::QueryRequest structureRequest(std::string value) {
    return {Jobs::QueryKind::Structure, std::move(value), "", "", "qa", "fixture prompt"};
}
Jobs::QueryResult result(uint64_t id) {
    await([&] { return Jobs::poll(id).ready; }, "Query did not complete");
    return Jobs::poll(id);
}

int main() {
    const auto root = fs::current_path() / ("data-hub-jobs-fixture-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    try {
        auto hf = std::make_shared<GRIM::DataCollection::HuggingFaceWebhook>();
        GRIM::DataCollection::DataStructuringConfig config;
        config.ollama_model = "fixture";
        auto structurer = std::make_shared<GRIM::DataCollection::DataStructurer>(config);
        auto stream = std::make_shared<GRIM::DataCollection::DatasetStream>(
            root / "models", root / "mass_dataset.jsonl");
        require(stream->refresh(), "Dataset fixture initialization failed");
        auto start = [&] { Jobs::start(hf, structurer, stream, root / "state", root / "downloads"); };
        rejects([] { Jobs::downloads(); }, "Missing service must fail");
        rejects([&] { Jobs::start({}, structurer, stream, root, root); }, "Missing dependency accepted");
        start();
        rejects(start, "Duplicate host start accepted");

        hfGate.arm("blocked-search");
        const auto discarded = Jobs::submit(search("blocked-search"));
        hfGate.waitEntered();
        const auto beforeRelease = std::chrono::steady_clock::now();
        Jobs::release(discarded);
        require(std::chrono::steady_clock::now() - beforeRelease < 200ms, "Release waited for backend");
        rejects([&] { Jobs::poll(discarded); }, "Released ID retained");
        std::vector<uint64_t> retained;
        for (int i = 0; i != 64; ++i) retained.push_back(Jobs::submit(search("queued")));
        rejects([] { Jobs::submit(search("overflow")); }, "Transient query bound missing");
        require(!Jobs::poll(retained.front()).ready, "HF lane allowed concurrent query");
        for (auto id : retained) Jobs::release(id);
        // Repeated queued cancellation must free both result slots and lane entries.
        for (int i = 0; i != 128; ++i) Jobs::release(Jobs::submit(search("cancelled")));
        hfGate.open();
        const auto next = Jobs::submit(search("fresh"));
        require(next > retained.back(), "Query IDs were reused");
        auto copied = result(next);
        require(copied.datasets.at(0).description == "search-token:filter", "Search payload changed");
        copied.datasets.clear();
        require(Jobs::poll(next).datasets.size() == 1, "Poll returned aliased status");
        Jobs::release(next);
        rejects([&] { Jobs::poll(discarded); }, "Released result resurrected on backend completion");
        const auto error = Jobs::submit(search("error"));
        require(!result(error).error.empty(), "Search failure was success shaped");
        Jobs::release(error);

        structureGate.arm("blocked-structure");
        const auto structure = Jobs::submit(structureRequest("blocked-structure"));
        structureGate.waitEntered();
        const auto preview = Jobs::submit({Jobs::QueryKind::Preview, "org/preview", "", "preview-token"});
        require(result(preview).preview == "preview:org/preview:preview-token",
                "Structuring blocked independent HF lane");
        Jobs::release(preview);
        const auto queuedStructure = Jobs::submit(structureRequest("second"));
        require(!Jobs::poll(queuedStructure).ready, "Structurer lane overlapped");
        Jobs::release(structure);
        structureGate.open();
        require(result(queuedStructure).structured.size() == 2, "Structure query did not complete");
        Jobs::release(queuedStructure);
        const auto failedStructure = Jobs::submit(structureRequest("fail"));
        require(!result(failedStructure).error.empty() && result(failedStructure).structured.empty(),
                "Failed structuring produced successful fallback");
        Jobs::release(failedStructure);

        Jobs::enqueueDownload("org/active", "Active");
        Jobs::enqueueDownload("org/remove", "Remove");
        Jobs::enqueueDownload("org/remaining", "Remaining");
        hfGate.arm("org/active");
        Jobs::processDownloads("download-token");
        hfGate.waitEntered();
        require(Jobs::downloads().activeId == "org/active", "Active ID missing");
        const auto providerQuery = Jobs::submit(search("provider-retiring"));
        Jobs::release(providerQuery);
        require(Jobs::downloads().processing && Jobs::downloads().activeId == "org/active",
                "Provider query release cancelled a surviving download");
        rejects([] { Jobs::processDownloads(""); }, "Duplicate processing accepted");
        Jobs::removeDownload("org/remove");
        Jobs::removeDownload("org/active");
        rejects([] { Jobs::enqueueDownload("org/active", "Duplicate"); },
                "Removed active ID was reused before completion");
        require(Jobs::downloads().items.at(0).progress == 0, "Shifted queue entry inherited active progress");
        hfGate.open();
        await([] { return !Jobs::downloads().processing; }, "Downloads did not continue without UI");
        auto downloaded = Jobs::downloads();
        require(downloaded.items.size() == 1 && downloaded.items[0].datasetId == "org/remaining" &&
                downloaded.items[0].status == "completed", "ID-based queue mutation failed");
        {
            std::lock_guard<std::mutex> lock(sourceMutex);
            require(sources.size() == 2 && sources[0] == "org/active:Active" &&
                    sources[1] == "org/remaining:Remaining", "Host source integration failed");
            require(fs::path(downloadPaths[0]).filename() == "active", "Dataset output path changed");
        }
        Jobs::clearCompletedDownloads();
        require(Jobs::downloads().items.empty(), "Completed queue clearing failed");
        failDownload = true;
        Jobs::enqueueDownload("org/retry", "Retry");
        Jobs::processDownloads("");
        await([] { return !Jobs::downloads().processing; }, "Failed download did not finish");
        require(Jobs::downloads().items[0].status == "failed" &&
                !Jobs::downloads().items[0].errorMessage.empty(), "Download error missing");
        failDownload = false;
        Jobs::retryDownload("org/retry");
        require(Jobs::downloads().items[0].retryCount == 1, "Retry count missing");
        Jobs::processDownloads("");
        await([] { return !Jobs::downloads().processing; }, "Retry did not finish");
        require(Jobs::downloads().items[0].status == "completed", "Retry failed");
        Jobs::clearDownloads();

        // A rejected start must be retryable after repairing queue storage.
        // Hold the HF lane so this checks the caller's persistence failure.
        Jobs::enqueueDownload("org/storage", "Storage");
        hfGate.arm("storage-blocker");
        const auto storageBlocker = Jobs::submit(search("storage-blocker"));
        hfGate.waitEntered();
        const auto staging = root / "state" / "download_queue.json.writing";
        fs::create_directory(staging);
        rejects([] { Jobs::processDownloads("storage-token"); }, "Queue persistence failure was hidden");
        require(!Jobs::downloads().processing && !Jobs::downloads().error.empty() &&
                Jobs::downloads().items[0].status == "pending",
                "Failed queue save left processing stuck or consumed pending work");
        fs::remove(staging);
        Jobs::processDownloads("storage-token");
        require(Jobs::downloads().error.empty(), "Explicit retry did not clear persistence error");
        Jobs::release(storageBlocker);
        hfGate.open();
        await([] { return !Jobs::downloads().processing; }, "Storage-repaired queue did not finish");
        require(Jobs::downloads().items[0].status == "completed", "Storage retry failed");
        Jobs::clearDownloads();

        // A worker-side save failure must not start the next network transfer.
        Jobs::enqueueDownload("org/worker-storage", "Worker storage");
        hfGate.arm("worker-storage-blocker");
        const auto workerBlocker = Jobs::submit(search("worker-storage-blocker"));
        hfGate.waitEntered();
        Jobs::processDownloads("");
        fs::create_directory(staging);
        size_t transfersBefore;
        {
            std::lock_guard<std::mutex> lock(sourceMutex);
            transfersBefore = downloadPaths.size();
        }
        Jobs::release(workerBlocker);
        hfGate.open();
        await([] {
            const auto status = Jobs::downloads();
            return !status.processing && status.activeId.empty();
        }, "Worker persistence failure did not settle");
        require(!Jobs::downloads().error.empty() && Jobs::downloads().items[0].status == "pending",
                "Worker persistence failure consumed pending download");
        {
            std::lock_guard<std::mutex> lock(sourceMutex);
            require(downloadPaths.size() == transfersBefore, "Transfer started after queue save failed");
        }
        fs::remove(staging);
        Jobs::clearDownloads();

        write(root / "mass_dataset.jsonl",
              "{\"id\":\"alpha\",\"content\":\"alpha\"}\n{\"id\":\"beta\",\"content\":\"beta\"}\n");
        require(stream->refresh(), "Batch dataset failed to load");
        structureGate.arm("alpha");
        Jobs::startBatch();
        structureGate.waitEntered();
        rejects([] { Jobs::startBatch(); }, "Overlapping batch accepted");
        write(root / "mass_dataset.jsonl",
              "{\"id\":\"replacement\",\"content\":\"new content\"}\n{\"id\":\"beta\",\"content\":\"beta\"}\n");
        require(stream->refresh(), "Concurrent dataset update failed");
        structureGate.open();
        await([] { return !Jobs::batchStatus().running; }, "Conflicting batch did not complete");
        auto batch = Jobs::batchStatus();
        require(batch.total == 2 && batch.completed == 2 && batch.failed == 2 &&
                batch.message.find("rejected") != std::string::npos, "Revision conflict was not surfaced");
        require(!stream->latest().catalog->getSequence(0).is_structured &&
                !stream->latest().catalog->getSequence(1).is_structured, "Batch wrote to shifted indices");

        write(root / "mass_dataset.jsonl",
              "{\"id\":\"good\",\"content\":\"good\"}\n{\"id\":\"bad\",\"content\":\"fail\"}\n"
              "{\"id\":\"last\",\"content\":\"last\"}\n");
        require(stream->refresh(), "Successful batch dataset failed to load");
        structureGate.arm("good");
        Jobs::startBatch();
        structureGate.waitEntered();
        const auto afterBatch = Jobs::submit(structureRequest("after batch"));
        require(!Jobs::poll(afterBatch).ready, "Query overlapped batch structurer");
        structureGate.open();
        await([] { return !Jobs::batchStatus().running; }, "Batch did not continue without provider");
        batch = Jobs::batchStatus();
        require(batch.total == 3 && batch.completed == 3 && batch.failed == 1,
                "Batch failed-entry accounting incorrect");
        auto catalog = stream->latest().catalog;
        require(catalog->getSequence(0).is_structured && !catalog->getSequence(1).is_structured &&
                catalog->getSequence(2).is_structured, "Batch marked failed output successful");
        require(catalog->getSequence(0).structured == "Q: good\n\nA: structured", "Batch output lost");
        require(result(afterBatch).error.empty(), "Serialized query after batch failed");
        Jobs::release(afterBatch);

        Jobs::enqueueDownload("org/shutdown", "Shutdown");
        Jobs::enqueueDownload("org/pending", "Pending");
        hfGate.arm("org/shutdown");
        Jobs::processDownloads("not-persisted-secret");
        hfGate.waitEntered();
        auto shutdown = std::async(std::launch::async, [] { Jobs::shutdown(); });
        await([] {
            try { Jobs::downloads(); return false; }
            catch (const std::runtime_error&) { return true; }
        }, "Shutdown did not promptly reject new work");
        require(shutdown.wait_for(20ms) == std::future_status::timeout, "Shutdown did not drain active backend");
        hfGate.open();
        shutdown.get();
        std::ifstream saved(root / "state" / "download_queue.json");
        nlohmann::json state;
        saved >> state;
        saved.close();
        require(state.at("queue").size() == 2, "Queue was not persisted");
        require(state.at("queue")[1].at("status") == "pending", "Shutdown consumed pending download");
        require(state.dump().find("not-persisted-secret") == std::string::npos, "Auth token persisted");
        state["queue"][0]["status"] = "downloading";
        state["queue"][0]["progress"] = 0.75;
        write(root / "state" / "download_queue.json", state.dump());
        start();
        require(!Jobs::downloads().processing && Jobs::downloads().items[0].status == "pending" &&
                Jobs::downloads().items[0].progress == 0, "Interrupted download was not restored pending");
        const auto restarted = Jobs::submit(search("restart"));
        require(restarted > afterBatch, "Host restart reused query IDs");
        require(result(restarted).error.empty(), "Restarted service not responsive");
        Jobs::release(restarted);
        Jobs::shutdown();
        Jobs::shutdown();
        write(root / "state" / "download_queue.json", "{\"queue\":null}");
        rejects(start, "Malformed persisted queue was silently treated as empty");
        require(!hfOverlap && !structureOverlap, "Shared backend use overlapped");
        fs::remove_all(root);
        std::cout << "DataHubJobs tests passed: async release, bounds, lanes, downloads, batch conflicts, persistence, shutdown\n";
        return 0;
    } catch (const std::exception& error) {
        hfGate.open();
        structureGate.open();
        std::cerr << error.what() << '\n';
        try { Jobs::shutdown(); }
        catch (const std::exception& failure) { std::cerr << "Shutdown: " << failure.what() << '\n'; }
        fs::remove_all(root);
        return 1;
    }
}
