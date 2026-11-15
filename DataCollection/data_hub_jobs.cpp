#include "data_hub_jobs.hpp"
#include "data_structurer.hpp"
#include "dataset_stream.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace GRIM::UI::DataHub {
void addDownloadedSource(const std::string& datasetId, const std::string& displayName);
}

#ifdef GRIM_DATA_HUB_JOBS_TEST
// Linked only by the focused host test; no runtime injection/provider callback.
namespace GRIM::DataCollection::DataHubJobs::TestBackend {
std::vector<std::string> structure(DataStructurer&, const std::string&,
                                  const std::string&, const std::string&, bool);
}
#endif

namespace GRIM::DataCollection::DataHubJobs {
namespace {
using json = nlohmann::json;
namespace fs = std::filesystem;
constexpr size_t maxQueries = 64;
constexpr size_t maxRequestBytes = 1024 * 1024;

std::string datasetName(const std::string& id) {
    const auto separator = id.find('/');
    const auto name = id.substr(separator == std::string::npos ? 0 : separator + 1);
    if (id.empty() || id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. /") != std::string::npos ||
        id.find(' ') != std::string::npos || name.empty() || name == "." || name == ".." ||
        name.find('/') != std::string::npos || separator == 0 ||
        id.substr(0, separator) == "." || id.substr(0, separator) == "..")
        throw std::runtime_error("Invalid Hugging Face dataset ID: " + id);
    return name;
}

std::vector<std::string> structure(DataStructurer& backend, const std::string& text,
                                  const std::string& mode, const std::string& prompt,
                                  bool batch) {
#ifdef GRIM_DATA_HUB_JOBS_TEST
    auto result = TestBackend::structure(backend, text, mode, prompt, batch);
#else
    auto result = batch ? backend.structureEntry(text) : backend.structureEntry(text, mode, prompt);
#endif
    if (result.empty() || std::any_of(result.begin(), result.end(),
                                     [](const auto& value) { return value.empty(); }))
        throw std::runtime_error("Structuring produced no usable output: " + backend.lastError());
    return result;
}

struct Query {
    QueryRequest request;
    QueryResult result;
};

struct Service {
    std::shared_ptr<HuggingFaceWebhook> hf;
    std::shared_ptr<DataStructurer> structurer;
    std::shared_ptr<DatasetStream> stream;
    fs::path stateDirectory, downloadDirectory;
    std::mutex mutex, persistenceMutex;
    std::condition_variable wake;
    bool stopping = false;
    std::map<uint64_t, Query> queries;
    std::deque<uint64_t> hfQueries, structureQueries;
    Downloads queue;
    std::string downloadToken;
    BatchStatus batch;
    DatasetStream::Snapshot batchSnapshot;
    bool batchPending = false;
    std::thread hfWorker, structureWorker;

    void requireRunning() const {
        if (stopping) throw std::runtime_error("DataHubJobs is shutting down");
    }
    auto findDownload(const std::string& id) {
        return std::find_if(queue.items.begin(), queue.items.end(),
                           [&](const auto& item) { return item.datasetId == id; });
    }
    void load() {
        const auto path = stateDirectory / "download_queue.json";
        if (!fs::exists(path)) return;
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot read download queue: " + path.string());
        json document;
        input >> document;
        if (!document.at("queue").is_array())
            throw std::runtime_error("Persisted download queue must be an array");
        std::set<std::string> ids;
        for (const auto& entry : document.at("queue")) {
            Download item;
            item.datasetId = entry.at("datasetId").get<std::string>();
            item.displayName = entry.at("displayName").get<std::string>();
            item.status = entry.at("status").get<std::string>();
            item.progress = entry.at("progress").get<float>();
            item.retryCount = entry.at("retryCount").get<int>();
            item.errorMessage = entry.at("errorMessage").get<std::string>();
            datasetName(item.datasetId);
            if (item.datasetId.empty() || item.displayName.empty() || !ids.insert(item.datasetId).second ||
                item.retryCount < 0 || !std::isfinite(item.progress) ||
                item.progress < 0 || item.progress > 1)
                throw std::runtime_error("Invalid persisted download queue entry");
            if (item.status == "downloading") {
                item.status = "pending";
                item.progress = 0;
            }
            if (item.status != "pending" && item.status != "failed" && item.status != "completed")
                throw std::runtime_error("Unknown persisted download status: " + item.status);
            queue.items.push_back(std::move(item));
        }
        ++queue.revision;
    }
    void persist(bool finalShutdownWrite = false) {
        // Serialize disk writers, then capture the newest state. Never hold the
        // reader mutex during filesystem operations or wait for IO in release().
        std::lock_guard<std::mutex> writer(persistenceMutex);
        json entries = json::array();
        {
            std::lock_guard<std::mutex> lock(mutex);
            // A caller may retain this service after shutdown has unpublished
            // it. It must never overwrite a replacement service's queue file.
            if (!finalShutdownWrite) requireRunning();
            for (const auto& item : queue.items)
                entries.push_back({{"datasetId", item.datasetId}, {"displayName", item.displayName},
                    {"status", item.status}, {"progress", item.progress},
                    {"retryCount", item.retryCount}, {"errorMessage", item.errorMessage}});
        }
        try {
            fs::create_directories(stateDirectory);
            const auto path = stateDirectory / "download_queue.json";
            auto staging = path;
            staging += ".writing";
            std::ofstream output(staging, std::ios::binary | std::ios::trunc);
            output << json{{"queue", std::move(entries)}}.dump(2) << '\n';
            output.close();
            if (!output) throw std::runtime_error("Cannot write download queue: " + staging.string());
#ifdef _WIN32
            if (!MoveFileExW(staging.c_str(), path.c_str(),
                             MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("Cannot replace download queue: Windows error " +
                                         std::to_string(GetLastError()));
#else
            fs::rename(staging, path);
#endif
        } catch (const std::exception& error) {
            std::lock_guard<std::mutex> lock(mutex);
            queue.error = error.what();
            queue.processing = false;
            downloadToken.clear();
            ++queue.revision;
            throw;
        }
    }
    bool persistWorker() {
        try { persist(); return true; }
        catch (const std::exception&) {
            // Disk failures are recorded and pause processing. During host
            // shutdown the final write instead owns persistence after draining.
            return false;
        }
    }
    void executeQuery(uint64_t id, QueryRequest request) {
        QueryResult result;
        try {
            if (request.kind == QueryKind::Structure) {
                result.structured = structure(*structurer, request.query, request.mode, request.prompt, false);
            } else {
                hf->setApiToken(request.token);
                if (request.kind == QueryKind::Search)
                    result.datasets = hf->searchDatasets(request.query, 20, request.filter);
                else
                    result.preview = hf->getDatasetPreviewSample(request.query);
                if (hf->hasError()) throw std::runtime_error(hf->getLastError());
            }
        } catch (const std::exception& error) {
            result.error = error.what();
        } catch (...) {
            result.error = "Unknown backend exception";
        }
        result.ready = true;
        std::lock_guard<std::mutex> lock(mutex);
        auto found = queries.find(id);
        if (found != queries.end()) found->second.result = std::move(result);
    }
    void executeDownload(Download item, const std::string& token) {
        std::string error;
        try {
            hf->setApiToken(token);
            const auto directory = downloadDirectory / datasetName(item.datasetId);
            fs::create_directories(directory);
            if (!hf->downloadDataset(item.datasetId, directory.string(), "train", "",
                [this, id = item.datasetId](size_t done, size_t total, const std::string&) {
                    std::lock_guard<std::mutex> lock(mutex);
                    const auto found = findDownload(id);
                    if (found == queue.items.end()) return; // Explicitly discarded queue entry.
                    if (total != 0)
                        found->progress = std::min(1.0f, static_cast<float>(done) / total);
                    ++queue.revision;
                }))
                throw std::runtime_error("Download failed for " + item.datasetId + ": " + hf->getLastError());
            GRIM::UI::DataHub::addDownloadedSource(item.datasetId, item.displayName);
        } catch (const std::exception& failure) {
            error = failure.what();
        } catch (...) {
            error = "Unknown download backend exception";
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto found = findDownload(item.datasetId);
            if (found != queue.items.end()) {
                found->errorMessage = error;
                if (error.empty()) { found->status = "completed"; found->progress = 1; }
                else { found->status = "failed"; }
            }
            queue.activeId.clear();
            ++queue.revision;
        }
        persistWorker();
    }
    void runHF() {
        for (;;) {
            uint64_t id = 0;
            QueryRequest request;
            Download item;
            std::string token;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || !hfQueries.empty() || queue.processing; });
                if (stopping) return;
                if (!hfQueries.empty()) {
                    id = hfQueries.front();
                    hfQueries.pop_front();
                    auto found = queries.find(id);
                    if (found == queries.end()) continue;
                    request = std::move(found->second.request);
                } else {
                    auto found = std::find_if(queue.items.begin(), queue.items.end(),
                                             [](const auto& value) { return value.status == "pending"; });
                    if (found == queue.items.end()) {
                        queue.processing = false;
                        downloadToken.clear();
                        ++queue.revision;
                        continue;
                    }
                    found->status = "downloading";
                    found->progress = 0;
                    queue.activeId = found->datasetId;
                    item = *found;
                    token = downloadToken;
                    ++queue.revision;
                }
            }
            if (id != 0) executeQuery(id, std::move(request));
            else {
                const bool saved = persistWorker();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!saved || stopping || !queue.processing) {
                        const auto found = findDownload(item.datasetId);
                        if (found != queue.items.end()) {
                            found->status = "pending";
                            found->progress = 0;
                        }
                        queue.activeId.clear();
                        ++queue.revision;
                        continue;
                    }
                }
                executeDownload(std::move(item), token);
            }
        }
    }
    void executeBatch(DatasetStream::Snapshot snapshot) {
        uint64_t expectedRevision = snapshot.revision;
        for (size_t i = 0; i < snapshot.catalog->massDatasetSize(); ++i) {
            const auto sequence = snapshot.catalog->getSequence(i);
            if (sequence.is_structured || sequence.content.empty()) continue;
            std::string error;
            try {
                const auto parts = structure(*structurer, sequence.content, "", "", true);
                std::string output;
                for (const auto& part : parts) {
                    if (!output.empty()) output += "\n\n";
                    output += part;
                }
                if (!stream->edit(expectedRevision, [&](DatasetTarget& target) {
                    if (i >= target.massDatasetSize() || target.getSequence(i).id != sequence.id)
                        throw std::runtime_error("Batch dataset identity changed: " + sequence.id);
                    return target.writeStructuredOutput(i, output);
                }))
                    throw std::runtime_error("Batch write rejected for " + sequence.id + ": " + stream->lastError());
                ++expectedRevision;
            } catch (const std::exception& failure) {
                error = failure.what();
            } catch (...) {
                error = "Unknown structuring backend exception";
            }
            std::lock_guard<std::mutex> lock(mutex);
            ++batch.completed;
            if (!error.empty()) { ++batch.failed; batch.message = error; }
            ++batch.revision;
        }
        std::lock_guard<std::mutex> lock(mutex);
        batch.running = false;
        if (batch.failed == 0) batch.message = "Batch completed";
        ++batch.revision;
    }
    void runStructure() {
        for (;;) {
            uint64_t id = 0;
            QueryRequest request;
            DatasetStream::Snapshot snapshot;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopping || batchPending || !structureQueries.empty(); });
                if (stopping) return;
                if (batchPending) {
                    snapshot = std::move(batchSnapshot);
                    batchPending = false;
                } else {
                    id = structureQueries.front();
                    structureQueries.pop_front();
                    auto found = queries.find(id);
                    if (found == queries.end()) continue;
                    request = std::move(found->second.request);
                }
            }
            if (id != 0) executeQuery(id, std::move(request));
            else executeBatch(std::move(snapshot));
        }
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            queries.clear();
            hfQueries.clear();
            structureQueries.clear();
            if (batchPending) {
                batchPending = false;
                batchSnapshot = {};
                batch.running = false;
                batch.message = "Host shutdown before batch started";
                ++batch.revision;
            }
        }
        wake.notify_all();
        if (hfWorker.joinable()) hfWorker.join();
        if (structureWorker.joinable()) structureWorker.join();
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.processing = false;
            downloadToken.clear();
            ++queue.revision;
        }
        persist(true);
    }
};

std::mutex serviceMutex, lifecycleMutex;
std::shared_ptr<Service> service;
uint64_t nextQueryId = 0; // Guarded by serviceMutex; survives host service restart.

std::shared_ptr<Service> requireService() {
    std::lock_guard<std::mutex> lock(serviceMutex);
    if (!service) throw std::runtime_error("DataHubJobs host service is not running");
    return service;
}

template<class Mutation>
void mutateDownloads(Mutation operation) {
    const auto owner = requireService();
    {
        std::lock_guard<std::mutex> lock(owner->mutex);
        owner->requireRunning();
        operation(*owner);
        ++owner->queue.revision;
    }
    owner->persist();
    owner->wake.notify_all();
}
}

void start(std::shared_ptr<HuggingFaceWebhook> hf, std::shared_ptr<DataStructurer> structurer,
           std::shared_ptr<DatasetStream> stream, fs::path stateDirectory, fs::path downloadDirectory) {
    std::lock_guard<std::mutex> lifecycle(lifecycleMutex);
    if (!hf || !structurer || !stream || stateDirectory.empty() || downloadDirectory.empty())
        throw std::runtime_error("DataHubJobs requires all host dependencies and directories");
    {
        std::lock_guard<std::mutex> lock(serviceMutex);
        if (service) throw std::runtime_error("DataHubJobs already started");
    }
    auto owner = std::make_shared<Service>();
    owner->hf = std::move(hf);
    owner->structurer = std::move(structurer);
    owner->stream = std::move(stream);
    owner->stateDirectory = std::move(stateDirectory);
    owner->downloadDirectory = std::move(downloadDirectory);
    owner->load();
    try {
        owner->hfWorker = std::thread([ptr = owner.get()] { ptr->runHF(); });
        owner->structureWorker = std::thread([ptr = owner.get()] { ptr->runStructure(); });
    } catch (...) {
        owner->stop();
        throw;
    }
    std::lock_guard<std::mutex> lock(serviceMutex);
    service = std::move(owner);
}

void shutdown() {
    std::lock_guard<std::mutex> lifecycle(lifecycleMutex);
    std::shared_ptr<Service> owner;
    {
        std::lock_guard<std::mutex> lock(serviceMutex);
        owner = std::move(service);
    }
    if (!owner) return; // Host teardown is idempotent.
    owner->stop();
}

uint64_t submit(QueryRequest request) {
    if (request.kind != QueryKind::Search && request.kind != QueryKind::Preview &&
        request.kind != QueryKind::Structure)
        throw std::runtime_error("Unknown DataHubJobs query kind");
    if (request.query.size() + request.filter.size() + request.token.size() +
        request.mode.size() + request.prompt.size() > maxRequestBytes)
        throw std::runtime_error("DataHubJobs query exceeds 1 MiB");
    if (request.kind != QueryKind::Search && request.query.empty())
        throw std::runtime_error("DataHubJobs query text is required");
    if (request.kind == QueryKind::Structure && request.mode != "qa" && request.mode != "thought" &&
        request.mode != "conversation" && request.mode != "instruct" && request.mode != "concept_block")
        throw std::runtime_error("Unsupported structuring mode: " + request.mode);
    std::lock_guard<std::mutex> global(serviceMutex);
    if (!service) throw std::runtime_error("DataHubJobs host service is not running");
    std::lock_guard<std::mutex> lock(service->mutex);
    service->requireRunning();
    if (service->queries.size() >= maxQueries)
        throw std::runtime_error("DataHubJobs query limit reached; release completed requests");
    if (nextQueryId == std::numeric_limits<uint64_t>::max())
        throw std::runtime_error("DataHubJobs query ID exhausted");
    const auto id = ++nextQueryId;
    const auto kind = request.kind;
    service->queries.emplace(id, Query{std::move(request), {}});
    if (kind == QueryKind::Structure) service->structureQueries.push_back(id);
    else service->hfQueries.push_back(id);
    service->wake.notify_all();
    return id;
}

QueryResult poll(uint64_t id) {
    const auto owner = requireService();
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->requireRunning();
    const auto found = owner->queries.find(id);
    if (found == owner->queries.end()) throw std::runtime_error("Unknown/released DataHubJobs query ID");
    return found->second.result;
}

void release(uint64_t id) {
    const auto owner = requireService();
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->requireRunning();
    if (owner->queries.erase(id) == 0) throw std::runtime_error("Unknown/released DataHubJobs query ID");
    std::erase(owner->hfQueries, id);
    std::erase(owner->structureQueries, id);
}

Downloads downloads() {
    const auto owner = requireService();
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->requireRunning();
    return owner->queue;
}
void enqueueDownload(const std::string& id, const std::string& name) {
    if (id.empty() || name.empty()) throw std::runtime_error("Download ID and display name are required");
    datasetName(id);
    mutateDownloads([&](Service& owner) {
        if (owner.queue.activeId == id || owner.findDownload(id) != owner.queue.items.end())
            throw std::runtime_error("Dataset is already queued or active: " + id);
        owner.queue.items.push_back({id, name, "pending"});
    });
}
void processDownloads(const std::string& token) {
    mutateDownloads([&](Service& owner) {
        if (owner.queue.processing) throw std::runtime_error("Download processing is already active");
        owner.downloadToken = token;
        owner.queue.error.clear();
        owner.queue.processing = true;
    });
}
void removeDownload(const std::string& id) {
    mutateDownloads([&](Service& owner) {
        const auto found = owner.findDownload(id);
        if (found == owner.queue.items.end()) throw std::runtime_error("Unknown download ID: " + id);
        owner.queue.items.erase(found);
    });
}
void retryDownload(const std::string& id) {
    mutateDownloads([&](Service& owner) {
        const auto found = owner.findDownload(id);
        if (found == owner.queue.items.end()) throw std::runtime_error("Unknown download ID: " + id);
        if (found->status != "failed" || owner.queue.activeId == id)
            throw std::runtime_error("Only failed, inactive downloads can be retried: " + id);
        if (found->retryCount == std::numeric_limits<int>::max())
            throw std::runtime_error("Download retry count exhausted");
        ++found->retryCount;
        found->status = "pending";
        found->progress = 0;
        found->errorMessage.clear();
    });
}
void clearDownloads() {
    mutateDownloads([](Service& owner) { owner.queue.items.clear(); });
}
void clearCompletedDownloads() {
    mutateDownloads([](Service& owner) {
        std::erase_if(owner.queue.items, [](const auto& item) { return item.status == "completed"; });
    });
}
void startBatch() {
    const auto owner = requireService();
    auto snapshot = owner->stream->latest();
    if (!snapshot.catalog) throw std::runtime_error("Batch requires a published dataset snapshot");
    size_t total = 0;
    std::set<std::string> ids;
    for (size_t i = 0; i < snapshot.catalog->massDatasetSize(); ++i) {
        const auto entry = snapshot.catalog->getSequence(i);
        if (entry.id.empty() || !ids.insert(entry.id).second)
            throw std::runtime_error("Batch requires unique stable dataset IDs");
        if (!entry.is_structured && !entry.content.empty()) ++total;
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->requireRunning();
    if (owner->batch.running) throw std::runtime_error("A structuring batch is already running");
    const auto revision = owner->batch.revision + 1;
    owner->batch = {true, revision, 0, total, 0, "Batch queued"};
    owner->batchSnapshot = std::move(snapshot);
    owner->batchPending = true;
    owner->wake.notify_all();
}
BatchStatus batchStatus() {
    const auto owner = requireService();
    std::lock_guard<std::mutex> lock(owner->mutex);
    owner->requireRunning();
    return owner->batch;
}
}
