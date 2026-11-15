#pragma once

#include "core/host_exports.hpp"
#include "huggingface_webhook.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace GRIM::DataCollection {
class DataStructurer;
class DatasetStream;

namespace DataHubJobs {
enum class QueryKind { Search, Preview, Structure };
struct QueryRequest {
    QueryKind kind = QueryKind::Search;
    std::string query, filter, token, mode, prompt;
};
struct QueryResult {
    bool ready = false;
    std::string error;
    std::vector<HFDatasetInfo> datasets;
    std::string preview;
    std::vector<std::string> structured;
};

// At most 64 unreleased requests. IDs never repeat during the host lifetime.
// Unknown/released IDs throw. Release discards queued work or an in-flight
// result without waiting for the backend; no provider code is retained.
GRIM_HOST_API uint64_t submit(QueryRequest request);
GRIM_HOST_API QueryResult poll(uint64_t id);
GRIM_HOST_API void release(uint64_t id);

struct Download {
    std::string datasetId, displayName, status;
    float progress = 0.0f;
    int retryCount = 0;
    std::string errorMessage;
};
struct Downloads {
    uint64_t revision = 0;
    bool processing = false;
    std::string activeId, error;
    std::vector<Download> items;
};
GRIM_HOST_API Downloads downloads();
GRIM_HOST_API void enqueueDownload(const std::string& id, const std::string& name);
GRIM_HOST_API void processDownloads(const std::string& token);
// Removing an active item hides it but does not cancel its host-owned transfer.
GRIM_HOST_API void removeDownload(const std::string& id);
GRIM_HOST_API void retryDownload(const std::string& id);
GRIM_HOST_API void clearDownloads();
GRIM_HOST_API void clearCompletedDownloads();

struct BatchStatus {
    bool running = false;
    uint64_t revision = 0;
    size_t completed = 0, total = 0, failed = 0;
    std::string message;
};
// completed counts attempted entries, including failed entries.
GRIM_HOST_API void startBatch();
GRIM_HOST_API BatchStatus batchStatus();

// Host lifecycle only. Dependencies must not also be used by provider workers.
// Shutdown rejects new work, drains active work, and leaves queued downloads
// pending. Provider retirement must never call shutdown.
void start(std::shared_ptr<HuggingFaceWebhook> hf,
           std::shared_ptr<DataStructurer> structurer,
           std::shared_ptr<DatasetStream> stream,
           std::filesystem::path stateDirectory,
           std::filesystem::path downloadDirectory);
void shutdown();
}
}
