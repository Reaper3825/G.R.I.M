#pragma once

#include "dataset_target.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

namespace GRIM::DataCollection {

// Process-owned, latest-version stream of source catalogs. Published catalogs
// are immutable; holding a Snapshot pins that version across later reloads.
// Authoring and pipeline writes are serialized here and publish verified source
// state immediately. Background reloads never migrate or rewrite source files.
class DatasetStream {
public:
    struct Snapshot {
        uint64_t revision = 0;
        std::shared_ptr<const DatasetTarget> catalog;
    };

    DatasetStream(std::filesystem::path modelStoreRoot,
                  std::filesystem::path massDatasetPath,
                  std::chrono::milliseconds interval = std::chrono::seconds(1));
    ~DatasetStream();
    DatasetStream(const DatasetStream&) = delete;
    DatasetStream& operator=(const DatasetStream&) = delete;

    // Initial load is synchronous. Subsequent file changes are polled on a
    // worker; an invalid update leaves the previous published version intact.
    void start();
    void stop();
    bool refresh();
    // Copy-on-write authoring. Refuse stale edits and invalid on-disk updates;
    // successful operations publish immediately under the same writer lock.
    bool edit(uint64_t expectedRevision, const std::function<bool(DatasetTarget&)>& operation);
    bool writeMassDataset(const std::filesystem::path& path,
        const std::vector<GRIM::Pipeline::TaggedEntry>& entries, bool replace);
    Snapshot latest() const;
    bool pullLatest(Snapshot& out, uint64_t& lastSeenRevision) const;
    std::string lastError() const;

private:
    struct FileStamp {
        bool exists = false;
        std::filesystem::file_time_type modified{};
        uintmax_t size = 0;
        bool operator==(const FileStamp& other) const {
            return exists == other.exists && modified == other.modified && size == other.size;
        }
    };
    using Stamps = std::array<FileStamp, 4>;
    Stamps sourceStamps() const;
    bool refreshLocked(bool force = false);
    std::shared_ptr<DatasetTarget> loadCatalog(const DatasetTarget* modelState, Stamps& stamps) const;
    void publish(std::shared_ptr<DatasetTarget> candidate, const Stamps& stamps);
    void run();

    const std::filesystem::path modelStoreRoot_;
    const std::filesystem::path massDatasetPath_;
    const std::chrono::milliseconds interval_;
    mutable std::mutex mutex_;
    std::mutex refreshMutex_;
    std::condition_variable wake_;
    std::thread worker_;
    bool stopping_ = false;
    Snapshot latest_;
    Stamps publishedStamps_{};
    std::string error_;
};

} // namespace GRIM::DataCollection
