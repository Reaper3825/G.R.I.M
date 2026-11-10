#include "dataset_stream.hpp"
#include "io/dataset_io_json.hpp"

#include <stdexcept>

namespace GRIM::DataCollection {
namespace fs = std::filesystem;

DatasetStream::DatasetStream(fs::path modelStoreRoot, fs::path massDatasetPath,
                             std::chrono::milliseconds interval)
    : modelStoreRoot_(std::move(modelStoreRoot)),
      massDatasetPath_(std::move(massDatasetPath)), interval_(interval) {
    if (interval_ <= std::chrono::milliseconds::zero())
        throw std::invalid_argument("DatasetStream refresh interval must be positive");
}

DatasetStream::~DatasetStream() { stop(); }

void DatasetStream::start() {
    // Lifecycle calls belong to the host thread. Refresh itself is serialized.
    if (worker_.joinable()) return;
    refresh();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = false;
    }
    worker_ = std::thread(&DatasetStream::run, this);
}

void DatasetStream::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}

DatasetStream::Stamps DatasetStream::sourceStamps() const {
    const auto directory = massDatasetPath_.parent_path();
    const std::array<fs::path, 4> paths = {massDatasetPath_,
        directory / "concept_blocks.fb", directory / "concept_blocks.jsonl",
        directory / "curriculum_registry.json"};
    Stamps result{};
    for (size_t i = 0; i < paths.size(); ++i) {
        result[i].exists = fs::exists(paths[i]);
        if (result[i].exists) {
            result[i].modified = fs::last_write_time(paths[i]);
            result[i].size = fs::file_size(paths[i]);
        }
    }
    return result;
}

bool DatasetStream::refresh() {
    std::lock_guard<std::mutex> refreshLock(refreshMutex_);
    return refreshLocked();
}

bool DatasetStream::refreshLocked(bool force) {
    try {
        const auto before = sourceStamps();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!force && latest_.catalog && before == publishedStamps_) {
                error_.clear();
                return true;
            }
        }
        const auto previous = latest();
        Stamps after;
        auto candidate = loadCatalog(previous.catalog.get(), after);
        publish(std::move(candidate), after);
        return true;
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = error.what();
        return false;
    }
}

std::shared_ptr<DatasetTarget> DatasetStream::loadCatalog(
    const DatasetTarget* modelState, Stamps& stamps) const {
    const auto before = sourceStamps();
    auto candidate = std::make_shared<DatasetTarget>(modelStoreRoot_, massDatasetPath_);
    if (before[0].exists) {
        // DatasetIOJson intentionally skips invalid lines for the editor.
        // The runtime stream instead rejects malformed/incomplete updates.
        if (!candidate->loadMassDataset(std::make_shared<GRIM::Pipeline::DatasetIOJson>(true)))
            throw std::runtime_error("Failed to load mass dataset");
    }
    if (!candidate->loadConceptBlocks(false))
        throw std::runtime_error("Failed to load ConceptBlock catalog");
    if (!candidate->loadCurriculumRegistry())
        throw std::runtime_error("Failed to load curriculum registry");
    if (modelState && !modelState->activeModelId().empty()) {
        candidate->setActiveModel(modelState->activeModelId(), modelState->activeModelName());
        if (!candidate->loadAssignments())
            throw std::runtime_error("Failed to load model assignments");
    }
    // Registry membership may contain stale block IDs while data is maintained.
    // Keep it browsable and preserve those references for repair. Exact block
    // lookup reports a missing ID when consumed; it must not hide every course.
    const auto after = sourceStamps();
    if (before != after)
        throw std::runtime_error("Dataset changed during reload; retrying on next refresh");
    stamps = after;
    return candidate;
}

void DatasetStream::publish(std::shared_ptr<DatasetTarget> candidate, const Stamps& stamps) {
    Snapshot retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        retired = std::move(latest_);
        latest_ = {retired.revision + 1, std::move(candidate)};
        publishedStamps_ = stamps;
        error_.clear();
    }
    // Retire potentially large old catalogs outside the reader lock.
}

bool DatasetStream::edit(uint64_t expectedRevision,
                         const std::function<bool(DatasetTarget&)>& operation) {
    std::lock_guard<std::mutex> writerLock(refreshMutex_);
    try {
        if (!refreshLocked()) return false;
        const auto current = latest();
        if (current.revision != expectedRevision)
            throw std::runtime_error("Dataset changed; refresh the editor before saving");
        auto candidate = std::make_shared<DatasetTarget>(*current.catalog);
        if (!operation(*candidate))
            throw std::runtime_error("Dataset operation failed; previous snapshot retained");
        Stamps stamps;
        auto verified = loadCatalog(candidate.get(), stamps);
        publish(std::move(verified), stamps);
        return true;
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = error.what();
        return false;
    }
}

bool DatasetStream::writeMassDataset(const fs::path& path,
    const std::vector<GRIM::Pipeline::TaggedEntry>& entries, bool replace) {
    std::lock_guard<std::mutex> writerLock(refreshMutex_);
    try {
        if (fs::weakly_canonical(path) != fs::weakly_canonical(massDatasetPath_))
            throw std::runtime_error("Pipeline output does not match the runtime source catalog");
        if (!refreshLocked()) return false;
        GRIM::Pipeline::DatasetIOJson io(true);
        const bool saved = replace ? io.saveAllEntries(path, entries) : io.appendEntries(path, entries);
        if (!saved) throw std::runtime_error("Pipeline catalog write failed");
        return refreshLocked(true);
    } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_ = error.what();
        return false;
    }
}

DatasetStream::Snapshot DatasetStream::latest() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

bool DatasetStream::pullLatest(Snapshot& out, uint64_t& lastSeenRevision) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!latest_.catalog || latest_.revision <= lastSeenRevision) return false;
    out = latest_;
    lastSeenRevision = latest_.revision;
    return true;
}

std::string DatasetStream::lastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_;
}

void DatasetStream::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!wake_.wait_for(lock, interval_, [this] { return stopping_; })) {
        lock.unlock();
        refresh();
        lock.lock();
    }
}

} // namespace GRIM::DataCollection
