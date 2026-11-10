#include "dataset_creation_service.hpp"

namespace GRIM::DataCollection {

DatasetCreationService::DatasetCreationService(std::shared_ptr<DatasetCatalog> catalog,
    std::shared_ptr<IConceptBindingAdapter> adapter, GeneratorAdapterOptions options,
    std::filesystem::path directory)
    : catalog_(std::move(catalog)), adapter_(std::move(adapter)),
      options_(options), directory_(std::move(directory)) {
    reloadAssets();
    worker_ = std::thread(&DatasetCreationService::run, this);
}

DatasetCreationService::~DatasetCreationService() { shutdown(); }

bool DatasetCreationService::reloadAssets() {
    std::lock_guard<std::mutex> loadLock(assetLoadMutex_);
    auto candidate = std::make_shared<ConceptGenerator>();
    const bool catalogLoaded = candidate->loadCatalog(directory_ / "LexiconPieces.json");
    const bool loaded = catalogLoaded && candidate->loadDataSource(directory_ / "GeneratorDataSource.json");
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded) {
        assetError_ = candidate->lastError();
        // An initial missing source file still permits explicit inline input.
        // Once assets exist, failed maintenance never replaces that version.
        if (catalogLoaded && !assets_) assets_ = std::move(candidate);
        return false;
    }
    assets_ = std::move(candidate);
    assetError_.clear();
    return true;
}

std::shared_ptr<const ConceptGenerator> DatasetCreationService::assets() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return assets_;
}

std::string DatasetCreationService::assetError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return assetError_;
}

bool DatasetCreationService::start(const Request& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || status_.filling || status_.running || !adapter_ || !assets_
        || request.frameIndex >= assets_->frames().size()
        || (request.document.text.empty() && assets_->documents().empty())
        || request.attempts == 0) return false;
    const auto revision = status_.revision;
    status_ = {};
    status_.revision = revision + 1;
    status_.running = true;
    status_.seed = request.seed;
    status_.queued = queue_.size();
    request_ = request;
    runAssets_ = assets_; // Pin recipe/source version for the whole run.
    wake_.notify_all();
    return true;
}

void DatasetCreationService::pause() {
    std::lock_guard<std::mutex> lock(mutex_);
    status_.paused = true;
    status_.running = false;
    ++status_.revision;
}

bool DatasetCreationService::resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || !status_.paused || !runAssets_
        || (!request_.continuous && status_.attempts >= request_.attempts)) return false;
    status_.paused = false;
    status_.running = true;
    ++status_.revision;
    wake_.notify_all();
    return true;
}

bool DatasetCreationService::saveQueue(size_t& added) {
    std::lock_guard<std::mutex> saveLock(saveMutex_);
    std::vector<GRIM::ConceptBlock> batch;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        batch = queue_;
    }
    added = 0;
    if (batch.empty()) return true;
    if (!catalog_->addConceptBlocks(batch, added)) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.erase(queue_.begin(), queue_.begin() + static_cast<ptrdiff_t>(batch.size()));
    status_.queued = queue_.size();
    ++status_.revision;
    return true;
}

DatasetCreationService::Status DatasetCreationService::status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return status_;
}

void DatasetCreationService::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        status_.running = false;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}

void DatasetCreationService::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        wake_.wait(lock, [this] { return stopping_ || status_.running; });
        if (stopping_) break;
        const auto request = request_;
        const auto assets = runAssets_;
        const auto iteration = status_.attempts;
        auto document = request.document;
        if (document.text.empty()) document = assets->documents()[iteration % assets->documents().size()];
        const auto recipe = assets->frames()[request.frameIndex];
        auto options = options_;
        options.seed = request.seed + iteration * 0x9E3779B97F4A7C15ULL;
        status_.filling = true;
        status_.message = "Filling " + document.id;
        ++status_.revision;
        lock.unlock();
        GeneratorBindingSet bindings;
        GeneratedConceptFrame frame;
        bool valid = false;
        try {
            bindings = adapter_->fill(document, recipe, {}, options);
            valid = bindings.error.empty() && assets->validateBindings(document, recipe, bindings);
            if (valid) frame = assets->assemble(request.frameIndex, bindings,
                request.seed, iteration, request.traversal);
        } catch (const std::exception& error) {
            valid = false;
            bindings.valid = false;
            bindings.error = error.what();
        }
        lock.lock();
        status_.filling = false;
        ++status_.attempts;
        status_.lastBindings = std::move(bindings);
        if (valid) {
            queue_.push_back(frame.conceptBlock);
            status_.lastFrame = std::move(frame);
            ++status_.generated;
            status_.message = "Valid bindings";
        } else {
            ++status_.failed;
            status_.message = status_.lastBindings.error.empty()
                ? "Binding validation failed" : status_.lastBindings.error;
        }
        status_.queued = queue_.size();
        if (!request.continuous && status_.attempts >= request.attempts) {
            status_.running = false;
            status_.paused = false;
        }
        ++status_.revision;
        // Bound continuous scheduling and allow pause/shutdown between requests.
        wake_.wait_for(lock, std::chrono::milliseconds(120), [this] {
            return stopping_ || !status_.running;
        });
    }
}

} // namespace GRIM::DataCollection
