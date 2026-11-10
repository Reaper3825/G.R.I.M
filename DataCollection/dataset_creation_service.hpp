#pragma once

#include "concept_binding_adapter.hpp"
#include "dataset_catalog.hpp"

namespace GRIM::DataCollection {

// Process-owned generator job. No UI callbacks or render-loop scheduling.
class DatasetCreationService {
public:
    struct Request {
        size_t frameIndex = 0;
        GeneratorDocument document; // Empty text selects configured source rows.
        uint64_t seed = 1337;
        uint64_t attempts = 1;
        bool continuous = false;
        GeneratorTraversal traversal = GeneratorTraversal::SeededRandom;
    };
    struct Status {
        uint64_t revision = 0;
        bool running = false;
        bool paused = false;
        bool filling = false;
        uint64_t attempts = 0;
        uint64_t generated = 0;
        uint64_t failed = 0;
        uint64_t seed = 0;
        size_t queued = 0;
        GeneratedConceptFrame lastFrame;
        GeneratorBindingSet lastBindings;
        std::string message;
    };

    DatasetCreationService(std::shared_ptr<DatasetCatalog> catalog,
                           std::shared_ptr<IConceptBindingAdapter> adapter,
                           GeneratorAdapterOptions options,
                           std::filesystem::path directory);
    ~DatasetCreationService();
    bool reloadAssets();
    std::shared_ptr<const ConceptGenerator> assets() const;
    std::string assetError() const;
    bool start(const Request& request);
    void pause();
    bool resume();
    bool saveQueue(size_t& added);
    Status status() const;
    void shutdown();

private:
    void run();
    std::shared_ptr<DatasetCatalog> catalog_;
    std::shared_ptr<IConceptBindingAdapter> adapter_;
    GeneratorAdapterOptions options_;
    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    std::mutex assetLoadMutex_;
    std::mutex saveMutex_;
    std::condition_variable wake_;
    std::thread worker_;
    bool stopping_ = false;
    Request request_;
    Status status_;
    std::shared_ptr<const ConceptGenerator> assets_;
    std::shared_ptr<const ConceptGenerator> runAssets_;
    std::string assetError_;
    std::vector<GRIM::ConceptBlock> queue_;
};

} // namespace GRIM::DataCollection
