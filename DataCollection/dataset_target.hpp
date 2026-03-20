//======================================================//
//  DatasetTarget — Cross-platform model folder and mass
//  dataset management with ID-based sequence referencing.
//
//  Owns:
//    - Model folder lifecycle (temp → commit / discard)
//    - Mass dataset load / search / append
//    - Per-model sequence assignment (ID references, not copies)
//    - Per-model structured output storage
//    - Source file cleanup after HF download merges
//
//  All paths are std::filesystem::path, relative to a
//  repo-root-derived model store root.  No hardcoded OS paths.
//
//  ID migration: index-based keys today, persistent IDs
//  once the tagging plan lands.  The UI framework (search,
//  filter, assign/remove) requires zero structural changes.
//======================================================//

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace GRIM { namespace Pipeline {
    class IDatasetIO;
    struct TaggedEntry;
}}

// Lightweight handle for a sequence in the mass dataset.
// `id` is a persistent unique ID assigned by StageTag.
struct SequenceHandle {
    size_t      index = 0;
    std::string id;
    std::string content;
    std::string structured;

    std::string source_type;
    std::string subject;
    std::string quality_tier;
    std::vector<std::string> tags;
    float       reliability_score = 0.0f;
};

// Truncated preview returned from search queries.
struct SearchResult {
    size_t      index;
    std::string id;
    std::string preview;
    float       relevance = 0.0f;
};

class DatasetTarget {
public:
    explicit DatasetTarget(const std::filesystem::path& model_store_root,
                           const std::filesystem::path& mass_dataset_path);
    ~DatasetTarget();

    // ── Model folder lifecycle ──────────────────────────

    std::filesystem::path createTempModelFolder();

    bool commitModelFolder(const std::filesystem::path& temp_path,
                           const std::string& model_id);

    bool discardTempModelFolder(const std::filesystem::path& temp_path);

    // ── Active model ────────────────────────────────────

    void        setActiveModel(const std::string& model_id);
    std::string activeModelId() const;

    // ── Mass dataset (single source of truth) ───────────

    bool   loadMassDataset();
    bool   loadMassDataset(std::shared_ptr<GRIM::Pipeline::IDatasetIO> io);
    size_t massDatasetSize() const;

    SequenceHandle getSequence(size_t index) const;

    bool appendToMassDataset(const std::vector<std::string>& entries);

    // ── Search & Filter ─────────────────────────────────

    std::vector<SearchResult> searchSequences(
        const std::string& query,
        size_t max_results = 8) const;

    std::vector<SearchResult> searchSequences(
        const std::string& query,
        const std::string& source_type_filter,
        const std::string& quality_filter,
        const std::string& subject_filter,
        size_t max_results = 8) const;

    // ── Per-model sequence assignment (ID refs) ─────────

    bool assignSequenceToModel(size_t seq_index);
    bool assignSequenceToModel(const std::string& seq_id);

    bool removeSequenceFromModel(size_t seq_index);

    std::vector<size_t> getAssignedSequences() const;
    bool   isAssigned(size_t seq_index) const;
    size_t assignedCount() const;

    // ── Assignment persistence ──────────────────────────
    // Stored as: model_store/<model_id>/dataset_refs.json
    // Format:    {"assigned": [0, 14, 27, 103, ...]}

    bool loadAssignments();
    bool saveAssignments() const;

    // ── Structured output I/O ───────────────────────────

    bool        writeStructuredOutput(size_t seq_index, const std::string& structured);
    std::string readStructuredOutput(size_t seq_index) const;

    // ── Cleanup ─────────────────────────────────────────

    bool deleteSourceFiles(const std::vector<std::filesystem::path>& files);

private:
    std::filesystem::path   modelStoreRoot_;
    std::filesystem::path   massDatasetPath_;
    std::string             activeModelId_;
    std::vector<SequenceHandle> sequences_;
    std::vector<size_t>     assignedIndices_;
};
