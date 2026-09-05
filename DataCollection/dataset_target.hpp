//======================================================//
//  DatasetTarget — Mass dataset access and per-model
//  configuration file management.
//
//  Owns:
//    - Mass dataset load / search / append
//    - Per-model sequence assignment (ID references, not copies)
//    - Structured output persistence back into mass_dataset.jsonl
//    - Per-model config file: <model_name>_configuration.json
//
//  All content lives in mass_dataset.jsonl.
//  Per-model assignment + config state lives in
//  model_store/<model_id>/<model_name>_configuration.json.
//======================================================//

#pragma once

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "concept_block.hpp"

namespace GRIM { namespace Pipeline {
    class IDatasetIO;
    struct TaggedEntry;
}}

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
    bool        verified = false;
    bool        is_structured = false;
};

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

    // ── Active model ────────────────────────────────────

    void        setActiveModel(const std::string& model_id,
                               const std::string& model_name = "");
    std::string activeModelId() const;
    std::string activeModelName() const;

    // ── Mass dataset (single source of truth) ───────────

    bool   loadMassDataset();
    bool   loadMassDataset(std::shared_ptr<GRIM::Pipeline::IDatasetIO> io);
    size_t massDatasetSize() const;

    SequenceHandle getSequence(size_t index) const;

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
    bool removeSequenceFromModel(const std::string& seq_id);

    bool   isAssigned(size_t seq_index) const;
    bool   isAssignedById(const std::string& seq_id) const;
    size_t assignedCount() const;

    // ── Curriculum ordering ─────────────────────────────

    bool moveSequenceUp(size_t curriculum_index);
    bool moveSequenceDown(size_t curriculum_index);
    bool moveSequence(size_t from_index, size_t to_index);
    const std::vector<std::string>& curriculumOrder() const;
    size_t curriculumIndexOf(const std::string& seq_id) const;

    // ── Phase markers ───────────────────────────────────

    struct PhaseMarker { size_t position = 0; std::string label; };

    void insertPhaseMarker(size_t position, const std::string& label);
    void removePhaseMarker(size_t marker_index);
    const std::vector<PhaseMarker>& phaseMarkers() const;

    // ── Bulk operations ─────────────────────────────────

    void assignMultiple(const std::vector<size_t>& seq_indices);
    void removeMultiple(const std::vector<size_t>& curriculum_indices);

    // ── Filtered pool access ────────────────────────────

    std::vector<size_t> filterSequences(
        const std::string& subject_filter,
        const std::string& quality_filter,
        const std::string& search_query) const;

    // ── Assignment persistence ──────────────────────────
    // Stored in: model_store/<model_id>/<model_name>_configuration.json

    bool loadAssignments();
    bool saveAssignments() const;

    // ── Structured output I/O ───────────────────────────

    bool        writeStructuredOutput(size_t seq_index, const std::string& structured);
    std::string readStructuredOutput(size_t seq_index) const;

    bool appendStructuredEntry(const std::string& content,
                               const std::string& structuredOutput,
                               const std::string& sourceType = "structurer",
                               const std::string& sourceUrl = "",
                               const std::string& qualityTier = "medium",
                               const std::string& subject = "general");

    // ── ConceptBlock CRUD ────────────────────────────────

    bool   loadConceptBlocks();
    bool   saveConceptBlocks() const;
    size_t conceptBlockCount() const;

    GRIM::ConceptBlock getConceptBlock(size_t index) const;
    GRIM::ConceptBlock getConceptBlockById(const std::string& cb_id) const;

    bool addConceptBlock(const GRIM::ConceptBlock& cb);
    bool addConceptBlocks(const std::vector<GRIM::ConceptBlock>& blocks,
                          size_t& added);
    bool updateConceptBlock(const std::string& cb_id, const GRIM::ConceptBlock& cb);
    bool removeConceptBlock(const std::string& cb_id);

    std::vector<size_t> searchConceptBlocks(const std::string& query,
                                            size_t max_results = 50) const;
    std::vector<size_t> filterConceptBlocks(const std::string& format_type,
                                            const std::string& search_query = "") const;

    // ── Curriculum registry CRUD ────────────────────────

    bool   loadCurriculumRegistry();
    bool   saveCurriculumRegistry() const;
    size_t curriculumCount() const;
    const std::vector<GRIM::Curriculum>& getCurriculums() const;

    GRIM::Curriculum getCurriculum(size_t index) const;
    GRIM::Curriculum getCurriculumById(const std::string& curr_id) const;
    size_t           getCurriculumIndexById(const std::string& curr_id) const;

    bool addCurriculum(const GRIM::Curriculum& curr);
    bool updateCurriculum(const std::string& curr_id, const GRIM::Curriculum& curr);
    bool removeCurriculum(const std::string& curr_id);

    // ── Concept block ↔ curriculum assignment ────────────

    const std::vector<GRIM::Course>& getCourses() const { return courses_; }
    GRIM::Course getCourseById(const std::string& id) const;
    bool saveCourse(const GRIM::Course& course);
    bool removeCourse(const std::string& id);
    bool assignCourse(const std::string& course_id, const std::string& curr_id, bool assigned);
    bool setCourseBlock(const std::string& course_id, const std::string& block_id, bool assigned);

    bool isConceptBlockInCurriculum(const std::string& cb_id,
                                    const std::string& curr_id) const;
    size_t conceptBlockCountInCurriculum(const std::string& curr_id) const;

    // ── Curriculum ↔ model assignment ────────────────────

    bool assignCurriculumToModel(const std::string& curr_id);
    bool removeCurriculumFromModel(const std::string& curr_id);
    bool isCurriculumAssigned(const std::string& curr_id) const;
    size_t assignedCurriculumCount() const;
    const std::vector<std::string>& assignedCurriculumOrder() const;

    // ── Curriculum manifest export ───────────────────────
    // Writes curriculum_manifest.json next to concept_blocks.fb.
    // Contains the union of concept_block_ids from all assigned
    // curricula so the GRIM-text DataLoader can filter at load time.

    bool exportCurriculumManifest() const;

private:
    std::filesystem::path       modelStoreRoot_;
    std::filesystem::path       massDatasetPath_;
    std::string                 activeModelId_;
    std::string                 activeModelName_;
    std::vector<SequenceHandle> sequences_;
    std::vector<std::string>    assignedOrder_;
    std::set<std::string>       assignedSet_;
    std::vector<PhaseMarker>    phaseMarkers_;

    // ConceptBlock storage
    std::vector<GRIM::ConceptBlock>                     conceptBlocks_;
    std::unordered_map<std::string, size_t>             cbIdIndex_;

    // Curriculum registry
    std::vector<GRIM::Curriculum>                       curriculums_;
    std::unordered_map<std::string, size_t>             currIdIndex_;
    std::vector<std::string>                            assignedCurrOrder_;
    std::set<std::string>                               assignedCurrSet_;

    std::filesystem::path configFilePath() const;
    std::filesystem::path conceptBlocksPath() const;
    std::filesystem::path legacyConceptBlocksPath() const;
    std::filesystem::path curriculumRegistryPath() const;
    void rebuildSequenceCache(const std::vector<GRIM::Pipeline::TaggedEntry>& entries);
    void rebuildCBIndex();
    void rebuildCurrIndex();
    void rebuildCurriculumBlocks();
    std::vector<GRIM::Course> courses_;
};
