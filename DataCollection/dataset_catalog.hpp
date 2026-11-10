#pragma once

#include "dataset_stream.hpp"
#include <stdexcept>

namespace GRIM::DataCollection {

// Lightweight catalog client. All reads copy values from immutable snapshots;
// writes go through the process stream rather than a UI-owned DatasetTarget.
class DatasetCatalog {
public:
    explicit DatasetCatalog(std::shared_ptr<DatasetStream> stream, bool stableReads = false)
        : stream_(std::move(stream)), stableReads_(stableReads) { pinLatest(); }
    bool loadMassDataset() { return refreshView(); }
    bool loadConceptBlocks() { return refreshView(); }
    bool loadCurriculumRegistry() { return refreshView(); }
    uint64_t revision() const { return stream_->latest().revision; }
    uint64_t viewRevision() const { return view().revision; }
    void pinLatest() {
        if (!stableReads_) return;
        const auto snapshot = stream_->latest();
        std::lock_guard<std::mutex> lock(viewMutex_);
        view_ = snapshot;
    }
    bool ready() const { return stream_->latest().catalog != nullptr; }
    std::string lastError() const { return stream_->lastError(); }
    std::vector<nlohmann::json> goalSelections(const std::string& query) const;
    nlohmann::json curriculumRegistry() const;
    nlohmann::json conceptBlock(const std::string& id) const;
    void importGoal(const nlohmann::json& selection);
private:
    DatasetStream::Snapshot view() const {
        if (!stableReads_) return stream_->latest();
        std::lock_guard<std::mutex> lock(viewMutex_);
        return view_;
    }
    bool refreshView() {
        const bool success = stream_->refresh();
        pinLatest();
        return success;
    }
    template<class Operation> auto read(Operation operation) const {
        const auto snapshot = view();
        if (!snapshot.catalog) {
            const DatasetTarget unavailable({}, {});
            return operation(unavailable);
        }
        return operation(*snapshot.catalog);
    }
    bool write(const std::function<bool(DatasetTarget&)>& operation, uint64_t expectedRevision = 0) {
        const bool success = stream_->edit(expectedRevision ? expectedRevision : view().revision, operation);
        if (success && stableReads_) pinLatest();
        return success;
    }
    std::shared_ptr<DatasetStream> stream_;
    const bool stableReads_;
    mutable std::mutex viewMutex_;
    DatasetStream::Snapshot view_;
public:
    auto setActiveModel(const std::string& model_id, const std::string& model_name = "") {
        if (!write([&](DatasetTarget& target) { target.setActiveModel(model_id, model_name); return true; })) throw std::runtime_error(stream_->lastError());
    }
    auto activeModelId() const {
        return read([&](const DatasetTarget& target) { return target.activeModelId(); });
    }
    auto activeModelName() const {
        return read([&](const DatasetTarget& target) { return target.activeModelName(); });
    }
    auto massDatasetSize() const {
        return read([&](const DatasetTarget& target) { return target.massDatasetSize(); });
    }
    auto getSequence(size_t index) const {
        return read([&](const DatasetTarget& target) { return target.getSequence(index); });
    }
    auto searchSequences( const std::string& query, size_t max_results = 8) const {
        return read([&](const DatasetTarget& target) { return target.searchSequences(query, max_results); });
    }
    auto searchSequences( const std::string& query, const std::string& source_type_filter, const std::string& quality_filter, const std::string& subject_filter, size_t max_results = 8) const {
        return read([&](const DatasetTarget& target) { return target.searchSequences(query, source_type_filter, quality_filter, subject_filter, max_results); });
    }
    auto assignSequenceToModel(size_t seq_index) {
        return write([&](DatasetTarget& target) { return target.assignSequenceToModel(seq_index); });
    }
    auto assignSequenceToModel(const std::string& seq_id) {
        return write([&](DatasetTarget& target) { return target.assignSequenceToModel(seq_id); });
    }
    auto removeSequenceFromModel(size_t seq_index) {
        return write([&](DatasetTarget& target) { return target.removeSequenceFromModel(seq_index); });
    }
    auto removeSequenceFromModel(const std::string& seq_id) {
        return write([&](DatasetTarget& target) { return target.removeSequenceFromModel(seq_id); });
    }
    auto isAssigned(size_t seq_index) const {
        return read([&](const DatasetTarget& target) { return target.isAssigned(seq_index); });
    }
    auto assignedCount() const {
        return read([&](const DatasetTarget& target) { return target.assignedCount(); });
    }
    auto moveSequenceUp(size_t curriculum_index) {
        return write([&](DatasetTarget& target) { return target.moveSequenceUp(curriculum_index); });
    }
    auto moveSequenceDown(size_t curriculum_index) {
        return write([&](DatasetTarget& target) { return target.moveSequenceDown(curriculum_index); });
    }
    auto curriculumOrder() const {
        return read([&](const DatasetTarget& target) { return target.curriculumOrder(); });
    }
    auto insertPhaseMarker(size_t position, const std::string& label) {
        if (!write([&](DatasetTarget& target) { target.insertPhaseMarker(position, label); return true; })) throw std::runtime_error(stream_->lastError());
    }
    auto phaseMarkers() const {
        return read([&](const DatasetTarget& target) { return target.phaseMarkers(); });
    }
    auto assignMultiple(const std::vector<size_t>& seq_indices) {
        if (!write([&](DatasetTarget& target) { target.assignMultiple(seq_indices); return true; })) throw std::runtime_error(stream_->lastError());
    }
    auto filterSequences( const std::string& subject_filter, const std::string& quality_filter, const std::string& search_query) const {
        return read([&](const DatasetTarget& target) { return target.filterSequences(subject_filter, quality_filter, search_query); });
    }
    auto loadAssignments() {
        return write([&](DatasetTarget& target) { return target.loadAssignments(); });
    }
    auto writeStructuredOutput(size_t seq_index, const std::string& structured, uint64_t expectedRevision = 0) {
        return write([&](DatasetTarget& target) { return target.writeStructuredOutput(seq_index, structured); }, expectedRevision);
    }
    auto appendStructuredEntry(const std::string& content, const std::string& structuredOutput, const std::string& sourceType = "structurer", const std::string& sourceUrl = "", const std::string& qualityTier = "medium", const std::string& subject = "general") {
        return write([&](DatasetTarget& target) { return target.appendStructuredEntry(content, structuredOutput, sourceType, sourceUrl, qualityTier, subject); });
    }
    auto conceptBlockCount() const {
        return read([&](const DatasetTarget& target) { return target.conceptBlockCount(); });
    }
    auto getConceptBlock(size_t index) const {
        return read([&](const DatasetTarget& target) { return target.getConceptBlock(index); });
    }
    auto getConceptBlockById(const std::string& cb_id) const {
        return read([&](const DatasetTarget& target) { return target.getConceptBlockById(cb_id); });
    }
    auto addConceptBlock(const GRIM::ConceptBlock& cb) {
        return write([&](DatasetTarget& target) { return target.addConceptBlock(cb); });
    }
    auto addConceptBlocks(const std::vector<GRIM::ConceptBlock>& blocks, size_t& added) {
        return write([&](DatasetTarget& target) { return target.addConceptBlocks(blocks, added); });
    }
    auto updateConceptBlock(const std::string& cb_id, const GRIM::ConceptBlock& cb, uint64_t expectedRevision = 0) {
        return write([&](DatasetTarget& target) { return target.updateConceptBlock(cb_id, cb); }, expectedRevision);
    }
    auto removeConceptBlock(const std::string& cb_id) {
        return write([&](DatasetTarget& target) { return target.removeConceptBlock(cb_id); });
    }
    auto searchGoalConceptBlocks(const std::string& query, size_t max_results = 100) const {
        return read([&](const DatasetTarget& target) { return target.searchGoalConceptBlocks(query, max_results); });
    }
    auto filterConceptBlocks(const std::string& format_type, const std::string& search_query = "") const {
        return read([&](const DatasetTarget& target) { return target.filterConceptBlocks(format_type, search_query); });
    }
    auto curriculumCount() const {
        return read([&](const DatasetTarget& target) { return target.curriculumCount(); });
    }
    auto getCurriculums() const {
        return read([&](const DatasetTarget& target) { return target.getCurriculums(); });
    }
    auto getCurriculumById(const std::string& curr_id) const {
        return read([&](const DatasetTarget& target) { return target.getCurriculumById(curr_id); });
    }
    auto addCurriculum(const GRIM::Curriculum& curr) {
        return write([&](DatasetTarget& target) { return target.addCurriculum(curr); });
    }
    auto updateCurriculum(const std::string& curr_id, const GRIM::Curriculum& curr) {
        return write([&](DatasetTarget& target) { return target.updateCurriculum(curr_id, curr); });
    }
    auto removeCurriculum(const std::string& curr_id) {
        return write([&](DatasetTarget& target) { return target.removeCurriculum(curr_id); });
    }
    auto getCourses() const {
        return read([&](const DatasetTarget& target) { return target.getCourses(); });
    }
    auto getCourseById(const std::string& id) const {
        return read([&](const DatasetTarget& target) { return target.getCourseById(id); });
    }
    auto saveCourse(const GRIM::Course& course) {
        return write([&](DatasetTarget& target) { return target.saveCourse(course); });
    }
    auto removeCourse(const std::string& id) {
        return write([&](DatasetTarget& target) { return target.removeCourse(id); });
    }
    auto assignCourse(const std::string& course_id, const std::string& curr_id, bool assigned) {
        return write([&](DatasetTarget& target) { return target.assignCourse(course_id, curr_id, assigned); });
    }
    auto setCourseBlock(const std::string& course_id, const std::string& block_id, bool assigned) {
        return write([&](DatasetTarget& target) { return target.setCourseBlock(course_id, block_id, assigned); });
    }
    auto assignCurriculumToModel(const std::string& curr_id) {
        return write([&](DatasetTarget& target) { return target.assignCurriculumToModel(curr_id); });
    }
    auto removeCurriculumFromModel(const std::string& curr_id) {
        return write([&](DatasetTarget& target) { return target.removeCurriculumFromModel(curr_id); });
    }
    auto isCurriculumAssigned(const std::string& curr_id) const {
        return read([&](const DatasetTarget& target) { return target.isCurriculumAssigned(curr_id); });
    }
};

} // namespace GRIM::DataCollection
