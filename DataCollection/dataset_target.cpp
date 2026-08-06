#include "dataset_target.hpp"
#include "io/concept_block_io_flatbuffer.hpp"
#include "io/dataset_io.hpp"
#include "io/dataset_io_json.hpp"
#include "pipeline/pipeline_context.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

// Legacy JSONL import helper. New persistence is concept_blocks.fb.
static GRIM::ConceptBlock conceptBlockFromJson(const json& j) {
    GRIM::ConceptBlock cb;
    cb.id                 = j.value("id", std::string());
    cb.name               = j.value("name", std::string());
    cb.prompt             = j.at("question").get<std::string>();
    cb.answer             = j.value("answer", std::string());
    cb.format_type        = j.value("format_type", std::string("chain_of_thought"));
    cb.source_sequence_id = j.value("source_sequence_id", std::string());
    cb.timestamp          = j.value("timestamp", int64_t(0));
    if (j.contains("goal") && j["goal"].is_object()) {
        cb.goal = GRIM::ConceptBlockGoal{
            j["goal"].value("target_state", std::string())};
    }

    if (j.contains("intermediates") && j["intermediates"].is_array()) {
        for (const auto& s : j["intermediates"]) {
            if (s.is_string()) cb.intermediates.push_back(s.get<std::string>());
        }
    }
    if (j.contains("explanation") && j["explanation"].is_array()) {
        for (const auto& s : j["explanation"]) {
            if (s.is_string()) cb.explanation.push_back(s.get<std::string>());
        }
    }

    if (j.contains("execution") && j["execution"].is_array()) {
        for (const auto& e : j["execution"]) {
            if (!e.is_object()) continue;
            GRIM::ConceptExecutionStep step;
            step.op     = e.value("op", std::string());
            step.result = e.value("result", 0.0);
            if (e.contains("args") && e["args"].is_array()) {
                for (const auto& a : e["args"]) {
                    if (a.is_number()) step.args.push_back(a.get<double>());
                }
            }
            if (e.contains("arg_slots") && e["arg_slots"].is_array()) {
                for (const auto& s : e["arg_slots"]) {
                    if (s.is_number_integer()) step.arg_slots.push_back(s.get<int>());
                }
            }
            cb.execution.push_back(std::move(step));
        }
    }

    cb.recomputeDerived();
    return cb;
}

// ─────────────────────────────────────────────────────────

static std::string generateEntryId(const std::string& content) {
    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 14695981039346656037ULL;
    for (size_t i = 0; i < content.size(); ++i) {
        uint8_t c = static_cast<uint8_t>(content[i]);
        h1 ^= c;
        h1 *= 1099511628211ULL;
        if (i + 1 < content.size()) {
            h2 ^= static_cast<uint8_t>(content[i + 1]);
            h2 *= 1099511628211ULL;
        }
    }
    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << h1
        << std::setw(16) << h2;
    return oss.str();
}

static std::string toLower(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
}

// ─── Construction ────────────────────────────────────────

DatasetTarget::DatasetTarget(const fs::path& model_store_root,
                             const fs::path& mass_dataset_path)
    : modelStoreRoot_(model_store_root)
    , massDatasetPath_(mass_dataset_path) {}

DatasetTarget::~DatasetTarget() = default;

// ─── Active model ────────────────────────────────────────

void DatasetTarget::setActiveModel(const std::string& model_id,
                                   const std::string& model_name) {
    activeModelId_ = model_id;
    activeModelName_ = model_name.empty() ? model_id : model_name;
    assignedOrder_.clear();
    assignedSet_.clear();
    phaseMarkers_.clear();
    assignedCurrOrder_.clear();
    assignedCurrSet_.clear();
}

std::string DatasetTarget::activeModelId() const { return activeModelId_; }
std::string DatasetTarget::activeModelName() const { return activeModelName_; }

// ─── Config file path ────────────────────────────────────

fs::path DatasetTarget::configFilePath() const {
    if (activeModelId_.empty()) return {};
    std::string filename = activeModelName_ + "_configuration.json";
    return modelStoreRoot_ / activeModelId_ / filename;
}

// ─── Mass dataset ────────────────────────────────────────

bool DatasetTarget::loadMassDataset() {
    auto io = std::make_shared<GRIM::Pipeline::DatasetIOJson>();
    return loadMassDataset(io);
}

bool DatasetTarget::loadMassDataset(std::shared_ptr<GRIM::Pipeline::IDatasetIO> io) {
    std::vector<GRIM::Pipeline::TaggedEntry> entries;
    if (!io->loadAllEntries(massDatasetPath_, entries)) {
        std::cerr << "[DatasetTarget] Failed to load mass dataset: "
                  << massDatasetPath_ << "\n";
        sequences_.clear();
        return false;
    }
    rebuildSequenceCache(entries);
    return true;
}

size_t DatasetTarget::massDatasetSize() const { return sequences_.size(); }

SequenceHandle DatasetTarget::getSequence(size_t index) const {
    if (index >= sequences_.size()) return {};
    return sequences_[index];
}

void DatasetTarget::rebuildSequenceCache(
    const std::vector<GRIM::Pipeline::TaggedEntry>& entries) {
    sequences_.clear();
    sequences_.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        SequenceHandle h;
        h.index            = i;
        h.id               = e.id;
        h.content          = e.content;
        h.structured       = e.structuredOutput;
        h.source_type      = e.sourceType;
        h.subject          = e.subject;
        h.quality_tier     = e.qualityTier;
        h.tags             = e.tags;
        h.reliability_score = e.reliabilityScore;
        h.verified         = e.verified;
        h.is_structured    = e.structured;
        sequences_.push_back(std::move(h));
    }
}

// ─── Search ──────────────────────────────────────────────

std::vector<SearchResult> DatasetTarget::searchSequences(
    const std::string& query, size_t max_results) const {
    return searchSequences(query, "", "", "", max_results);
}

std::vector<SearchResult> DatasetTarget::searchSequences(
    const std::string& query,
    const std::string& source_type_filter,
    const std::string& quality_filter,
    const std::string& subject_filter,
    size_t max_results) const {

    std::vector<SearchResult> results;
    std::string lowerQuery = toLower(query);

    for (const auto& seq : sequences_) {
        if (!source_type_filter.empty() && seq.source_type != source_type_filter)
            continue;
        if (!quality_filter.empty() && seq.quality_tier != quality_filter)
            continue;
        if (!subject_filter.empty() && seq.subject != subject_filter)
            continue;

        if (!lowerQuery.empty()) {
            std::string lowerContent = toLower(seq.content);
            if (lowerContent.find(lowerQuery) == std::string::npos)
                continue;
        }

        SearchResult r;
        r.index = seq.index;
        r.id = seq.id;
        r.preview = seq.content.substr(0, std::min(seq.content.size(), size_t(120)));
        if (seq.content.size() > 120) r.preview += "...";
        r.relevance = 1.0f;
        results.push_back(std::move(r));

        if (results.size() >= max_results) break;
    }
    return results;
}

// ─── Assignment ──────────────────────────────────────────

bool DatasetTarget::assignSequenceToModel(size_t seq_index) {
    if (seq_index >= sequences_.size()) return false;
    return assignSequenceToModel(sequences_[seq_index].id);
}

bool DatasetTarget::assignSequenceToModel(const std::string& seq_id) {
    if (seq_id.empty() || activeModelId_.empty()) return false;
    if (assignedSet_.count(seq_id)) return true;
    assignedOrder_.push_back(seq_id);
    assignedSet_.insert(seq_id);
    return saveAssignments();
}

bool DatasetTarget::removeSequenceFromModel(size_t seq_index) {
    if (seq_index >= sequences_.size()) return false;
    return removeSequenceFromModel(sequences_[seq_index].id);
}

bool DatasetTarget::removeSequenceFromModel(const std::string& seq_id) {
    if (!assignedSet_.count(seq_id)) return false;
    assignedSet_.erase(seq_id);
    assignedOrder_.erase(
        std::remove(assignedOrder_.begin(), assignedOrder_.end(), seq_id),
        assignedOrder_.end());
    // Adjust phase markers when an entry is removed
    for (auto it = phaseMarkers_.begin(); it != phaseMarkers_.end(); ) {
        if (it->position >= assignedOrder_.size() && !assignedOrder_.empty())
            it->position = assignedOrder_.size();
        ++it;
    }
    return saveAssignments();
}

bool DatasetTarget::isAssigned(size_t seq_index) const {
    if (seq_index >= sequences_.size()) return false;
    return assignedSet_.count(sequences_[seq_index].id) > 0;
}

bool DatasetTarget::isAssignedById(const std::string& seq_id) const {
    return assignedSet_.count(seq_id) > 0;
}

size_t DatasetTarget::assignedCount() const { return assignedOrder_.size(); }

// ─── Curriculum ordering ─────────────────────────────────

bool DatasetTarget::moveSequenceUp(size_t curriculum_index) {
    if (curriculum_index == 0 || curriculum_index >= assignedOrder_.size())
        return false;
    std::swap(assignedOrder_[curriculum_index], assignedOrder_[curriculum_index - 1]);
    return saveAssignments();
}

bool DatasetTarget::moveSequenceDown(size_t curriculum_index) {
    if (curriculum_index + 1 >= assignedOrder_.size())
        return false;
    std::swap(assignedOrder_[curriculum_index], assignedOrder_[curriculum_index + 1]);
    return saveAssignments();
}

bool DatasetTarget::moveSequence(size_t from_index, size_t to_index) {
    if (from_index >= assignedOrder_.size() || to_index >= assignedOrder_.size())
        return false;
    if (from_index == to_index) return true;
    std::string id = assignedOrder_[from_index];
    assignedOrder_.erase(assignedOrder_.begin() + static_cast<ptrdiff_t>(from_index));
    assignedOrder_.insert(assignedOrder_.begin() + static_cast<ptrdiff_t>(to_index), id);
    return saveAssignments();
}

const std::vector<std::string>& DatasetTarget::curriculumOrder() const {
    return assignedOrder_;
}

size_t DatasetTarget::curriculumIndexOf(const std::string& seq_id) const {
    auto it = std::find(assignedOrder_.begin(), assignedOrder_.end(), seq_id);
    if (it == assignedOrder_.end()) return SIZE_MAX;
    return static_cast<size_t>(std::distance(assignedOrder_.begin(), it));
}

// ─── Phase markers ───────────────────────────────────────

void DatasetTarget::insertPhaseMarker(size_t position, const std::string& label) {
    PhaseMarker pm;
    pm.position = position;
    pm.label = label;
    phaseMarkers_.push_back(pm);
    std::sort(phaseMarkers_.begin(), phaseMarkers_.end(),
              [](const PhaseMarker& a, const PhaseMarker& b) {
                  return a.position < b.position;
              });
    saveAssignments();
}

void DatasetTarget::removePhaseMarker(size_t marker_index) {
    if (marker_index >= phaseMarkers_.size()) return;
    phaseMarkers_.erase(phaseMarkers_.begin() + static_cast<ptrdiff_t>(marker_index));
    saveAssignments();
}

const std::vector<DatasetTarget::PhaseMarker>& DatasetTarget::phaseMarkers() const {
    return phaseMarkers_;
}

// ─── Bulk operations ─────────────────────────────────────

void DatasetTarget::assignMultiple(const std::vector<size_t>& seq_indices) {
    for (size_t idx : seq_indices) {
        if (idx >= sequences_.size()) continue;
        const auto& id = sequences_[idx].id;
        if (assignedSet_.count(id)) continue;
        assignedOrder_.push_back(id);
        assignedSet_.insert(id);
    }
    saveAssignments();
}

void DatasetTarget::removeMultiple(const std::vector<size_t>& curriculum_indices) {
    std::vector<size_t> sorted = curriculum_indices;
    std::sort(sorted.rbegin(), sorted.rend());
    for (size_t ci : sorted) {
        if (ci >= assignedOrder_.size()) continue;
        assignedSet_.erase(assignedOrder_[ci]);
        assignedOrder_.erase(assignedOrder_.begin() + static_cast<ptrdiff_t>(ci));
    }
    saveAssignments();
}

// ─── Filtered pool access ────────────────────────────────

std::vector<size_t> DatasetTarget::filterSequences(
    const std::string& subject_filter,
    const std::string& quality_filter,
    const std::string& search_query) const {

    std::vector<size_t> result;
    std::string lowerQuery = toLower(search_query);

    for (const auto& seq : sequences_) {
        if (!subject_filter.empty() && subject_filter != "All"
            && seq.subject != subject_filter)
            continue;
        if (!quality_filter.empty() && quality_filter != "All"
            && seq.quality_tier != quality_filter)
            continue;
        if (!lowerQuery.empty()) {
            std::string lc = toLower(seq.content);
            if (lc.find(lowerQuery) == std::string::npos)
                continue;
        }
        result.push_back(seq.index);
    }
    return result;
}

// ─── Assignment persistence ─────────────────────────────

bool DatasetTarget::loadAssignments() {
    assignedOrder_.clear();
    assignedSet_.clear();
    phaseMarkers_.clear();
    assignedCurrOrder_.clear();
    assignedCurrSet_.clear();
    fs::path cfgPath = configFilePath();
    if (cfgPath.empty() || !fs::exists(cfgPath)) return true;

    try {
        std::ifstream file(cfgPath);
        if (!file.is_open()) return false;
        json j = json::parse(file);
        if (j.contains("assigned_sequence_ids") && j["assigned_sequence_ids"].is_array()) {
            for (const auto& id : j["assigned_sequence_ids"]) {
                if (!id.is_string()) continue;
                std::string s = id.get<std::string>();
                if (assignedSet_.count(s)) continue;
                assignedOrder_.push_back(s);
                assignedSet_.insert(s);
            }
        }
        if (j.contains("assigned_curriculum_ids") && j["assigned_curriculum_ids"].is_array()) {
            for (const auto& id : j["assigned_curriculum_ids"]) {
                if (!id.is_string()) continue;
                std::string s = id.get<std::string>();
                if (assignedCurrSet_.count(s)) continue;
                assignedCurrOrder_.push_back(s);
                assignedCurrSet_.insert(s);
            }
        }
        if (j.contains("curriculum_phases") && j["curriculum_phases"].is_array()) {
            for (const auto& pm : j["curriculum_phases"]) {
                if (!pm.contains("position") || !pm.contains("label")) continue;
                PhaseMarker marker;
                marker.position = pm["position"].get<size_t>();
                marker.label    = pm["label"].get<std::string>();
                phaseMarkers_.push_back(marker);
            }
            std::sort(phaseMarkers_.begin(), phaseMarkers_.end(),
                      [](const PhaseMarker& a, const PhaseMarker& b) {
                          return a.position < b.position;
                      });
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[DatasetTarget] Error loading assignments: " << e.what() << "\n";
        return false;
    }
}

bool DatasetTarget::saveAssignments() const {
    fs::path cfgPath = configFilePath();
    if (cfgPath.empty()) return false;

    std::error_code ec;
    fs::create_directories(cfgPath.parent_path(), ec);
    if (ec) return false;

    json j;
    if (fs::exists(cfgPath)) {
        try {
            std::ifstream existing(cfgPath);
            j = json::parse(existing);
        } catch (...) {
            j = json::object();
        }
    }

    j["model_id"] = activeModelId_;
    j["assigned_sequence_ids"] = json::array();
    for (const auto& id : assignedOrder_) {
        j["assigned_sequence_ids"].push_back(id);
    }

    j["assigned_curriculum_ids"] = json::array();
    for (const auto& id : assignedCurrOrder_) {
        j["assigned_curriculum_ids"].push_back(id);
    }

    j["curriculum_phases"] = json::array();
    for (const auto& pm : phaseMarkers_) {
        j["curriculum_phases"].push_back({
            {"position", pm.position},
            {"label",    pm.label}
        });
    }

    fs::path tmpPath = cfgPath;
    tmpPath += ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        if (!out.is_open()) return false;
        out << j.dump(2) << "\n";
        if (!out.good()) return false;
    }
    fs::rename(tmpPath, cfgPath, ec);
    if (ec) return false;

    // Keep curriculum manifest in sync with assignments.
    exportCurriculumManifest();
    return true;
}

// ─── Structured output I/O ──────────────────────────────

bool DatasetTarget::writeStructuredOutput(size_t seq_index,
                                          const std::string& structured) {
    if (seq_index >= sequences_.size()) return false;

    auto io = std::make_shared<GRIM::Pipeline::DatasetIOJson>();
    std::vector<GRIM::Pipeline::TaggedEntry> entries;
    if (!io->loadAllEntries(massDatasetPath_, entries)) return false;

    const std::string& targetId = sequences_[seq_index].id;
    bool found = false;
    for (auto& e : entries) {
        if (e.id == targetId) {
            e.structured = true;
            e.structuredOutput = structured;
            found = true;
            break;
        }
    }
    if (!found) return false;

    if (!io->saveAllEntries(massDatasetPath_, entries)) return false;

    rebuildSequenceCache(entries);
    return true;
}

std::string DatasetTarget::readStructuredOutput(size_t seq_index) const {
    if (seq_index >= sequences_.size()) return {};
    return sequences_[seq_index].structured;
}

bool DatasetTarget::appendStructuredEntry(
    const std::string& content,
    const std::string& structuredOutput,
    const std::string& sourceType,
    const std::string& sourceUrl,
    const std::string& qualityTier,
    const std::string& subject) {

    GRIM::Pipeline::TaggedEntry entry;
    entry.id = generateEntryId(content);
    entry.content = content;
    entry.sourceUrl = sourceUrl;
    entry.sourceType = sourceType;
    entry.qualityTier = qualityTier;
    entry.subject = subject;
    entry.reliabilityScore = 0.8f;
    entry.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    entry.verified = false;
    entry.structured = true;
    entry.structuredOutput = structuredOutput;

    auto io = std::make_shared<GRIM::Pipeline::DatasetIOJson>();
    std::vector<GRIM::Pipeline::TaggedEntry> batch = {entry};
    if (!io->appendEntries(massDatasetPath_, batch)) return false;

    loadMassDataset(io);
    return true;
}

// ─── ConceptBlock persistence ────────────────────────────

fs::path DatasetTarget::conceptBlocksPath() const {
    return massDatasetPath_.parent_path() / "concept_blocks.fb";
}

fs::path DatasetTarget::legacyConceptBlocksPath() const {
    return massDatasetPath_.parent_path() / "concept_blocks.jsonl";
}

void DatasetTarget::rebuildCBIndex() {
    cbIdIndex_.clear();
    for (size_t i = 0; i < conceptBlocks_.size(); ++i)
        cbIdIndex_[conceptBlocks_[i].id] = i;
}

bool DatasetTarget::loadConceptBlocks() {
    conceptBlocks_.clear();
    cbIdIndex_.clear();
    const fs::path path = conceptBlocksPath();
    const fs::path legacy_path = legacyConceptBlocksPath();

    bool legacy_is_newer = false;
    if (fs::exists(path) && fs::exists(legacy_path)) {
        std::error_code fb_time_error;
        std::error_code legacy_time_error;
        const auto fb_time = fs::last_write_time(path, fb_time_error);
        const auto legacy_time = fs::last_write_time(legacy_path, legacy_time_error);
        legacy_is_newer = !fb_time_error && !legacy_time_error
            && legacy_time > fb_time;
    }

    if (fs::exists(path) && !legacy_is_newer) {
        std::string error;
        if (!GRIM::ConceptBlockIO::loadFlatBuffer(path, conceptBlocks_, &error)) {
            std::cerr << "[DatasetTarget] Error loading " << path.string()
                      << ": " << error << "\n";
            return false;
        }
        rebuildCBIndex();
        return true;
    }

    // One-time, non-destructive migration. This also refreshes the FB when a
    // legacy maintenance script has updated the JSONL more recently. The
    // JSONL remains in place as a rollback artifact.
    if (!fs::exists(legacy_path)) return true;

    std::ifstream file(legacy_path);
    if (!file.is_open()) return false;

    std::string line;
    size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        if (line.empty()) continue;
        try {
            conceptBlocks_.push_back(conceptBlockFromJson(json::parse(line)));
        } catch (const std::exception& ex) {
            std::cerr << "[DatasetTarget] Skipping invalid legacy concept block at line "
                      << line_number << ": " << ex.what() << "\n";
        }
    }
    rebuildCBIndex();
    if (!saveConceptBlocks()) {
        std::cerr << "[DatasetTarget] Failed to migrate " << legacy_path.string()
                  << " to " << path.string() << "\n";
        return false;
    }
    std::cout << "[DatasetTarget] Migrated " << conceptBlocks_.size()
              << " concept blocks to " << path.string() << "\n";
    return true;
}

bool DatasetTarget::saveConceptBlocks() const {
    std::string error;
    if (!GRIM::ConceptBlockIO::saveFlatBuffer(
            conceptBlocksPath(), conceptBlocks_, &error)) {
        std::cerr << "[DatasetTarget] Error saving concept blocks: "
                  << error << "\n";
        return false;
    }
    return true;
}

size_t DatasetTarget::conceptBlockCount() const {
    return conceptBlocks_.size();
}

GRIM::ConceptBlock DatasetTarget::getConceptBlock(size_t index) const {
    if (index >= conceptBlocks_.size()) return {};
    return conceptBlocks_[index];
}

GRIM::ConceptBlock DatasetTarget::getConceptBlockById(const std::string& cb_id) const {
    auto it = cbIdIndex_.find(cb_id);
    if (it == cbIdIndex_.end()) return {};
    return conceptBlocks_[it->second];
}

bool DatasetTarget::addConceptBlock(const GRIM::ConceptBlock& cb) {
    if (cb.id.empty()) return false;
    if (cbIdIndex_.count(cb.id)) return false;
    conceptBlocks_.push_back(cb);
    cbIdIndex_[cb.id] = conceptBlocks_.size() - 1;
    return saveConceptBlocks();
}

bool DatasetTarget::updateConceptBlock(const std::string& cb_id,
                                       const GRIM::ConceptBlock& cb) {
    auto it = cbIdIndex_.find(cb_id);
    if (it == cbIdIndex_.end()) return false;
    conceptBlocks_[it->second] = cb;
    conceptBlocks_[it->second].id = cb_id;
    if (cb.id != cb_id) {
        cbIdIndex_.erase(cb.id);
        cbIdIndex_[cb_id] = it->second;
    }
    return saveConceptBlocks();
}

bool DatasetTarget::removeConceptBlock(const std::string& cb_id) {
    auto it = cbIdIndex_.find(cb_id);
    if (it == cbIdIndex_.end()) return false;
    conceptBlocks_.erase(conceptBlocks_.begin() +
                         static_cast<ptrdiff_t>(it->second));
    rebuildCBIndex();

    // Remove from any curriculum that contains this block.
    bool currChanged = false;
    for (auto& curr : curriculums_) {
        if (curr.removeBlock(cb_id))
            currChanged = true;
    }
    if (currChanged) saveCurriculumRegistry();

    return saveConceptBlocks();
}

std::vector<size_t> DatasetTarget::searchConceptBlocks(
    const std::string& query, size_t max_results) const {
    std::vector<size_t> results;
    std::string lq = toLower(query);
    for (size_t i = 0; i < conceptBlocks_.size(); ++i) {
        const auto& cb = conceptBlocks_[i];
        if (!lq.empty()) {
            bool match = toLower(cb.name).find(lq) != std::string::npos
                      || toLower(cb.prompt).find(lq) != std::string::npos
                      || toLower(cb.answer).find(lq) != std::string::npos;
            if (!match) {
                for (const auto& line : cb.intermediates) {
                    if (toLower(line).find(lq) != std::string::npos) {
                        match = true;
                        break;
                    }
                }
            }
            if (!match) {
                for (const auto& line : cb.explanation) {
                    if (toLower(line).find(lq) != std::string::npos) {
                        match = true;
                        break;
                    }
                }
            }
            if (!match) continue;
        }
        results.push_back(i);
        if (results.size() >= max_results) break;
    }
    return results;
}

std::vector<size_t> DatasetTarget::filterConceptBlocks(
    const std::string& format_type,
    const std::string& search_query) const {
    std::vector<size_t> results;
    std::string lq = toLower(search_query);
    for (size_t i = 0; i < conceptBlocks_.size(); ++i) {
        const auto& cb = conceptBlocks_[i];
        if (!format_type.empty() && format_type != "All"
            && cb.format_type != format_type)
            continue;
        if (!lq.empty()) {
            bool match = toLower(cb.name).find(lq) != std::string::npos
                      || toLower(cb.prompt).find(lq) != std::string::npos
                      || toLower(cb.answer).find(lq) != std::string::npos;
            if (!match) {
                for (const auto& line : cb.intermediates) {
                    if (toLower(line).find(lq) != std::string::npos) {
                        match = true;
                        break;
                    }
                }
            }
            if (!match) {
                for (const auto& line : cb.explanation) {
                    if (toLower(line).find(lq) != std::string::npos) {
                        match = true;
                        break;
                    }
                }
            }
            if (!match) continue;
        }
        results.push_back(i);
    }
    return results;
}

// ─── Curriculum registry ─────────────────────────────────

fs::path DatasetTarget::curriculumRegistryPath() const {
    return massDatasetPath_.parent_path() / "curriculum_registry.json";
}

void DatasetTarget::rebuildCurrIndex() {
    currIdIndex_.clear();
    for (size_t i = 0; i < curriculums_.size(); ++i)
        currIdIndex_[curriculums_[i].id] = i;
}

bool DatasetTarget::loadCurriculumRegistry() {
    curriculums_.clear();
    currIdIndex_.clear();
    fs::path path = curriculumRegistryPath();
    if (!fs::exists(path)) return true;

    try {
        std::ifstream file(path);
        if (!file.is_open()) return false;
        json j = json::parse(file);
        if (j.contains("curriculums") && j["curriculums"].is_array()) {
            for (const auto& cj : j["curriculums"]) {
                GRIM::Curriculum curr;
                curr.id        = cj.value("id", std::string());
                curr.name      = cj.value("name", std::string());
                curr.timestamp = cj.value("timestamp", int64_t(0));
                curr.format_as_concept = cj.value("format_as_concept", true);
                if (cj.contains("concept_block_ids") && cj["concept_block_ids"].is_array()) {
                    for (const auto& bid : cj["concept_block_ids"]) {
                        if (bid.is_string())
                            curr.concept_block_ids.push_back(bid.get<std::string>());
                    }
                }
                if (!curr.id.empty())
                    curriculums_.push_back(std::move(curr));
            }
        }
        rebuildCurrIndex();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[DatasetTarget] Error loading curriculum registry: " << e.what() << "\n";
        return false;
    }
}

bool DatasetTarget::saveCurriculumRegistry() const {
    fs::path path = curriculumRegistryPath();
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);

    json j;
    j["curriculums"] = json::array();
    for (const auto& curr : curriculums_) {
        json cj;
        cj["id"]                = curr.id;
        cj["name"]              = curr.name;
        cj["timestamp"]         = curr.timestamp;
        cj["format_as_concept"] = curr.format_as_concept;
        cj["concept_block_ids"] = curr.concept_block_ids;
        j["curriculums"].push_back(std::move(cj));
    }

    fs::path tmpPath = path;
    tmpPath += ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        if (!out.is_open()) return false;
        out << j.dump(2) << "\n";
        if (!out.good()) return false;
    }
    fs::rename(tmpPath, path, ec);
    return !ec;
}

size_t DatasetTarget::curriculumCount() const {
    return curriculums_.size();
}

const std::vector<GRIM::Curriculum>& DatasetTarget::getCurriculums() const {
    return curriculums_;
}

GRIM::Curriculum DatasetTarget::getCurriculum(size_t index) const {
    if (index >= curriculums_.size()) return {};
    return curriculums_[index];
}

GRIM::Curriculum DatasetTarget::getCurriculumById(const std::string& curr_id) const {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return {};
    return curriculums_[it->second];
}

size_t DatasetTarget::getCurriculumIndexById(const std::string& curr_id) const {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return SIZE_MAX;
    return it->second;
}

bool DatasetTarget::addCurriculum(const GRIM::Curriculum& curr) {
    if (curr.id.empty()) return false;
    if (currIdIndex_.count(curr.id)) return false;
    curriculums_.push_back(curr);
    currIdIndex_[curr.id] = curriculums_.size() - 1;
    return saveCurriculumRegistry();
}

bool DatasetTarget::updateCurriculum(const std::string& curr_id,
                                     const GRIM::Curriculum& curr) {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return false;
    curriculums_[it->second] = curr;
    curriculums_[it->second].id = curr_id;
    return saveCurriculumRegistry();
}

bool DatasetTarget::removeCurriculum(const std::string& curr_id) {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return false;
    curriculums_.erase(curriculums_.begin() +
                       static_cast<ptrdiff_t>(it->second));
    rebuildCurrIndex();

    // Also remove from model assignment if assigned.
    if (assignedCurrSet_.count(curr_id)) {
        assignedCurrSet_.erase(curr_id);
        assignedCurrOrder_.erase(
            std::remove(assignedCurrOrder_.begin(), assignedCurrOrder_.end(), curr_id),
            assignedCurrOrder_.end());
        saveAssignments();
    }
    return saveCurriculumRegistry();
}

// ─── Concept block ↔ curriculum assignment ───────────────

bool DatasetTarget::addConceptBlockToCurriculum(const std::string& cb_id,
                                                const std::string& curr_id) {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return false;
    if (!curriculums_[it->second].addBlock(cb_id)) return false;
    return saveCurriculumRegistry();
}

bool DatasetTarget::removeConceptBlockFromCurriculum(const std::string& cb_id,
                                                     const std::string& curr_id) {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return false;
    if (!curriculums_[it->second].removeBlock(cb_id)) return false;
    return saveCurriculumRegistry();
}

bool DatasetTarget::isConceptBlockInCurriculum(const std::string& cb_id,
                                               const std::string& curr_id) const {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return false;
    return curriculums_[it->second].containsBlock(cb_id);
}

size_t DatasetTarget::conceptBlockCountInCurriculum(const std::string& curr_id) const {
    auto it = currIdIndex_.find(curr_id);
    if (it == currIdIndex_.end()) return 0;
    return curriculums_[it->second].concept_block_ids.size();
}

// ─── Curriculum ↔ model assignment ───────────────────────

bool DatasetTarget::assignCurriculumToModel(const std::string& curr_id) {
    if (curr_id.empty() || activeModelId_.empty()) return false;
    if (assignedCurrSet_.count(curr_id)) return true;
    assignedCurrOrder_.push_back(curr_id);
    assignedCurrSet_.insert(curr_id);
    return saveAssignments();
}

bool DatasetTarget::removeCurriculumFromModel(const std::string& curr_id) {
    if (!assignedCurrSet_.count(curr_id)) return false;
    assignedCurrSet_.erase(curr_id);
    assignedCurrOrder_.erase(
        std::remove(assignedCurrOrder_.begin(), assignedCurrOrder_.end(), curr_id),
        assignedCurrOrder_.end());
    return saveAssignments();
}

bool DatasetTarget::isCurriculumAssigned(const std::string& curr_id) const {
    return assignedCurrSet_.count(curr_id) > 0;
}

size_t DatasetTarget::assignedCurriculumCount() const {
    return assignedCurrOrder_.size();
}

const std::vector<std::string>& DatasetTarget::assignedCurriculumOrder() const {
    return assignedCurrOrder_;
}

// ─── Curriculum manifest export ──────────────────────────

bool DatasetTarget::exportCurriculumManifest() const {
    // Collect the union of concept_block_ids from all assigned curricula,
    // partitioned by format_as_concept flag.
    std::set<std::string> concept_set;
    std::set<std::string> plaintext_set;
    for (const auto& curr_id : assignedCurrOrder_) {
        auto it = currIdIndex_.find(curr_id);
        if (it == currIdIndex_.end()) continue;
        const auto& curr = curriculums_[it->second];
        for (const auto& cb_id : curr.concept_block_ids) {
            if (curr.format_as_concept) {
                concept_set.insert(cb_id);
            } else {
                plaintext_set.insert(cb_id);
            }
        }
    }
    // Concept formatting takes priority if a block appears in both.
    for (const auto& id : concept_set) {
        plaintext_set.erase(id);
    }

    fs::path manifest_path = massDatasetPath_.parent_path() / "curriculum_manifest.json";

    // If no curricula are assigned, remove the manifest so the DataLoader
    // falls back to loading all concept blocks.
    if (assignedCurrOrder_.empty()) {
        std::error_code ec;
        fs::remove(manifest_path, ec);
        return true;
    }

    json j;
    j["model_id"] = activeModelId_;
    j["curriculum_ids"] = json::array();
    for (const auto& curr_id : assignedCurrOrder_) {
        j["curriculum_ids"].push_back(curr_id);
    }
    j["concept_block_ids"] = json::array();
    for (const auto& cb_id : concept_set) {
        j["concept_block_ids"].push_back(cb_id);
    }
    j["plaintext_block_ids"] = json::array();
    for (const auto& cb_id : plaintext_set) {
        j["plaintext_block_ids"].push_back(cb_id);
    }

    fs::path tmpPath = manifest_path;
    tmpPath += ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        if (!out.is_open()) return false;
        out << j.dump(2) << "\n";
        if (!out.good()) return false;
    }
    std::error_code ec;
    fs::rename(tmpPath, manifest_path, ec);
    return !ec;
}
