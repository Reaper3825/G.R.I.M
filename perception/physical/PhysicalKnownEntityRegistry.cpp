#include "PhysicalKnownEntityRegistry.hpp"
#include "PhysicalIdentityStore.hpp"
#include "PhysicalFaceIdentityMatcher.hpp"
#include "PhysicalWorldStateLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace GRIM { namespace Perception { namespace Physical {
namespace {

constexpr float kMinimumEnrollmentQuality = 0.65f;
constexpr uint32_t kRecognitionEvidenceRequired = 3;
constexpr size_t kMaximumTemplatesPerIdentity = 24;

struct CandidateEvidence {
    std::string persistent_entity_id;
    uint32_t consecutive_hits = 0;
    float last_score = 0.0f;
    uint64_t last_evidence_frame = 0;
    bool logged_first_attempt = false;
    bool logged_first_accepted_candidate = false;
};

struct RegistryState {
    std::mutex mutex;
    bool loaded = false;
    std::vector<PhysicalIdentityProfile> profiles;
    std::unordered_map<uint64_t, PhysicalKnownEntityRecord> records_by_known_id;
    std::unordered_map<std::string, uint64_t> known_id_by_persistent_id;
    std::unordered_map<uint64_t, uint64_t> known_id_by_track_id;
    std::unordered_map<uint64_t, PhysicalFaceEmbedding> latest_face_by_track_id;
    std::unordered_map<uint64_t, CandidateEvidence> candidate_by_track_id;
    uint64_t next_known_entity_id = 1;
    uint64_t revision = 0;
};

RegistryState& State() { static RegistryState state; return state; }

std::string GeneratePersistentId() {
    std::array<unsigned char, 16> bytes{};
    std::random_device random;
    for (auto& byte : bytes) byte = static_cast<unsigned char>(random());
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0fU) | 0x40U);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3fU) | 0x80U);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (size_t i = 0; i < bytes.size(); ++i) {
        out << std::setw(2) << static_cast<unsigned int>(bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) out << '-';
    }
    return out.str();
}

std::string TrimAndValidateName(const std::string& input) {
    auto first = std::find_if_not(input.begin(), input.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    auto last = std::find_if_not(input.rbegin(), input.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) throw std::invalid_argument("Physical entity name must not be empty");
    std::string name(first, last);
    if (name.size() > 96) throw std::invalid_argument("Physical entity name must be at most 96 bytes");
    for (unsigned char c : name) {
        if (c < 0x20 || c == 0x7f) {
            throw std::invalid_argument("Physical entity name must not contain control characters");
        }
    }
    return name;
}

void LoadProfilesLocked(RegistryState& state) {
    if (state.loaded) return;
    state.profiles = LoadPhysicalIdentityProfiles();
    for (const auto& profile : state.profiles) {
        PhysicalKnownEntityRecord record;
        record.known_entity_id = state.next_known_entity_id++;
        record.persistent_entity_id = profile.persistent_entity_id;
        record.name = profile.display_name;
        record.face_template_count = profile.face_templates.size();
        record.identity_state = profile.face_templates.empty()
            ? PhysicalEntityIdentityState::NamedOnly
            : PhysicalEntityIdentityState::Enrolled;
        state.known_id_by_persistent_id[profile.persistent_entity_id] = record.known_entity_id;
        state.records_by_known_id.emplace(record.known_entity_id, std::move(record));
    }
    state.loaded = true;
    ++state.revision;
    size_t template_count = 0;
    for (const auto& profile : state.profiles) {
        template_count += profile.face_templates.size();
    }
    LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
        std::string("PhysicalIdentityRegistry: loaded profiles=")
        + std::to_string(state.profiles.size())
        + " face_templates=" + std::to_string(template_count));
}

PhysicalIdentityProfile* FindProfileLocked(RegistryState& state, const std::string& id) {
    for (auto& profile : state.profiles) {
        if (profile.persistent_entity_id == id) return &profile;
    }
    return nullptr;
}

void SaveProfilesLocked(const RegistryState& state) {
    SavePhysicalIdentityProfiles(state.profiles);
}

bool FaceBelongsToEntity(const PhysicalFaceEmbedding& face, const PhysicalWorldEntity& entity) {
    if (entity.class_label != "person" || entity.track_state != PhysicalEntityTrackState::Confirmed) return false;
    const cv::Point2f centre(face.model_bbox.x + face.model_bbox.width * 0.5f,
                             face.model_bbox.y + face.model_bbox.height * 0.5f);
    return centre.x >= entity.model_box.x && centre.x <= entity.model_box.x + entity.model_box.width &&
           centre.y >= entity.model_box.y && centre.y <= entity.model_box.y + entity.model_box.height * 0.65f;
}

void AssociateFacesLocked(RegistryState& state, const PhysicalWorldStateSnapshot& snapshot,
                          const std::vector<PhysicalFaceEmbedding>& faces) {
    for (const auto& face : faces) {
        const PhysicalWorldEntity* owner = nullptr;
        float owner_area = 0.0f;
        for (const auto& entity : snapshot.entities) {
            if (!FaceBelongsToEntity(face, entity)) continue;
            const float area = entity.model_box.area();
            if (!owner || area < owner_area) { owner = &entity; owner_area = area; }
        }
        if (!owner) continue;
        auto existing = state.latest_face_by_track_id.find(owner->object_id);
        if (existing == state.latest_face_by_track_id.end() ||
            existing->second.source_frame_counter < face.source_frame_counter ||
            (existing->second.source_frame_counter == face.source_frame_counter &&
             existing->second.quality_score < face.quality_score)) {
            state.latest_face_by_track_id[owner->object_id] = face;
        }
    }
}

void BindTrackLocked(RegistryState& state, PhysicalKnownEntityRecord& record,
                     const PhysicalWorldEntity& entity,
                     PhysicalEntityIdentityState identity_state, float score) {
    if (record.object_id != 0 && record.object_id != entity.object_id) {
        state.known_id_by_track_id.erase(record.object_id);
    }
    record.object_id = entity.object_id;
    record.currently_tracked = true;
    record.last_observation = entity;
    record.identity_state = identity_state;
    record.last_face_match_score = score;
    if (std::find(record.track_history.begin(), record.track_history.end(), entity.object_id) ==
        record.track_history.end()) record.track_history.push_back(entity.object_id);
    state.known_id_by_track_id[entity.object_id] = record.known_entity_id;
}

bool DeleteKnownEntityLocked(RegistryState& state, uint64_t known_entity_id) {
    auto record = state.records_by_known_id.find(known_entity_id);
    if (record == state.records_by_known_id.end()) return false;
    const std::string persistent_id = record->second.persistent_entity_id;
    if (record->second.object_id != 0) state.known_id_by_track_id.erase(record->second.object_id);
    state.known_id_by_persistent_id.erase(persistent_id);
    state.profiles.erase(std::remove_if(state.profiles.begin(), state.profiles.end(),
        [&](const auto& profile) { return profile.persistent_entity_id == persistent_id; }),
        state.profiles.end());
    state.records_by_known_id.erase(record);
    SaveProfilesLocked(state);
    ++state.revision;
    return true;
}

} // anonymous namespace

void ObservePhysicalWorldStateForKnownEntities(PhysicalWorldStateSnapshot& snapshot,
    const std::vector<PhysicalFaceEmbedding>& face_embeddings) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    AssociateFacesLocked(state, snapshot, face_embeddings);

    std::unordered_set<uint64_t> live_tracks;
    for (const auto& entity : snapshot.entities) live_tracks.insert(entity.object_id);
    for (auto it = state.latest_face_by_track_id.begin();
         it != state.latest_face_by_track_id.end();) {
        if (live_tracks.count(it->first) == 0) it = state.latest_face_by_track_id.erase(it);
        else ++it;
    }
    for (auto it = state.candidate_by_track_id.begin();
         it != state.candidate_by_track_id.end();) {
        if (live_tracks.count(it->first) == 0) it = state.candidate_by_track_id.erase(it);
        else ++it;
    }
    for (auto& item : state.records_by_known_id) {
        auto& record = item.second;
        const bool live = record.object_id != 0 && live_tracks.count(record.object_id) != 0;
        if (record.currently_tracked != live) ++state.revision;
        record.currently_tracked = live;
        if (!live && record.object_id != 0) {
            state.known_id_by_track_id.erase(record.object_id);
            record.object_id = 0;
        }
    }

    for (auto& entity : snapshot.entities) {
        auto binding = state.known_id_by_track_id.find(entity.object_id);
        if (binding == state.known_id_by_track_id.end()) {
            auto face_it = state.latest_face_by_track_id.find(entity.object_id);
            if (face_it != state.latest_face_by_track_id.end()) {
                const auto& face = face_it->second;
                entity.identity_face_embedding_present = true;
                entity.identity_face_quality = face.quality_score;
                auto& evidence = state.candidate_by_track_id[entity.object_id];
                if (face.source_frame_counter > evidence.last_evidence_frame) {
                    const auto match = MatchPhysicalFaceIdentity(face, state.profiles);
                    if (!evidence.logged_first_attempt) {
                        LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
                            std::string("PhysicalIdentityRegistry: first match attempt person#")
                            + std::to_string(entity.object_id)
                            + " nearest_cosine=" + std::to_string(match.best_similarity)
                            + " accepted=" + (match.accepted ? "true" : "false")
                            + " embedding_model='" + face.embedding_model_id + "'");
                        evidence.logged_first_attempt = true;
                    }
                    if (match.accepted) {
                        if (evidence.persistent_entity_id == match.persistent_entity_id) {
                            ++evidence.consecutive_hits;
                        } else {
                            evidence.persistent_entity_id = match.persistent_entity_id;
                            evidence.consecutive_hits = 1;
                        }
                        if (!evidence.logged_first_accepted_candidate) {
                            const auto* profile = FindProfileLocked(
                                state, match.persistent_entity_id);
                            LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
                                std::string("PhysicalIdentityRegistry: candidate person#")
                                + std::to_string(entity.object_id) + " nearest='"
                                + (profile ? profile->display_name : std::string("unknown"))
                                + "' cosine=" + std::to_string(match.best_similarity));
                            evidence.logged_first_accepted_candidate = true;
                        }
                    } else {
                        // Preserve the nearest score/name for UI diagnosis, but
                        // clear accumulated evidence so rejected observations
                        // can never contribute to automatic recognition.
                        evidence.persistent_entity_id = match.persistent_entity_id;
                        evidence.consecutive_hits = 0;
                    }
                    evidence.last_score = match.best_similarity;
                    evidence.last_evidence_frame = face.source_frame_counter;
                    ++state.revision;
                }
                entity.identity_state = evidence.consecutive_hits == 0
                    ? PhysicalEntityIdentityState::Unknown : PhysicalEntityIdentityState::Candidate;
                entity.identity_confidence = evidence.last_score;
                entity.identity_evidence_hits = evidence.consecutive_hits;
                entity.identity_evidence_required = kRecognitionEvidenceRequired;
                if (const auto* candidate = FindProfileLocked(
                        state, evidence.persistent_entity_id)) {
                    entity.identity_candidate_display_name = candidate->display_name;
                }
                if (evidence.consecutive_hits >= kRecognitionEvidenceRequired) {
                    const auto known = state.known_id_by_persistent_id.find(evidence.persistent_entity_id);
                    if (known != state.known_id_by_persistent_id.end()) {
                        auto record = state.records_by_known_id.find(known->second);
                        if (record != state.records_by_known_id.end() && !record->second.currently_tracked) {
                            BindTrackLocked(state, record->second, entity,
                                PhysicalEntityIdentityState::Recognized, evidence.last_score);
                            ++record->second.automatic_relink_count;
                            record->second.last_automatic_relink_score = evidence.last_score;
                            LOG_DEBUG(PHYSICAL_WORLD_STATE_LOG_TAG,
                                std::string("PhysicalIdentityRegistry: recognized '")
                                + record->second.name + "' on person#"
                                + std::to_string(entity.object_id)
                                + " cosine=" + std::to_string(evidence.last_score)
                                + " evidence="
                                + std::to_string(evidence.consecutive_hits) + "/"
                                + std::to_string(kRecognitionEvidenceRequired));
                            binding = state.known_id_by_track_id.find(entity.object_id);
                            ++state.revision;
                        }
                    }
                }
            }
        }

        if (binding != state.known_id_by_track_id.end()) {
            auto record_it = state.records_by_known_id.find(binding->second);
            if (record_it == state.records_by_known_id.end()) continue;
            auto& record = record_it->second;
            record.last_observation = entity;
            record.currently_tracked = true;
            auto face = state.latest_face_by_track_id.find(entity.object_id);
            if (face != state.latest_face_by_track_id.end() &&
                record.last_face_quality != face->second.quality_score) {
                record.last_face_quality = face->second.quality_score;
                ++state.revision;
            }
            if (face != state.latest_face_by_track_id.end()) {
                entity.identity_face_embedding_present = true;
                entity.identity_face_quality = face->second.quality_score;
            }
            entity.persistent_entity_id = record.persistent_entity_id;
            entity.display_name = record.name;
            entity.identity_state = record.face_template_count == 0
                ? PhysicalEntityIdentityState::NamedOnly : record.identity_state;
            entity.identity_confidence = record.last_face_match_score;
            entity.identity_evidence_hits = kRecognitionEvidenceRequired;
            entity.identity_evidence_required = kRecognitionEvidenceRequired;
        }
    }
}

void AssignPhysicalEntityName(const PhysicalWorldEntity& entity, const std::string& requested_name) {
    if (entity.object_id == 0) throw std::invalid_argument("Cannot name a physical entity with object_id 0");
    const std::string name = TrimAndValidateName(requested_name);
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    const auto mapping = state.known_id_by_track_id.find(entity.object_id);
    if (mapping != state.known_id_by_track_id.end()) {
        auto& record = state.records_by_known_id.at(mapping->second);
        auto* profile = FindProfileLocked(state, record.persistent_entity_id);
        if (!profile) throw std::logic_error("identity profile is missing");
        record.name = name;
        record.last_observation = entity;
        profile->display_name = name;
        profile->updated_at_unix_seconds = static_cast<uint64_t>(std::time(nullptr));
        SaveProfilesLocked(state);
        ++state.revision;
        return;
    }
    const uint64_t now = static_cast<uint64_t>(std::time(nullptr));
    PhysicalIdentityProfile profile;
    profile.persistent_entity_id = GeneratePersistentId();
    profile.display_name = name;
    profile.created_at_unix_seconds = now;
    profile.updated_at_unix_seconds = now;
    state.profiles.push_back(profile);
    PhysicalKnownEntityRecord record;
    record.known_entity_id = state.next_known_entity_id++;
    record.persistent_entity_id = profile.persistent_entity_id;
    record.name = name;
    BindTrackLocked(state, record, entity, PhysicalEntityIdentityState::NamedOnly, 0.0f);
    state.known_id_by_persistent_id[record.persistent_entity_id] = record.known_entity_id;
    state.records_by_known_id.emplace(record.known_entity_id, std::move(record));
    SaveProfilesLocked(state);
    ++state.revision;
}

void RenamePhysicalKnownEntity(uint64_t known_entity_id,
                               const std::string& requested_name) {
    const std::string name = TrimAndValidateName(requested_name);
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    auto record = state.records_by_known_id.find(known_entity_id);
    if (record == state.records_by_known_id.end()) {
        throw std::invalid_argument("Selected identity no longer exists");
    }
    auto* profile = FindProfileLocked(state, record->second.persistent_entity_id);
    if (!profile) throw std::logic_error("identity profile is missing");
    record->second.name = name;
    profile->display_name = name;
    profile->updated_at_unix_seconds = static_cast<uint64_t>(std::time(nullptr));
    SaveProfilesLocked(state);
    ++state.revision;
}

void EnrollPhysicalEntityFace(uint64_t object_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    const auto mapping = state.known_id_by_track_id.find(object_id);
    if (mapping == state.known_id_by_track_id.end()) throw std::invalid_argument("Name the selected person before enrolling a face");
    auto face = state.latest_face_by_track_id.find(object_id);
    if (face == state.latest_face_by_track_id.end()) throw std::invalid_argument("No recognition embedding is available for this person");
    if (face->second.quality_score < kMinimumEnrollmentQuality) throw std::invalid_argument("Face quality is too low for enrollment");
    auto& record = state.records_by_known_id.at(mapping->second);
    auto* profile = FindProfileLocked(state, record.persistent_entity_id);
    if (!profile) throw std::logic_error("identity profile is missing");
    if (profile->face_templates.size() >= kMaximumTemplatesPerIdentity) throw std::invalid_argument("This identity has reached its face-template limit");
    for (const auto& stored : profile->face_templates) {
        if (stored.model_id == face->second.embedding_model_id &&
            ComputePhysicalFaceCosineSimilarity(stored.embedding, face->second.embedding) > 0.995f) {
            throw std::invalid_argument("This face sample is a duplicate of an enrollment");
        }
    }
    PhysicalFaceTemplateRecord stored;
    stored.model_id = face->second.embedding_model_id;
    stored.embedding = face->second.embedding;
    stored.quality_score = face->second.quality_score;
    stored.enrolled_at_unix_seconds = static_cast<uint64_t>(std::time(nullptr));
    profile->face_templates.push_back(std::move(stored));
    profile->updated_at_unix_seconds = static_cast<uint64_t>(std::time(nullptr));
    record.face_template_count = profile->face_templates.size();
    record.last_face_quality = face->second.quality_score;
    record.identity_state = PhysicalEntityIdentityState::Enrolled;
    SaveProfilesLocked(state);
    ++state.revision;
}

bool ForgetPhysicalEntityFaceData(uint64_t known_entity_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    auto record = state.records_by_known_id.find(known_entity_id);
    if (record == state.records_by_known_id.end()) return false;
    auto* profile = FindProfileLocked(state, record->second.persistent_entity_id);
    if (!profile || profile->face_templates.empty()) return false;
    profile->face_templates.clear();
    profile->updated_at_unix_seconds = static_cast<uint64_t>(std::time(nullptr));
    record->second.face_template_count = 0;
    record->second.identity_state = PhysicalEntityIdentityState::NamedOnly;
    record->second.last_face_match_score = 0.0f;
    SaveProfilesLocked(state);
    ++state.revision;
    return true;
}

bool DeletePhysicalKnownEntity(uint64_t known_entity_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    return DeleteKnownEntityLocked(state, known_entity_id);
}

void LinkPhysicalEntityToKnownEntity(const PhysicalWorldEntity& entity, uint64_t known_entity_id) {
    if (entity.object_id == 0 || known_entity_id == 0) throw std::invalid_argument("LinkPhysicalEntityToKnownEntity requires non-zero IDs");
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    auto record = state.records_by_known_id.find(known_entity_id);
    if (record == state.records_by_known_id.end()) throw std::invalid_argument("Selected known entity no longer exists");
    if (record->second.currently_tracked && record->second.object_id != entity.object_id) throw std::invalid_argument("Known entity is already bound to a live track");
    const auto existing = state.known_id_by_track_id.find(entity.object_id);
    if (existing != state.known_id_by_track_id.end() && existing->second != known_entity_id) throw std::invalid_argument("Live track is already bound to another identity");
    BindTrackLocked(state, record->second, entity,
                    record->second.face_template_count == 0
                        ? PhysicalEntityIdentityState::NamedOnly
                        : PhysicalEntityIdentityState::Enrolled,
                    0.0f);
    ++state.revision;
}

bool ClearPhysicalEntityName(uint64_t object_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    const auto mapping = state.known_id_by_track_id.find(object_id);
    return mapping != state.known_id_by_track_id.end() && DeleteKnownEntityLocked(state, mapping->second);
}

std::string ResolvePhysicalEntityName(uint64_t object_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    const auto mapping = state.known_id_by_track_id.find(object_id);
    if (mapping == state.known_id_by_track_id.end()) return {};
    const auto record = state.records_by_known_id.find(mapping->second);
    return record == state.records_by_known_id.end() ? std::string{} : record->second.name;
}

std::vector<PhysicalKnownEntityRecord> GetPhysicalKnownEntitiesSnapshot() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    std::vector<PhysicalKnownEntityRecord> out;
    for (const auto& item : state.records_by_known_id) out.push_back(item.second);
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.currently_tracked != b.currently_tracked) return a.currently_tracked > b.currently_tracked;
        return a.name < b.name;
    });
    return out;
}

uint64_t GetPhysicalKnownEntityRegistryRevision() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    return state.revision;
}

void ResetPhysicalKnownEntityRegistry() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    LoadProfilesLocked(state);
    state.known_id_by_track_id.clear();
    state.latest_face_by_track_id.clear();
    state.candidate_by_track_id.clear();
    for (auto& item : state.records_by_known_id) {
        item.second.object_id = 0;
        item.second.currently_tracked = false;
        item.second.track_history.clear();
        item.second.identity_state = item.second.face_template_count == 0
            ? PhysicalEntityIdentityState::NamedOnly
            : PhysicalEntityIdentityState::Enrolled;
        item.second.last_face_match_score = 0.0f;
    }
    ++state.revision;
}

}}} // namespace GRIM::Perception::Physical
