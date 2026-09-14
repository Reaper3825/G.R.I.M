#include "PhysicalKnownEntityRegistry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

struct RegistryState {
    std::mutex mutex;
    std::unordered_map<uint64_t, PhysicalKnownEntityRecord> records_by_known_id;
    std::unordered_map<uint64_t, uint64_t> known_id_by_track_id;
    uint64_t next_known_entity_id = 1;
    uint64_t revision = 0;
};

RegistryState& State() {
    static RegistryState state;
    return state;
}

// Conservative re-association gates. This is deliberately not presented as
// biometric recognition: it only reconnects an unambiguous same-class track
// near the last known geometry during a bounded absence.
constexpr uint64_t kMaxAutomaticRelinkGapFrames = 900;
constexpr float kMaxAutomaticRelinkCentreDistance = 0.55f;
constexpr float kMinAutomaticRelinkAreaSimilarity = 0.40f;
constexpr float kMinAutomaticRelinkScore = 0.55f;
constexpr float kMinAutomaticRelinkMargin = 0.12f;

struct AutomaticRelinkCandidate {
    uint64_t known_entity_id = 0;
    const PhysicalWorldEntity* live_entity = nullptr;
    float score = 0.0f;
};

float BoxArea(const cv::Rect2f& box) {
    return std::max(0.0f, box.width) * std::max(0.0f, box.height);
}

float AutomaticRelinkScore(const PhysicalKnownEntityRecord& known,
                           const PhysicalWorldEntity& live,
                           const PhysicalWorldStateSnapshot& snapshot)
{
    if (known.last_observation.class_id != live.class_id) return -1.0f;
    if (known.last_observation.class_label != live.class_label) return -1.0f;
    if (live.track_state != PhysicalEntityTrackState::Confirmed) return -1.0f;
    if (snapshot.source_frame_counter <
        known.last_observation.last_seen_frame_counter) return -1.0f;
    const uint64_t gap = snapshot.source_frame_counter -
        known.last_observation.last_seen_frame_counter;
    if (gap > kMaxAutomaticRelinkGapFrames) return -1.0f;

    const float width = static_cast<float>(std::max(1, snapshot.model_image_width));
    const float height = static_cast<float>(std::max(1, snapshot.model_image_height));
    const float diagonal = std::sqrt(width * width + height * height);
    const float dx = live.model_centre.x - known.last_observation.model_centre.x;
    const float dy = live.model_centre.y - known.last_observation.model_centre.y;
    const float normalized_distance = std::sqrt(dx * dx + dy * dy) / diagonal;
    if (normalized_distance > kMaxAutomaticRelinkCentreDistance) return -1.0f;

    const float old_area = BoxArea(known.last_observation.model_box);
    const float new_area = BoxArea(live.model_box);
    if (old_area <= 0.0f || new_area <= 0.0f) return -1.0f;
    const float area_similarity = std::min(old_area, new_area) /
                                  std::max(old_area, new_area);
    if (area_similarity < kMinAutomaticRelinkAreaSimilarity) return -1.0f;

    const float distance_score = 1.0f -
        normalized_distance / kMaxAutomaticRelinkCentreDistance;
    return 0.65f * distance_score + 0.25f * area_similarity
         + 0.10f * std::clamp(live.confidence, 0.0f, 1.0f);
}

std::string TrimAndValidateName(const std::string& input) {
    auto first = std::find_if_not(input.begin(), input.end(),
        [](unsigned char c) { return std::isspace(c) != 0; });
    auto last = std::find_if_not(input.rbegin(), input.rend(),
        [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) {
        throw std::invalid_argument("Physical entity name must not be empty");
    }
    std::string name(first, last);
    if (name.size() > 96) {
        throw std::invalid_argument("Physical entity name must be at most 96 bytes");
    }
    for (unsigned char c : name) {
        if (c < 0x20 || c == 0x7f) {
            throw std::invalid_argument(
                "Physical entity name must not contain control characters");
        }
    }
    return name;
}

} // anonymous namespace

void ObservePhysicalWorldStateForKnownEntities(
    const PhysicalWorldStateSnapshot& snapshot)
{
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.records_by_known_id.empty()) return;

    std::unordered_map<uint64_t, const PhysicalWorldEntity*> live_by_id;
    live_by_id.reserve(snapshot.entities.size());
    bool membership_changed = false;
    for (const auto& entity : snapshot.entities) {
        if (entity.object_id == 0) continue;
        live_by_id[entity.object_id] = &entity;
    }
    for (auto& item : state.records_by_known_id) {
        auto& record = item.second;
        const auto live = live_by_id.find(record.object_id);
        const bool is_live = live != live_by_id.end();
        if (record.currently_tracked != is_live) membership_changed = true;
        record.currently_tracked = is_live;
        if (is_live) {
            record.last_observation = *live->second;
        }
    }

    // Reconnect stale known identities to newly confirmed tracks. A match is
    // accepted only when it is the best choice from both directions and has a
    // clear margin over the runner-up. Ambiguous scenes remain unlinked.
    std::vector<AutomaticRelinkCandidate> candidates;
    for (const auto& item : state.records_by_known_id) {
        const auto& known = item.second;
        if (known.currently_tracked) continue;
        for (const auto& entity : snapshot.entities) {
            if (state.known_id_by_track_id.find(entity.object_id) !=
                state.known_id_by_track_id.end()) continue;
            const float score = AutomaticRelinkScore(known, entity, snapshot);
            if (score >= kMinAutomaticRelinkScore) {
                candidates.push_back({item.first, &entity, score});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const AutomaticRelinkCandidate& a,
           const AutomaticRelinkCandidate& b) { return a.score > b.score; });

    std::unordered_map<uint64_t, bool> linked_known_ids;
    std::unordered_map<uint64_t, bool> linked_track_ids;
    for (const auto& candidate : candidates) {
        const uint64_t track_id = candidate.live_entity->object_id;
        if (linked_known_ids[candidate.known_entity_id] ||
            linked_track_ids[track_id]) continue;

        float next_known_score = -1.0f;
        float next_track_score = -1.0f;
        bool best_for_known = true;
        bool best_for_track = true;
        for (const auto& other : candidates) {
            if (&other == &candidate) continue;
            if (other.known_entity_id == candidate.known_entity_id) {
                if (other.score > candidate.score) best_for_known = false;
                else next_known_score = std::max(next_known_score, other.score);
            }
            if (other.live_entity->object_id == track_id) {
                if (other.score > candidate.score) best_for_track = false;
                else next_track_score = std::max(next_track_score, other.score);
            }
        }
        if (!best_for_known || !best_for_track) continue;
        if (next_known_score >= 0.0f &&
            candidate.score - next_known_score < kMinAutomaticRelinkMargin) continue;
        if (next_track_score >= 0.0f &&
            candidate.score - next_track_score < kMinAutomaticRelinkMargin) continue;

        auto record_it = state.records_by_known_id.find(candidate.known_entity_id);
        if (record_it == state.records_by_known_id.end()) continue;
        auto& record = record_it->second;
        record.object_id = track_id;
        record.last_observation = *candidate.live_entity;
        record.currently_tracked = true;
        record.track_history.push_back(track_id);
        ++record.automatic_relink_count;
        record.last_automatic_relink_score = candidate.score;
        state.known_id_by_track_id[track_id] = candidate.known_entity_id;
        linked_known_ids[candidate.known_entity_id] = true;
        linked_track_ids[track_id] = true;
        membership_changed = true;
    }
    if (membership_changed) ++state.revision;
}

void AssignPhysicalEntityName(const PhysicalWorldEntity& entity,
                              const std::string& requested_name)
{
    if (entity.object_id == 0) {
        throw std::invalid_argument(
            "Cannot name a physical entity with object_id 0");
    }
    const std::string name = TrimAndValidateName(requested_name);
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto mapping = state.known_id_by_track_id.find(entity.object_id);
    if (mapping != state.known_id_by_track_id.end()) {
        auto record = state.records_by_known_id.find(mapping->second);
        if (record == state.records_by_known_id.end()) {
            throw std::logic_error("Known entity track map is internally inconsistent");
        }
        if (record->second.object_id != entity.object_id) {
            throw std::invalid_argument(
                "Cannot rename a historical track; select its current known entity record");
        }
        record->second.name = name;
        record->second.last_observation = entity;
        ++state.revision;
        return;
    }

    PhysicalKnownEntityRecord record;
    record.known_entity_id = state.next_known_entity_id++;
    record.object_id = entity.object_id;
    record.name = name;
    record.last_observation = entity;
    record.currently_tracked = true;
    record.track_history.push_back(entity.object_id);
    state.known_id_by_track_id[entity.object_id] = record.known_entity_id;
    state.records_by_known_id.emplace(record.known_entity_id, std::move(record));
    ++state.revision;
}

void LinkPhysicalEntityToKnownEntity(const PhysicalWorldEntity& live_entity,
                                     uint64_t known_entity_id)
{
    if (live_entity.object_id == 0 || known_entity_id == 0) {
        throw std::invalid_argument(
            "LinkPhysicalEntityToKnownEntity requires non-zero IDs");
    }
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto record_it = state.records_by_known_id.find(known_entity_id);
    if (record_it == state.records_by_known_id.end()) {
        throw std::invalid_argument("Selected known entity no longer exists");
    }
    auto& record = record_it->second;
    if (record.currently_tracked && record.object_id != live_entity.object_id) {
        throw std::invalid_argument(
            "Known entity is still tracked; only out-of-frame identities can be linked");
    }
    const auto existing = state.known_id_by_track_id.find(live_entity.object_id);
    if (existing != state.known_id_by_track_id.end() &&
        existing->second != known_entity_id) {
        throw std::invalid_argument(
            "Live track is already bound to a different known entity");
    }
    if (std::find(record.track_history.begin(), record.track_history.end(),
                  live_entity.object_id) == record.track_history.end()) {
        record.track_history.push_back(live_entity.object_id);
    }
    state.known_id_by_track_id[live_entity.object_id] = known_entity_id;
    record.object_id = live_entity.object_id;
    record.last_observation = live_entity;
    record.currently_tracked = true;
    ++state.revision;
}

bool ClearPhysicalEntityName(uint64_t object_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto mapping = state.known_id_by_track_id.find(object_id);
    if (mapping == state.known_id_by_track_id.end()) return false;
    const uint64_t known_id = mapping->second;
    const auto record = state.records_by_known_id.find(known_id);
    if (record != state.records_by_known_id.end()) {
        for (uint64_t track_id : record->second.track_history) {
            state.known_id_by_track_id.erase(track_id);
        }
        state.records_by_known_id.erase(record);
    } else {
        state.known_id_by_track_id.erase(mapping);
    }
    ++state.revision;
    return true;
}

std::string ResolvePhysicalEntityName(uint64_t object_id) {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto mapping = state.known_id_by_track_id.find(object_id);
    if (mapping == state.known_id_by_track_id.end()) return {};
    const auto record = state.records_by_known_id.find(mapping->second);
    return record == state.records_by_known_id.end()
        ? std::string{} : record->second.name;
}

std::vector<PhysicalKnownEntityRecord> GetPhysicalKnownEntitiesSnapshot() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    std::vector<PhysicalKnownEntityRecord> out;
    out.reserve(state.records_by_known_id.size());
    for (const auto& item : state.records_by_known_id) out.push_back(item.second);
    std::sort(out.begin(), out.end(),
        [](const PhysicalKnownEntityRecord& a,
           const PhysicalKnownEntityRecord& b) {
            if (a.name != b.name) return a.name < b.name;
            return a.object_id < b.object_id;
        });
    return out;
}

uint64_t GetPhysicalKnownEntityRegistryRevision() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.revision;
}

void ResetPhysicalKnownEntityRegistry() {
    auto& state = State();
    std::lock_guard<std::mutex> lock(state.mutex);
    state.records_by_known_id.clear();
    state.known_id_by_track_id.clear();
    state.next_known_entity_id = 1;
    ++state.revision;
}

}}} // namespace GRIM::Perception::Physical
