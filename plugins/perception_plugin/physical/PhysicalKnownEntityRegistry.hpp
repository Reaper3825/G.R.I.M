#pragma once

#include "PhysicalWorldStateResult.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// Durable identity profile plus its optional live tracker binding.
struct PhysicalKnownEntityRecord {
    uint64_t            known_entity_id = 0; // stable inside tracker session
    uint64_t            object_id = 0;
    std::string         name;
    std::string         persistent_entity_id;
    PhysicalWorldEntity last_observation;
    bool                currently_tracked = false;
    std::vector<uint64_t> track_history;
    uint32_t            automatic_relink_count = 0;
    float               last_automatic_relink_score = 0.0f;
    size_t              face_template_count = 0;
    float               last_face_match_score = 0.0f;
    float               last_face_quality = 0.0f;
    PhysicalEntityIdentityState identity_state = PhysicalEntityIdentityState::Unknown;
};

// Refresh last observations and live/stale state for already-named records.
// Unnamed detections are intentionally not inserted into the registry.
void ObservePhysicalWorldStateForKnownEntities(
    PhysicalWorldStateSnapshot& snapshot,
    const std::vector<PhysicalFaceEmbedding>& face_embeddings = {});

// Create or rename a binding. Throws std::invalid_argument for bad input.
void AssignPhysicalEntityName(const PhysicalWorldEntity& entity,
                              const std::string& name);
void RenamePhysicalKnownEntity(uint64_t known_entity_id,
                               const std::string& name);

// Attach a newly-created live tracker identity to a named identity that is
// currently out of frame. The stable known_entity_id and name are retained;
// object_id advances and the prior tracker IDs remain in track_history.
void LinkPhysicalEntityToKnownEntity(const PhysicalWorldEntity& live_entity,
                                     uint64_t known_entity_id);

bool ClearPhysicalEntityName(uint64_t object_id);
// Explicit biometric operations. Enrollment only uses the latest good face
// associated with the selected live person track; unknown faces are never
// persisted automatically.
void EnrollPhysicalEntityFace(uint64_t object_id);
bool ForgetPhysicalEntityFaceData(uint64_t known_entity_id);
bool DeletePhysicalKnownEntity(uint64_t known_entity_id);
std::string ResolvePhysicalEntityName(uint64_t object_id);
std::vector<PhysicalKnownEntityRecord> GetPhysicalKnownEntitiesSnapshot();
// Changes when a binding is edited or a known entity enters/leaves tracking.
uint64_t GetPhysicalKnownEntityRegistryRevision();

// Called with tracker reset so restarted IDs never inherit stale names.
void ResetPhysicalKnownEntityRegistry();

}}} // namespace GRIM::Perception::Physical
