#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalFaceTemplateRecord {
    std::string model_id;
    std::vector<float> embedding;
    float quality_score = 0.0f;
    uint64_t enrolled_at_unix_seconds = 0;
};

struct PhysicalIdentityProfile {
    std::string persistent_entity_id;
    std::string display_name;
    uint64_t created_at_unix_seconds = 0;
    uint64_t updated_at_unix_seconds = 0;
    std::vector<PhysicalFaceTemplateRecord> face_templates;
};

// Canonical local-only biometric profile store. This is intentionally not
// UnifiedMemoryStorage: face vectors have model-specific dimensions and an
// explicit enrollment/deletion lifecycle.
std::string GetPhysicalIdentityStorePath();
std::vector<PhysicalIdentityProfile> LoadPhysicalIdentityProfiles();
void SavePhysicalIdentityProfiles(
    const std::vector<PhysicalIdentityProfile>& profiles);
std::vector<PhysicalIdentityProfile> LoadPhysicalIdentityProfilesFromPath(
    const std::string& path);
void SavePhysicalIdentityProfilesToPath(
    const std::vector<PhysicalIdentityProfile>& profiles,
    const std::string& path);

// Public for deterministic focused tests and migration tooling.
void ValidatePhysicalIdentityProfiles(
    const std::vector<PhysicalIdentityProfile>& profiles);

}}} // namespace GRIM::Perception::Physical
