#include "PhysicalIdentityStore.hpp"

#include "memory/atomic_writer.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#ifndef GRIM_ROOT_DIR
#error "GRIM_ROOT_DIR must be defined by the build system"
#endif

namespace GRIM { namespace Perception { namespace Physical {

namespace {

constexpr int kSchemaVersion = 1;
constexpr size_t kMaximumProfiles = 512;
constexpr size_t kMaximumTemplatesPerProfile = 24;
constexpr size_t kMaximumEmbeddingDimension = 4096;

void ValidateName(const std::string& name) {
    if (name.empty() || name.size() > 96) {
        throw std::runtime_error("identity display_name must contain 1..96 bytes");
    }
    for (unsigned char c : name) {
        if (c < 0x20 || c == 0x7f) {
            throw std::runtime_error("identity display_name contains control characters");
        }
    }
}

void ValidateTemplate(const PhysicalFaceTemplateRecord& face) {
    if (face.model_id.empty()) {
        throw std::runtime_error("face template model_id is empty");
    }
    if (face.embedding.empty() ||
        face.embedding.size() > kMaximumEmbeddingDimension) {
        throw std::runtime_error("face template embedding dimension is invalid");
    }
    if (!std::isfinite(face.quality_score) || face.quality_score < 0.0f ||
        face.quality_score > 1.0f) {
        throw std::runtime_error("face template quality_score is outside [0,1]");
    }
    double norm2 = 0.0;
    for (float value : face.embedding) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("face template contains a non-finite value");
        }
        norm2 += static_cast<double>(value) * value;
    }
    const double norm = std::sqrt(norm2);
    if (std::fabs(norm - 1.0) > 0.02) {
        throw std::runtime_error("face template must be L2-normalized");
    }
}

} // anonymous namespace

std::string GetPhysicalIdentityStorePath() {
    const std::filesystem::path directory =
        std::filesystem::path(GRIM_ROOT_DIR) / "data" / "perception" /
        "physical" / "identities";
    return (directory / "profiles.json").string();
}

void ValidatePhysicalIdentityProfiles(
    const std::vector<PhysicalIdentityProfile>& profiles) {
    if (profiles.size() > kMaximumProfiles) {
        throw std::runtime_error("physical identity profile count exceeds safety cap");
    }
    std::unordered_set<std::string> ids;
    for (const auto& profile : profiles) {
        if (profile.persistent_entity_id.empty() ||
            !ids.insert(profile.persistent_entity_id).second) {
            throw std::runtime_error("physical identity ID is empty or duplicated");
        }
        ValidateName(profile.display_name);
        if (profile.face_templates.size() > kMaximumTemplatesPerProfile) {
            throw std::runtime_error("physical identity template count exceeds safety cap");
        }
        for (const auto& face : profile.face_templates) ValidateTemplate(face);
    }
}

std::vector<PhysicalIdentityProfile> LoadPhysicalIdentityProfiles() {
    return LoadPhysicalIdentityProfilesFromPath(GetPhysicalIdentityStorePath());
}

std::vector<PhysicalIdentityProfile> LoadPhysicalIdentityProfilesFromPath(
    const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        if (ec) {
            throw std::runtime_error("failed checking identity store: " + ec.message());
        }
        return {};
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("failed opening identity store: " + path);
    nlohmann::json root;
    input >> root;
    if (root.value("schema_version", 0) != kSchemaVersion) {
        throw std::runtime_error("unsupported physical identity schema_version");
    }

    std::vector<PhysicalIdentityProfile> out;
    for (const auto& item : root.value("profiles", nlohmann::json::array())) {
        PhysicalIdentityProfile profile;
        profile.persistent_entity_id = item.value("persistent_entity_id", "");
        profile.display_name = item.value("display_name", "");
        profile.created_at_unix_seconds = item.value("created_at_unix_seconds", uint64_t{0});
        profile.updated_at_unix_seconds = item.value("updated_at_unix_seconds", uint64_t{0});
        for (const auto& stored : item.value("face_templates", nlohmann::json::array())) {
            PhysicalFaceTemplateRecord face;
            face.model_id = stored.value("model_id", "");
            face.embedding = stored.value("embedding", std::vector<float>{});
            face.quality_score = stored.value("quality_score", 0.0f);
            face.enrolled_at_unix_seconds =
                stored.value("enrolled_at_unix_seconds", uint64_t{0});
            profile.face_templates.push_back(std::move(face));
        }
        out.push_back(std::move(profile));
    }
    ValidatePhysicalIdentityProfiles(out);
    return out;
}

void SavePhysicalIdentityProfiles(
    const std::vector<PhysicalIdentityProfile>& profiles) {
    SavePhysicalIdentityProfilesToPath(profiles, GetPhysicalIdentityStorePath());
}

void SavePhysicalIdentityProfilesToPath(
    const std::vector<PhysicalIdentityProfile>& profiles,
    const std::string& path) {
    ValidatePhysicalIdentityProfiles(profiles);
    nlohmann::json root;
    root["schema_version"] = kSchemaVersion;
    root["profiles"] = nlohmann::json::array();
    for (const auto& profile : profiles) {
        nlohmann::json item;
        item["persistent_entity_id"] = profile.persistent_entity_id;
        item["display_name"] = profile.display_name;
        item["created_at_unix_seconds"] = profile.created_at_unix_seconds;
        item["updated_at_unix_seconds"] = profile.updated_at_unix_seconds;
        item["face_templates"] = nlohmann::json::array();
        for (const auto& face : profile.face_templates) {
            item["face_templates"].push_back({
                {"model_id", face.model_id},
                {"embedding", face.embedding},
                {"quality_score", face.quality_score},
                {"enrolled_at_unix_seconds", face.enrolled_at_unix_seconds}
            });
        }
        root["profiles"].push_back(std::move(item));
    }
    GRIM::AtomicWriter::writeString(path, root.dump(2) + "\n");
}

}}} // namespace GRIM::Perception::Physical
