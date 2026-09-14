#include "perception/physical/PhysicalFaceIdentityMatcher.hpp"
#include "perception/physical/PhysicalIdentityStore.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace PE = GRIM::Perception::Physical;

namespace {

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> Unit(float x, float y) {
    const float norm = std::sqrt(x * x + y * y);
    return {x / norm, y / norm};
}

PE::PhysicalIdentityProfile Profile(
    std::string id, std::string name, std::vector<float> embedding) {
    PE::PhysicalIdentityProfile profile;
    profile.persistent_entity_id = std::move(id);
    profile.display_name = std::move(name);
    profile.created_at_unix_seconds = 1;
    profile.updated_at_unix_seconds = 1;
    PE::PhysicalFaceTemplateRecord face;
    face.model_id = "test-model:v1";
    face.embedding = std::move(embedding);
    face.quality_score = 0.9f;
    face.enrolled_at_unix_seconds = 1;
    profile.face_templates.push_back(std::move(face));
    return profile;
}

} // anonymous namespace

int main() {
    try {
        const auto directory = std::filesystem::temp_directory_path() /
            "grim_physical_identity_focused_test";
        std::filesystem::create_directories(directory);
        const auto path = (directory / "profiles.json").string();

        std::vector<PE::PhysicalIdentityProfile> profiles{
            Profile("person-a", "Alice", Unit(1.0f, 0.0f)),
            Profile("person-b", "Bob", Unit(0.0f, 1.0f))
        };
        PE::SavePhysicalIdentityProfilesToPath(profiles, path);
        auto loaded = PE::LoadPhysicalIdentityProfilesFromPath(path);
        Require(loaded.size() == 2, "identity store round trip lost profiles");
        Require(loaded[0].face_templates[0].embedding.size() == 2,
                "identity store round trip lost embedding dimension");

        // A second save exercises atomic replacement of an existing store.
        loaded[0].display_name = "Alice Renamed";
        PE::SavePhysicalIdentityProfilesToPath(loaded, path);
        loaded = PE::LoadPhysicalIdentityProfilesFromPath(path);
        Require(loaded[0].display_name == "Alice Renamed",
                "identity store atomic replacement did not persist update");

        PE::PhysicalFaceEmbedding observation;
        observation.embedding_model_id = "test-model:v1";
        observation.embedding = Unit(0.99f, 0.05f);
        auto match = PE::MatchPhysicalFaceIdentity(observation, loaded);
        Require(match.accepted && match.persistent_entity_id == "person-a",
                "clear face match was not accepted");

        observation.embedding_model_id = "different-model";
        match = PE::MatchPhysicalFaceIdentity(observation, loaded);
        Require(!match.accepted,
                "embeddings from incompatible model spaces were compared");

        observation.embedding_model_id = "test-model:v1";
        observation.embedding = Unit(1.0f, 1.0f);
        match = PE::MatchPhysicalFaceIdentity(observation, loaded);
        Require(!match.accepted,
                "ambiguous match was accepted without runner-up margin");

        std::filesystem::remove(path);
        std::filesystem::remove(directory);
        std::cout << "physical_identity_tests: PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "physical_identity_tests: FAIL: " << e.what() << '\n';
        return 1;
    }
}
