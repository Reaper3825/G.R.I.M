#include "PhysicalFaceIdentityMatcher.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

float ComputePhysicalFaceCosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {
    if (a.empty() || a.size() != b.size()) return -1.0f;
    double dot = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
    }
    return static_cast<float>(dot);
}

PhysicalFaceIdentityMatch MatchPhysicalFaceIdentity(
    const PhysicalFaceEmbedding& face,
    const std::vector<PhysicalIdentityProfile>& profiles,
    const PhysicalFaceIdentityMatcherConfig& config) {
    if (!(config.minimum_similarity >= -1.0f && config.minimum_similarity <= 1.0f) ||
        !(config.minimum_runner_up_margin >= 0.0f &&
          config.minimum_runner_up_margin <= 2.0f)) {
        throw std::invalid_argument("PhysicalFaceIdentityMatcherConfig is invalid");
    }
    PhysicalFaceIdentityMatch result;
    for (const auto& profile : profiles) {
        float profile_score = -1.0f;
        for (const auto& stored : profile.face_templates) {
            if (stored.model_id != face.embedding_model_id) continue;
            profile_score = std::max(profile_score,
                ComputePhysicalFaceCosineSimilarity(stored.embedding, face.embedding));
        }
        if (profile_score > result.best_similarity) {
            result.runner_up_similarity = result.best_similarity;
            result.best_similarity = profile_score;
            result.persistent_entity_id = profile.persistent_entity_id;
        } else if (profile_score > result.runner_up_similarity) {
            result.runner_up_similarity = profile_score;
        }
    }
    result.accepted = result.best_similarity >= config.minimum_similarity &&
        (result.runner_up_similarity < -0.5f ||
         result.best_similarity - result.runner_up_similarity >=
            config.minimum_runner_up_margin);
    return result;
}

}}} // namespace GRIM::Perception::Physical
