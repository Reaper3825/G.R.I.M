#pragma once

#include "PhysicalIdentityStore.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <string>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

struct PhysicalFaceIdentityMatcherConfig {
    // OpenCV's published SFace LFW operating point for cosine similarity.
    // Keep the independent runner-up margin below to reject ambiguous matches.
    float minimum_similarity = 0.363f;
    float minimum_runner_up_margin = 0.08f;
};

struct PhysicalFaceIdentityMatch {
    std::string persistent_entity_id;
    float best_similarity = -1.0f;
    float runner_up_similarity = -1.0f;
    bool accepted = false;
};

float ComputePhysicalFaceCosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b);

PhysicalFaceIdentityMatch MatchPhysicalFaceIdentity(
    const PhysicalFaceEmbedding& face,
    const std::vector<PhysicalIdentityProfile>& profiles,
    const PhysicalFaceIdentityMatcherConfig& config = {});

}}} // namespace GRIM::Perception::Physical
