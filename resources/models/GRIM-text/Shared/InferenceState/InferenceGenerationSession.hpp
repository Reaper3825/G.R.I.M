#pragma once

#include "InferenceReturnPolicy.hpp"
#include "../Forward/GeneratedSequence.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../Sampling/Sampling.hpp"

namespace GRIM::Batching { struct BatchPayload; }

namespace GRIM::Inference {

// Geometry remains owned by the compact prefill payload and model config.
// Data-only session state; creation and operations are free functions below.
// GPU forwards, KV storage, and local-atom retrieval execution remain with
// their respective owners.
struct InferenceGenerationSession {
    HyperParameters::GenerationHP config;
    HyperParameters::GenerationStreamCallback* stream_callback;
    Sampling::SamplingPipeline pipeline;
    GeneratedSequence sequence;
    ConceptSpanTracker span_tracker;
    std::size_t acknowledged_span_events = 0;
};

InferenceGenerationSession createInferenceGenerationSession(
    Batching::BatchPayload& prefill,
    const HyperParameters::ModelHP& model_hp,
    const HyperParameters::GenerationHP& generation_hp,
    HyperParameters::GenerationStreamCallback* stream_callback = nullptr,
    bool inspect_only = false);

int sampledTokenCount(const InferenceGenerationSession& session);
int remainingTokenCapacity(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp);
int remainingContextTokenCapacity(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp);
InferenceReturnReason returnReason(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp);
bool finished(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp);
bool shouldReturn(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp);
Sampling::SampleResult selectNextToken(InferenceGenerationSession& session,
    const Batching::BatchPayload& prefill, const float* logits);
void commitToken(InferenceGenerationSession& session, const Sampling::SampleResult& sample,
                 TokenOrigin origin = TokenOrigin::SampledAction);
void commitForcedToken(InferenceGenerationSession& session, int token_id);
void acknowledgeSpanReturn(InferenceGenerationSession& session);
GeneratedSequence takeSequence(InferenceGenerationSession& session, InferenceReturnReason reason);

} // namespace GRIM::Inference
