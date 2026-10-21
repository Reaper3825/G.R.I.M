#pragma once

#include "../Forward/GeneratedSequence.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../Sampling/Sampling.hpp"

namespace GRIM::Batching { struct BatchPayload; }

namespace GRIM::Inference {

// One generated trajectory and its sampling policy. GPU forwards, KV storage,
// and local-atom retrieval execution remain with their respective owners.
class InferenceGenerationSession {
public:
    InferenceGenerationSession(
        Batching::BatchPayload& prefill,
        const HyperParameters::ModelHP& model_hp,
        const HyperParameters::GenerationHP& generation_hp,
        HyperParameters::GenerationStreamCallback* stream_callback = nullptr,
        bool inspect_only = false);

    GeneratedSequence& sequence() { return sequence_; }
    int committedNewTokens() const;
    int remainingTokenCapacity() const;
    bool finished() const;
    Sampling::SampleResult selectNextToken(const float* logits);
    void commitToken(const Sampling::SampleResult& sample,
                     TokenOrigin origin = TokenOrigin::SampledAction);
    void commitForcedToken(int token_id);
    GeneratedSequence takeSequence();

private:
    HyperParameters::GenerationHP cfg_;
    int max_seq_len_;
    int vocab_size_;
    int prompt_len_;
    HyperParameters::GenerationStreamCallback* stream_callback_;
    Sampling::SamplingPipeline pipeline_;
    GeneratedSequence sequence_;
};

} // namespace GRIM::Inference
