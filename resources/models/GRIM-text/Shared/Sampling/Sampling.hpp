//======================================================//
//  Sampling.hpp
//  Inference sampling strategies for GRIM-text
//
//  Supported strategies:
//    - Greedy (argmax)
//    - Top-K filtering
//    - Top-P (nucleus) filtering
//    - Min-P filtering
//    - Typical sampling (locally typical decoding)
//    - Combined Top-K + Top-P
//    - No-repeat n-gram blocking
//    - Frequency/presence penalty
//    - Temperature scaling
//    - Repetition penalty (sign-aware)
//
//  Rule 20: No fallbacks. Crash on invalid config.
//  Rule 26: Single sampling path via SamplingPipeline.
//======================================================//

#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <limits>
#include <functional>

namespace GRIM {
namespace Sampling {

//======================================================//
//  Sampling Strategy Enum
//======================================================//
enum class Strategy {
    GREEDY,         // Argmax - deterministic
    TOP_K,          // Top-K only
    TOP_P,          // Top-P (nucleus) only
    MIN_P,          // Min-P (relative threshold)
    TYPICAL,        // Locally typical sampling
    TOP_K_TOP_P,    // Combined: Top-K first, then Top-P within survivors
};

//======================================================//
//  SamplingConfig - All parameters for the sampling pipeline
//======================================================//
struct SamplingConfig {
    // Core strategy
    Strategy strategy = Strategy::TOP_P;
    bool do_sample = true;        // false = force greedy regardless of strategy

    // Temperature
    float temperature = 1.0f;     // Must be > 0 for sampling strategies

    // Top-K
    int top_k = 50;               // Number of top tokens to keep (0 = disabled)

    // Top-P (Nucleus)
    float top_p = 0.9f;           // Cumulative probability threshold (0,1)

    // Min-P (relative minimum probability)
    float min_p = 0.0f;           // Tokens below max_prob * min_p are removed (0 = disabled)

    // Typical sampling
    float typical_p = 1.0f;       // Typical probability mass (1.0 = disabled, <1.0 = active)

    // Repetition control
    float repetition_penalty = 1.0f;      // Standard rep penalty (1.0 = disabled)
    int repetition_penalty_window = 64;   // How far back to look
    float frequency_penalty = 0.0f;       // Additive penalty per occurrence (0 = disabled)
    float presence_penalty = 0.0f;        // Additive penalty if token appeared at all (0 = disabled)
    int no_repeat_ngram_size = 0;         // Block n-grams of this size from repeating (0 = disabled)

    // Token masking
    std::vector<int> bad_token_ids;       // Tokens to force -inf
    int eos_token_id = 3;                 // EOS token ID
    int bos_token_id = 2;                 // BOS token ID (always masked in generation)
    int pad_token_id = 1;                 // PAD token ID (always masked in generation)
    int unk_token_id = 0;                 // UNK token ID (always masked in generation)

    // RNG
    unsigned int seed = 0;                // 0 = random device seed
};

//======================================================//
//  SampleResult - Output of a single sampling step
//======================================================//
struct SampleResult {
    int token_id = -1;
    float probability = 0.0f;
    float log_probability = -std::numeric_limits<float>::infinity();
};

//======================================================//
//  Softmax with temperature scaling
//  Returns normalized probability distribution
//======================================================//
std::vector<float> softmaxTemperature(const std::vector<float>& logits, float temperature);

//======================================================//
//  Filter Functions - Each modifies probabilities in-place
//  All filters zero out excluded tokens; caller must renormalize after.
//======================================================//

// Top-K: Keep only the top_k highest-probability tokens
void filterTopK(std::vector<float>& probs, int top_k);

// Top-P (Nucleus): Keep smallest set of tokens whose cumulative prob >= top_p
void filterTopP(std::vector<float>& probs, float top_p);

// Min-P: Remove tokens whose probability < max_probability * min_p
void filterMinP(std::vector<float>& probs, float min_p);

// Typical: Keep tokens whose information content is closest to the expected
// information content of the distribution (locally typical decoding)
void filterTypical(std::vector<float>& probs, float typical_p);

// Renormalize a probability vector after filtering
void renormalize(std::vector<float>& probs);

//======================================================//
//  Penalty Functions - Modify logits before softmax
//======================================================//

// Standard repetition penalty (sign-aware, multiplicative)
// Positive logits are divided by penalty, negative logits are multiplied
void applyRepetitionPenalty(std::vector<float>& logits,
                           const std::vector<int>& history,
                           float penalty,
                           int window);

// Frequency penalty: penalize proportional to occurrence count in history
void applyFrequencyPenalty(std::vector<float>& logits,
                           const std::vector<int>& history,
                           float frequency_penalty,
                           int window);

// Presence penalty: flat penalty if token appeared in history
void applyPresencePenalty(std::vector<float>& logits,
                          const std::vector<int>& history,
                          float presence_penalty,
                          int window);

// No-repeat n-gram: set logits to -inf for any token that would complete
// a repeated n-gram
void applyNoRepeatNgram(std::vector<float>& logits,
                        const std::vector<int>& history,
                        int ngram_size);

// Mask special tokens (UNK, PAD, BOS) and any explicit bad_token_ids
void applyTokenMask(std::vector<float>& logits,
                    const SamplingConfig& config);

//======================================================//
//  SamplingPipeline - The single sampling path
//
//  Pipeline order (matches llama.cpp / HuggingFace convention):
//    1. Apply penalties to logits (repetition, frequency, presence)
//    2. Apply n-gram blocking to logits
//    3. Mask special/bad tokens
//    4. Temperature scaling + softmax → probabilities
//    5. Apply filters (top-k, min-p, top-p, typical) to probabilities
//    6. Renormalize
//    7. Sample or argmax
//======================================================//
class SamplingPipeline {
public:
    explicit SamplingPipeline(const SamplingConfig& config);

    // Select a single next token from logits given generation history.
    // logits: raw model output logits [vocab_size]
    // history: all tokens generated so far (prompt + generated)
    // vocab_size: must match logits.size()
    SampleResult selectNextToken(const std::vector<float>& logits,
                                 const std::vector<int>& history,
                                 int vocab_size);

    // Reset RNG state (e.g., for a new generation session)
    void resetRng(unsigned int seed = 0);

    // Access config (read-only)
    const SamplingConfig& config() const { return config_; }

private:
    SamplingConfig config_;
    std::mt19937 rng_;
};

//======================================================//
//  Utility: Create SamplingConfig from root-derived generation fields
//======================================================//

/// Convert a HyperParameters::SamplingStrategy enum value to Sampling::Strategy
Strategy convertStrategy(int legacy_strategy);

/// Build a SamplingConfig from GenerationHP/root generation fields.
SamplingConfig buildSamplingConfigFromGenerationFields(
    int strategy,        // Cast from SamplingStrategy
    bool do_sample,
    float temperature,
    int top_k,
    float top_p,
    float min_p,
    float typical_p,
    float repetition_penalty,
    int repetition_penalty_window,
    float frequency_penalty,
    float presence_penalty,
    int no_repeat_ngram_size,
    int eos_token_id,
    int bos_token_id,
    int pad_token_id,
    int unk_token_id,
    const std::vector<int>& bad_token_ids,
    unsigned int seed);

} // namespace Sampling
} // namespace GRIM
