//======================================================//
//  Sampling.cu
//  Implementation of inference sampling strategies
//
//  Pipeline order (matches llama.cpp / HuggingFace):
//    1. Apply penalties to logits (repetition, frequency, presence)
//    2. Apply n-gram blocking to logits
//    3. Mask special/bad tokens
//    4. Temperature scaling + softmax → probabilities
//    5. Apply filters (top-k, min-p, top-p, typical) to probs
//    6. Renormalize
//    7. Sample or argmax
//
//  Rule 20: No fallbacks. Crash on invalid config.
//  Rule 26: Single sampling path via SamplingPipeline.
//======================================================//

#include "Sampling.hpp"

#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace GRIM {
namespace Sampling {

//======================================================//
//  Softmax with Temperature
//======================================================//
std::vector<float> softmaxTemperature(const std::vector<float>& logits, float temperature) {
    if (logits.empty()) {
        throw std::runtime_error("softmaxTemperature: empty logits");
    }
    if (temperature <= 0.0f) {
        throw std::runtime_error("softmaxTemperature: temperature must be > 0, got " +
                                 std::to_string(temperature));
    }

    const size_t n = logits.size();
    std::vector<float> probs(n);

    // Find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // exp((logit - max) / temperature)
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float scaled = (logits[i] - max_logit) / temperature;
        // Clamp to avoid exp overflow/underflow
        scaled = std::max(scaled, -88.0f);  // exp(-88) ≈ 6e-39
        probs[i] = std::exp(scaled);
        sum += probs[i];
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error("softmaxTemperature: probability sum is zero or non-finite "
                                 "(temperature=" + std::to_string(temperature) + ")");
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) {
        probs[i] *= inv_sum;
    }

    return probs;
}

//======================================================//
//  Top-K Filter
//======================================================//
void filterTopK(std::vector<float>& probs, int top_k) {
    if (top_k <= 0) return;  // Disabled

    const int n = static_cast<int>(probs.size());
    if (top_k >= n) return;  // Keep all

    // Find the k-th largest probability via partial sort
    std::vector<std::pair<float, int>> indexed(n);
    for (int i = 0; i < n; ++i) {
        indexed[i] = {probs[i], i};
    }

    // Partial sort: top_k elements at front, descending by probability
    std::nth_element(indexed.begin(), indexed.begin() + top_k, indexed.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    float threshold = indexed[top_k - 1].first;

    // Zero out everything below threshold (keep exactly top_k or more if ties)
    int kept = 0;
    for (int i = 0; i < n; ++i) {
        if (probs[i] < threshold) {
            probs[i] = 0.0f;
        } else {
            ++kept;
        }
    }

    // If we somehow zeroed everything (degenerate), keep the single max
    if (kept == 0) {
        int max_idx = static_cast<int>(std::distance(
            probs.begin(), std::max_element(probs.begin(), probs.end())));
        probs[max_idx] = 1.0f;
    }
}

//======================================================//
//  Top-P (Nucleus) Filter
//======================================================//
void filterTopP(std::vector<float>& probs, float top_p) {
    if (top_p >= 1.0f) return;  // Disabled
    if (top_p <= 0.0f) {
        throw std::runtime_error("filterTopP: top_p must be in (0, 1], got " +
                                 std::to_string(top_p));
    }

    const int n = static_cast<int>(probs.size());

    // Sort indices by descending probability
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](int a, int b) { return probs[a] > probs[b]; });

    // Accumulate probability mass; zero out tokens beyond threshold
    float cumulative = 0.0f;
    bool threshold_reached = false;

    for (int i = 0; i < n; ++i) {
        int idx = indices[i];
        if (threshold_reached) {
            probs[idx] = 0.0f;
        } else {
            cumulative += probs[idx];
            if (cumulative >= top_p) {
                threshold_reached = true;
                // Keep this token (it pushed us over the threshold)
            }
        }
    }
}

//======================================================//
//  Min-P Filter
//  Removes tokens whose probability < max_prob * min_p
//======================================================//
void filterMinP(std::vector<float>& probs, float min_p) {
    if (min_p <= 0.0f) return;  // Disabled
    if (min_p >= 1.0f) {
        throw std::runtime_error("filterMinP: min_p must be in (0, 1), got " +
                                 std::to_string(min_p));
    }

    float max_prob = *std::max_element(probs.begin(), probs.end());
    float threshold = max_prob * min_p;

    bool kept_any = false;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] < threshold) {
            probs[i] = 0.0f;
        } else {
            kept_any = true;
        }
    }

    // Safety: if nothing survived, keep the max token
    if (!kept_any) {
        int max_idx = static_cast<int>(std::distance(
            probs.begin(), std::max_element(probs.begin(), probs.end())));
        probs[max_idx] = max_prob;
    }
}

//======================================================//
//  Typical Sampling Filter
//  Keeps tokens whose information content is closest to
//  the expected information content (entropy) of the distribution.
//  Reference: https://arxiv.org/abs/2202.00666
//======================================================//
void filterTypical(std::vector<float>& probs, float typical_p) {
    if (typical_p >= 1.0f) return;  // Disabled
    if (typical_p <= 0.0f) {
        throw std::runtime_error("filterTypical: typical_p must be in (0, 1), got " +
                                 std::to_string(typical_p));
    }

    const int n = static_cast<int>(probs.size());

    // Compute entropy H = -Σ p_i * log(p_i)
    float entropy = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (probs[i] > 0.0f) {
            entropy -= probs[i] * std::log(probs[i]);
        }
    }

    // Compute |information_content - entropy| for each token
    // information_content_i = -log(p_i)
    std::vector<std::pair<float, int>> surprisal_deviation(n);
    for (int i = 0; i < n; ++i) {
        if (probs[i] > 0.0f) {
            float info = -std::log(probs[i]);
            surprisal_deviation[i] = {std::abs(info - entropy), i};
        } else {
            surprisal_deviation[i] = {std::numeric_limits<float>::infinity(), i};
        }
    }

    // Sort by deviation (ascending — most typical first)
    std::sort(surprisal_deviation.begin(), surprisal_deviation.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Accumulate probability mass of most-typical tokens
    float cumulative = 0.0f;
    std::unordered_set<int> keep_set;

    for (int i = 0; i < n; ++i) {
        int idx = surprisal_deviation[i].second;
        keep_set.insert(idx);
        cumulative += probs[idx];
        if (cumulative >= typical_p) break;
    }

    // Zero out non-typical tokens
    for (int i = 0; i < n; ++i) {
        if (keep_set.find(i) == keep_set.end()) {
            probs[i] = 0.0f;
        }
    }
}

//======================================================//
//  Renormalize
//======================================================//
void renormalize(std::vector<float>& probs) {
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f) {
        throw std::runtime_error("renormalize: probability sum is zero after filtering");
    }
    float inv = 1.0f / sum;
    for (auto& p : probs) {
        p *= inv;
    }
}

//======================================================//
//  Repetition Penalty (sign-aware, multiplicative)
//======================================================//
void applyRepetitionPenalty(std::vector<float>& logits,
                           const std::vector<int>& history,
                           float penalty,
                           int window) {
    if (penalty <= 1.0f || history.empty()) return;

    const int vocab = static_cast<int>(logits.size());
    const int start = std::max(0, static_cast<int>(history.size()) - window);

    std::unordered_set<int> seen;
    seen.reserve(static_cast<size_t>(window));

    for (int i = start; i < static_cast<int>(history.size()); ++i) {
        int tid = history[i];
        if (tid < 0 || tid >= vocab) continue;
        if (!seen.insert(tid).second) continue;

        // Sign-aware: positive logits divided, negative multiplied
        if (logits[tid] > 0.0f) {
            logits[tid] /= penalty;
        } else {
            logits[tid] *= penalty;
        }
    }
}

//======================================================//
//  Frequency Penalty (additive, proportional to count)
//======================================================//
void applyFrequencyPenalty(std::vector<float>& logits,
                           const std::vector<int>& history,
                           float frequency_penalty,
                           int window) {
    if (frequency_penalty == 0.0f || history.empty()) return;

    const int vocab = static_cast<int>(logits.size());
    const int start = std::max(0, static_cast<int>(history.size()) - window);

    std::unordered_map<int, int> counts;
    counts.reserve(static_cast<size_t>(window));

    for (int i = start; i < static_cast<int>(history.size()); ++i) {
        int tid = history[i];
        if (tid >= 0 && tid < vocab) {
            counts[tid]++;
        }
    }

    for (const auto& [tid, count] : counts) {
        logits[tid] -= frequency_penalty * static_cast<float>(count);
    }
}

//======================================================//
//  Presence Penalty (additive, flat per unique token)
//======================================================//
void applyPresencePenalty(std::vector<float>& logits,
                          const std::vector<int>& history,
                          float presence_penalty,
                          int window) {
    if (presence_penalty == 0.0f || history.empty()) return;

    const int vocab = static_cast<int>(logits.size());
    const int start = std::max(0, static_cast<int>(history.size()) - window);

    std::unordered_set<int> seen;
    seen.reserve(static_cast<size_t>(window));

    for (int i = start; i < static_cast<int>(history.size()); ++i) {
        int tid = history[i];
        if (tid >= 0 && tid < vocab) {
            seen.insert(tid);
        }
    }

    for (int tid : seen) {
        logits[tid] -= presence_penalty;
    }
}

//======================================================//
//  No-Repeat N-gram Blocking
//  Blocks any token that would complete a repeated n-gram
//======================================================//
void applyNoRepeatNgram(std::vector<float>& logits,
                        const std::vector<int>& history,
                        int ngram_size) {
    if (ngram_size <= 0) return;
    if (static_cast<int>(history.size()) < ngram_size) return;

    const int vocab = static_cast<int>(logits.size());
    const int hist_len = static_cast<int>(history.size());

    // The (ngram_size-1) most recent tokens form the "context" for the next prediction
    // We scan history for any earlier occurrence of this same context
    // and block the token that followed it
    const int ctx_len = ngram_size - 1;

    // Extract current context (last ctx_len tokens)
    std::vector<int> current_ctx(history.end() - ctx_len, history.end());

    // Scan all possible positions where this context may have appeared before
    for (int pos = 0; pos <= hist_len - ngram_size; ++pos) {
        bool match = true;
        for (int j = 0; j < ctx_len; ++j) {
            if (history[pos + j] != current_ctx[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            // The token at pos + ctx_len completed this n-gram before — block it
            int blocked_token = history[pos + ctx_len];
            if (blocked_token >= 0 && blocked_token < vocab) {
                logits[blocked_token] = -std::numeric_limits<float>::infinity();
            }
        }
    }
}

//======================================================//
//  Token Masking (special tokens + bad_token_ids)
//======================================================//
void applyTokenMask(std::vector<float>& logits,
                    const SamplingConfig& config) {
    const int vocab = static_cast<int>(logits.size());
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    // Always mask UNK, PAD, BOS — these should never be generated
    if (config.unk_token_id >= 0 && config.unk_token_id < vocab) {
        logits[config.unk_token_id] = neg_inf;
    }
    if (config.pad_token_id >= 0 && config.pad_token_id < vocab) {
        logits[config.pad_token_id] = neg_inf;
    }
    if (config.bos_token_id >= 0 && config.bos_token_id < vocab) {
        logits[config.bos_token_id] = neg_inf;
    }

    // Mask explicit bad tokens
    for (int tid : config.bad_token_ids) {
        if (tid >= 0 && tid < vocab) {
            logits[tid] = neg_inf;
        }
    }
}

//======================================================//
//  SamplingPipeline
//======================================================//

SamplingPipeline::SamplingPipeline(const SamplingConfig& config)
    : config_(config)
{
    // Validate config up front (Rule 20: crash on bad config)
    if (config_.do_sample && config_.strategy != Strategy::GREEDY) {
        if (config_.temperature <= 0.0f) {
            throw std::runtime_error("SamplingPipeline: temperature must be > 0 for sampling, got " +
                                     std::to_string(config_.temperature));
        }
    }
    if (config_.strategy == Strategy::TOP_K && config_.top_k <= 0) {
        throw std::runtime_error("SamplingPipeline: TOP_K strategy requires top_k > 0, got " +
                                 std::to_string(config_.top_k));
    }
    if (config_.strategy == Strategy::TOP_P &&
        (config_.top_p <= 0.0f || config_.top_p >= 1.0f)) {
        throw std::runtime_error("SamplingPipeline: TOP_P strategy requires top_p in (0,1), got " +
                                 std::to_string(config_.top_p));
    }
    if (config_.strategy == Strategy::MIN_P &&
        (config_.min_p <= 0.0f || config_.min_p >= 1.0f)) {
        throw std::runtime_error("SamplingPipeline: MIN_P strategy requires min_p in (0,1), got " +
                                 std::to_string(config_.min_p));
    }
    if (config_.strategy == Strategy::TYPICAL &&
        (config_.typical_p <= 0.0f || config_.typical_p >= 1.0f)) {
        throw std::runtime_error("SamplingPipeline: TYPICAL strategy requires typical_p in (0,1), got " +
                                 std::to_string(config_.typical_p));
    }

    resetRng(config_.seed);
}

void SamplingPipeline::resetRng(unsigned int seed) {
    if (seed == 0) {
        std::random_device rd;
        rng_.seed(rd());
    } else {
        rng_.seed(seed);
    }
}

SampleResult SamplingPipeline::sample(const std::vector<float>& logits,
                                      const std::vector<int>& history,
                                      int vocab_size) {
    if (logits.empty()) {
        throw std::runtime_error("SamplingPipeline::sample: empty logits");
    }
    if (static_cast<int>(logits.size()) != vocab_size) {
        throw std::runtime_error("SamplingPipeline::sample: logits.size()=" +
                                 std::to_string(logits.size()) + " != vocab_size=" +
                                 std::to_string(vocab_size));
    }

    // Work on a mutable copy of logits
    std::vector<float> work_logits = logits;

    //--------------------------------------------------//
    // Step 1: Apply penalties to logits
    //--------------------------------------------------//
    if (config_.repetition_penalty > 1.0f) {
        applyRepetitionPenalty(work_logits, history,
                              config_.repetition_penalty,
                              config_.repetition_penalty_window);
    }
    if (config_.frequency_penalty != 0.0f) {
        applyFrequencyPenalty(work_logits, history,
                              config_.frequency_penalty,
                              config_.repetition_penalty_window);
    }
    if (config_.presence_penalty != 0.0f) {
        applyPresencePenalty(work_logits, history,
                             config_.presence_penalty,
                             config_.repetition_penalty_window);
    }

    //--------------------------------------------------//
    // Step 2: N-gram blocking
    //--------------------------------------------------//
    if (config_.no_repeat_ngram_size > 0) {
        applyNoRepeatNgram(work_logits, history, config_.no_repeat_ngram_size);
    }

    //--------------------------------------------------//
    // Step 3: Mask special/bad tokens
    //--------------------------------------------------//
    applyTokenMask(work_logits, config_);

    //--------------------------------------------------//
    // Step 4+5+6+7: Greedy or sampled
    //--------------------------------------------------//
    const bool use_sampling = config_.do_sample && config_.strategy != Strategy::GREEDY;

    if (!use_sampling) {
        // Greedy: argmax on raw logits (no temperature needed)
        int best = static_cast<int>(std::distance(
            work_logits.begin(), std::max_element(work_logits.begin(), work_logits.end())));
        if (best < 0 || best >= vocab_size) {
            throw std::runtime_error("SamplingPipeline: greedy argmax out of range");
        }

        // Compute probability for the greedy token efficiently:
        // P(best) = exp(logit_best) / sum(exp(logits)) = 1 / sum(exp(logits - logit_best))
        // This avoids allocating a full vocab-sized probability vector.
        float max_logit = work_logits[best];
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            float shifted = work_logits[i] - max_logit;
            shifted = std::max(shifted, -88.0f);
            sum_exp += std::exp(shifted);
        }
        float prob = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1e-10f;
        prob = std::max(prob, 1e-10f);

        SampleResult result;
        result.token_id = best;
        result.probability = prob;
        result.log_probability = std::log(prob);
        return result;
    }

    //--------------------------------------------------//
    // Step 4: Temperature + softmax → probabilities
    //--------------------------------------------------//
    auto probs = softmaxTemperature(work_logits, config_.temperature);

    //--------------------------------------------------//
    // Step 5: Apply filters in order
    //--------------------------------------------------//
    switch (config_.strategy) {
        case Strategy::TOP_K:
            filterTopK(probs, config_.top_k);
            break;

        case Strategy::TOP_P:
            filterTopP(probs, config_.top_p);
            break;

        case Strategy::MIN_P:
            filterMinP(probs, config_.min_p);
            break;

        case Strategy::TYPICAL:
            filterTypical(probs, config_.typical_p);
            break;

        case Strategy::TOP_K_TOP_P:
            // Combined: top-k first to reduce candidates, then top-p for dynamic cutoff
            filterTopK(probs, config_.top_k);
            filterTopP(probs, config_.top_p);
            break;

        case Strategy::GREEDY:
            // Already handled above, but just in case
            break;
    }

    // Apply min-p as an additional global filter (when strategy isn't already MIN_P)
    if (config_.strategy != Strategy::MIN_P && config_.min_p > 0.0f) {
        filterMinP(probs, config_.min_p);
    }

    // Apply typical as an additional filter (when strategy isn't already TYPICAL)
    if (config_.strategy != Strategy::TYPICAL && config_.typical_p < 1.0f) {
        filterTypical(probs, config_.typical_p);
    }

    //--------------------------------------------------//
    // Step 6: Renormalize
    //--------------------------------------------------//
    renormalize(probs);

    //--------------------------------------------------//
    // Step 7: Sample from distribution
    //--------------------------------------------------//
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int sampled = dist(rng_);

    if (sampled < 0 || sampled >= vocab_size) {
        throw std::runtime_error("SamplingPipeline: sampled token out of range: " +
                                 std::to_string(sampled));
    }

    float prob = std::max(probs[sampled], 1e-10f);

    SampleResult result;
    result.token_id = sampled;
    result.probability = prob;
    result.log_probability = std::log(prob);
    return result;
}

//======================================================//
//  Bridge: Convert legacy SamplingStrategy → Sampling::Strategy
//======================================================//
Strategy convertStrategy(int legacy_strategy) {
    // Maps from SamplingStrategy enum values:
    //   GREEDY=0, TOP_K=1, TOP_P=2, MIN_P=3, TYPICAL=4, TOP_K_TOP_P=5, BEAM_SEARCH=6
    switch (legacy_strategy) {
        case 0: return Strategy::GREEDY;
        case 1: return Strategy::TOP_K;
        case 2: return Strategy::TOP_P;
        case 3: return Strategy::MIN_P;
        case 4: return Strategy::TYPICAL;
        case 5: return Strategy::TOP_K_TOP_P;
        case 6:
            throw std::runtime_error("convertStrategy: BEAM_SEARCH is not supported");
        default:
            throw std::runtime_error("convertStrategy: unknown strategy value " +
                                     std::to_string(legacy_strategy));
    }
}

//======================================================//
//  Bridge: Build SamplingConfig from GenerationConfig fields
//======================================================//
SamplingConfig buildFromGenerationConfig(
    int strategy,
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
    unsigned int seed)
{
    SamplingConfig sc;
    sc.strategy = convertStrategy(strategy);
    sc.do_sample = do_sample;
    sc.temperature = temperature;
    sc.top_k = top_k;
    sc.top_p = top_p;
    sc.min_p = min_p;
    sc.typical_p = typical_p;
    sc.repetition_penalty = repetition_penalty;
    sc.repetition_penalty_window = repetition_penalty_window;
    sc.frequency_penalty = frequency_penalty;
    sc.presence_penalty = presence_penalty;
    sc.no_repeat_ngram_size = no_repeat_ngram_size;
    sc.eos_token_id = eos_token_id;
    sc.bos_token_id = bos_token_id;
    sc.pad_token_id = pad_token_id;
    sc.unk_token_id = unk_token_id;
    sc.bad_token_ids = bad_token_ids;
    sc.seed = seed;
    return sc;
}

} // namespace Sampling
} // namespace GRIM
