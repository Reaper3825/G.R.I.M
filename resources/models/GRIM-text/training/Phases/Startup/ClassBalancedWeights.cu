//======================================================//
//  Startup/ClassBalancedWeights.cu
//
//  Implementation of computeAndUploadClassBalancedWeights.
//  Lifted verbatim from Phase1_Startup step 10b — same math,
//  same logging, same cuda calls.
//======================================================//

#include "ClassBalancedWeights.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace GRIMText::Training {

void computeAndUploadClassBalancedWeights(
    const std::vector<TrainingSequence>& train_seqs,
    std::uint32_t vocab_size,
    float beta,
    GRIM::TrainingState& ts,
    TrainingLogger& logger) {

    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();

    // Count target frequencies across ALL training sequences
    std::vector<std::int64_t> target_counts(vocab_size, 0);
    std::int64_t total_targets = 0;
    for (const auto& seq : train_seqs) {
        for (int tgt : seq.targets) {
            if (tgt >= 0 && tgt < static_cast<int>(vocab_size)) {
                target_counts[tgt]++;
                total_targets++;
            }
        }
    }

    if (total_targets <= 0) {
        throw std::runtime_error("[class_balanced] total_targets=0 — no valid targets in training data");
    }

    // Compute weights: w_v = 1/freq(v)^β, clamped for unseen tokens.
    // Unseen tokens get weight = max_weight (same as the rarest seen token).
    std::vector<float> h_class_weights(vocab_size);
    float max_weight = 0.0f;
    int seen_count = 0;
    int unseen_count = 0;

    for (std::uint32_t v = 0; v < vocab_size; v++) {
        if (target_counts[v] > 0) {
            const float freq = static_cast<float>(target_counts[v]) / static_cast<float>(total_targets);
            h_class_weights[v] = 1.0f / std::pow(freq, beta);
            max_weight = std::max(max_weight, h_class_weights[v]);
            seen_count++;
        } else {
            h_class_weights[v] = 0.0f;  // placeholder, filled after loop
            unseen_count++;
        }
    }

    // Unseen tokens: clamp to max_weight (the rarest seen token's weight).
    // Prevents infinite weights while still giving rare tokens maximum upweight.
    if (max_weight <= 0.0f) max_weight = 1.0f;  // Safety: shouldn't happen
    for (std::uint32_t v = 0; v < vocab_size; v++) {
        if (target_counts[v] == 0) {
            h_class_weights[v] = max_weight;
        }
    }

    if (vocab_size > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("[class_balanced] vocab_size exceeds Tensor shape int capacity");
    }

    // Upload to Tensor-owned GPU storage
    const std::size_t weights_bytes = static_cast<std::size_t>(vocab_size) * sizeof(float);
    ts.class_weights_tensor = GRIM::Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(vocab_size)),
        false,
        stream,
        "phase1_class_weights");
    cudaMemcpyAsync(ts.class_weights_tensor.data, h_class_weights.data(), weights_bytes,
                     cudaMemcpyHostToDevice, stream);
    ts.class_weights_vocab_size = static_cast<int>(vocab_size);
    cudaStreamSynchronize(stream);

    // Log top-10 highest and lowest weight tokens for verification
    std::vector<std::pair<float, std::uint32_t>> weight_pairs;
    for (std::uint32_t v = 0; v < vocab_size; v++) {
        if (target_counts[v] > 0) {
            weight_pairs.push_back({h_class_weights[v], v});
        }
    }
    std::sort(weight_pairs.begin(), weight_pairs.end());

    std::ostringstream cb_msg;
    cb_msg << "[CLASS_BALANCED] β=" << beta
           << " total_targets=" << total_targets
           << " seen_tokens=" << seen_count
           << " unseen_tokens=" << unseen_count
           << " max_weight=" << max_weight;
    logger.log(cb_msg.str());

    // Lowest weights = most frequent tokens
    cb_msg.str("");
    cb_msg << "[CLASS_BALANCED] Lowest weights (most frequent): ";
    for (int i = 0; i < std::min(10, static_cast<int>(weight_pairs.size())); i++) {
        cb_msg << "tok" << weight_pairs[i].second
               << "(w=" << std::fixed << std::setprecision(2) << weight_pairs[i].first
               << ",cnt=" << target_counts[weight_pairs[i].second] << ") ";
    }
    logger.log(cb_msg.str());

    // Highest weights = rarest seen tokens
    cb_msg.str("");
    cb_msg << "[CLASS_BALANCED] Highest weights (rarest seen): ";
    for (int i = std::max(0, static_cast<int>(weight_pairs.size()) - 10);
         i < static_cast<int>(weight_pairs.size()); i++) {
        cb_msg << "tok" << weight_pairs[i].second
               << "(w=" << std::fixed << std::setprecision(2) << weight_pairs[i].first
               << ",cnt=" << target_counts[weight_pairs[i].second] << ") ";
    }
    logger.log(cb_msg.str());

    // Log ratio: max_weight / min_weight shows the dynamic range
    if (!weight_pairs.empty()) {
        const float ratio = weight_pairs.back().first / weight_pairs.front().first;
        cb_msg.str("");
        cb_msg << "[CLASS_BALANCED] Dynamic range: max/min weight ratio = "
               << std::fixed << std::setprecision(1) << ratio << "x";
        logger.log(cb_msg.str());
    }
}

} // namespace GRIMText::Training
