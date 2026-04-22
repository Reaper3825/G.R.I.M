#include "PhysicalClassPolicy.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Stable ordering for the ranked summary: ascending priority_rank, then
// canonical_label lexicographic. Both fields are deterministic so the UI
// never reshuffles row positions between frames for the same set.
bool ClassSummaryComesBefore(const PhysicalClassPolicyClassSummary& a,
                             const PhysicalClassPolicyClassSummary& b) {
    if (a.priority_rank != b.priority_rank) return a.priority_rank < b.priority_rank;
    return a.canonical_label < b.canonical_label;
}

// Update the running max — used to bubble per-signal confidences into
// PhysicalClassPolicyClassSummary::max_confidence.
inline void AccumulateMaxConfidence(float& running_max, float candidate) {
    if (candidate > running_max) running_max = candidate;
}

// IoU on cv::Rect2f. Empty boxes (w<=0 or h<=0) yield 0 — consistent with
// "cannot overlap" rather than throwing, since legitimately-empty boxes
// arise during track spawn before the first smoothing step.
inline float IouRect2f(const cv::Rect2f& a, const cv::Rect2f& b) {
    if (a.width <= 0.0f || a.height <= 0.0f) return 0.0f;
    if (b.width <= 0.0f || b.height <= 0.0f) return 0.0f;
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(a.x + a.width,  b.x + b.width);
    const float iy2 = std::min(a.y + a.height, b.y + b.height);
    const float iw  = ix2 - ix1;
    const float ih  = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float uni   = a.width * a.height + b.width * b.height - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
}

// Intersection-over-Min-area. Captures the "small box mostly contained in
// a large box" case that pure IoU misses: a 50x50 ghost track overlapping
// fully with a 500x500 confirmed track has IoU=0.01 but IoMin=1.0. This is
// the regime that produces ghost couch/chair fragments hanging off a
// confirmed couch — IoU alone never fires.
inline float IoMinRect2f(const cv::Rect2f& a, const cv::Rect2f& b) {
    if (a.width <= 0.0f || a.height <= 0.0f) return 0.0f;
    if (b.width <= 0.0f || b.height <= 0.0f) return 0.0f;
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(a.x + a.width,  b.x + b.width);
    const float iy2 = std::min(a.y + a.height, b.y + b.height);
    const float iw  = ix2 - ix1;
    const float ih  = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float min_area = std::min(a.width * a.height, b.width * b.height);
    return (min_area > 0.0f) ? (inter / min_area) : 0.0f;
}

// Greedy same-canonical NMS: sort items by `score` (descending) and
// suppress later items that share the same canonical_label AND overlap the
// pivot's box with max(IoU, IoMin) >= threshold. Used post-merge so that
// two items whose ORIGINAL classes differed (e.g. chair + couch on the
// same physical object) are now both labelled "couch" and collapse into
// one. Returns the number of items dropped.
template <typename ItemT, typename GetBox, typename GetScore>
size_t SuppressOverlappingSameCanonical(
    std::vector<ItemT>& items,
    float                threshold,
    GetBox               get_box,
    GetScore             get_score)
{
    if (threshold <= 0.0f || items.size() < 2) return 0;
    std::vector<size_t> order(items.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return get_score(items[a]) > get_score(items[b]); });
    std::vector<uint8_t> suppressed(items.size(), 0);
    size_t dropped = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const size_t pivot = order[i];
        if (suppressed[pivot]) continue;
        const auto&         pivot_box   = get_box(items[pivot]);
        const std::string&  pivot_label = items[pivot].class_label;
        for (size_t j = i + 1; j < order.size(); ++j) {
            const size_t cand = order[j];
            if (suppressed[cand]) continue;
            if (items[cand].class_label != pivot_label) continue;
            const auto& cand_box = get_box(items[cand]);
            const float iou   = IouRect2f(pivot_box, cand_box);
            const float iomin = IoMinRect2f(pivot_box, cand_box);
            if (std::max(iou, iomin) < threshold) continue;
            suppressed[cand] = 1;
            ++dropped;
        }
    }
    std::vector<ItemT> kept;
    kept.reserve(items.size() - dropped);
    for (size_t k = 0; k < items.size(); ++k) {
        if (!suppressed[k]) kept.push_back(std::move(items[k]));
    }
    items = std::move(kept);
    return dropped;
}

} // anonymous namespace

PhysicalClassPolicy::PhysicalClassPolicy()  = default;
PhysicalClassPolicy::~PhysicalClassPolicy() = default;

void PhysicalClassPolicy::ConfigurePhysicalClassPolicy(
    const PhysicalClassPolicyConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);

    // ── Validate up front. Rule 20: silent fallbacks forbidden. ──
    if (cfg.emit_only_top_rank < 0) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalClassPolicy: emit_only_top_rank must be >= 0 (got "
                             + std::to_string(cfg.emit_only_top_rank) + ")";
        throw std::runtime_error(last_error_reason_);
    }
    if (!(cfg.default_confidence_floor >= 0.0f && cfg.default_confidence_floor <= 1.0f)) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalClassPolicy: default_confidence_floor must be in [0, 1] (got "
                             + std::to_string(cfg.default_confidence_floor) + ")";
        throw std::runtime_error(last_error_reason_);
    }
    if (!(cfg.post_merge_nms_iou >= 0.0f && cfg.post_merge_nms_iou <= 1.0f)) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalClassPolicy: post_merge_nms_iou must be in [0, 1] (got "
                             + std::to_string(cfg.post_merge_nms_iou) + ")";
        throw std::runtime_error(last_error_reason_);
    }

    // Build the alias map. Every canonical maps to itself first so that
    // an unaliased label still hits the priority lookup correctly.
    std::unordered_map<std::string, std::string> alias_to_canonical;
    for (size_t i = 0; i < cfg.merge_rules.size(); ++i) {
        const auto& rule = cfg.merge_rules[i];
        if (rule.canonical_label.empty()) {
            state_             = PhysicalImageOperatorState::ModelLoadFailed;
            last_error_reason_ = "ConfigurePhysicalClassPolicy: merge_rules[" + std::to_string(i)
                                 + "].canonical_label is empty";
            throw std::runtime_error(last_error_reason_);
        }
        if (rule.source_labels.empty()) {
            state_             = PhysicalImageOperatorState::ModelLoadFailed;
            last_error_reason_ = "ConfigurePhysicalClassPolicy: merge_rules[" + std::to_string(i)
                                 + "].source_labels is empty (canonical=\""
                                 + rule.canonical_label + "\")";
            throw std::runtime_error(last_error_reason_);
        }
        // Canonical maps to itself; record before aliases so a self-rule
        // is always idempotent and conflict checks see it.
        {
            auto it = alias_to_canonical.find(rule.canonical_label);
            if (it != alias_to_canonical.end() && it->second != rule.canonical_label) {
                state_             = PhysicalImageOperatorState::ModelLoadFailed;
                last_error_reason_ = "ConfigurePhysicalClassPolicy: alias \""
                                     + rule.canonical_label + "\" already maps to \""
                                     + it->second + "\" (cannot also be canonical of itself)";
                throw std::runtime_error(last_error_reason_);
            }
            alias_to_canonical[rule.canonical_label] = rule.canonical_label;
        }
        for (const auto& alias : rule.source_labels) {
            if (alias.empty()) {
                state_             = PhysicalImageOperatorState::ModelLoadFailed;
                last_error_reason_ = "ConfigurePhysicalClassPolicy: merge_rules[" + std::to_string(i)
                                     + "] contains an empty source_label (canonical=\""
                                     + rule.canonical_label + "\")";
                throw std::runtime_error(last_error_reason_);
            }
            auto it = alias_to_canonical.find(alias);
            if (it != alias_to_canonical.end() && it->second != rule.canonical_label) {
                state_             = PhysicalImageOperatorState::ModelLoadFailed;
                last_error_reason_ = "ConfigurePhysicalClassPolicy: alias \"" + alias
                                     + "\" maps to \"" + it->second + "\" AND \""
                                     + rule.canonical_label + "\"";
                throw std::runtime_error(last_error_reason_);
            }
            alias_to_canonical[alias] = rule.canonical_label;
        }
    }

    // Build the priority maps. Conflicting declarations for the same
    // canonical are rejected.
    std::unordered_map<std::string, int32_t> canonical_to_rank;
    std::unordered_map<std::string, float>   canonical_to_floor;
    for (size_t i = 0; i < cfg.priority_rules.size(); ++i) {
        const auto& rule = cfg.priority_rules[i];
        if (rule.canonical_label.empty()) {
            state_             = PhysicalImageOperatorState::ModelLoadFailed;
            last_error_reason_ = "ConfigurePhysicalClassPolicy: priority_rules[" + std::to_string(i)
                                 + "].canonical_label is empty";
            throw std::runtime_error(last_error_reason_);
        }
        if (!(rule.confidence_floor >= 0.0f && rule.confidence_floor <= 1.0f)) {
            state_             = PhysicalImageOperatorState::ModelLoadFailed;
            last_error_reason_ = "ConfigurePhysicalClassPolicy: priority_rules[" + std::to_string(i)
                                 + "].confidence_floor must be in [0, 1] (got "
                                 + std::to_string(rule.confidence_floor) + ", canonical=\""
                                 + rule.canonical_label + "\")";
            throw std::runtime_error(last_error_reason_);
        }
        {
            auto it = canonical_to_rank.find(rule.canonical_label);
            if (it != canonical_to_rank.end() && it->second != rule.priority_rank) {
                state_             = PhysicalImageOperatorState::ModelLoadFailed;
                last_error_reason_ = "ConfigurePhysicalClassPolicy: canonical \""
                                     + rule.canonical_label + "\" declared with rank "
                                     + std::to_string(it->second) + " AND "
                                     + std::to_string(rule.priority_rank);
                throw std::runtime_error(last_error_reason_);
            }
        }
        {
            auto it = canonical_to_floor.find(rule.canonical_label);
            if (it != canonical_to_floor.end() && it->second != rule.confidence_floor) {
                state_             = PhysicalImageOperatorState::ModelLoadFailed;
                last_error_reason_ = "ConfigurePhysicalClassPolicy: canonical \""
                                     + rule.canonical_label + "\" declared with floor "
                                     + std::to_string(it->second) + " AND "
                                     + std::to_string(rule.confidence_floor);
                throw std::runtime_error(last_error_reason_);
            }
        }
        canonical_to_rank[rule.canonical_label]  = rule.priority_rank;
        canonical_to_floor[rule.canonical_label] = rule.confidence_floor;
    }

    cfg_                  = cfg;
    alias_to_canonical_   = std::move(alias_to_canonical);
    canonical_to_rank_    = std::move(canonical_to_rank);
    canonical_to_floor_   = std::move(canonical_to_floor);
    state_                = PhysicalImageOperatorState::ModelLoaded;
    last_error_reason_.clear();

    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
              std::string("ConfigurePhysicalClassPolicy: merge_rules=")
              + std::to_string(cfg_.merge_rules.size())
              + " priority_rules=" + std::to_string(cfg_.priority_rules.size())
              + " default_rank=" + std::to_string(cfg_.default_priority_rank)
              + " default_floor=" + std::to_string(cfg_.default_confidence_floor)
              + " emit_only_top_rank=" + std::to_string(cfg_.emit_only_top_rank));
}

void PhysicalClassPolicy::ResetPhysicalClassPolicy() {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_                  = PhysicalClassPolicyConfig{};
    alias_to_canonical_.clear();
    canonical_to_rank_.clear();
    canonical_to_floor_.clear();
    state_                = PhysicalImageOperatorState::NoModelConfigured;
    last_error_reason_.clear();
    inference_count_      = 0;
    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG, "ResetPhysicalClassPolicy: rule set cleared");
}

PhysicalImageOperatorState PhysicalClassPolicy::GetPhysicalClassPolicyState() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_;
}

std::string PhysicalClassPolicy::GetPhysicalClassPolicyLastError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_reason_;
}

bool PhysicalClassPolicy::IsPhysicalClassPolicyReady() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_ == PhysicalImageOperatorState::ModelLoaded;
}

PhysicalClassPolicyConfig PhysicalClassPolicy::GetPhysicalClassPolicyConfig() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return cfg_;
}

void PhysicalClassPolicy::ApplyPhysicalClassPolicyToPerceptionResults(
    PhysicalPerceptionPrimitiveResults& results,
    PhysicalClassPolicyOutput& out)
{
    std::lock_guard<std::mutex> lk(mutex_);

    out = PhysicalClassPolicyOutput{};
    out.state              = state_;
    out.last_error_reason  = last_error_reason_;
    out.inference_count    = inference_count_;
    out.last_frame_counter = results.source_frame_counter;

    if (state_ != PhysicalImageOperatorState::ModelLoaded) {
        // Not ready — surface state and stop. No silent skip.
        return;
    }

    const auto t_start = std::chrono::steady_clock::now();

    try {
        // Local helpers that close over the rule maps.
        auto resolve_canonical = [&](const std::string& src) -> const std::string& {
            auto it = alias_to_canonical_.find(src);
            return (it != alias_to_canonical_.end()) ? it->second : src;
        };
        auto rank_for_canonical = [&](const std::string& canon) -> int32_t {
            auto it = canonical_to_rank_.find(canon);
            return (it != canonical_to_rank_.end()) ? it->second : cfg_.default_priority_rank;
        };
        auto floor_for_canonical = [&](const std::string& canon) -> float {
            auto it = canonical_to_floor_.find(canon);
            return (it != canonical_to_floor_.end()) ? it->second : cfg_.default_confidence_floor;
        };
        const bool   priority_cutoff_active = cfg_.emit_only_top_rank > 0;
        const int32_t cutoff_rank           = cfg_.emit_only_top_rank;

        // Per-canonical accumulator for the ranked summary. Use a map so
        // the final sort is over a small distinct set, not over every item.
        std::unordered_map<std::string, PhysicalClassPolicyClassSummary> summary;

        // Decision for one item — canonical label, whether to keep, and
        // (when keeping) the rank used for the cutoff test.
        struct ItemDecision {
            std::string canonical;
            int32_t     rank   = 0;
            bool        keep   = true;
            bool        relabeled = false;       // canonical != original label
            bool        dropped_by_confidence = false;
            bool        dropped_by_priority   = false;
        };
        auto decide = [&](const std::string& original_label,
                          float confidence) -> ItemDecision {
            ItemDecision d;
            d.canonical = resolve_canonical(original_label);
            d.relabeled = (d.canonical != original_label);
            d.rank      = rank_for_canonical(d.canonical);
            const float floor = floor_for_canonical(d.canonical);
            if (confidence < floor) {
                d.keep = false;
                d.dropped_by_confidence = true;
                return d;
            }
            if (priority_cutoff_active && d.rank > cutoff_rank) {
                d.keep = false;
                d.dropped_by_priority = true;
                return d;
            }
            return d;
        };

        // ── 1. Object detector ──────────────────────────────────────────
        {
            auto& dets = results.object_detector.detections;
            const size_t before = dets.size();
            std::vector<PhysicalObjectDetection> kept;
            kept.reserve(before);
            for (auto& d : dets) {
                const auto dec = decide(d.class_label, d.confidence);
                if (dec.relabeled) ++out.detections_relabeled;
                if (!dec.keep) {
                    if (dec.dropped_by_confidence || dec.dropped_by_priority) {
                        ++out.detections_dropped;
                    }
                    continue;
                }
                d.class_label = dec.canonical;
                auto& row = summary[dec.canonical];
                if (row.canonical_label.empty()) {
                    row.canonical_label = dec.canonical;
                    row.priority_rank   = dec.rank;
                }
                ++row.detection_count;
                AccumulateMaxConfidence(row.max_confidence, d.confidence);
                kept.push_back(std::move(d));
            }
            dets = std::move(kept);
        }

        // ── 2. Entity tracker ───────────────────────────────────────────
        {
            auto& tracks = results.entity_tracker.tracks;
            std::vector<PhysicalEntityTrack> kept;
            kept.reserve(tracks.size());
            for (auto& t : tracks) {
                // Tracks carry smoothed_confidence as the running estimate;
                // last_detection_confidence is per-frame. Use the smoothed
                // value for the policy decision so a single noisy miss does
                // not evict an otherwise stable track.
                const auto dec = decide(t.class_label, t.smoothed_confidence);
                if (dec.relabeled) ++out.tracks_relabeled;
                if (!dec.keep) {
                    if (dec.dropped_by_confidence || dec.dropped_by_priority) {
                        ++out.tracks_dropped;
                    }
                    continue;
                }
                t.class_label = dec.canonical;
                auto& row = summary[dec.canonical];
                if (row.canonical_label.empty()) {
                    row.canonical_label = dec.canonical;
                    row.priority_rank   = dec.rank;
                }
                ++row.track_count;
                AccumulateMaxConfidence(row.max_confidence, t.smoothed_confidence);
                kept.push_back(std::move(t));
            }
            tracks = std::move(kept);
        }

        // ── 3. Instance segmenter masks ─────────────────────────────────
        {
            auto& masks = results.instance_segmenter.segmentation.instances;
            std::vector<PhysicalInstanceMask> kept;
            kept.reserve(masks.size());
            for (auto& m : masks) {
                // Masks carry both detection_confidence (from the prompting
                // box) and mask_confidence (SAM 2 IoU prediction). The
                // policy is about CLASS identity, so use detection_confidence
                // as the gating signal — mask_confidence is a quality score
                // for the mask shape, not the class.
                const auto dec = decide(m.class_label, m.detection_confidence);
                if (dec.relabeled) ++out.instance_masks_relabeled;
                if (!dec.keep) {
                    if (dec.dropped_by_confidence || dec.dropped_by_priority) {
                        ++out.instance_masks_dropped;
                    }
                    continue;
                }
                m.class_label = dec.canonical;
                auto& row = summary[dec.canonical];
                if (row.canonical_label.empty()) {
                    row.canonical_label = dec.canonical;
                    row.priority_rank   = dec.rank;
                }
                ++row.instance_mask_count;
                AccumulateMaxConfidence(row.max_confidence, m.detection_confidence);
                kept.push_back(std::move(m));
            }
            masks = std::move(kept);
        }

        // ── 4. Image classifier top-K ───────────────────────────────────
        {
            auto& topk = results.image_classifier.top_k;
            std::vector<PhysicalImageClassification> kept;
            kept.reserve(topk.size());
            for (auto& c : topk) {
                const auto dec = decide(c.class_label, c.score);
                if (dec.relabeled) ++out.classifications_relabeled;
                if (!dec.keep) {
                    if (dec.dropped_by_confidence || dec.dropped_by_priority) {
                        ++out.classifications_dropped;
                    }
                    continue;
                }
                c.class_label = dec.canonical;
                auto& row = summary[dec.canonical];
                if (row.canonical_label.empty()) {
                    row.canonical_label = dec.canonical;
                    row.priority_rank   = dec.rank;
                }
                ++row.classification_count;
                AccumulateMaxConfidence(row.max_confidence, c.score);
                kept.push_back(std::move(c));
            }
            topk = std::move(kept);
        }

        // ── 5. Post-merge cross-class de-duplication ────────────────────
        // After merge re-labels (e.g. chair → couch), two items whose
        // ORIGINAL classes differed may now share the same canonical label
        // and overlap on the same physical object. Greedy NMS collapses
        // them. Disabled when post_merge_nms_iou == 0. Counters bump the
        // existing *_dropped fields so callers see one combined drop count.
        if (cfg_.post_merge_nms_iou > 0.0f) {
            const float thr = cfg_.post_merge_nms_iou;

            const size_t det_dropped = SuppressOverlappingSameCanonical(
                results.object_detector.detections, thr,
                [](const PhysicalObjectDetection& d) -> const cv::Rect2f& { return d.model_box; },
                [](const PhysicalObjectDetection& d) { return d.confidence; });
            out.detections_dropped += static_cast<uint32_t>(det_dropped);

            const size_t trk_dropped = SuppressOverlappingSameCanonical(
                results.entity_tracker.tracks, thr,
                [](const PhysicalEntityTrack& t) -> const cv::Rect2f& { return t.smoothed_model_box; },
                [](const PhysicalEntityTrack& t) { return t.smoothed_confidence; });
            out.tracks_dropped += static_cast<uint32_t>(trk_dropped);

            const size_t msk_dropped = SuppressOverlappingSameCanonical(
                results.instance_segmenter.segmentation.instances, thr,
                [](const PhysicalInstanceMask& m) -> const cv::Rect2f& { return m.prompt_model_box; },
                [](const PhysicalInstanceMask& m) { return m.detection_confidence; });
            out.instance_masks_dropped += static_cast<uint32_t>(msk_dropped);

            // Rebuild the per-canonical summary from the survivors so the
            // ranked rows exactly reflect what downstream consumers see.
            // Image classifier top-K has no box so it is unaffected by NMS
            // but must be re-counted from the (already filtered) vector.
            summary.clear();
            auto bump_summary = [&](const std::string& canon, float conf,
                                    uint32_t det = 0, uint32_t trk = 0,
                                    uint32_t msk = 0, uint32_t cls = 0) {
                auto& row = summary[canon];
                if (row.canonical_label.empty()) {
                    row.canonical_label = canon;
                    row.priority_rank   = rank_for_canonical(canon);
                }
                row.detection_count       += det;
                row.track_count           += trk;
                row.instance_mask_count   += msk;
                row.classification_count  += cls;
                AccumulateMaxConfidence(row.max_confidence, conf);
            };
            for (const auto& d : results.object_detector.detections)
                bump_summary(d.class_label, d.confidence, 1, 0, 0, 0);
            for (const auto& t : results.entity_tracker.tracks)
                bump_summary(t.class_label, t.smoothed_confidence, 0, 1, 0, 0);
            for (const auto& m : results.instance_segmenter.segmentation.instances)
                bump_summary(m.class_label, m.detection_confidence, 0, 0, 1, 0);
            for (const auto& c : results.image_classifier.top_k)
                bump_summary(c.class_label, c.score, 0, 0, 0, 1);
        }

        // ── 6. Build the ranked summary in deterministic order ──────────
        out.ranked_classes.reserve(summary.size());
        for (auto& kv : summary) out.ranked_classes.push_back(std::move(kv.second));
        std::sort(out.ranked_classes.begin(), out.ranked_classes.end(),
                  &ClassSummaryComesBefore);

        ++inference_count_;
        out.inference_count = inference_count_;
        out.state           = PhysicalImageOperatorState::ModelLoaded;
        last_error_reason_.clear();
        out.last_error_reason.clear();
    } catch (const std::exception& e) {
        state_             = PhysicalImageOperatorState::InferenceFailed;
        last_error_reason_ = std::string("ApplyPhysicalClassPolicyToPerceptionResults: ") + e.what();
        out.state              = state_;
        out.last_error_reason  = last_error_reason_;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
    }

    const auto t_end = std::chrono::steady_clock::now();
    out.last_apply_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
}

}}} // namespace GRIM::Perception::Physical
