//======================================================//
//  AutogradSelectorSupervisionLoss.cu
//  Decode-time selector supervision loss primitive
//======================================================//

#include "AutogradSelectorSupervisionLoss.hpp"
#include "AutogradTraining.hpp"

#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Execution/DecodeTimeNumPolicy.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIM {
namespace Autograd {

float addSelectorSupervisionLoss(
    AutogradContext& ctx,
    AutogradIntermediates& intermediates
) {
    const auto* cfg = ctx.config;
    if (!cfg) {
        throw std::runtime_error("addSelectorSupervisionLoss: ctx.config is NULL");
    }
    const auto selector_hp = HyperParameters::decodeTimeSelectorConstructionHP(*cfg);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*cfg);

    float selector_supervision_loss = 0.0f;

    // Keep SelectorForwardResult objects alive until backward completes — stored in
    // autograd_intermediates so they survive past computeAutogradLoss() return.
    auto& selector_fwd_results = intermediates.selector_fwd_results;
    selector_fwd_results.clear();
    intermediates.selector_h_t_inputs.clear();
    intermediates.selector_slot_feature_inputs.clear();

    const bool selector_supervision_configured =
        selector_hp.enabled && selector_hp.supervision_weight > 0.0f;
    const bool selector_targets_supplied =
        ctx.payload && !ctx.payload->slot_selection_targets.empty();

    bool selector_targets_request_ce = false;
    if (selector_targets_supplied) {
        for (const auto& row_targets : ctx.payload->slot_selection_targets) {
            for (const auto& target : row_targets) {
                if (target.kind != Execution::SlotSelectionTargetKind::Ignore) {
                    selector_targets_request_ce = true;
                    break;
                }
            }
            if (selector_targets_request_ce) break;
        }
    }

    if (!selector_supervision_configured || !selector_targets_request_ce) {
        return selector_supervision_loss;
    }

    if (!execution_hp.enabled) {
        throw std::runtime_error("addSelectorSupervisionLoss: selector supervision configured but execution_block_enabled=false; final candidate state cannot be built");
    }
    if (!ctx.model) {
        throw std::runtime_error("addSelectorSupervisionLoss: selector supervision configured but ctx.model is NULL");
    }
    if (!ctx.payload) {
        throw std::runtime_error("addSelectorSupervisionLoss: selector supervision configured but ctx.payload is NULL");
    }
    const auto& payload = *ctx.payload;
    if (intermediates.exec_memories.empty()) {
        throw std::runtime_error("addSelectorSupervisionLoss: selector supervision configured but no ExecutionMemory snapshots exist; materializeTrainingGraphActivations must run ExecutionBlock first");
    }
    if (!intermediates.loss_tensor.data) {
        throw std::runtime_error("addSelectorSupervisionLoss: loss_tensor is NULL before selector CE accumulation");
    }

    auto* selector = ctx.model->getDecodeTimeSlotSelectorLayer();
    auto* policy = ctx.model->getDecodeTimeNumPolicy();

    if (!selector) {
        throw std::runtime_error("addSelectorSupervisionLoss: selector supervision configured but DecodeTimeSlotSelectorLayer is NULL");
    }
    if (!policy) {
        throw std::runtime_error("addSelectorSupervisionLoss: selector supervision configured but DecodeTimeNumPolicy is NULL");
    }

    const int d_model = selector_hp.d_model;
    const int seq_len = payload.max_seq_len;
    const Tensor* live_lm_head_input = intermediates.liveLmHeadInputOrNull();
    const float* d_hidden = live_lm_head_input ? live_lm_head_input->data : nullptr;
    if (!d_hidden) {
        throw std::runtime_error("addSelectorSupervisionLoss: encoder output tensor is NULL for selector supervision");
    }

    int ce_count = 0;

    // First pass: validate final-state supervision contract and count examples
    // that MUST emit CE. Any non-Ignore target that cannot be represented is a
    // data/model contract bug, not a denominator skip.
    for (int b = 0; b < payload.batch_size; ++b) {
        if (!payload.execution_active.empty()
            && !payload.execution_active[b]) {
            continue;
        }
        if (b >= static_cast<int>(payload.slot_selection_targets.size())) {
            continue;
        }
        const auto& row_targets = payload.slot_selection_targets[b];
        if (row_targets.empty()) continue;
        if (b >= static_cast<int>(intermediates.exec_memories.size())) {
            continue;
        }
        const auto& mem = intermediates.exec_memories[b];
        if (!mem.valid_mask.data) continue;

        const int row_len = std::min(seq_len, static_cast<int>(row_targets.size()));
        int row_supervised_t = -1;
        for (int t = 0; t < row_len; ++t) {
            if (row_targets[t].kind == Execution::SlotSelectionTargetKind::Ignore) {
                continue;
            }
            if (row_supervised_t >= 0) {
                throw std::runtime_error(
                    "addSelectorSupervisionLoss: selector supervision row=" + std::to_string(b)
                    + " has multiple non-Ignore per-token targets (first="
                    + std::to_string(row_supervised_t) + ", second=" + std::to_string(t)
                    + "). Current selector training is final-state only; provide one final-position target or add timestep memory snapshots.");
            }
            row_supervised_t = t;
        }
        if (row_supervised_t >= 0) {
            if (row_supervised_t != row_len - 1) {
                throw std::runtime_error(
                    "addSelectorSupervisionLoss: selector supervision row=" + std::to_string(b)
                    + " target at token=" + std::to_string(row_supervised_t)
                    + " but available candidates are final-state only at token="
                    + std::to_string(row_len - 1)
                    + "; refusing to train against the wrong candidate state");
            }
            ce_count++;
        }
    }

    if (ce_count == 0) {
        return selector_supervision_loss;
    }

    const float per_pos_weight = selector_hp.supervision_weight
                               / static_cast<float>(ce_count);

    selector_fwd_results.reserve(static_cast<size_t>(ce_count));
    intermediates.selector_h_t_inputs.reserve(static_cast<size_t>(ce_count));
    intermediates.selector_slot_feature_inputs.reserve(static_cast<size_t>(ce_count));

    // Second pass: autograd forward + CE for each supervised final position.
    for (int b = 0; b < payload.batch_size; ++b) {
        if (!payload.execution_active.empty()
            && !payload.execution_active[b]) {
            continue;
        }
        if (b >= static_cast<int>(payload.slot_selection_targets.size())) {
            continue;
        }
        const auto& row_targets = payload.slot_selection_targets[b];
        if (row_targets.empty()) continue;
        if (b >= static_cast<int>(intermediates.exec_memories.size())) {
            continue;
        }
        const auto& mem = intermediates.exec_memories[b];
        if (!mem.valid_mask.data) continue;

        policy->buildCandidateSet(
            mem.valid_mask.data,
            mem.values.data,
            mem.recent_write_mask.data,
            mem.usage.data,
            policy->hp().num_slots,
            policy->hp().scratch_slots,
            ctx.stream);
        const auto& cands = policy->candidates();

        const int row_len = std::min(seq_len, static_cast<int>(row_targets.size()));
        for (int t = 0; t < row_len; ++t) {
            const auto& tgt = row_targets[t];
            if (tgt.kind == Execution::SlotSelectionTargetKind::Ignore) {
                continue;
            }
            if (t != row_len - 1) {
                throw std::runtime_error(
                    "addSelectorSupervisionLoss: selector target row=" + std::to_string(b)
                    + " token=" + std::to_string(t)
                    + " reached second pass despite final-state prevalidation");
            }

            // Hidden state for this final position — owned detached copy.
            // Selector supervision intentionally trains selector parameters only.
            float* h_t_ptr = const_cast<float*>(
                d_hidden + (static_cast<size_t>(b) * seq_len + t) * d_model);
            Tensor h_t_owned = Tensor::empty(
                TensorContract::TensorShape::make_BSM(1, d_model),
                false,
                ctx.stream,
                "selector_h_t_owned_detached");
            cudaMemcpyAsync(h_t_owned.data, h_t_ptr,
                            static_cast<size_t>(d_model) * sizeof(float),
                            cudaMemcpyDeviceToDevice, ctx.stream);
            intermediates.selector_h_t_inputs.push_back(std::move(h_t_owned));
            Tensor& h_t_input = intermediates.selector_h_t_inputs.back();

            int num_live = cands.num_live_slots > 0 ? cands.num_live_slots : 0;
            if (num_live <= 0 || !cands.d_slot_features) {
                throw std::runtime_error(
                    "addSelectorSupervisionLoss: selector target row=" + std::to_string(b)
                    + " token=" + std::to_string(t)
                    + " has no live slot candidates; non-Ignore selector target cannot emit meaningful CE");
            }

            Tensor slot_feat_owned = Tensor::empty(
                TensorContract::TensorShape::make_BSM(num_live, kSlotFeatureDim),
                false,
                ctx.stream,
                "selector_slot_features_owned_detached");
            cudaMemcpyAsync(slot_feat_owned.data, cands.d_slot_features,
                            static_cast<size_t>(num_live) * kSlotFeatureDim * sizeof(float),
                            cudaMemcpyDeviceToDevice, ctx.stream);
            intermediates.selector_slot_feature_inputs.push_back(std::move(slot_feat_owned));
            Tensor& slot_feat_input = intermediates.selector_slot_feature_inputs.back();

            int target_idx = -1;
            if (tgt.kind == Execution::SlotSelectionTargetKind::Null) {
                target_idx = 0;
            } else if (tgt.kind == Execution::SlotSelectionTargetKind::Slot) {
                for (int c = 0; c < cands.num_live_slots; ++c) {
                    if (cands.live_slot_ids[c] == tgt.slot_id) {
                        target_idx = 1 + c;
                        break;
                    }
                }
                if (target_idx < 0) {
                    throw std::runtime_error(
                        "addSelectorSupervisionLoss: selector target slot=" + std::to_string(tgt.slot_id)
                        + " not present in final candidate set for row=" + std::to_string(b)
                        + " token=" + std::to_string(t)
                        + "; refusing to shrink CE denominator silently");
                }
            } else {
                throw std::runtime_error(
                    "addSelectorSupervisionLoss: selector target row=" + std::to_string(b)
                    + " token=" + std::to_string(t)
                    + " has illegal target kind=" + std::to_string(static_cast<int>(tgt.kind)));
            }

            SelectorForwardResult fwd = selector->forward(
                h_t_input, slot_feat_input, num_live, ctx.stream, ctx.cublas_handle);

            Tensor ce = autograd::cross_entropy_logits(
                fwd.scores, target_idx, ctx.stream);
            Tensor ce_scaled = autograd::scale_scalar(ce, per_pos_weight, ctx.stream);

            intermediates.loss_tensor = autograd::add(
                intermediates.loss_tensor, ce_scaled, ctx.stream);

            float ce_val = 0.0f;
            cudaMemcpyAsync(&ce_val, ce.data, sizeof(float),
                            cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            selector_supervision_loss += per_pos_weight * ce_val;

            selector_fwd_results.push_back(std::move(fwd));
        }
    }

    return selector_supervision_loss;
}

}  // namespace Autograd
}  // namespace GRIM
