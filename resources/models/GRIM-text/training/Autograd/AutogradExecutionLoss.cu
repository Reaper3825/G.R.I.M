//======================================================//
//  AutogradExecutionLoss.cu
//  Autograd-owned execution auxiliary loss assembly
//======================================================//

#include "AutogradExecutionLoss.hpp"
#include "AutogradTraining.hpp"

#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_internal.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/TensorContract/GradFns/NormalizedEntropyGradFn.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {
namespace Autograd {
namespace {

constexpr int kWarpSize = 32;
constexpr int kDivOpIdx = 3;

__global__ void kernelDivInvalidPenaltyForward(
    float* __restrict__ out_loss,
    const float* __restrict__ p_op,
    int div_flag,
    float weight,
    int div_op_idx
) {
    if (threadIdx.x != 0) return;
    out_loss[0] = div_flag ? (weight * p_op[div_op_idx]) : 0.0f;
}

__global__ void kernelDivInvalidPenaltyBackward(
    float* __restrict__ grad_p_op,
    const float* __restrict__ grad_out,
    int div_flag,
    float weight,
    int div_op_idx,
    int num_ops
) {
    if (threadIdx.x != 0) return;
    for (int k = 0; k < num_ops; ++k) {
        grad_p_op[k] = 0.0f;
    }
    if (div_flag) {
        grad_p_op[div_op_idx] = weight * grad_out[0];
    }
}

struct DivInvalidPenaltyGradFn : public GradFn {
    int saved_div_flag_ = 0;
    float weight_ = 0.0f;
    int div_op_idx_ = kDivOpIdx;
    int num_ops_ = 0;

    float* grad_p_op_ = nullptr;
    std::shared_ptr<float> owned_grad_p_op_;
    std::shared_ptr<GradFn> p_op_grad_fn_;
    TensorContract::TensorShape p_op_shape_;
    bool p_op_requires_grad_ = false;

    DivInvalidPenaltyGradFn() { op_name = "autograd_exec_div_invalid_penalty"; }

    void capture(int div_flag, float weight, int div_op_idx, int num_ops, Tensor& p_op_t, cudaStream_t stream) {
        saved_div_flag_ = div_flag;
        weight_ = weight;
        div_op_idx_ = div_op_idx;
        num_ops_ = num_ops;
        p_op_requires_grad_ = p_op_t.requires_grad;
        p_op_shape_ = p_op_t.shape;
        p_op_grad_fn_ = p_op_t.grad_fn;
        register_input(p_op_t.grad_fn);

        float* buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buf), static_cast<size_t>(num_ops_) * sizeof(float), "autograd_exec_div_penalty_grad_p_op");
        cudaMemsetAsync(buf, 0, static_cast<size_t>(num_ops_) * sizeof(float), stream);
        owned_grad_p_op_ = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
        grad_p_op_ = owned_grad_p_op_.get();
    }

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        if (applied) return;
        applied = true;

        kernelDivInvalidPenaltyBackward<<<1, 1, 0, stream>>>(
            grad_p_op_, grad_output.data, saved_div_flag_, weight_, div_op_idx_, num_ops_);
        CUDA_CHECK_KERNEL();

        if (p_op_requires_grad_ && p_op_grad_fn_) {
            Tensor view;
            view.data = grad_p_op_;
            view.shape = p_op_shape_;
            view.owns_data = false;
            view.stream = stream;
            p_op_grad_fn_->apply(view, stream, backward_payload, backward_bindings);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        grad_p_op_ = nullptr;
        owned_grad_p_op_.reset();
        p_op_grad_fn_.reset();
    }
};

void requireTensor(const Tensor& tensor, const char* label, int row, int step) {
    if (!tensor.data) {
        throw std::runtime_error(
            std::string("addExecutionAuxiliaryLoss: retained tensor '") + label
            + "' is NULL at row=" + std::to_string(row)
            + " step=" + std::to_string(step));
    }
}

void requirePositiveTemperature(float temperature, int row, int step) {
    if (!std::isfinite(temperature) || temperature <= 0.0f) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: selection_temperature must be finite and > 0 at row="
            + std::to_string(row) + " step=" + std::to_string(step)
            + ", got " + std::to_string(temperature));
    }
}

int requireTransitionTarget(
    int target,
    int num_transitions,
    int row,
    int step,
    const char* label)
{
    if (target < 0 || target >= num_transitions) {
        throw std::runtime_error(
            std::string("addExecutionAuxiliaryLoss: ") + label + "=" + std::to_string(target)
            + " is outside [0," + std::to_string(num_transitions) + ") at row="
            + std::to_string(row) + " step=" + std::to_string(step));
    }
    return target;
}

int requireValueSlotTarget(int slot_index, int scratch_slots, int num_slots, int row, int step, const char* label) {
    if (slot_index < scratch_slots || slot_index >= num_slots) {
        throw std::runtime_error(
            std::string("addExecutionAuxiliaryLoss: ") + label + "=" + std::to_string(slot_index)
            + " is outside execution value-slot range [" + std::to_string(scratch_slots)
            + "," + std::to_string(num_slots) + ") at row="
            + std::to_string(row) + " step=" + std::to_string(step));
    }
    return slot_index - scratch_slots;
}

int requireWriteTarget(int slot_index, int scratch_slots, int num_slots, int row, int step) {
    if (slot_index < scratch_slots || slot_index >= num_slots) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: transition result slot_index=" + std::to_string(slot_index)
            + " is outside writable slot range [" + std::to_string(scratch_slots)
            + "," + std::to_string(num_slots) + ") at row="
            + std::to_string(row) + " step=" + std::to_string(step));
    }
    return slot_index;
}

void requireInitializedValueTarget(
    const Tensor& state_before_valid,
    int slot_index,
    int num_slots,
    int row,
    int step,
    const char* label,
    cudaStream_t stream
) {
    requireTensor(state_before_valid, "state_before_valid", row, step);
    if (state_before_valid.numel() != static_cast<size_t>(num_slots)) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: state_before_valid has "
            + std::to_string(state_before_valid.numel())
            + " elements but expected num_slots=" + std::to_string(num_slots)
            + " at row=" + std::to_string(row)
            + " step=" + std::to_string(step));
    }

    float valid = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(
        &valid,
        state_before_valid.data + slot_index,
        sizeof(float),
        cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!std::isfinite(valid) || valid < 0.5f) {
        throw std::runtime_error(
            std::string("addExecutionAuxiliaryLoss: refusing masked ") + label
            + " CE target at row=" + std::to_string(row)
            + " step=" + std::to_string(step)
            + " slot_index=" + std::to_string(slot_index)
            + " state_before_valid=" + std::to_string(valid)
            + "; transition argument targets must reference slots initialized by "
              "bootstrap or an earlier write");
    }
}

void readScalarFromDevice(const Tensor& tensor, float& out_value, cudaStream_t stream, const char* label) {
    if (!tensor.data) {
        throw std::runtime_error(std::string("addExecutionAuxiliaryLoss: cannot read NULL scalar tensor '") + label + "'");
    }
    CUDA_CHECK(cudaMemcpyAsync(&out_value, tensor.data, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

Tensor makeDivInvalidPenalty(Tensor& p_op_tensor, bool div_was_clamped, float weight, int num_ops, cudaStream_t stream) {
    Tensor loss = Tensor::zeros({1, 1}, stream, "autograd_exec_div_invalid_penalty");
    kernelDivInvalidPenaltyForward<<<1, 1, 0, stream>>>(
        loss.data,
        p_op_tensor.data,
        div_was_clamped ? 1 : 0,
        weight,
        kDivOpIdx);
    CUDA_CHECK_KERNEL();

    auto grad_fn = std::make_shared<DivInvalidPenaltyGradFn>();
    grad_fn->capture(div_was_clamped ? 1 : 0, weight, kDivOpIdx, num_ops, p_op_tensor, stream);
    loss.grad_fn = grad_fn;
    loss.requires_grad = true;
    loss.is_leaf = false;
    return loss;
}

}  // namespace

ExecutionAuxiliaryLossSummary addExecutionAuxiliaryLoss(
    AutogradContext& ctx,
    const Batching::BatchPayload& payload,
    Forward::ModelForwardOutputs& forward_outputs,
    AutogradLossState& loss_state
) {
    ExecutionAuxiliaryLossSummary summary{};

    if (!ctx.config) {
        throw std::runtime_error("addExecutionAuxiliaryLoss: ctx.config is NULL");
    }
    if (!ctx.execution_block_enabled) {
        throw std::runtime_error("addExecutionAuxiliaryLoss: ctx.execution_block_enabled is false");
    }
    if (!loss_state.loss_tensor.data) {
        throw std::runtime_error("addExecutionAuxiliaryLoss: loss_tensor is NULL before execution loss assembly");
    }

    const auto model_hp = HyperParameters::modelHP(*ctx.config);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*ctx.config);

    if (!model_hp.execution_block_enabled || forward_outputs.exec_outputs_per_row.empty()) {
        return summary;
    }

    if (static_cast<int>(forward_outputs.exec_outputs_per_row.size()) != payload.batch_size) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: exec_outputs_per_row size="
            + std::to_string(forward_outputs.exec_outputs_per_row.size())
            + " does not match payload.batch_size=" + std::to_string(payload.batch_size));
    }
    if (!payload.execution_active.empty()
        && static_cast<int>(payload.execution_active.size()) != payload.batch_size) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: execution_active size="
            + std::to_string(payload.execution_active.size())
            + " does not match payload.batch_size=" + std::to_string(payload.batch_size));
    }
    if (!payload.transition_target_mask.empty()
        && static_cast<int>(payload.transition_target_mask.size()) != payload.batch_size) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: transition_target_mask size="
            + std::to_string(payload.transition_target_mask.size())
            + " does not match payload.batch_size=" + std::to_string(payload.batch_size));
    }
    if (!payload.execution_gate_targets.empty()
        && static_cast<int>(payload.execution_gate_targets.size()) != payload.batch_size) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: execution_gate_targets size does not match batch_size");
    }

    const bool need_transition_targets =
        (model_hp.structured_ce_enabled && model_hp.execution_block_structured_ce_weight > 0.0f)
        || (model_hp.execution_block_stop_ce_weight > 0.0f);

    if (need_transition_targets) {
        if (payload.transition_targets.empty()) {
            throw std::runtime_error(
                "addExecutionAuxiliaryLoss: transition-target losses are enabled but payload.transition_targets is empty");
        }
        if (static_cast<int>(payload.transition_targets.size()) != payload.batch_size) {
            throw std::runtime_error(
                "addExecutionAuxiliaryLoss: transition_targets size="
                + std::to_string(payload.transition_targets.size())
                + " does not match payload.batch_size=" + std::to_string(payload.batch_size));
        }
    }

    loss_state.exec_op_ce_added = false;
    loss_state.exec_arg_ce_added = false;
    loss_state.exec_write_ce_added = false;
    loss_state.exec_execute_ce_added = false;
    loss_state.exec_stop_ce_added = false;

    enum class ExecLossFlag {
        Op,
        Arg,
        Write,
        Execute,
        Stop
    };

    Tensor exec_loss_sum;
    bool have_exec_loss_sum = false;
    Tensor entropy_loss_sum;
    bool have_entropy_loss_sum = false;
    int ce_tensor_count = 0;
    float ce_scalar_sum = 0.0f;
    int monitored_entropy_rows = 0;
    const bool have_step_mask = !payload.transition_target_mask.empty();
    const float ce_weight = model_hp.execution_block_structured_ce_weight;
    const float execute_ce_weight = model_hp.execution_block_execute_ce_weight;
    const float stop_ce_weight = model_hp.execution_block_stop_ce_weight;
    const int num_slots = execution_hp.num_slots;
    const int num_scratch_slots = execution_hp.num_scratch_slots;
    const int num_ops = execution_hp.num_ops;

    if (execution_hp.div_invalid_penalty_weight > 0.0f && num_ops <= kDivOpIdx) {
        throw std::runtime_error(
            "addExecutionAuxiliaryLoss: division penalty requested but num_ops="
            + std::to_string(num_ops) + " does not contain division op index "
            + std::to_string(kDivOpIdx));
    }

    auto addExecLossTerm = [&](Tensor&& contribution, const char* name, int row, int step, ExecLossFlag flag) {
        if (!contribution.data || !contribution.grad_fn) {
            throw std::runtime_error(
                std::string("addExecutionAuxiliaryLoss: execution loss tensor '") + name
                + "' has no data/grad_fn at row=" + std::to_string(row)
                + " step=" + std::to_string(step));
        }
        if (contribution.numel() != 1) {
            throw std::runtime_error(
                std::string("addExecutionAuxiliaryLoss: execution loss tensor '") + name
                + "' is not scalar at row=" + std::to_string(row)
                + " step=" + std::to_string(step)
                + " numel=" + std::to_string(contribution.numel()));
        }
        if (!have_exec_loss_sum) {
            exec_loss_sum = std::move(contribution);
            have_exec_loss_sum = true;
        } else {
            exec_loss_sum = autograd::add(exec_loss_sum, contribution, ctx.stream);
        }
        switch (flag) {
            case ExecLossFlag::Op:
                loss_state.exec_op_ce_added = true;
                break;
            case ExecLossFlag::Arg:
                loss_state.exec_arg_ce_added = true;
                break;
            case ExecLossFlag::Write:
                loss_state.exec_write_ce_added = true;
                break;
            case ExecLossFlag::Execute:
                loss_state.exec_execute_ce_added = true;
                break;
            case ExecLossFlag::Stop:
                loss_state.exec_stop_ce_added = true;
                break;
        }
        summary.scalar_loss_terms++;
    };

    for (int b = 0; b < payload.batch_size; ++b) {
        const auto gate_target = payload.execution_gate_targets.empty()
            ? Execution::ExecutionGateTarget::UNSUPERVISED
            : payload.execution_gate_targets[b];
        if (gate_target != Execution::ExecutionGateTarget::UNSUPERVISED
            && execute_ce_weight > 0.0f) {
            auto& gate_logits = forward_outputs.exec_outputs_per_row[b].gate.logits;
            requireTensor(gate_logits, "execution_gate_logits", b, -1);
            const int target = gate_target == Execution::ExecutionGateTarget::EXECUTE ? 1 : 0;
            Tensor gate_ce = autograd::cross_entropy_logits(gate_logits, target, ctx.stream);
            float gate_ce_value = 0.0f;
            readScalarFromDevice(gate_ce, gate_ce_value, ctx.stream, "execution_gate_ce");
            ce_scalar_sum += gate_ce_value;
            ce_tensor_count++;
            Tensor scaled_gate = autograd::scale_scalar(gate_ce, execute_ce_weight, ctx.stream);
            addExecLossTerm(std::move(scaled_gate), "execution_gate_ce", b, -1, ExecLossFlag::Execute);
            summary.execute_targets++;
        }

        if (!payload.execution_active.empty() && !payload.execution_active[b]) {
            continue;
        }

        auto& row_steps = forward_outputs.exec_outputs_per_row[b].steps;
        Tensor row_entropy_sum;
        bool have_row_entropy_sum = false;
        int row_entropy_terms = 0;
        const std::vector<Execution::TransitionInvocation>* target_row = nullptr;
        if (!payload.transition_targets.empty()) {
            target_row = &payload.transition_targets[b];
            int real_transition_targets = 0;
            if (have_step_mask) {
                bool saw_padding = false;
                for (uint8_t mask_value : payload.transition_target_mask[b]) {
                    if (mask_value == 0) {
                        saw_padding = true;
                    } else {
                        if (saw_padding) {
                            throw std::runtime_error(
                                "addExecutionAuxiliaryLoss: transition target mask must be a contiguous prefix");
                        }
                        ++real_transition_targets;
                    }
                }
            } else {
                real_transition_targets = static_cast<int>(target_row->size());
            }
            if (static_cast<int>(row_steps.size()) != real_transition_targets) {
                throw std::runtime_error(
                    "addExecutionAuxiliaryLoss: row " + std::to_string(b)
                    + " produced " + std::to_string(row_steps.size())
                    + " execution outputs but transition target mask has "
                    + std::to_string(real_transition_targets) + " real steps");
            }
        }

        for (int k = 0; k < static_cast<int>(row_steps.size()); ++k) {
            if (have_step_mask
                && b < static_cast<int>(payload.transition_target_mask.size())
                && k < static_cast<int>(payload.transition_target_mask[b].size())
                && payload.transition_target_mask[b][k] == 0) {
                continue;
            }

            auto& sout = row_steps[k];
            summary.active_steps++;

            if (stop_ce_weight > 0.0f) {
                requireTensor(sout.stop_logits_tensor, "stop_logits_tensor", b, k);
                const int stop_target =
                    (k + 1 == static_cast<int>(row_steps.size())) ? 1 : 0;
                Tensor stop_ce = autograd::cross_entropy_logits(
                    sout.stop_logits_tensor, stop_target, ctx.stream);
                float stop_ce_value = 0.0f;
                readScalarFromDevice(stop_ce, stop_ce_value, ctx.stream, "stop_control_ce");
                ce_scalar_sum += stop_ce_value;
                ce_tensor_count++;
                Tensor scaled_stop = autograd::scale_scalar(stop_ce, stop_ce_weight, ctx.stream);
                addExecLossTerm(std::move(scaled_stop), "stop_control_ce", b, k, ExecLossFlag::Stop);
                summary.stop_targets++;
            }

            Tensor p_arg1_live;
            Tensor p_arg2_live;
            Tensor p_op_live;
            Tensor p_write_live;
            bool have_p_arg1_live = false;
            bool have_p_arg2_live = false;
            bool have_p_op_live = false;
            bool have_p_write_live = false;

            auto ensurePArg1Live = [&]() {
                if (!have_p_arg1_live) {
                    requirePositiveTemperature(sout.selection_temperature, b, k);
                    requireTensor(sout.arg1_logits_tensor, "arg1_logits_tensor", b, k);
                    p_arg1_live = autograd::softmax(sout.arg1_logits_tensor, sout.selection_temperature, ctx.stream);
                    have_p_arg1_live = true;
                }
            };

            auto ensurePArg2Live = [&]() {
                if (!have_p_arg2_live) {
                    requirePositiveTemperature(sout.selection_temperature, b, k);
                    requireTensor(sout.arg2_logits_tensor, "arg2_logits_tensor", b, k);
                    p_arg2_live = autograd::softmax(sout.arg2_logits_tensor, sout.selection_temperature, ctx.stream);
                    have_p_arg2_live = true;
                }
            };

            auto ensurePOpLive = [&]() {
                if (!have_p_op_live) {
                    requirePositiveTemperature(sout.selection_temperature, b, k);
                    requireTensor(sout.op_logits_tensor, "op_logits_tensor", b, k);
                    p_op_live = autograd::softmax(sout.op_logits_tensor, sout.selection_temperature, ctx.stream);
                    have_p_op_live = true;
                }
            };

            auto ensurePWriteLive = [&]() {
                if (!have_p_write_live) {
                    requirePositiveTemperature(sout.selection_temperature, b, k);
                    requireTensor(sout.write_logits_tensor, "write_logits_tensor", b, k);
                    p_write_live = autograd::softmax(sout.write_logits_tensor, sout.selection_temperature, ctx.stream);
                    have_p_write_live = true;
                }
            };

            auto addRowEntropyTerm = [&](Tensor& probs, const char* name) {
                auto& staging = forward_outputs.appendNormalizedEntropyBackwardStaging(
                    probs.shape,
                    ctx.stream);
                Tensor entropy = autograd::normalized_entropy(
                    probs,
                    staging.saved_probs,
                    staging.grad_probs,
                    ctx.stream);
                if (!entropy.data || !entropy.grad_fn) {
                    throw std::runtime_error(
                        std::string("addExecutionAuxiliaryLoss: entropy tensor '") + name
                        + "' has no data/grad_fn at row=" + std::to_string(b)
                        + " step=" + std::to_string(k));
                }
                if (!have_row_entropy_sum) {
                    row_entropy_sum = std::move(entropy);
                    have_row_entropy_sum = true;
                } else {
                    row_entropy_sum = autograd::add(row_entropy_sum, entropy, ctx.stream);
                }
                row_entropy_terms++;
            };

            if (target_row && model_hp.structured_ce_enabled && ce_weight > 0.0f) {
                const auto& target = (*target_row)[k];
                if (payload.compiled_slot_bindings.empty() ||
                    b >= static_cast<int>(payload.compiled_slot_bindings.size())) {
                    throw std::runtime_error(
                        "addExecutionAuxiliaryLoss: row " + std::to_string(b) +
                        " has transition SlotId targets but no compiled slot bindings");
                }
                if (payload.compiled_transition_bindings.empty() ||
                    b >= static_cast<int>(
                        payload.compiled_transition_bindings.size())) {
                    throw std::runtime_error(
                        "addExecutionAuxiliaryLoss: row " + std::to_string(b) +
                        " has a TransitionId target but no compiled "
                        "transition bindings");
                }
                if (target.arguments.size() != 2 ||
                    target.results.size() != 1) {
                    throw std::runtime_error(
                        "addExecutionAuxiliaryLoss: current execution heads "
                        "require two transition arguments and one result");
                }
                const auto& realized = sout.record.invocation;
                if (!realized.transition_id.valid() ||
                    realized.arguments.size() != 2 ||
                    realized.results.size() != 1) {
                    throw std::runtime_error(
                        "addExecutionAuxiliaryLoss: realized transition trace "
                        "does not satisfy the current executor contract");
                }
                const bool transition_match =
                    realized.transition_id == target.transition_id;
                const bool arguments_match =
                    realized.arguments == target.arguments;
                const bool results_match =
                    realized.results == target.results;
                const bool exact_target_match =
                    transition_match && arguments_match && results_match;

                char target_probe[768];
                snprintf(
                    target_probe,
                    sizeof(target_probe),
                    "[TEMP_TRANSITION_TARGET_MEASUREMENT] row=%d step=%d "
                    "exact_target_match=%d transition_match=%d "
                    "arguments_match=%d results_match=%d "
                    "expected={transition_id=%llu,args=[%llu,%llu],results=[%llu]} "
                    "realized={transition_id=%llu,args=[%llu,%llu],results=[%llu]}",
                    b,
                    k,
                    exact_target_match ? 1 : 0,
                    transition_match ? 1 : 0,
                    arguments_match ? 1 : 0,
                    results_match ? 1 : 0,
                    static_cast<unsigned long long>(
                        target.transition_id.serialized()),
                    static_cast<unsigned long long>(
                        target.arguments[0].serialized()),
                    static_cast<unsigned long long>(
                        target.arguments[1].serialized()),
                    static_cast<unsigned long long>(
                        target.results[0].serialized()),
                    static_cast<unsigned long long>(
                        realized.transition_id.serialized()),
                    static_cast<unsigned long long>(
                        realized.arguments[0].serialized()),
                    static_cast<unsigned long long>(
                        realized.arguments[1].serialized()),
                    static_cast<unsigned long long>(
                        realized.results[0].serialized()));
                GRIM::Logging::EmitModuleInfo(
                    GRIM::Logging::ModuleId::ExecutionBlock,
                    target_probe);

                const auto& slot_bindings = payload.compiled_slot_bindings[b];
                const auto& transition_bindings =
                    payload.compiled_transition_bindings[b];
                const int arg1_slot_index = Execution::requireSlotIndex(
                    slot_bindings,
                    target.arguments[0],
                    "addExecutionAuxiliaryLoss transition argument 0").dense();
                const int arg2_slot_index = Execution::requireSlotIndex(
                    slot_bindings,
                    target.arguments[1],
                    "addExecutionAuxiliaryLoss transition argument 1").dense();
                const int write_slot_index = Execution::requireSlotIndex(
                    slot_bindings,
                    target.results[0],
                    "addExecutionAuxiliaryLoss transition result 0").dense();
                const int op_target = requireTransitionTarget(
                    Execution::requireTransitionIndex(
                        transition_bindings,
                        target.transition_id,
                        "addExecutionAuxiliaryLoss transition target").dense(),
                    num_ops,
                    b,
                    k,
                    "transition_index");
                const int arg1_target = requireValueSlotTarget(arg1_slot_index, num_scratch_slots, num_slots, b, k, "transition argument 0");
                const int arg2_target = requireValueSlotTarget(arg2_slot_index, num_scratch_slots, num_slots, b, k, "transition argument 1");
                const int write_target = requireWriteTarget(write_slot_index, num_scratch_slots, num_slots, b, k);
                requireInitializedValueTarget(
                    sout.state_before_valid,
                    arg1_slot_index,
                    num_slots,
                    b,
                    k,
                    "arg1",
                    ctx.stream);
                requireInitializedValueTarget(
                    sout.state_before_valid,
                    arg2_slot_index,
                    num_slots,
                    b,
                    k,
                    "arg2",
                    ctx.stream);
                requirePositiveTemperature(sout.selection_temperature, b, k);
                const float inverse_selection_temperature = 1.0f / sout.selection_temperature;

                auto accumulateCe = [&](Tensor& logits, int target, const char* name, ExecLossFlag flag) {
                    // Match the policy sampled by the forward pass. Keeping the
                    // temperature scaling in the graph also supplies the correct
                    // 1/T chain-rule factor to the selector logits.
                    Tensor policy_logits = autograd::mul_scalar(
                        logits,
                        inverse_selection_temperature,
                        ctx.stream);
                    Tensor ce = autograd::cross_entropy_logits(policy_logits, target, ctx.stream);
                    float ce_value = 0.0f;
                    readScalarFromDevice(ce, ce_value, ctx.stream, name);
                    ce_scalar_sum += ce_value;
                    ce_tensor_count++;
                    Tensor scaled = autograd::scale_scalar(ce, ce_weight, ctx.stream);
                    addExecLossTerm(std::move(scaled), name, b, k, flag);
                };

                requireTensor(sout.op_logits_tensor, "op_logits_tensor", b, k);
                requireTensor(sout.arg1_logits_tensor, "arg1_logits_tensor", b, k);
                requireTensor(sout.arg2_logits_tensor, "arg2_logits_tensor", b, k);
                requireTensor(sout.write_logits_tensor, "write_logits_tensor", b, k);

                accumulateCe(sout.op_logits_tensor, op_target, "selection_ce_op", ExecLossFlag::Op);
                accumulateCe(sout.arg1_logits_tensor, arg1_target, "selection_ce_arg1", ExecLossFlag::Arg);
                accumulateCe(sout.arg2_logits_tensor, arg2_target, "selection_ce_arg2", ExecLossFlag::Arg);
                accumulateCe(sout.write_logits_tensor, write_target, "selection_ce_write", ExecLossFlag::Write);
            }

            if (execution_hp.div_invalid_penalty_weight > 0.0f && sout.div_was_clamped) {
                ensurePOpLive();
                Tensor& p_op = p_op_live;
                Tensor penalty = makeDivInvalidPenalty(
                    p_op,
                    sout.div_was_clamped,
                    execution_hp.div_invalid_penalty_weight,
                    num_ops,
                    ctx.stream);
                addExecLossTerm(std::move(penalty), "div_invalid_penalty", b, k, ExecLossFlag::Op);
            }

            if (model_hp.execution_block_entropy_aux_weight > 0.0f) {
                ensurePArg1Live();
                ensurePArg2Live();
                ensurePOpLive();
                ensurePWriteLive();
                addRowEntropyTerm(p_arg1_live, "entropy_arg1");
                addRowEntropyTerm(p_arg2_live, "entropy_arg2");
                addRowEntropyTerm(p_op_live, "entropy_op");
                addRowEntropyTerm(p_write_live, "entropy_write");
            }
        }

        if (model_hp.execution_block_entropy_aux_weight > 0.0f && row_entropy_terms > 0) {
            Tensor row_entropy_loss = autograd::scale_scalar(
                row_entropy_sum,
                -model_hp.execution_block_entropy_aux_weight / static_cast<float>(row_entropy_terms),
                ctx.stream);
            float row_entropy_value = 0.0f;
            readScalarFromDevice(row_entropy_loss, row_entropy_value, ctx.stream, "execution_entropy_loss");
            summary.entropy_monitor += row_entropy_value;
            monitored_entropy_rows++;

            if (!have_entropy_loss_sum) {
                entropy_loss_sum = std::move(row_entropy_loss);
                have_entropy_loss_sum = true;
            } else {
                entropy_loss_sum = autograd::add(entropy_loss_sum, row_entropy_loss, ctx.stream);
            }
        }
    }

    if (ce_tensor_count > 0) {
        summary.structured_ce = ce_scalar_sum / static_cast<float>(ce_tensor_count);
    }

    if (have_exec_loss_sum) {
        if (summary.scalar_loss_terms <= 0) {
            throw std::runtime_error(
                "addExecutionAuxiliaryLoss: have_exec_loss_sum=true but scalar_loss_terms<=0");
        }
        const float norm = 1.0f / static_cast<float>(summary.scalar_loss_terms);
        Tensor normalized_exec_loss = autograd::scale_scalar(exec_loss_sum, norm, ctx.stream);
        loss_state.loss_tensor = autograd::add(loss_state.loss_tensor, normalized_exec_loss, ctx.stream);
    }

    if (have_entropy_loss_sum) {
        if (monitored_entropy_rows <= 0) {
            throw std::runtime_error(
                "addExecutionAuxiliaryLoss: have_entropy_loss_sum=true but monitored_entropy_rows<=0");
        }
        Tensor normalized_entropy_loss = autograd::scale_scalar(
            entropy_loss_sum,
            1.0f / static_cast<float>(monitored_entropy_rows),
            ctx.stream);
        loss_state.loss_tensor = autograd::add(loss_state.loss_tensor, normalized_entropy_loss, ctx.stream);
        summary.entropy_monitor /= static_cast<float>(monitored_entropy_rows);
    }

    return summary;
}

}  // namespace Autograd
}  // namespace GRIM
