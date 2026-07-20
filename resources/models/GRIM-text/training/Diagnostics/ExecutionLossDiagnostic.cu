//======================================================//
//  ExecutionLossDiagnostic.cu
//  Execution-objective decomposition diagnostic
//======================================================//

#include "ExecutionLossDiagnostic.hpp"

#include "../Autograd/AutogradTraining.hpp"
#include "../Phases/Phase1_Startup.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/Telemetry/TelemetryLattice_GPU.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM::Diagnostics {

namespace {

constexpr int kDivisionOpIndex = 3;
constexpr float kProbabilityFloor = 1e-10f;

struct HeadAccumulator {
    double raw_ce_sum = 0.0;
    double weighted_sum = 0.0;
    int target_count = 0;
    int correct_count = 0;

    float rawMean() const {
        return target_count > 0
            ? static_cast<float>(raw_ce_sum / static_cast<double>(target_count))
            : 0.0f;
    }

    float accuracy() const {
        return target_count > 0
            ? static_cast<float>(correct_count) / static_cast<float>(target_count)
            : 0.0f;
    }

    float mixedContribution(int scalar_loss_terms) const {
        return scalar_loss_terms > 0
            ? static_cast<float>(weighted_sum / static_cast<double>(scalar_loss_terms))
            : 0.0f;
    }
};

struct ClassificationObservation {
    float cross_entropy = 0.0f;
    int predicted_class = -1;
};

std::string formatScalar(float value, int precision = 6) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
        if (value == 0.0f) value = 0.0f;
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0.0f ? "inf" : "-inf");
    }
    return oss.str();
}

std::vector<float> copyTensorToHost(
    const GRIM::Tensor& tensor,
    const char* label,
    int row,
    int step)
{
    if (!tensor.data) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: ") + label +
            " is NULL at row=" + std::to_string(row) +
            " step=" + std::to_string(step));
    }
    const auto count = tensor.numel();
    if (count <= 0) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: ") + label +
            " is empty at row=" + std::to_string(row) +
            " step=" + std::to_string(step));
    }

    std::vector<float> host(static_cast<std::size_t>(count));
    const cudaError_t copy_error = cudaMemcpy(
        host.data(),
        tensor.data,
        static_cast<std::size_t>(count) * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (copy_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: cudaMemcpy failed for ") + label +
            " at row=" + std::to_string(row) +
            " step=" + std::to_string(step) +
            " error=" + cudaGetErrorString(copy_error));
    }
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (!std::isfinite(host[i])) {
            throw std::runtime_error(
                std::string("runExecutionLossDiagnostic: non-finite ") + label +
                "[" + std::to_string(i) + "]=" + std::to_string(host[i]) +
                " at row=" + std::to_string(row) +
                " step=" + std::to_string(step));
        }
    }
    return host;
}

ClassificationObservation observeClassification(
    const GRIM::Tensor& logits,
    int target,
    const char* label,
    int row,
    int step,
    float temperature = 1.0f)
{
    const std::vector<float> values = copyTensorToHost(logits, label, row, step);
    if (target < 0 || target >= static_cast<int>(values.size())) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: target ") + std::to_string(target) +
            " is outside [0," + std::to_string(values.size()) + ") for " + label +
            " at row=" + std::to_string(row) +
            " step=" + std::to_string(step));
    }
    if (!std::isfinite(temperature) || temperature <= 0.0f) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: invalid temperature ") +
            std::to_string(temperature) + " for " + label +
            " at row=" + std::to_string(row) +
            " step=" + std::to_string(step));
    }

    const double inverse_temperature = 1.0 / static_cast<double>(temperature);
    const auto max_it = std::max_element(values.begin(), values.end());
    const double max_logit = static_cast<double>(*max_it) * inverse_temperature;
    double exp_sum = 0.0;
    for (float value : values) {
        exp_sum += std::exp(static_cast<double>(value) * inverse_temperature - max_logit);
    }
    if (!std::isfinite(exp_sum) || exp_sum <= 0.0) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: invalid logsumexp for ") + label +
            " at row=" + std::to_string(row) +
            " step=" + std::to_string(step));
    }

    ClassificationObservation observation;
    observation.cross_entropy = static_cast<float>(
        max_logit + std::log(exp_sum) -
        static_cast<double>(values[static_cast<std::size_t>(target)]) * inverse_temperature);
    observation.predicted_class = static_cast<int>(std::distance(values.begin(), max_it));
    return observation;
}

float normalizedEntropy(
    const GRIM::Tensor& probabilities,
    const char* label,
    int row,
    int step)
{
    const std::vector<float> values = copyTensorToHost(probabilities, label, row, step);
    if (values.size() <= 1) return 0.0f;

    double entropy = 0.0;
    for (float probability : values) {
        if (probability < 0.0f || probability > 1.0f + 1e-4f) {
            throw std::runtime_error(
                std::string("runExecutionLossDiagnostic: invalid probability in ") + label +
                " at row=" + std::to_string(row) +
                " step=" + std::to_string(step) +
                " value=" + std::to_string(probability));
        }
        const double p = std::max(static_cast<double>(probability),
                                  static_cast<double>(kProbabilityFloor));
        entropy -= p * std::log(p);
    }
    return static_cast<float>(entropy / std::log(static_cast<double>(values.size())));
}

float targetProbability(
    const GRIM::Tensor& probabilities,
    int target,
    const char* label,
    int row,
    int step)
{
    const std::vector<float> values = copyTensorToHost(probabilities, label, row, step);
    if (target < 0 || target >= static_cast<int>(values.size())) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: probability target outside range for ") + label);
    }
    return values[static_cast<std::size_t>(target)];
}

void accumulateClassification(
    HeadAccumulator& accumulator,
    const ClassificationObservation& observation,
    int target,
    float weight)
{
    accumulator.raw_ce_sum += observation.cross_entropy;
    accumulator.weighted_sum += static_cast<double>(weight) * observation.cross_entropy;
    accumulator.target_count++;
    if (observation.predicted_class == target) accumulator.correct_count++;
}

void requireTelemetryCapacityOrThrow(GRIMText::Training::TrainingContext& ctx) {
    if (!ctx.telemetry.enabled) return;
    if (!ctx.telemetry.lattice) {
        throw std::runtime_error(
            "runExecutionLossDiagnostic: telemetry is enabled but lattice is NULL");
    }
    constexpr int required_streams =
        static_cast<int>(GRIM::Telemetry::MetricStream::EXEC_LOSS_SCALAR_TERM_COUNT) + 1;
    if (ctx.telemetry.lattice->config().num_streams < required_streams) {
        throw std::runtime_error(
            "runExecutionLossDiagnostic: telemetry lattice has " +
            std::to_string(ctx.telemetry.lattice->config().num_streams) +
            " streams but execution-loss diagnostics require " +
            std::to_string(required_streams));
    }
}

void writeTelemetry(
    GRIMText::Training::TrainingContext& ctx,
    GRIM::Telemetry::MetricStream stream,
    float value)
{
    if (!ctx.telemetry.enabled) return;
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string("runExecutionLossDiagnostic: refusing non-finite telemetry stream ") +
            GRIM::Telemetry::getMetricStreamName(stream));
    }
    ctx.telemetry.last_obs[static_cast<int>(stream)] = value;
}

} // namespace

void runExecutionLossDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Forward::ModelForwardOutputs& forward_outputs,
    const GRIM::Autograd::LossResult& loss_result,
    int batch_idx)
{
    requireTelemetryCapacityOrThrow(ctx);

    const auto model_hp = GRIM::HyperParameters::modelHP(ctx.config);
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(ctx.config);

    HeadAccumulator gate;
    HeadAccumulator stop;
    HeadAccumulator op;
    HeadAccumulator arg1;
    HeadAccumulator arg2;
    HeadAccumulator write;

    double div_penalty_pre_norm = 0.0;
    double entropy_row_contribution_sum = 0.0;
    int entropy_row_count = 0;
    int active_steps = 0;
    int teacher_forced_steps = 0;
    int scalar_loss_terms = 0;
    int reinforce_term_count = 0;

    if (model_hp.execution_block_enabled) {
        if (static_cast<int>(forward_outputs.exec_outputs_per_row.size()) != payload.batch_size) {
            throw std::runtime_error(
                "runExecutionLossDiagnostic: execution output row count " +
                std::to_string(forward_outputs.exec_outputs_per_row.size()) +
                " != payload.batch_size " + std::to_string(payload.batch_size));
        }
        if (!payload.execution_active.empty() &&
            static_cast<int>(payload.execution_active.size()) != payload.batch_size) {
            throw std::runtime_error(
                "runExecutionLossDiagnostic: execution_active size does not match batch_size");
        }
        if (!payload.execution_gate_targets.empty() &&
            static_cast<int>(payload.execution_gate_targets.size()) != payload.batch_size) {
            throw std::runtime_error(
                "runExecutionLossDiagnostic: execution_gate_targets size does not match batch_size");
        }
        if (!payload.teacher_step_mask.empty() &&
            static_cast<int>(payload.teacher_step_mask.size()) != payload.batch_size) {
            throw std::runtime_error(
                "runExecutionLossDiagnostic: teacher_step_mask size does not match batch_size");
        }

        const float structured_weight = model_hp.execution_block_structured_ce_weight;
        const float execute_weight = model_hp.execution_block_execute_ce_weight;
        const float stop_weight = model_hp.execution_block_stop_ce_weight;
        const float entropy_weight = model_hp.execution_block_entropy_aux_weight;
        const float div_weight = execution_hp.div_invalid_penalty_weight;
        const bool need_teacher_targets =
            (model_hp.structured_ce_enabled && structured_weight > 0.0f) ||
            stop_weight > 0.0f ||
            execution_hp.arg_reinforce_weight > 0.0f;
        if (need_teacher_targets &&
            (payload.teacher_steps.empty() ||
             static_cast<int>(payload.teacher_steps.size()) != payload.batch_size)) {
            throw std::runtime_error(
                "runExecutionLossDiagnostic: teacher-dependent losses require one teacher row per batch row");
        }

        for (int row = 0; row < payload.batch_size; ++row) {
            const auto& row_output = forward_outputs.exec_outputs_per_row[static_cast<std::size_t>(row)];
            const auto gate_target = payload.execution_gate_targets.empty()
                ? GRIM::Execution::ExecutionGateTarget::UNSUPERVISED
                : payload.execution_gate_targets[static_cast<std::size_t>(row)];
            if (gate_target != GRIM::Execution::ExecutionGateTarget::UNSUPERVISED &&
                execute_weight > 0.0f) {
                const int target = gate_target == GRIM::Execution::ExecutionGateTarget::EXECUTE ? 1 : 0;
                const auto observation = observeClassification(
                    row_output.gate.logits, target, "execution_gate_logits", row, -1);
                accumulateClassification(gate, observation, target, execute_weight);
                scalar_loss_terms++;
            }

            const bool row_active = payload.execution_active.empty()
                ? true
                : payload.execution_active[static_cast<std::size_t>(row)];
            if (!row_active) continue;

            const std::vector<GRIM::Batching::TeacherStep>* teacher_row = nullptr;
            if (!payload.teacher_steps.empty()) {
                teacher_row = &payload.teacher_steps[static_cast<std::size_t>(row)];
                int real_teacher_steps = 0;
                if (!payload.teacher_step_mask.empty()) {
                    bool saw_padding = false;
                    for (uint8_t mask_value :
                         payload.teacher_step_mask[static_cast<std::size_t>(row)]) {
                        if (mask_value == 0) {
                            saw_padding = true;
                        } else {
                            if (saw_padding) {
                                throw std::runtime_error(
                                    "runExecutionLossDiagnostic: teacher step mask must be a contiguous prefix");
                            }
                            ++real_teacher_steps;
                        }
                    }
                } else {
                    real_teacher_steps = static_cast<int>(teacher_row->size());
                }
                if (static_cast<int>(row_output.steps.size()) != real_teacher_steps) {
                    throw std::runtime_error(
                        "runExecutionLossDiagnostic: row=" + std::to_string(row) +
                        " output step count=" + std::to_string(row_output.steps.size()) +
                        " real teacher step count=" + std::to_string(real_teacher_steps));
                }
            }

            double row_entropy_sum = 0.0;
            int row_entropy_terms = 0;
            for (int step_index = 0;
                 step_index < static_cast<int>(row_output.steps.size());
                 ++step_index) {
                if (!payload.teacher_step_mask.empty() &&
                    step_index < static_cast<int>(
                        payload.teacher_step_mask[static_cast<std::size_t>(row)].size()) &&
                    payload.teacher_step_mask[static_cast<std::size_t>(row)]
                                             [static_cast<std::size_t>(step_index)] == 0) {
                    continue;
                }

                const auto& step_output = row_output.steps[static_cast<std::size_t>(step_index)];
                active_steps++;
                if (step_output.teacher_forced_transition) teacher_forced_steps++;

                if (stop_weight > 0.0f) {
                    const int target = step_index + 1 == static_cast<int>(row_output.steps.size()) ? 1 : 0;
                    const auto observation = observeClassification(
                        step_output.stop_logits_tensor, target, "stop_logits", row, step_index);
                    accumulateClassification(stop, observation, target, stop_weight);
                    scalar_loss_terms++;
                }

                if (model_hp.structured_ce_enabled && structured_weight > 0.0f) {
                    const auto& teacher = (*teacher_row)[static_cast<std::size_t>(step_index)];
                    const int op_target = teacher.op_id;
                    const int arg1_target = teacher.arg1_slot - execution_hp.num_scratch_slots;
                    const int arg2_target = teacher.arg2_slot - execution_hp.num_scratch_slots;
                    const int write_target = teacher.write_slot;

                    const auto op_observation = observeClassification(
                        step_output.op_logits_tensor, op_target, "op_logits", row, step_index,
                        step_output.selection_temperature);
                    const auto arg1_observation = observeClassification(
                        step_output.arg1_logits_tensor, arg1_target, "arg1_logits", row, step_index,
                        step_output.selection_temperature);
                    const auto arg2_observation = observeClassification(
                        step_output.arg2_logits_tensor, arg2_target, "arg2_logits", row, step_index,
                        step_output.selection_temperature);
                    const auto write_observation = observeClassification(
                        step_output.write_logits_tensor, write_target, "write_logits", row, step_index,
                        step_output.selection_temperature);

                    accumulateClassification(op, op_observation, op_target, structured_weight);
                    accumulateClassification(arg1, arg1_observation, arg1_target, structured_weight);
                    accumulateClassification(arg2, arg2_observation, arg2_target, structured_weight);
                    accumulateClassification(write, write_observation, write_target, structured_weight);
                    scalar_loss_terms += 4;
                }

                if (div_weight > 0.0f && step_output.div_was_clamped) {
                    if (execution_hp.num_ops <= kDivisionOpIndex) {
                        throw std::runtime_error(
                            "runExecutionLossDiagnostic: division penalty enabled without division op");
                    }
                    div_penalty_pre_norm += static_cast<double>(div_weight) * targetProbability(
                        step_output.p_op,
                        kDivisionOpIndex,
                        "p_op",
                        row,
                        step_index);
                    scalar_loss_terms++;
                }

                if (execution_hp.arg_reinforce_weight > 0.0f &&
                    step_output.record.op_id ==
                        (*teacher_row)[static_cast<std::size_t>(step_index)].op_id) {
                    // The REINFORCE baseline is stateful and is updated while the
                    // loss is assembled. Count its denominator effect exactly;
                    // its value intentionally remains visible in the residual.
                    scalar_loss_terms++;
                    reinforce_term_count++;
                }

                if (entropy_weight > 0.0f) {
                    row_entropy_sum += normalizedEntropy(
                        step_output.p_arg1, "p_arg1", row, step_index);
                    row_entropy_sum += normalizedEntropy(
                        step_output.p_arg2, "p_arg2", row, step_index);
                    row_entropy_sum += normalizedEntropy(
                        step_output.p_op, "p_op", row, step_index);
                    row_entropy_sum += normalizedEntropy(
                        step_output.p_write, "p_write", row, step_index);
                    row_entropy_terms += 4;
                }
            }

            if (entropy_weight > 0.0f && row_entropy_terms > 0) {
                entropy_row_contribution_sum +=
                    -static_cast<double>(entropy_weight) *
                    (row_entropy_sum / static_cast<double>(row_entropy_terms));
                entropy_row_count++;
            }
        }
    }

    const float gate_contribution = gate.mixedContribution(scalar_loss_terms);
    const float stop_contribution = stop.mixedContribution(scalar_loss_terms);
    const float op_contribution = op.mixedContribution(scalar_loss_terms);
    const float arg1_contribution = arg1.mixedContribution(scalar_loss_terms);
    const float arg2_contribution = arg2.mixedContribution(scalar_loss_terms);
    const float write_contribution = write.mixedContribution(scalar_loss_terms);
    const float div_contribution = scalar_loss_terms > 0
        ? static_cast<float>(div_penalty_pre_norm / static_cast<double>(scalar_loss_terms))
        : 0.0f;
    const float entropy_contribution = entropy_row_count > 0
        ? static_cast<float>(entropy_row_contribution_sum /
                             static_cast<double>(entropy_row_count))
        : 0.0f;
    const float reconstructed =
        gate_contribution + stop_contribution + op_contribution +
        arg1_contribution + arg2_contribution + write_contribution +
        div_contribution + entropy_contribution;
    const float residual = loss_result.execution_loss - reconstructed;
    const float teacher_forced_ratio = active_steps > 0
        ? static_cast<float>(teacher_forced_steps) / static_cast<float>(active_steps)
        : 0.0f;

    const float finite_values[] = {
        gate.rawMean(), stop.rawMean(), op.rawMean(), arg1.rawMean(), arg2.rawMean(),
        write.rawMean(), static_cast<float>(div_penalty_pre_norm), entropy_contribution,
        gate_contribution, stop_contribution, op_contribution, arg1_contribution,
        arg2_contribution, write_contribution, div_contribution, reconstructed, residual,
        gate.accuracy(), stop.accuracy(), op.accuracy(), arg1.accuracy(),
        arg2.accuracy(), write.accuracy(), teacher_forced_ratio};
    for (float value : finite_values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "runExecutionLossDiagnostic: non-finite reconstructed execution diagnostic");
        }
    }

    std::ostringstream summary;
    summary << "[ExecutionLossDiagnostic] batch=" << (batch_idx + 1)
            << " aggregate=" << formatScalar(loss_result.execution_loss)
            << " reconstructed=" << formatScalar(reconstructed)
            << " residual=" << formatScalar(residual)
            << " scalar_terms=" << scalar_loss_terms
            << " active_steps=" << active_steps
            << " reinforce_terms=" << reinforce_term_count
            << " teacher_forced_ratio=" << formatScalar(teacher_forced_ratio, 4)
            << " raw_ce={gate:" << formatScalar(gate.rawMean(), 4)
            << ",stop:" << formatScalar(stop.rawMean(), 4)
            << ",op:" << formatScalar(op.rawMean(), 4)
            << ",arg1:" << formatScalar(arg1.rawMean(), 4)
            << ",arg2:" << formatScalar(arg2.rawMean(), 4)
            << ",write:" << formatScalar(write.rawMean(), 4) << "}"
            << " accuracy={gate:" << formatScalar(gate.accuracy(), 4)
            << ",stop:" << formatScalar(stop.accuracy(), 4)
            << ",op:" << formatScalar(op.accuracy(), 4)
            << ",arg1:" << formatScalar(arg1.accuracy(), 4)
            << ",arg2:" << formatScalar(arg2.accuracy(), 4)
            << ",write:" << formatScalar(write.accuracy(), 4) << "}";
    ctx.logging.logger->log(summary.str());

    std::ostringstream equation;
    equation << "L_exec = (sum weighted scalar terms / N_scalar) + mean_row(L_entropy)\n"
             << "  aggregate=" << formatScalar(loss_result.execution_loss)
             << " reconstructed=" << formatScalar(reconstructed)
             << " residual=" << formatScalar(residual)
             << " N_scalar=" << scalar_loss_terms << "\n"
             << "  head       count  raw_mean_CE  accuracy  mixed_contribution\n"
             << "  gate       " << gate.target_count << "  " << formatScalar(gate.rawMean())
             << "  " << formatScalar(gate.accuracy()) << "  " << formatScalar(gate_contribution) << "\n"
             << "  stop       " << stop.target_count << "  " << formatScalar(stop.rawMean())
             << "  " << formatScalar(stop.accuracy()) << "  " << formatScalar(stop_contribution) << "\n"
             << "  op         " << op.target_count << "  " << formatScalar(op.rawMean())
             << "  " << formatScalar(op.accuracy()) << "  " << formatScalar(op_contribution) << "\n"
             << "  arg1       " << arg1.target_count << "  " << formatScalar(arg1.rawMean())
             << "  " << formatScalar(arg1.accuracy()) << "  " << formatScalar(arg1_contribution) << "\n"
             << "  arg2       " << arg2.target_count << "  " << formatScalar(arg2.rawMean())
             << "  " << formatScalar(arg2.accuracy()) << "  " << formatScalar(arg2_contribution) << "\n"
             << "  write      " << write.target_count << "  " << formatScalar(write.rawMean())
             << "  " << formatScalar(write.accuracy()) << "  " << formatScalar(write_contribution) << "\n"
             << "  div pre_norm=" << formatScalar(static_cast<float>(div_penalty_pre_norm))
             << " mixed_contribution=" << formatScalar(div_contribution) << "\n"
             << "  entropy contribution=" << formatScalar(entropy_contribution) << "\n"
             << "  teacher_forced_steps=" << teacher_forced_steps << "/" << active_steps
             << " reinforce_terms_in_residual=" << reinforce_term_count;
    EQ_LOG(
        ctx.logging.tape.get(),
        GRIM::Logging::LogGroup::Loss,
        GRIM::Logging::LogPhase::LOSS_COMPUTATION,
        -1,
        "EXECUTION_LOSS_DECOMPOSITION",
        equation.str().c_str());

    using GRIM::Telemetry::MetricStream;
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_GATE_CE_RAW, gate.rawMean());
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_STOP_CE_RAW, stop.rawMean());
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_OP_CE_RAW, op.rawMean());
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_ARG1_CE_RAW, arg1.rawMean());
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_ARG2_CE_RAW, arg2.rawMean());
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_WRITE_CE_RAW, write.rawMean());
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_DIV_PRE_NORM,
                   static_cast<float>(div_penalty_pre_norm));
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_ENTROPY_CONTRIBUTION, entropy_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_GATE_CONTRIBUTION, gate_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_STOP_CONTRIBUTION, stop_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_OP_CONTRIBUTION, op_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_ARG1_CONTRIBUTION, arg1_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_ARG2_CONTRIBUTION, arg2_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_WRITE_CONTRIBUTION, write_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_DIV_CONTRIBUTION, div_contribution);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_RECONSTRUCTED, reconstructed);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_RESIDUAL, residual);
    writeTelemetry(ctx, MetricStream::EXEC_GATE_ACCURACY, gate.accuracy());
    writeTelemetry(ctx, MetricStream::EXEC_STOP_ACCURACY, stop.accuracy());
    writeTelemetry(ctx, MetricStream::EXEC_OP_ACCURACY, op.accuracy());
    writeTelemetry(ctx, MetricStream::EXEC_ARG1_ACCURACY, arg1.accuracy());
    writeTelemetry(ctx, MetricStream::EXEC_ARG2_ACCURACY, arg2.accuracy());
    writeTelemetry(ctx, MetricStream::EXEC_WRITE_ACCURACY, write.accuracy());
    writeTelemetry(ctx, MetricStream::EXEC_TEACHER_FORCED_RATIO, teacher_forced_ratio);
    writeTelemetry(ctx, MetricStream::EXEC_LOSS_SCALAR_TERM_COUNT,
                   static_cast<float>(scalar_loss_terms));
}

} // namespace GRIM::Diagnostics
