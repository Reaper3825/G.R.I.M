//======================================================//
//  Startup/Batching/PlannedBatches.cu
//======================================================//

#include "PlannedBatches.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/Batching/BatchDeviceStorage.hpp"
#include "../../../../Shared/Batching/EpochBatching.hpp"      // buildEpochBatches
#include "../../../../Shared/Batching/PackerPolicy.hpp"
#include "../../../../Shared/Batching/Batching_GPU.hpp"       // buildBatches
#include "../../../../Shared/UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

namespace {

int trainingMtpK(const TrainingContext& ctx) {
    return GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "mtp_enabled")
        ? GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "mtp_k")
        : 0;
}

//======================================================//
// Per-batch payload builder. Reads ALL static run-invariant inputs (cache
// geometry, execution-block sizes, vocab size, token layout) from the
// Phase1-authored TrainingContext config + SequenceData.vocab_size. This is the single
// Phase1 entry point for GRIM::Batching::buildBatchPayload.
//======================================================//
GRIM::Batching::BatchPayload buildPayloadFromAssignmentImpl(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment,
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views,
    int mtp_k)
{
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    const int execution_block_num_slots = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "execution_block_num_slots");
    const int execution_block_num_ops = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "execution_block_num_ops");
    const int execution_block_num_steps = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "execution_block_num_steps");
    const auto number_encoder_hp = GRIM::HyperParameters::numberEncoderConstructionHP(ctx.config);
    const int number_encoder_digit_slots = number_encoder_hp.enabled ? number_encoder_hp.max_digit_slots : 0;
    const int number_encoder_max_abs_pow10 = number_encoder_hp.max_abs_pow10;
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "vocab_size");
    const std::uint32_t actual_vocab_size = ctx.data.vocab_size;
    const auto layout = GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
        actual_vocab_size,
        "PlannedBatches::buildPayloadFromAssignmentImpl");

    if (actual_vocab_size == 0) {
        throw std::runtime_error(
            "FATAL: PlannedBatches SequenceData.vocab_size is zero");
    }
    if (vocab_size != static_cast<int>(actual_vocab_size)) {
        throw std::runtime_error(
            "FATAL: PlannedBatches runtime vocab mismatch: actual_vocab_size=" +
            std::to_string(actual_vocab_size) +
            " model vocab_size=" + std::to_string(vocab_size));
    }
    if (layout.total_vocab() != vocab_size) {
        throw std::runtime_error(
            "FATAL: PlannedBatches token_layout.total_vocab=" +
            std::to_string(layout.total_vocab()) +
            " != model vocab_size=" + std::to_string(vocab_size));
    }

    return GRIM::Batching::buildBatchPayload(
        assignment, views, vocab_size,
        layout,
        static_cast<std::size_t>(fixed_shape.batch_size),
        static_cast<std::size_t>(fixed_shape.max_seq_len),
        execution_block_num_slots,
        execution_block_num_ops,
        execution_block_num_steps,
        mtp_k,
        number_encoder_digit_slots,
        number_encoder_max_abs_pow10);
}

//======================================================//
// Validate that a Phase1-authored payload is complete and ready for Phase2
// indexing. Mirrors the per-batch invariants Phase2 used to assert at
// runtime; surfacing them at startup is the whole point of the contract.
//======================================================//
void validatePlannedPayloadOrThrow(
    const GRIM::Batching::BatchPayload& payload,
    const char* split,
    int batch_idx,
    int expected_mtp_k)
{
    payload.validate("PlannedBatches");

    if (expected_mtp_k < 0) {
        std::ostringstream oss;
        oss << "FATAL: PlannedBatches " << split << " batch " << batch_idx
            << " expected_mtp_k=" << expected_mtp_k
            << " (must be >= 0)";
        throw std::runtime_error(oss.str());
    }
    if (static_cast<int>(payload.mtp_shifted_targets.size()) != expected_mtp_k) {
        std::ostringstream oss;
        oss << "FATAL: PlannedBatches " << split << " batch " << batch_idx
            << " has mtp_shifted_targets.size()=" << payload.mtp_shifted_targets.size()
            << " != configured mtp_k=" << expected_mtp_k;
        throw std::runtime_error(oss.str());
    }
    if (static_cast<int>(payload.mtp_valid_counts.size()) != expected_mtp_k) {
        std::ostringstream oss;
        oss << "FATAL: PlannedBatches " << split << " batch " << batch_idx
            << " has mtp_valid_counts.size()=" << payload.mtp_valid_counts.size()
            << " != configured mtp_k=" << expected_mtp_k;
        throw std::runtime_error(oss.str());
    }

    if (!payload.device_storage) {
        std::ostringstream oss;
        oss << "FATAL: PlannedBatches " << split << " batch " << batch_idx
            << " has no attached BatchDeviceStorage owner";
        throw std::runtime_error(oss.str());
    }

    if (payload.batch_size <= 0) {
        std::ostringstream oss;
        oss << "FATAL: PlannedBatches " << split << " batch " << batch_idx
            << " has batch_size=" << payload.batch_size
            << " — scheduler produced empty batch; fix the upstream filter";
        throw std::runtime_error(oss.str());
    }
}

void log_fn_to(TrainingContext& ctx, const std::string& msg) {
    if (ctx.logging.logger) {
        ctx.logging.logger->log(msg);
    }
}

} // namespace

GRIM::Batching::BatchPayload buildTrainPayload(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment)
{
    return buildPayloadFromAssignmentImpl(
        ctx, assignment, ctx.data.train_views,
    trainingMtpK(ctx));
}

GRIM::Batching::BatchPayload buildValPayload(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment)
{
    return buildPayloadFromAssignmentImpl(
        ctx, assignment, ctx.data.val_views,
        trainingMtpK(ctx));
}

void PlannedBatchesReady(TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error(
            "FATAL: PlannedBatchesReady requires an allocated model — "
            "call ModelAllocated before this step");
    }
    if (ctx.data.vocab_size == 0) {
        throw std::runtime_error(
            "FATAL: PlannedBatchesReady requires LoadTrainingData to author ctx.data.vocab_size");
    }
    (void)GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
        ctx.data.vocab_size,
        "PlannedBatchesReady");
    const int fixed_batch_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "batch_size");
    const int fixed_max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "max_seq_len");
    if (fixed_batch_size <= 0 || fixed_max_seq_len <= 0) {
        std::ostringstream oss;
        oss << "FATAL: PlannedBatchesReady requires configured batching hyperparameters "
            << "(batch_size=" << fixed_batch_size
            << " max_seq_len=" << fixed_max_seq_len
            << ")";
        throw std::runtime_error(oss.str());
    }
    if (ctx.data.train_views.empty()) {
        throw std::runtime_error(
            "FATAL: PlannedBatchesReady requires LoadTrainingData to have loaded "
            "ctx.data.train_views");
    }

    const auto schedule_hp =
        GRIM::HyperParameters::trainingScheduleHP(ctx.config);
    const int num_epochs = schedule_hp.epochs;
    if (num_epochs <= 0) {
        throw std::runtime_error(
            "FATAL: PlannedBatchesReady requires hyperparameters.epochs > 0 (got " +
            std::to_string(num_epochs) + ")");
    }

    auto log = [&](const std::string& msg) { log_fn_to(ctx, msg); };

    //======================================================//
    // Train: build ONE fixed BatchSchedule.
    //
    // We delegate to GRIM::Batching::buildEpochBatches with epoch=0 so the
    // packing policy (greedy, RANDOM ordering, bucket settings) matches the
    // historical per-epoch behaviour exactly. Per the plan's "fixed batch
    // membership" rule, this schedule is authored once and never rebuilt.
    // Normal-mode per-epoch diversity comes from `epoch_batch_order`.
    //======================================================//
    log("[PlannedBatches] Building fixed train schedule (epochs=" +
        std::to_string(num_epochs) + ")");

    ctx.fixed_train_schedule = GRIM::Batching::buildEpochBatches(
        ctx.data.train_seq_lengths,
        static_cast<uint32_t>(fixed_max_seq_len),
        static_cast<uint32_t>(fixed_batch_size),
        /*global_step=*/0,
        /*epoch=*/0,
        /*data_seed=*/ctx.rng.data_seed,
        log);

    const int num_train_batches =
        static_cast<int>(ctx.fixed_train_schedule.batches.size());
    if (num_train_batches <= 0) {
        throw std::runtime_error(
            "FATAL: PlannedBatchesReady produced 0 training batches");
    }

    auto shared_device_storage = GRIM::Batching::createBatchDeviceStorage(
        ctx.config,
        ctx.requireTrainingState("PlannedBatchesReady").stream_ctrl.getPrimaryStream());
    const int mtp_k = trainingMtpK(ctx);

    //======================================================//
    // Train payloads: materialize a host BatchPayload per assignment.
    //
    // Every payload is validated immediately; a contract violation here is
    // caught at startup, not in the training hot loop (Rule 20).
    //======================================================//
    ctx.train_payloads.clear();
    ctx.train_payloads.reserve(num_train_batches);
    for (int i = 0; i < num_train_batches; ++i) {
        auto payload = buildTrainPayload(ctx, ctx.fixed_train_schedule.batches[i]);
        GRIM::Batching::attachBatchDeviceStorage(
            payload,
            shared_device_storage,
            "PlannedBatchesReady(train)");
        validatePlannedPayloadOrThrow(payload, "train", i, mtp_k);
        ctx.train_payloads.push_back(std::move(payload));
    }
    log("[PlannedBatches] Built " + std::to_string(ctx.train_payloads.size()) +
        " train payloads");

    //======================================================//
    // Validation: build a fixed BatchSchedule + payload vector.
    //
    // Validation never shuffles: default PackerPolicy preserves source order.
    // val_payloads are iterated in order.
    //======================================================//
    if (!ctx.data.val_views.empty()) {
        GRIM::Batching::PackerPolicy val_policy;

        ctx.fixed_val_schedule = GRIM::Batching::buildBatches(
            ctx.data.val_seq_lengths,
            static_cast<uint32_t>(fixed_max_seq_len),
            static_cast<uint32_t>(fixed_batch_size),
            val_policy);

        const int num_val_batches =
            static_cast<int>(ctx.fixed_val_schedule.batches.size());
        log("[PlannedBatches] Built " + std::to_string(num_val_batches) +
            " val batches");
        if (ctx.fixed_val_schedule.discarded_tail_sequences > 0) {
            log("[PlannedBatches] Discarded " +
                std::to_string(ctx.fixed_val_schedule.discarded_tail_sequences) +
                " validation sequence(s) that did not fill a full fixed batch");
        }

        ctx.val_payloads.clear();
        ctx.val_payloads.reserve(num_val_batches);
        for (int i = 0; i < num_val_batches; ++i) {
            auto payload = buildValPayload(ctx, ctx.fixed_val_schedule.batches[i]);
            GRIM::Batching::attachBatchDeviceStorage(
                payload,
                shared_device_storage,
                "PlannedBatchesReady(val)");
            validatePlannedPayloadOrThrow(payload, "val", i, mtp_k);
            ctx.val_payloads.push_back(std::move(payload));
        }
        log("[PlannedBatches] Built " + std::to_string(ctx.val_payloads.size()) +
            " val payloads");
    } else {
        ctx.fixed_val_schedule = GRIM::Batching::BatchSchedule{};
        ctx.val_payloads.clear();
        log("[PlannedBatches] No validation data configured — skipping val schedule");
    }

    //======================================================//
    // Per-epoch executable order.
    //
    // Normal training: deterministic permutations of [0, num_train_batches).
    // Single-batch overfit: diagnostic mode authored here as repeated index 0.
    // Phase2 treats this vector as the complete executable step plan and does
    // not branch on the diagnostic hyperparameter.
    //======================================================//
    ctx.epoch_batch_order.assign(num_epochs, std::vector<int>{});
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto& order = ctx.epoch_batch_order[epoch];
        if (schedule_hp.single_batch_overfit_enabled) {
            order.assign(schedule_hp.single_batch_overfit_max_steps, 0);
        } else {
            order.resize(num_train_batches);
            std::iota(order.begin(), order.end(), 0);

            const bool shuffle_this_epoch =
                schedule_hp.shuffle_train_enabled &&
                (schedule_hp.shuffle_train_epochs == 0 ||
                 epoch < schedule_hp.shuffle_train_epochs);

            if (shuffle_this_epoch) {
                std::mt19937_64 epoch_rng(
                    ctx.rng.data_seed + static_cast<uint64_t>(epoch));
                std::shuffle(order.begin(), order.end(), epoch_rng);
            }
        }
    }
    if (schedule_hp.single_batch_overfit_enabled) {
        log("[PlannedBatches] Authored single-batch overfit diagnostic order for " +
            std::to_string(num_epochs) + " epochs (" +
            std::to_string(schedule_hp.single_batch_overfit_max_steps) +
            " repeated steps/epoch)");
    } else {
        log("[PlannedBatches] Authored epoch_batch_order for " +
            std::to_string(num_epochs) + " epochs (" +
            std::to_string(num_train_batches) + " batches/epoch)");
    }
}

} // namespace GRIMText::Training
