//======================================================//
//  Startup/Batching/PlannedBatches.cu
//======================================================//

#include "PlannedBatches.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/Batching/BatchDeviceStorage.hpp"
#include "../../../../Shared/Batching/EpochBatching.hpp"      // buildEpochBatches
#include "../../../../Shared/Batching/PackerPolicy.hpp"
#include "../../../../Shared/Batching/Batching_GPU.hpp"       // buildBatches
#include "../../../../Shared/AtomInsertion/AtomInsertionData.hpp"
#include "../../../../Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp"
#include "../../../../Shared/UnigramByte/UniByte.hpp"
#include "../../../../Shared/UnigramByte/TokenLayout.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <limits>
#include <vector>

namespace GRIMText::Training {

namespace {

//======================================================//
// Per-batch payload builder. Reads ALL static run-invariant inputs (cache
// geometry, vocab size, token layout) from the
// Phase1-authored TrainingContext config + SequenceData.vocab_size. This is the single
// Phase1 entry point for GRIM::Batching::buildBatchPayload.
//======================================================//
GRIM::Batching::BatchPayload buildPayloadFromAssignmentImpl(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment,
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views)
{
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    const auto model_hp = GRIM::HyperParameters::modelHP(ctx.config);
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
        model_hp.local_atom_retrieval_enabled);
}

std::vector<GRIM::AtomInsertion::AtomInsertionExample>
splitAtomInsertionExample(
    const GRIM::AtomInsertion::AtomInsertionExample& source,
    int max_seq_len,
    const char* caller) {
    const std::size_t max_bytes = static_cast<std::size_t>(max_seq_len - 2);
    if (source.transformerInputSize() <=
        static_cast<std::size_t>(max_seq_len)) {
        return {source};
    }
    if (max_bytes == 0) {
        throw std::runtime_error(
            std::string(caller) + ": atom byte window has zero content capacity");
    }

    auto cut_is_valid = [&](std::size_t gap) {
        if (gap >= source.valid_utf8_gaps.size() ||
            source.valid_utf8_gaps[gap] == 0) {
            return false;
        }
        for (const auto& span : source.spans) {
            if (span.begin_gap < gap && gap < span.end_gap) {
                return false;
            }
        }
        return true;
    };

    std::vector<GRIM::AtomInsertion::AtomInsertionExample> windows;
    std::size_t begin = 0;
    const std::size_t byte_count = source.byteSize();
    while (begin < byte_count) {
        std::size_t end = std::min(begin + max_bytes, byte_count);
        while (end > begin && !cut_is_valid(end)) {
            --end;
        }
        if (end == begin) {
            throw std::runtime_error(
                std::string(caller) +
                ": an atom span or UTF-8 code point exceeds the configured byte window");
        }

        std::vector<GRIM::AtomInsertion::AtomInsertionSpanLabel> window_spans;
        for (const auto& span : source.spans) {
            const bool owns_start = span.begin_gap >= begin &&
                (span.begin_gap < end || end == byte_count);
            if (!owns_start) continue;
            if (span.end_gap > end) {
                throw std::runtime_error(
                    std::string(caller) +
                    ": selected byte window splits an atom span");
            }
            window_spans.push_back(
                GRIM::AtomInsertion::AtomInsertionSpanLabel{
                    span.begin_gap - begin,
                    span.end_gap - begin,
                    span.type});
        }

        std::string annotated_window;
        const std::string plain_window =
            source.plain_text_bytes.substr(begin, end - begin);
        annotated_window.reserve(plain_window.size() + window_spans.size() * 16);
        for (std::size_t gap = 0; gap <= plain_window.size(); ++gap) {
            for (const auto& span : window_spans) {
                if (span.begin_gap < span.end_gap && span.end_gap == gap) {
                    annotated_window += GRIM::Tokenizer::atomTokenText(
                        GRIM::Tokenizer::atomTypeToCloseTokenId(span.type));
                }
            }
            for (const auto& span : window_spans) {
                if (span.begin_gap == gap) {
                    annotated_window += GRIM::Tokenizer::atomTokenText(
                        GRIM::Tokenizer::atomTypeToOpenTokenId(span.type));
                    if (span.end_gap == gap) {
                        annotated_window += GRIM::Tokenizer::atomTokenText(
                            GRIM::Tokenizer::atomTypeToCloseTokenId(span.type));
                    }
                }
            }
            if (gap < plain_window.size()) {
                annotated_window.push_back(plain_window[gap]);
            }
        }

        auto window = GRIM::AtomInsertion::buildAtomInsertionExample(
            annotated_window,
            true,
            caller);
        if (window.plain_text_bytes != plain_window ||
            window.transformerInputSize() >
                static_cast<std::size_t>(max_seq_len)) {
            throw std::runtime_error(
                std::string(caller) +
                ": atom byte-window reconstruction changed source geometry");
        }
        windows.push_back(std::move(window));
        begin = end;
    }
    return windows;
}

std::vector<GRIM::AtomInsertion::AtomInsertionExample>
buildAtomInsertionExamples(
    const TrainingContext& ctx,
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence*>& views,
    int max_seq_len,
    const char* caller) {
    const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(ctx.config);
    GRIM::Tokenizer::UniByte tokenizer(tokenizer_hp);
    const std::uint32_t loaded_vocab_size =
        GRIM::TokenizerArtifacts::loadSharedTokenizerVocabulary(
            tokenizer_hp,
            tokenizer);
    if (loaded_vocab_size != ctx.data.vocab_size) {
        throw std::runtime_error(
            std::string(caller) + ": shared tokenizer vocab size=" +
            std::to_string(loaded_vocab_size) +
            " does not match GRMT vocab size=" +
            std::to_string(ctx.data.vocab_size));
    }

    std::vector<GRIM::AtomInsertion::AtomInsertionExample> examples;
    examples.reserve(views.size());
    for (std::size_t index = 0; index < views.size(); ++index) {
        const auto* sequence = views[index];
        if (!sequence) {
            throw std::runtime_error(
                std::string(caller) + ": NULL GRMT sequence view at index=" +
                std::to_string(index));
        }
        std::size_t begin = 0;
        std::size_t end = sequence->token_ids.size();
        if (begin < end &&
            sequence->token_ids[begin] == GRIM::Tokenizer::BOS_TOKEN_ID) {
            ++begin;
        }
        if (begin < end &&
            sequence->token_ids[end - 1] == GRIM::Tokenizer::EOS_TOKEN_ID) {
            --end;
        }
        std::vector<int> annotated_token_ids(
            sequence->token_ids.begin() + static_cast<std::ptrdiff_t>(begin),
            sequence->token_ids.begin() + static_cast<std::ptrdiff_t>(end));
        const GRIM::Tokenizer::DecodeRequest decode_request(annotated_token_ids);
        const std::string annotated_source = tokenizer.decode(decode_request);
        auto example = GRIM::AtomInsertion::buildAtomInsertionExample(
            annotated_source,
            true,
            caller);
        auto windows = splitAtomInsertionExample(
            example,
            max_seq_len,
            caller);
        for (auto& window : windows) {
            examples.push_back(std::move(window));
        }
    }
    return examples;
}

std::vector<std::uint32_t> atomInsertionSequenceLengths(
    const std::vector<GRIM::AtomInsertion::AtomInsertionExample>& examples) {
    std::vector<std::uint32_t> lengths;
    lengths.reserve(examples.size());
    for (const auto& example : examples) {
        if (example.transformerInputSize() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error(
                "atomInsertionSequenceLengths: sequence length exceeds uint32_t");
        }
        lengths.push_back(
            static_cast<std::uint32_t>(example.transformerInputSize()));
    }
    return lengths;
}

GRIM::Batching::BatchPayload buildAtomPayloadFromAssignment(
    const GRIM::Batching::BatchAssignment& assignment,
    const std::vector<GRIM::AtomInsertion::AtomInsertionExample>& all_examples,
    int vocab_size,
    int max_seq_len,
    const char* caller) {
    std::vector<GRIM::AtomInsertion::AtomInsertionExample> batch_examples;
    batch_examples.reserve(assignment.seq_ids.size());
    for (const std::uint32_t sequence_id : assignment.seq_ids) {
        if (sequence_id >= all_examples.size()) {
            throw std::runtime_error(
                std::string(caller) + ": atom assignment sequence id is out of range");
        }
        batch_examples.push_back(all_examples[sequence_id]);
    }
    auto payload = GRIM::AtomInsertion::buildAtomInsertionBatchPayload(
        batch_examples,
        vocab_size,
        max_seq_len,
        true,
        caller);
    payload.seq_ids = assignment.seq_ids;
    payload.validate(caller);
    return payload;
}

//======================================================//
// Validate that a Phase1-authored payload is complete and ready for Phase2
// indexing. Mirrors the per-batch invariants Phase2 used to assert at
// runtime; surfacing them at startup is the whole point of the contract.
//======================================================//
void validatePlannedPayloadOrThrow(
    const GRIM::Batching::BatchPayload& payload,
    const char* split,
    int batch_idx)
{
    payload.validate("PlannedBatches");

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
    return buildPayloadFromAssignmentImpl(ctx, assignment, ctx.data.train_views);
}

GRIM::Batching::BatchPayload buildValPayload(
    const TrainingContext& ctx,
    const GRIM::Batching::BatchAssignment& assignment)
{
    return buildPayloadFromAssignmentImpl(ctx, assignment, ctx.data.val_views);
}

void PlannedBatchesReady(TrainingContext& ctx) {
    if (ctx.data.vocab_size == 0) {
        throw std::runtime_error(
            "FATAL: PlannedBatchesReady requires LoadTrainingData to author ctx.data.vocab_size");
    }
    (void)GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
        ctx.data.vocab_size,
        "PlannedBatchesReady");
    const int fixed_batch_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "batch_size");
    const int fixed_max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "max_seq_len");
    const int vocab_size =
        GRIM::HyperParameters::snapshotTrainingConfigField<int>(
            ctx.config, "vocab_size");
    const bool atom_insertion_enabled =
        GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(ctx.config)
            .enabled;
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

    std::vector<GRIM::AtomInsertion::AtomInsertionExample> train_atom_examples;
    std::vector<GRIM::AtomInsertion::AtomInsertionExample> val_atom_examples;
    std::vector<std::uint32_t> train_atom_lengths;
    std::vector<std::uint32_t> val_atom_lengths;
    if (atom_insertion_enabled) {
        train_atom_examples = buildAtomInsertionExamples(
            ctx,
            ctx.data.train_views,
            fixed_max_seq_len,
            "PlannedBatchesReady(train atom examples)");
        train_atom_lengths = atomInsertionSequenceLengths(train_atom_examples);
        if (!ctx.data.val_views.empty()) {
            val_atom_examples = buildAtomInsertionExamples(
                ctx,
                ctx.data.val_views,
                fixed_max_seq_len,
                "PlannedBatchesReady(val atom examples)");
            val_atom_lengths = atomInsertionSequenceLengths(val_atom_examples);
        }
        log("[PlannedBatches] Atom insertion mode: authored byte-gap examples");
    }

    //======================================================//
    // Train: build ONE fixed BatchSchedule.
    //
    // Resolve curriculum policy before packing. Generic epoch batch shuffles
    // must not subsequently destroy the authored concept-block order.
    if (!ctx.current_curriculum_metadata)
        throw std::runtime_error("PlannedBatches: missing current curriculum metadata");
    const auto ordering = GRIM::Batching::parseCurriculumOrdering(schedule_hp.batch_strategy);
    std::vector<std::string> train_block_ids;
    train_block_ids.reserve(ctx.data.train_views.size());
    for (const auto* row : ctx.data.train_views) {
        if (!row) throw std::runtime_error("PlannedBatches: null training row");
        train_block_ids.push_back(row->concept_block_id);
    }
    log("[PlannedBatches] Building fixed train schedule: batch_strategy=" + schedule_hp.batch_strategy +
        " randomize_course_order=" + std::to_string(ctx.current_curriculum_metadata->randomize_course_order) +
        " randomize_concept_block_order=" + std::to_string(ctx.current_curriculum_metadata->randomize_concept_block_order));
    ctx.fixed_train_schedule = GRIM::Batching::buildEpochBatches(
        atom_insertion_enabled ? train_atom_lengths : ctx.data.train_seq_lengths,
        static_cast<uint32_t>(fixed_max_seq_len),
        static_cast<uint32_t>(fixed_batch_size),
        /*global_step=*/0,
        /*epoch=*/0,
        /*data_seed=*/ctx.rng.data_seed,
        *ctx.current_curriculum_metadata,
        train_block_ids,
        ordering,
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

    //======================================================//
    // Train payloads: materialize a host BatchPayload per assignment.
    //
    // Every payload is validated immediately; a contract violation here is
    // caught at startup, not in the training hot loop (Rule 20).
    //======================================================//
    ctx.train_payloads.clear();
    ctx.train_payloads.reserve(num_train_batches);
    for (int i = 0; i < num_train_batches; ++i) {
        auto payload = atom_insertion_enabled
            ? buildAtomPayloadFromAssignment(
                ctx.fixed_train_schedule.batches[i],
                train_atom_examples,
                vocab_size,
                fixed_max_seq_len,
                "PlannedBatchesReady(train atom payload)")
            : buildTrainPayload(ctx, ctx.fixed_train_schedule.batches[i]);
        GRIM::Batching::attachBatchDeviceStorage(
            payload,
            shared_device_storage,
            "PlannedBatchesReady(train)");
        validatePlannedPayloadOrThrow(payload, "train", i);
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
            atom_insertion_enabled ? val_atom_lengths : ctx.data.val_seq_lengths,
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
            auto payload = atom_insertion_enabled
                ? buildAtomPayloadFromAssignment(
                    ctx.fixed_val_schedule.batches[i],
                    val_atom_examples,
                    vocab_size,
                    fixed_max_seq_len,
                    "PlannedBatchesReady(val atom payload)")
                : buildValPayload(ctx, ctx.fixed_val_schedule.batches[i]);
            GRIM::Batching::attachBatchDeviceStorage(
                payload,
                shared_device_storage,
                "PlannedBatchesReady(val)");
            validatePlannedPayloadOrThrow(payload, "val", i);
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
    // Normal training: replay the initial concept-block plan without batch shuffling.
    // Single-batch overfit: diagnostic mode authored here as repeated index 0.
    // Phase2 treats this vector as the complete executable step plan and does
    // not branch on the diagnostic hyperparameter.
    //======================================================//
    if (schedule_hp.shuffle_train_enabled)
        log("[PlannedBatches] shuffle_train is superseded by batch_strategy and curriculum flags; "
            "fixed block plan is replayed each epoch");
    ctx.epoch_batch_order.assign(num_epochs, std::vector<int>{});
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto& order = ctx.epoch_batch_order[epoch];
        if (schedule_hp.single_batch_overfit_enabled) {
            order.assign(schedule_hp.single_batch_overfit_max_steps, 0);
        } else {
            order.resize(num_train_batches);
            std::iota(order.begin(), order.end(), 0);

            // Fixed payload membership can only replay the initial block plan.
            // Shuffling packed batches would violate course order and split
            // multi-row blocks. Fresh per-epoch block plans need rematerialization.

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
