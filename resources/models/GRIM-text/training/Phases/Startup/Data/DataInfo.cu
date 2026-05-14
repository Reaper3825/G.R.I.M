#include "DataInfo.hpp"

#include "../SlidingWindow.hpp"
#include "../../ConfigDump.hpp"
#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/UnigramByte/Unigram.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace Internal {

void validatePaths(const PathConfig& paths) {
    if (!fs::exists(paths.vocab_path)) {
        throw std::runtime_error("Vocabulary file does not exist: " + paths.vocab_path);
    }
    if (!fs::exists(paths.data_path)) {
        throw std::runtime_error("Training data file does not exist: " + paths.data_path);
    }
    if (paths.output_model_path.empty()) {
        throw std::runtime_error("Output model path not configured");
    }
    if (paths.checkpoint_dir.empty()) {
        throw std::runtime_error("Checkpoint directory not configured");
    }
    if (paths.log_dir.empty()) {
        throw std::runtime_error("Log directory not configured");
    }

    auto model_parent = fs::path(paths.output_model_path).parent_path();
    if (!model_parent.empty()) {
        fs::create_directories(model_parent);
    }
    fs::create_directories(paths.checkpoint_dir);
    fs::create_directories(paths.log_dir);
}

std::unique_ptr<GRIM::Tokenizer::UniByte> initializeTokenizer(
    const std::string& vocab_path,
    const GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    TrainingLogger& logger)
{
    logger.log("Loading tokenizer configuration...");

    auto tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>(tokenizer_hp);
    if (!tokenizer->load(vocab_path)) {
        throw std::runtime_error("Failed to load vocabulary: " + vocab_path);
    }

    return tokenizer;
}

SequenceData loadTrainingData(
    const std::string& data_path,
    int max_seq_len,
    int min_seq_valid_tokens,
    int sliding_window_stride,
    bool add_bos_token,
    bool add_eos_token,
    const GRIM::Tokenizer::UniByte& tokenizer,
    TrainingLogger& logger)
{
    SequenceData data;

    GRMTDataLoader loader;
    if (!loader.load(data_path)) {
        throw std::runtime_error("Failed to load training data");
    }

    data.vocab_size = loader.vocabSize();
    auto all_sequences = loader.getSequences();

    const int bos_id = GRIM::Tokenizer::BOS_TOKEN_ID;
    const int eos_id = GRIM::Tokenizer::EOS_TOKEN_ID;

    size_t val_size = all_sequences.size() / 10;
    data.train_seqs.assign(all_sequences.begin() + val_size, all_sequences.end());
    data.val_seqs.assign(all_sequences.begin(), all_sequences.begin() + val_size);

    applySlidingWindows(data.train_seqs, "train",
                        max_seq_len, sliding_window_stride, min_seq_valid_tokens,
                        add_bos_token, add_eos_token, bos_id, eos_id, logger);
    applySlidingWindows(data.val_seqs, "val",
                        max_seq_len, sliding_window_stride, min_seq_valid_tokens,
                        add_bos_token, add_eos_token, bos_id, eos_id, logger);

    data.train_views.reserve(data.train_seqs.size());
    data.train_seq_lengths.reserve(data.train_seqs.size());
    for (uint32_t i = 0; i < data.train_seqs.size(); ++i) {
        data.train_views.push_back(&data.train_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.train_seqs[i].token_ids.size());
        data.train_seq_lengths.push_back(len);
    }

    data.val_views.reserve(data.val_seqs.size());
    data.val_seq_lengths.reserve(data.val_seqs.size());
    for (uint32_t i = 0; i < data.val_seqs.size(); ++i) {
        data.val_views.push_back(&data.val_seqs[i]);
        const uint32_t len = static_cast<uint32_t>(data.val_seqs[i].token_ids.size());
        data.val_seq_lengths.push_back(len);
    }

    return data;
}

} // namespace Internal

namespace {

std::uint32_t maxSequenceLen(const std::vector<TrainingSequence>& sequences) {
    std::uint32_t max_len = 0;
    for (const auto& seq : sequences) {
        max_len = std::max(max_len, static_cast<std::uint32_t>(seq.token_ids.size()));
    }
    return max_len;
}

} // namespace

DataInfo summarizeDataInfoOrThrow(
    const DataLoadInputs& inputs,
    const SequenceData& data,
    std::uint32_t tokenizer_vocab_size)
{
    if (data.vocab_size == 0) {
        throw std::runtime_error("FATAL: training data missing vocab_size; regenerate GRMT with tokenizer.totalVocabSize()");
    }
    if (data.train_views.size() != data.train_seqs.size()) {
        throw std::runtime_error("FATAL: train view count does not match train sequence count (views=" +
                                 std::to_string(data.train_views.size()) +
                                 " seqs=" + std::to_string(data.train_seqs.size()) + ")");
    }
    if (data.val_views.size() != data.val_seqs.size()) {
        throw std::runtime_error("FATAL: val view count does not match val sequence count (views=" +
                                 std::to_string(data.val_views.size()) +
                                 " seqs=" + std::to_string(data.val_seqs.size()) + ")");
    }

    DataInfo info;
    info.data_path = inputs.data_path;
    info.vocab_path = inputs.vocab_path;
    info.tokenizer_vocab_size = tokenizer_vocab_size;
    info.actual_vocab_size = data.vocab_size;
    info.train_sequence_count = data.train_seqs.size();
    info.val_sequence_count = data.val_seqs.size();
    info.max_train_seq_len = maxSequenceLen(data.train_seqs);
    info.max_val_seq_len = maxSequenceLen(data.val_seqs);
    return info;
}

void DataInfoReady(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    EmitModuleInfo(ModuleId::Training, "[Phase1] Validating paths...", 0);
    Internal::validatePaths(ctx.config.paths);
    EmitModuleInfo(ModuleId::Training, "[Phase1] ✓ All paths validated", 0);

    const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(ctx.config);
    const auto data_hp = GRIM::HyperParameters::dataLoadingHP(ctx.config);
    ctx.tokenizer = Internal::initializeTokenizer(
        ctx.config.paths.vocab_path, tokenizer_hp, *ctx.logging.logger);

    DataLoadInputs data_inputs;
    data_inputs.data_path = ctx.config.paths.data_path;
    data_inputs.vocab_path = ctx.config.paths.vocab_path;
    data_inputs.max_seq_len = ctx.config.max_seq_len;
    data_inputs.min_seq_valid_tokens = data_hp.min_seq_valid_tokens;
    data_inputs.sliding_window_stride = data_hp.sliding_window_stride;
    data_inputs.add_bos = tokenizer_hp.add_bos;
    data_inputs.add_eos = tokenizer_hp.add_eos;

    ctx.data = Internal::loadTrainingData(
        data_inputs.data_path,
        data_inputs.max_seq_len,
        data_inputs.min_seq_valid_tokens,
        data_inputs.sliding_window_stride,
        data_inputs.add_bos,
        data_inputs.add_eos,
        *ctx.tokenizer,
        *ctx.logging.logger);

    const uint32_t tokenizer_vocab_size = ctx.tokenizer->totalVocabSize();
    ctx.data_info = summarizeDataInfoOrThrow(data_inputs, ctx.data, tokenizer_vocab_size);
    ctx.config.actual_vocab_size = ctx.data_info.actual_vocab_size;
    if (ctx.config.actual_vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        throw std::runtime_error("FATAL: training data vocab_size must include special+byte+atom ranges (>= " +
            std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) + ")");
    }

    GRIM::HyperParameters::DerivationContext hp_ctx;
    hp_ctx.train_sequence_count = static_cast<int>(ctx.data.train_seqs.size());
    hp_ctx.validation_interval = ctx.config.hyperparameters.validation_interval;
    ctx.derived_schedule = GRIM::HyperParameters::computeDerivedSchedule(
        ctx.config.hyperparameters, hp_ctx);
    auto hp_logger = [&](const std::string& msg) { ctx.logging.logger->log(msg); };
    GRIM::HyperParameters::applyTrainingHyperparameterPolicy(
        ctx.config.hyperparameters, ctx.derived_schedule, hp_ctx, hp_logger);

    GRIMText::Training::DataStatsSnapshot data_stats;
    data_stats.data_path = ctx.data_info.data_path;
    data_stats.vocab_path = ctx.data_info.vocab_path;
    data_stats.tokenizer_vocab_size = ctx.data_info.tokenizer_vocab_size;
    data_stats.actual_vocab_size = ctx.data_info.actual_vocab_size;
    data_stats.train_sequence_count = ctx.data_info.train_sequence_count;
    data_stats.val_sequence_count = ctx.data_info.val_sequence_count;
    data_stats.memory_device = ctx.memory_snapshot.device;
    data_stats.memory_device_name = ctx.memory_snapshot.device_name;
    data_stats.memory_total_bytes = ctx.memory_snapshot.total_bytes;
    data_stats.memory_free_bytes = ctx.memory_snapshot.free_bytes;

    dumpAllHyperparameters(
        ctx.config.hyperparameters,
        &ctx.derived_schedule,
        &data_stats,
        [](const std::string& msg) { GRIM::Logging::EmitModuleInfo(GRIM::Logging::ModuleId::Training, msg, 0); });
}

} // namespace GRIMText::Training

