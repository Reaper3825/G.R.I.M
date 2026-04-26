#include "DataInfo.hpp"

#include "../../Phase1_Startup.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

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

} // namespace GRIMText::Training

