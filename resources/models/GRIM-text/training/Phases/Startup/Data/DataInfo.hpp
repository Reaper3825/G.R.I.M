#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace GRIMText::Training {

struct SequenceData;
struct TrainingContext;

struct DataLoadInputs {
    std::string data_path;
    std::string vocab_path;
    int max_seq_len = 0;
    int min_seq_valid_tokens = 0;
    int sliding_window_stride = 0;
    bool add_bos = false;
    bool add_eos = false;
};

struct DataInfo {
    std::string data_path;
    std::string vocab_path;
    std::uint32_t tokenizer_vocab_size = 0;
    std::uint32_t actual_vocab_size = 0;
    std::size_t train_sequence_count = 0;
    std::size_t val_sequence_count = 0;
    std::uint32_t max_train_seq_len = 0;
    std::uint32_t max_val_seq_len = 0;
};

DataInfo summarizeDataInfoOrThrow(
    const DataLoadInputs& inputs,
    const SequenceData& data,
    std::uint32_t tokenizer_vocab_size);

void DataInfoReady(TrainingContext& ctx);

} // namespace GRIMText::Training

