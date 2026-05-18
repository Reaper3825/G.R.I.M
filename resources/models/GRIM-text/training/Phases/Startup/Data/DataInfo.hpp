#pragma once

#include <cstddef>
#include <cstdint>

namespace GRIMText::Training {

struct DataInfo {
    std::uint32_t actual_vocab_size = 0;
    std::size_t train_sequence_count = 0;
    std::size_t val_sequence_count = 0;
};

} // namespace GRIMText::Training

