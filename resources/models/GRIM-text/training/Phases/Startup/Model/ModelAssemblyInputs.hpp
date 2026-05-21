#pragma once

#include <cstdint>
#include <string>

namespace GRIMText::Training {

struct ModelAssemblyInputs {
    std::uint32_t actual_vocab_size = 0;
    std::uint64_t weight_init_seed = 0;
    std::string vocab_path;
};

} // namespace GRIMText::Training

