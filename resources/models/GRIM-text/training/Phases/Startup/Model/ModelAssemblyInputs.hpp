#pragma once

#include "../Capacity/RunCapacity.hpp"

#include <cstdint>
#include <string>

namespace GRIMText::Training {

struct ModelAssemblyInputs {
    RunCapacity capacity;
    std::uint32_t actual_vocab_size = 0;
    std::uint64_t weight_init_seed = 0;
    std::string vocab_path;
};

} // namespace GRIMText::Training

