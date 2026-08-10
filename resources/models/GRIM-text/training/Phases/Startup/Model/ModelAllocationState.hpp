#pragma once

#include <cstdint>

namespace GRIMText::Training {

struct TrainingContext;

void ModelAllocated(
	TrainingContext& ctx,
	std::uint32_t actual_vocab_size,
	std::uint64_t weight_init_seed);

} // namespace GRIMText::Training

