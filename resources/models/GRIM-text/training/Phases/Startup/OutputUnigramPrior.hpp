#pragma once

#include <cstdint>

namespace GRIMText::Training {

struct TrainingContext;

// Authors the fresh-initialization LM-head bias from the frequency of valid
// training targets. Disabled configurations and checkpoint restores leave the
// temporary prior empty.
void authorOutputUnigramPrior(
    TrainingContext& ctx,
    std::uint32_t vocab_size);

} // namespace GRIMText::Training
