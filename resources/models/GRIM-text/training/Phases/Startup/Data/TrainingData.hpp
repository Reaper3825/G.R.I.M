#pragma once

namespace GRIMText::Training {

struct TrainingContext;

void LoadInferenceTokenizer(TrainingContext& ctx);
void LoadTrainingData(TrainingContext& ctx);

} // namespace GRIMText::Training