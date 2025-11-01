#pragma once

#include "weight_provider.hpp"
#include "nlp/nlp.hpp"

namespace GRIM {

// Provider that extracts weights from NLP grammar database
class NLPGrammarProvider : public IWeightProvider {
public:
    explicit NLPGrammarProvider(int priority = 75);
    ~NLPGrammarProvider() override = default;

    std::unordered_map<std::string, float> getWeights(const std::string& category) const override;
    int getPriority() const override { return priority_; }
    std::string getMergeStrategy() const override { return "additive"; }
    std::string getName() const override { return "NLPGrammarProvider"; }
    bool init() override;

private:
    int priority_;
    
    // Cache extracted from grammar database
    std::unordered_map<std::string, std::unordered_map<std::string, float>> weightCache_;
    
    void extractCommandWeights();
    void extractQuestionWeights();
    void extractLocationWeights();
    void extractBanterWeights();
};

} // namespace GRIM
