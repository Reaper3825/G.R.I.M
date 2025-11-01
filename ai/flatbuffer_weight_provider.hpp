#pragma once

#include "weight_provider.hpp"
#include <memory>
#include <vector>
#include <fstream>

namespace GRIM {

// Provider that loads weights from FlatBuffer binary files
class FlatBufferWeightProvider : public IWeightProvider {
public:
    explicit FlatBufferWeightProvider(const std::string& filePath, int priority = 50);
    ~FlatBufferWeightProvider() override = default;

    std::unordered_map<std::string, float> getWeights(const std::string& category) const override;
    int getPriority() const override { return priority_; }
    std::string getMergeStrategy() const override { return mergeStrategy_; }
    std::string getName() const override { return "FlatBufferWeightProvider(" + filePath_ + ")"; }
    bool init() override;

private:
    std::string filePath_;
    int priority_;
    std::string mergeStrategy_;
    std::vector<uint8_t> buffer_;  // FlatBuffer binary data
    
    // Category -> (Token -> Weight) mapping
    std::unordered_map<std::string, std::unordered_map<std::string, float>> weightCache_;
};

} // namespace GRIM
