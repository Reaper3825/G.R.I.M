#include "flatbuffer_weight_provider.hpp"
#include "classifier_weights_generated.h"
#include "logger.hpp"
#include <fstream>
#include <algorithm>

namespace GRIM {

FlatBufferWeightProvider::FlatBufferWeightProvider(const std::string& filePath, int priority)
    : filePath_(filePath), priority_(priority), mergeStrategy_("override") {}

bool FlatBufferWeightProvider::init() {
    std::ifstream file(filePath_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        logError("FlatBufferWeightProvider", "Failed to open FlatBuffer weights file: " + filePath_);
        return false;
    }

    // Read entire file into buffer
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    buffer_.resize(size);
    if (!file.read(reinterpret_cast<char*>(buffer_.data()), size)) {
        logError("FlatBufferWeightProvider", "Failed to read FlatBuffer weights file: " + filePath_);
        return false;
    }
    file.close();

    // Parse FlatBuffer
    auto config = ClassifierWeights::GetClassifierConfig(buffer_.data());
    if (!config) {
        logError("FlatBufferWeightProvider", "Invalid FlatBuffer format in: " + filePath_);
        return false;
    }

    // Cache merge strategy and priority
    if (config->merge_strategy() && config->merge_strategy()->size() > 0) {
        mergeStrategy_ = config->merge_strategy()->str();
    } else {
        mergeStrategy_ = "override";  // Default strategy
    }
    if (config->priority() != 0) {
        priority_ = config->priority();
    }

    // Extract all category weights into cache
    if (config->categories()) {
        for (auto categoryWeights : *config->categories()) {
            std::string category = categoryWeights->category()->str();
            std::unordered_map<std::string, float> weights;

            if (categoryWeights->weights()) {
                for (auto entry : *categoryWeights->weights()) {
                    weights[entry->token()->str()] = entry->weight();
                }
            }

            weightCache_[category] = std::move(weights);
        }
    }

    logDebug("FlatBufferWeightProvider", "Loaded FlatBuffer weights from " + filePath_ + " (" + 
            std::to_string(weightCache_.size()) + " categories, priority=" + 
            std::to_string(priority_) + ", strategy=" + mergeStrategy_ + ")");

    return true;
}

std::unordered_map<std::string, float> FlatBufferWeightProvider::getWeights(const std::string& category) const {
    auto it = weightCache_.find(category);
    if (it != weightCache_.end()) {
        return it->second;
    }
    return {};  // Empty map if category not found
}

} // namespace GRIM
