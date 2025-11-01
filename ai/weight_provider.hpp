#pragma once

#include <string>
#include <unordered_map>

namespace GRIM {

// Abstract interface for weight providers
class IWeightProvider {
public:
    virtual ~IWeightProvider() = default;

    // Get weights for a specific category (command, question, banter)
    // Returns map of token -> weight
    virtual std::unordered_map<std::string, float> getWeights(const std::string& category) const = 0;

    // Get provider priority (higher = more important during merge)
    virtual int getPriority() const = 0;

    // Get merge strategy: "override", "additive", "max"
    virtual std::string getMergeStrategy() const = 0;

    // Provider name for debugging
    virtual std::string getName() const = 0;

    // Initialize the provider (load data, connect to sources, etc.)
    virtual bool init() = 0;
};

} // namespace GRIM
