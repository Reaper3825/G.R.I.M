#pragma once
#include "commands_core.hpp"
#include <string>
#include <vector>
#include <utility>

namespace GRIM {
namespace CommandExecution {

// Try to execute a learned command from memory
CommandResult tryLearnedCommand(const std::string& cmd, const std::string& arg);

// Try to infer command using RL system
CommandResult tryRLInference(const std::string& cmd, const std::string& arg);

// Record an unknown command in memory for later analysis
void recordUnknownCommand(const std::string& cmd, const std::string& arg);

// Store a new learned command mapping
void storeLearnedCommand(const std::string& phrase, const std::string& action, float confidence = 0.75f);

// Find similar learned commands using fuzzy matching
// Returns pairs of (normalized_command, score) sorted by score descending
std::vector<std::pair<std::string, float>> findSimilarLearnedCommands(const std::string& input, float minSimilarity = 0.6f);

} // namespace CommandExecution
} // namespace GRIM
