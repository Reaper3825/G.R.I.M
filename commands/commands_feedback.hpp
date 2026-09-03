#pragma once
#include <string>
#include <optional>

namespace GRIM {
namespace Feedback {

// Check if there's a pending feedback request
bool hasPending();

// Check if there's a pending clarification request
bool hasPendingClarification();

// Set pending feedback for a command
void setPending(const std::string& command);

// Set pending clarification for a command
void setPendingClarification(const std::string& command);

// Clear pending feedback
void clearPending();

// Clear pending clarification
void clearPendingClarification();

// Get the pending feedback command
std::optional<std::string> getPending();

// Get the pending clarification command
std::optional<std::string> getPendingClarification();

// Process user's feedback response (yes/no or clarification)
// Returns true if feedback was handled, false if should continue normal processing
bool processFeedbackResponse(const std::string& originalCmd, const std::string& userResponse);

// Process user's clarification response
// Returns true if clarification was handled, false if should continue normal processing
bool processClarificationResponse(const std::string& originalInput, const std::string& userResponse);

} // namespace Feedback
} // namespace GRIM
