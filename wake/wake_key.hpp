#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "console_history.hpp"
#include "timer.hpp"
#include "nlp/nlp.hpp"

namespace WakeKey {

void start(ConsoleHistory* history,
           std::vector<Timer>& timers,
           nlohmann::json& longTermMemory,
           NLP& nlp);

void stop();
bool isRunning();
void update();

// External local controllers (for example physical hand gestures) use the
// same wake/capture path as the configured key binding.
bool requestWake(const std::string& source);

} // namespace WakeKey
