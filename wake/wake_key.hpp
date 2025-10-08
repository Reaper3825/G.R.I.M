#pragma once
#include <vector>
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

} // namespace WakeKey
