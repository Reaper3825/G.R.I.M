#pragma once
#include <whisper.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "nlp.hpp"
#include "console_history.hpp"
#include "timer.hpp"

void runVoiceStream(whisper_context *ctx,
                    ConsoleHistory* history,
                    std::vector<Timer>& timers,
                    nlohmann::json& longTermMemory,
                    NLP& nlp);
