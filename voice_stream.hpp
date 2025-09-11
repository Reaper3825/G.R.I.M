#pragma once
#include <whisper.h>
struct ConsoleHistory;
void runVoiceStream(whisper_context *ctx, ConsoleHistory* history);
