#pragma once
#include "commands_core.hpp"

// Voice commands
CommandResult cmdVoice(const std::string& arg);
CommandResult cmdVoiceStream(const std::string& arg);
CommandResult cmd_testTTS(const std::string& arg);
CommandResult cmd_testSAPI(const std::string& arg);
CommandResult cmd_ttsDevice(const std::string& arg);
CommandResult cmd_listVoices(const std::string& arg);
CommandResult cmdNevermind(const std::string& arg);

// ? NEW: Speaker embedding commands (XTTS v2)
CommandResult cmd_createEmbedding(const std::string& arg);
CommandResult cmd_listEmbeddings(const std::string& arg);
