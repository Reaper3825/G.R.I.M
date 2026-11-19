#pragma once
#include "commands/command_result.hpp"
#include "plugins/runtime_service_client.hpp"
#include "memory/unified_memory.hpp"

namespace memoryProvider {
void stageManual(const std::string& raw);
std::vector<GRIM::UnifiedMemoryObject> search(const std::string& query);
inline std::string errorMessage(const std::string& code) {
    return grim::runtime::take(grim::runtime::host->get_user_error_message(grim::runtime::host.instance(), code.c_str()));
}
}

CommandResult cmdRemember(const std::string&);
CommandResult cmdRecall(const std::string&);
