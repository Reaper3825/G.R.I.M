#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "core/grim_platform.h"
#include "console_history.hpp"

bool processEvents(HWND hwnd,
                   std::string& buffer,
                   std::filesystem::path& currentDir,

                   nlohmann::json& longTermMemory,
                   ConsoleHistory& uiHistory);
