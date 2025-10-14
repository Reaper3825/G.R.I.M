#pragma once
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <windows.h>
#include "console_history.hpp"
#include "timer.hpp"

bool processEvents(HWND hwnd,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& uiTimers,
                   nlohmann::json& longTermMemory,
                   ConsoleHistory& uiHistory);
