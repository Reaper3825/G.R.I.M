#pragma once
#include <string>
#include "console_history.hpp"

// Find a font from resources or system. Logs to history if provided.
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history = nullptr);
