#pragma once
#include <string>
#include <SFML/Graphics.hpp>
#include "console_history.hpp"

// Returns the full path to the resources folder relative to the binary
std::string getResourcePath();

// Existing helpers you already use
std::string loadTextResource(const std::string& filename, int argc, char** argv);
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history);
