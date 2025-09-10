#pragma once
#include <string>
#include <SFML/Graphics.hpp>
#include "console_history.hpp"

// Returns the full path to the resources folder
// - In portable mode (GRIM_PORTABLE_ONLY=1), points to ./resources next to exe
// - In install mode, points to ${CMAKE_INSTALL_DATADIR}/grim/resources
std::string getResourcePath();

// Load a text resource (e.g. JSON, rules) from the resources folder
// Returns an empty string if not found
std::string loadTextResource(const std::string& filename, int argc, char** argv);

// Locate any usable .ttf font in resources/ or fall back to system fonts
// Returns empty string and logs error to history if none found
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history);
