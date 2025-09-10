#pragma once
#include <string>
#include <SFML/Graphics.hpp>
#include "console_history.hpp"

// Returns the base resource path (portable or system-specific)
std::string getResourcePath();

// Loads the contents of a text resource (e.g., config, JSON) as a string
std::string loadTextResource(const std::string& filename, int argc, char** argv);

// Attempts to find any usable font in resources or system fonts
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history);
