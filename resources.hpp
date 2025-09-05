#pragma once
#include <string>
#include "console_history.hpp"

// Load a text-based resource (e.g. JSON config) from exe dir, ./resources,
// or system install dir (depending on build mode).
std::string loadTextResource(const std::string& name, int argc, char** argv);

// Find a usable font (first .ttf in resources/, else fallback system fonts).
// Logs an error message to history if provided.
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history = nullptr);
