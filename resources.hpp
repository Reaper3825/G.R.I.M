#pragma once
#include <string>
#include <SFML/Graphics.hpp>
#include "console_history.hpp"

/// Locate the base resource path.
/// - In portable mode, resolves relative to the executable.
/// - In system mode, resolves relative to current working directory.
/// @return Path to the resources directory as a string.
std::string getResourcePath();

/// Load the contents of a text resource (JSON, config, etc.) as a string.
/// @param filename Name of the resource file (e.g., "nlp_rules.json").
/// @param argc/argv Passed from main for future portability hooks (currently unused).
/// @return File contents as a string, or empty string if missing.
std::string loadTextResource(const std::string& filename, int argc, char** argv);

/// Attempt to find any usable font in resources/ directory.
/// Looks for .ttf or .otf.  
/// @param argc/argv Passed from main for portability (currently unused).  
/// @param history Optional history object for logging errors. If null, logs to stderr.  
/// @return Full path to the first font found, or empty string if none.
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history = nullptr);
