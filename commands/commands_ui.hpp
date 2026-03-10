#pragma once
#include "commands_core.hpp"

// UI control commands
CommandResult cmdToggleOverlayConsole(const std::string& arg);
CommandResult cmdToggleSettings(const std::string& arg);

// MMO UI surface commands (backed by UISurfaceRegistry)
CommandResult cmdCreateSurface(const std::string& arg);
CommandResult cmdShowSurface(const std::string& arg);
CommandResult cmdHideSurface(const std::string& arg);
CommandResult cmdDestroySurface(const std::string& arg);
