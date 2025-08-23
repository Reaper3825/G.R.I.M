// commands.hpp
#pragma once
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// Handle a command string and return a response
std::string handleCommand(const std::string& raw, fs::path& currentDir);
