#pragma once
#include <cstdint>
#include <string>

// Plugin-owned OS boundary for open and search tools.
// Windows returns ShellExecuteA's native status; other platforms return zero.
int64_t shellOpenTarget(const std::string& target);
