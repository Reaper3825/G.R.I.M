#pragma once
#include <cstdint>
#include <string>

// Thin OS boundary, shared by plugin API open_target and legacy launch_app.
// Windows returns ShellExecuteA's native status; other platforms return zero.
int64_t shellOpenTarget(const std::string& target);
