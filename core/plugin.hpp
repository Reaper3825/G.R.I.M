#pragma once
#include <string>

// ---------- Legacy host registration types ----------
struct CommandResult;
using PluginCommandFunc = CommandResult(*)(const std::string&);

// ---------- Export macros (Windows: __declspec, macOS/Linux: visibility attribute) ----------
#if defined(__APPLE__) || defined(__linux__)
  #define GRIM_HOST_API __attribute__((visibility("default")))
#else
  #if defined(GRIM_BUILD_HOST)
    #define GRIM_HOST_API __declspec(dllexport)
  #else
    #define GRIM_HOST_API __declspec(dllimport)
  #endif
#endif

extern "C" {
// Legacy host adapters into ToolRegistry. Dynamic plugins use plugin_api.hpp.
GRIM_HOST_API void grim_register_command(const char* name, PluginCommandFunc func);
GRIM_HOST_API void grim_unregister_command(const char* name);
}
