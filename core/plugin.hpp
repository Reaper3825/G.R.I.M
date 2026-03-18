#pragma once
#include <string>
#include <vector>

// ---------- Stable ABI types ----------
struct CommandResult;
using PluginCommandFunc = CommandResult(*)(const std::string&);

struct PluginCommand {
    std::string name;
    PluginCommandFunc func;
};

// ---------- Export macros (Windows: __declspec, macOS/Linux: visibility attribute) ----------
#if defined(__APPLE__) || defined(__linux__)
  #define GRIM_HOST_API __attribute__((visibility("default")))
  #if defined(GRIM_BUILD_PLUGIN)
    #define GRIM_PLUGIN_API __attribute__((visibility("default")))
  #else
    #define GRIM_PLUGIN_API
  #endif
#else
  #if defined(GRIM_BUILD_HOST)
    #define GRIM_HOST_API __declspec(dllexport)
  #else
    #define GRIM_HOST_API __declspec(dllimport)
  #endif
  #if defined(GRIM_BUILD_PLUGIN)
    #define GRIM_PLUGIN_API __declspec(dllexport)
  #else
    #define GRIM_PLUGIN_API
  #endif
#endif

extern "C" {
// Host API (plugins call these)
GRIM_HOST_API void grim_register_command(const char* name, PluginCommandFunc func);
GRIM_HOST_API void grim_unregister_command(const char* name);
}

#if defined(GRIM_BUILD_PLUGIN)
// Plugin entry (host calls this in each DLL)
extern "C" GRIM_PLUGIN_API void registerGrimPlugin();
#endif
