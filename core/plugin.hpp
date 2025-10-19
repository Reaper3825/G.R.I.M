#pragma once
#include <string>
#include <vector>

// ---------- Stable ABI types ----------
struct CommandResult;
using CommandFunc = CommandResult(*)(const std::string&);

struct PluginCommand {
    std::string name;
    CommandFunc func;
};

// ---------- Export macros ----------
#if defined(GRIM_BUILD_HOST)
  // Building the GRIM host (exe): host symbols are exported
  #define GRIM_HOST_API __declspec(dllexport)
#else
  // From plugins: host symbols are imported
  #define GRIM_HOST_API __declspec(dllimport)
#endif

#if defined(GRIM_BUILD_PLUGIN)
  // Building a plugin DLL: plugin entry is exported
  #define GRIM_PLUGIN_API __declspec(dllexport)
#else
  // Not building a plugin: no export needed
  #define GRIM_PLUGIN_API
#endif

extern "C" {
// Host API (plugins call these)
GRIM_HOST_API void grim_register_command(const char* name, CommandFunc func);
GRIM_HOST_API void grim_unregister_command(const char* name);
}

#if defined(GRIM_BUILD_PLUGIN)
// Plugin entry (host calls this in each DLL)
extern "C" GRIM_PLUGIN_API void registerGrimPlugin();
#endif
