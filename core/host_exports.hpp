#pragma once

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
