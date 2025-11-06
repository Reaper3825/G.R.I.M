// pch.hpp
#pragma once

// ---------------------------------------------------------
// Standard Library (MUST come first for nlohmann/json)
// ---------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------
// Windows / SAPI (only compiled on Windows, after STL)
// ---------------------------------------------------------
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <sapi.h>
  #include <sphelper.h>
  #include <atlbase.h>
#endif

// ---------------------------------------------------------
// External libraries (after standard library and Windows)
// ---------------------------------------------------------
#include <nlohmann/json.hpp>

#include "logger.hpp"