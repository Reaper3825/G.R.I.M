// pch.hpp
#pragma once

// ---------------------------------------------------------
// External libraries
// ---------------------------------------------------------
#include <nlohmann/json.hpp>


// ---------------------------------------------------------
// Windows / SAPI (only compiled on Windows)
// ---------------------------------------------------------
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #define NOMINMAX
  #include <sapi.h>
  #include <sphelper.h>
  #include <atlbase.h>
#endif

// ---------------------------------------------------------
// Standard Library
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
#include "logger.hpp"