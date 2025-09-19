// pch.hpp
#pragma once

// ---------------------------------------------------------
// External libraries
// ---------------------------------------------------------
#include <nlohmann/json.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

// ---------------------------------------------------------
// Windows / SAPI (only compiled on Windows)
// ---------------------------------------------------------
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <sapi.h>
  #include <sphelper.h>
  #include <atlbase.h>
#endif

// ---------------------------------------------------------
// Standard Library
// ---------------------------------------------------------
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <map>
#include <set>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <thread>
#include <chrono>
#include <iomanip>
#include <regex>
#include <optional>
#include <memory>
#include <functional>
#include <fstream>
#include <mutex>
#include <atomic>
#include <random>
#include <future>
#include <cmath>