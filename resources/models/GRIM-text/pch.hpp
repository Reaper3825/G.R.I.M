//======================================================//
//  Precompiled Header (PCH)
//  Common includes for GRIM-text training and inference
//  
//  This file is pre-compiled to speed up compilation times
//  Include the most frequently used, stable headers here
//======================================================//

#pragma once

//======================================================//
// Standard Library - Core
//======================================================//
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <variant>

//======================================================//
// Standard Library - Containers
//======================================================//
#include <deque>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//======================================================//
// CUDA Headers (only when USE_CUDA is defined)
//======================================================//
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#endif

//======================================================//
// Third-Party Libraries
//======================================================//
#include <nlohmann/json.hpp>
#include <flatbuffers/flatbuffers.h>

//======================================================//
// Common Project Headers
// (Only include stable headers that rarely change)
//======================================================//
// Note: Don't include implementation files here, only interfaces
// that are stable and used across many translation units
