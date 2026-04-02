//======================================================//
//  Detectors.hpp
//  Structural Pattern Detectors for Atom Detection
//
//  Pure parsing functions that detect numeric patterns in text.
//  No class state, no CUDA — just pattern matching.
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cstddef>
#include <string>

namespace GRIM {
namespace Tokenizer {
namespace Detector {

// Detect integers: 42, -17, +5
bool detectInteger(const std::string& text, size_t pos, size_t& end);

// Detect floats: 3.14, -2.5e10, .5
bool detectFloat(const std::string& text, size_t pos, size_t& end);

// Detect hex: 0xFF, 0x1A2B
bool detectHex(const std::string& text, size_t pos, size_t& end);

// Detect binary: 0b1010
bool detectBinary(const std::string& text, size_t pos, size_t& end);

} // namespace Detector
} // namespace Tokenizer
} // namespace GRIM
