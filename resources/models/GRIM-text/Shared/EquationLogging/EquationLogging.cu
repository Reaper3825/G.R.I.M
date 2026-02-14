/**
 * EquationLogging.cu - Compilation unit for EquationLogger
 *
 * The EquationLogger is now a pure host-side class (header-only inline methods).
 * This .cu file exists solely as a compilation unit so CMakeLists.txt doesn't
 * need updating. All device-side ring buffer, kernels, and global pointers
 * have been deleted (Rule 20: no dead code, Rule 26: YAGNI).
 */

#include "EquationLogging.hpp"

// All implementation is inline in EquationLogging.hpp.
// This file kept as a compilation unit placeholder.
