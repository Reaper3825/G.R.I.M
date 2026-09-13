#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace GRIM::Tools::Math {

enum class CalculatorError {
    None,
    EmptyExpression,
    ExpressionTooLong,
    UnexpectedToken,
    MissingClosingParenthesis,
    UnknownIdentifier,
    InvalidArgumentCount,
    DivisionByZero,
    DomainError,
    NonFiniteResult,
    NestingLimitExceeded
};

struct CalculatorResult {
    bool success = false;
    double value = 0.0;
    std::string answer;
    CalculatorError error = CalculatorError::None;
    std::string error_message;
    std::size_t error_offset = 0;
};

// Evaluates a calculator expression without executing code or consulting any
// global state. This registry-neutral signature can be wrapped by any future
// tool registry: equation in, structured result out.
//
// Supported syntax:
//   numbers:    12, -2.5, 1e6
//   operators:  +, -, *, /, %, ^
//   grouping:   ( and )
//   constants:  pi, e
//   functions:  abs, sqrt, sin, cos, tan, asin, acos, atan,
//               ln, log, log10, exp, floor, ceil, round,
//               min(a,b), max(a,b), pow(a,b)
CalculatorResult calculate(std::string_view equation);

} // namespace GRIM::Tools::Math
