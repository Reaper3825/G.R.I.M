#include "Calculator.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM::Tools::Math {
namespace {

constexpr std::size_t kMaximumExpressionLength = 4096;
constexpr std::size_t kMaximumNestingDepth = 64;

class ParseFailure final : public std::runtime_error {
public:
    ParseFailure(CalculatorError error, std::size_t offset, std::string message)
        : std::runtime_error(std::move(message)), error(error), offset(offset) {}

    CalculatorError error;
    std::size_t offset;
};

class Parser {
public:
    explicit Parser(std::string_view input) : input_(input) {}

    double parse() {
        const double value = parseExpression();
        skipWhitespace();
        if (!atEnd()) {
            fail(CalculatorError::UnexpectedToken,
                 "unexpected token '" + std::string(1, peek()) + "'");
        }
        requireFinite(value);
        return value;
    }

private:
    double parseExpression() {
        double lhs = parseTerm();
        for (;;) {
            if (consume('+')) {
                lhs += parseTerm();
            } else if (consume('-')) {
                lhs -= parseTerm();
            } else {
                return checked(lhs);
            }
        }
    }

    double parseTerm() {
        double lhs = parseUnary();
        for (;;) {
            if (consume('*')) {
                lhs *= parseUnary();
            } else if (consume('/')) {
                const std::size_t operator_offset = previousOffset();
                const double rhs = parseUnary();
                if (rhs == 0.0) {
                    failAt(CalculatorError::DivisionByZero, operator_offset,
                           "division by zero");
                }
                lhs /= rhs;
            } else if (consume('%')) {
                const std::size_t operator_offset = previousOffset();
                const double rhs = parseUnary();
                if (rhs == 0.0) {
                    failAt(CalculatorError::DivisionByZero, operator_offset,
                           "modulo by zero");
                }
                lhs = std::fmod(lhs, rhs);
            } else {
                return checked(lhs);
            }
        }
    }

    double parseUnary() {
        if (consume('+')) return parseUnary();
        if (consume('-')) return checked(-parseUnary());
        return parsePower();
    }

    // Exponentiation is right-associative: 2^3^2 == 2^(3^2). Unary signs are
    // accepted in exponents while -2^2 remains -(2^2).
    double parsePower() {
        double base = parsePrimary();
        if (consume('^')) {
            const std::size_t operator_offset = previousOffset();
            const double exponent = parseUnary();
            errno = 0;
            const double value = std::pow(base, exponent);
            if (errno == EDOM || std::isnan(value)) {
                failAt(CalculatorError::DomainError, operator_offset,
                       "power is outside the real-number domain");
            }
            return checkedAt(value, operator_offset);
        }
        return base;
    }

    double parsePrimary() {
        skipWhitespace();
        if (atEnd()) {
            fail(CalculatorError::UnexpectedToken, "expected a number, constant, function, or '('");
        }

        if (consume('(')) {
            enterNesting();
            const double value = parseExpression();
            if (!consume(')')) {
                leaveNesting();
                fail(CalculatorError::MissingClosingParenthesis, "expected ')'");
            }
            leaveNesting();
            return value;
        }

        if (isIdentifierStart(peek())) {
            return parseIdentifier();
        }
        return parseNumber();
    }

    double parseIdentifier() {
        skipWhitespace();
        const std::size_t start = position_;
        while (!atEnd() && isIdentifierPart(peek())) ++position_;
        std::string name(input_.substr(start, position_ - start));
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (name == "pi") return 3.14159265358979323846;
        if (name == "e") return 2.71828182845904523536;

        if (!consume('(')) {
            failAt(CalculatorError::UnknownIdentifier, start,
                   "unknown constant or missing '(' after function '" + name + "'");
        }

        enterNesting();
        std::vector<double> arguments;
        if (!consume(')')) {
            for (;;) {
                arguments.push_back(parseExpression());
                if (consume(')')) break;
                if (!consume(',')) {
                    leaveNesting();
                    fail(CalculatorError::MissingClosingParenthesis,
                         "expected ',' or ')' in function call");
                }
            }
        }
        leaveNesting();
        return callFunction(name, arguments, start);
    }

    double callFunction(const std::string& name,
                        const std::vector<double>& arguments,
                        std::size_t offset) {
        const auto unary = [&](auto function) {
            requireArgumentCount(name, arguments, 1, offset);
            return function(arguments[0]);
        };
        const auto binary = [&](auto function) {
            requireArgumentCount(name, arguments, 2, offset);
            return function(arguments[0], arguments[1]);
        };

        double value = 0.0;
        errno = 0;
        if (name == "abs") value = unary([](double x) { return std::fabs(x); });
        else if (name == "sqrt") value = unary([](double x) { return std::sqrt(x); });
        else if (name == "sin") value = unary([](double x) { return std::sin(x); });
        else if (name == "cos") value = unary([](double x) { return std::cos(x); });
        else if (name == "tan") value = unary([](double x) { return std::tan(x); });
        else if (name == "asin") value = unary([](double x) { return std::asin(x); });
        else if (name == "acos") value = unary([](double x) { return std::acos(x); });
        else if (name == "atan") value = unary([](double x) { return std::atan(x); });
        else if (name == "ln") value = unary([](double x) { return std::log(x); });
        else if (name == "log" || name == "log10") value = unary([](double x) { return std::log10(x); });
        else if (name == "exp") value = unary([](double x) { return std::exp(x); });
        else if (name == "floor") value = unary([](double x) { return std::floor(x); });
        else if (name == "ceil") value = unary([](double x) { return std::ceil(x); });
        else if (name == "round") value = unary([](double x) { return std::round(x); });
        else if (name == "min") value = binary([](double a, double b) { return std::min(a, b); });
        else if (name == "max") value = binary([](double a, double b) { return std::max(a, b); });
        else if (name == "pow") value = binary([](double a, double b) { return std::pow(a, b); });
        else {
            failAt(CalculatorError::UnknownIdentifier, offset,
                   "unknown function '" + name + "'");
        }

        if (errno == EDOM || std::isnan(value)) {
            failAt(CalculatorError::DomainError, offset,
                   "function '" + name + "' is outside the real-number domain");
        }
        return checkedAt(value, offset);
    }

    double parseNumber() {
        skipWhitespace();
        const std::size_t start = position_;
        std::string remaining(input_.substr(position_));
        char* end = nullptr;
        errno = 0;
        const double value = std::strtod(remaining.c_str(), &end);
        if (end == remaining.c_str()) {
            fail(CalculatorError::UnexpectedToken, "expected a number");
        }
        position_ += static_cast<std::size_t>(end - remaining.c_str());
        if (errno == ERANGE || !std::isfinite(value)) {
            failAt(CalculatorError::NonFiniteResult, start,
                   "numeric literal is outside the supported finite range");
        }
        return value;
    }

    static void requireArgumentCount(const std::string& name,
                                     const std::vector<double>& arguments,
                                     std::size_t expected,
                                     std::size_t offset) {
        if (arguments.size() != expected) {
            throw ParseFailure(
                CalculatorError::InvalidArgumentCount, offset,
                "function '" + name + "' expects " + std::to_string(expected) +
                " argument" + (expected == 1 ? "" : "s"));
        }
    }

    void enterNesting() {
        if (++nesting_depth_ > kMaximumNestingDepth) {
            fail(CalculatorError::NestingLimitExceeded,
                 "expression nesting limit exceeded");
        }
    }

    void leaveNesting() { --nesting_depth_; }

    bool consume(char expected) {
        skipWhitespace();
        if (!atEnd() && peek() == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (!atEnd() && std::isspace(static_cast<unsigned char>(peek()))) ++position_;
    }

    bool atEnd() const { return position_ == input_.size(); }
    char peek() const { return input_[position_]; }
    std::size_t previousOffset() const { return position_ == 0 ? 0 : position_ - 1; }

    static bool isIdentifierStart(char c) {
        return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
    }

    static bool isIdentifierPart(char c) {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    }

    double checked(double value) const { return checkedAt(value, position_); }

    double checkedAt(double value, std::size_t offset) const {
        if (!std::isfinite(value)) {
            failAt(CalculatorError::NonFiniteResult, offset,
                   "calculation produced a non-finite result");
        }
        return value;
    }

    void requireFinite(double value) const {
        if (!std::isfinite(value)) {
            fail(CalculatorError::NonFiniteResult,
                 "calculation produced a non-finite result");
        }
    }

    [[noreturn]] void fail(CalculatorError error, std::string message) const {
        failAt(error, position_, std::move(message));
    }

    [[noreturn]] static void failAt(CalculatorError error,
                                    std::size_t offset,
                                    std::string message) {
        throw ParseFailure(error, offset, std::move(message));
    }

    std::string_view input_;
    std::size_t position_ = 0;
    std::size_t nesting_depth_ = 0;
};

std::string formatAnswer(double value) {
    if (value == 0.0) value = 0.0; // normalize negative zero
    std::ostringstream stream;
    stream << std::setprecision(15) << std::defaultfloat << value;
    return stream.str();
}

} // namespace

CalculatorResult calculate(std::string_view equation) {
    CalculatorResult result;

    const auto first_non_space = equation.find_first_not_of(" \t\r\n\f\v");
    if (first_non_space == std::string_view::npos) {
        result.error = CalculatorError::EmptyExpression;
        result.error_message = "expression is empty";
        return result;
    }
    if (equation.size() > kMaximumExpressionLength) {
        result.error = CalculatorError::ExpressionTooLong;
        result.error_message = "expression exceeds the 4096-character limit";
        result.error_offset = kMaximumExpressionLength;
        return result;
    }

    try {
        result.value = Parser(equation).parse();
        result.answer = formatAnswer(result.value);
        result.success = true;
        return result;
    } catch (const ParseFailure& failure) {
        result.error = failure.error;
        result.error_message = failure.what();
        result.error_offset = failure.offset;
        return result;
    }
}

} // namespace GRIM::Tools::Math
