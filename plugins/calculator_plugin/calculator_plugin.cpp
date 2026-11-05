#include "core/plugin_api.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <locale>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class CalculatorError {
    EmptyExpression, ExpressionTooLong, UnexpectedToken,
    MissingClosingParenthesis, UnknownIdentifier, InvalidArgumentCount,
    DivisionByZero, DomainError, NonFiniteResult, NestingLimitExceeded
};

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
        // Bound recursive signs and exponent chains as well as grouped calls.
        enterNesting();
        struct Guard {
            Parser& parser;
            ~Guard() { parser.leaveNesting(); }
        } guard{*this};
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
    stream.imbue(std::locale::classic());
    stream << std::setprecision(15) << std::defaultfloat << value;
    return stream.str();
}


const char* errorCode(CalculatorError error) {
    switch (error) {
    case CalculatorError::EmptyExpression: return "ERR_CALC_EMPTY_EXPRESSION";
    case CalculatorError::ExpressionTooLong: return "ERR_CALC_EXPRESSION_TOO_LONG";
    case CalculatorError::UnexpectedToken: return "ERR_CALC_UNEXPECTED_TOKEN";
    case CalculatorError::MissingClosingParenthesis: return "ERR_CALC_MISSING_CLOSING_PARENTHESIS";
    case CalculatorError::UnknownIdentifier: return "ERR_CALC_UNKNOWN_IDENTIFIER";
    case CalculatorError::InvalidArgumentCount: return "ERR_CALC_INVALID_ARGUMENT_COUNT";
    case CalculatorError::DivisionByZero: return "ERR_CALC_DIVISION_BY_ZERO";
    case CalculatorError::DomainError: return "ERR_CALC_DOMAIN_ERROR";
    case CalculatorError::NonFiniteResult: return "ERR_CALC_NON_FINITE_RESULT";
    case CalculatorError::NestingLimitExceeded: return "ERR_CALC_NESTING_LIMIT_EXCEEDED";
    }
    return "ERR_CALC_UNKNOWN";
}

std::string jsonString(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20 || c >= 0x7f) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

GrimResult calculate(const GrimToolCall* call, GrimToolResultWriter* writer, void*) {
    if (!call || !writer || !writer->complete || (!call->arguments && call->arguments_size))
        return GRIM_ERROR_INVALID_PARAM;
    const std::string_view equation(call->arguments ? call->arguments : "", call->arguments_size);
    GrimToolResult result;
    result.category = "math";
    std::string message;
    std::string output;
    try {
        if (equation.size() > kMaximumExpressionLength)
            throw ParseFailure(CalculatorError::ExpressionTooLong, kMaximumExpressionLength,
                               "expression exceeds the 4096-character limit");
        if (equation.find_first_not_of(" \t\r\n\f\v") == std::string_view::npos)
            throw ParseFailure(CalculatorError::EmptyExpression, 0, "expression is empty");
        const double value = Parser(equation).parse();
        message = formatAnswer(value);
        std::ostringstream numeric;
        numeric.imbue(std::locale::classic());
        numeric << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        output = "{\"success\":true,\"value\":" + numeric.str() +
                 ",\"answer\":" + jsonString(message) + "}";
        result.success = 1;
        result.error_code = "ERR_NONE";
        result.color_rgb = 0x64FF64;
    } catch (const ParseFailure& failure) {
        result.error_code = errorCode(failure.error);
        message = failure.what();
        output = "{\"success\":false,\"error\":" + jsonString(result.error_code) +
                 ",\"error_message\":" + jsonString(message) +
                 ",\"error_offset\":" + std::to_string(failure.offset) + "}";
        result.color_rgb = 0xFF6464;
    } catch (...) {
        // Never let C++ exceptions cross the plugin ABI.
        return GRIM_ERROR_UNKNOWN;
    }
    result.message = message.c_str();
    result.structured_output_json = output.c_str();
    return writer->complete(writer, &result);
}

} // namespace

extern "C" PLUGIN_EXPORT const GrimPluginInfo* grim_plugin_get_info() {
    static const GrimPluginInfo info{GRIM_API_VERSION, "Calculator Plugin", "1.0.0",
        "GRIM Project", "Native arithmetic expression evaluation", 0};
    return &info;
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* api) {
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION)
        return GRIM_ERROR_API_VERSION_MISMATCH;
    if (!api->register_tools) return GRIM_ERROR_INVALID_PARAM;
    GrimToolRegistration tool;
    tool.handler = calculate;
    tool.usage_type = GRIM_TOOL_USAGE_MODEL;
    tool.short_description = "Evaluate arithmetic expressions with constants and mathematical functions";
    auto& d = tool.descriptor;
    d.tool_id = d.display_name = "calculate";
    d.version = "1.0.0";
    d.description = "Evaluate finite double-precision arithmetic: + - * / % ^, parentheses, pi, e; "
        "abs, sqrt, sin, cos, tan, asin, acos, atan, ln, log/log10, exp, floor, ceil, round, "
        "min(a,b), max(a,b), pow(a,b). Trigonometric angles are radians; ^ is right-associative. "
        "Maximum 4096 input bytes and 64 active recursive parsing levels. "
        "Returns an answer and numeric value, or an error code, message, and zero-based byte offset.";
    d.usage = "calculate {t}";
    d.category = "math";
    d.argument_format = "raw_text";
    d.is_informational = 1;
    const GrimToolBinding binding{"t", "t"};
    tool.bindings = &binding;
    tool.binding_count = 1;
    GrimToolParameter parameter;
    parameter.name = "expression";
    parameter.type = "string";
    parameter.description = "Raw mathematical expression to evaluate";
    parameter.required = 1;
    d.parameters = &parameter;
    d.parameter_count = 1;
    const char* examples[] = {"calculate 2 + 3 * 4", "calculate sqrt(144) + 2^3"};
    d.examples = {examples, 2};
    return api->register_tools(api, &tool, 1);
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {}
