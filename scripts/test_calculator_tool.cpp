#include "Tools/Math/Calculator/Calculator.hpp"

#include <cmath>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void expectAnswer(std::string_view expression, std::string_view expected) {
    const auto result = GRIM::Tools::Math::calculate(expression);
    if (!result.success || result.answer != expected) {
        std::cerr << "FAIL: " << expression << " expected " << expected
                  << ", got " << (result.success ? result.answer : result.error_message) << '\n';
        ++failures;
    }
}

void expectError(std::string_view expression, GRIM::Tools::Math::CalculatorError expected) {
    const auto result = GRIM::Tools::Math::calculate(expression);
    if (result.success || result.error != expected) {
        std::cerr << "FAIL: " << expression << " expected error "
                  << static_cast<int>(expected) << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    using GRIM::Tools::Math::CalculatorError;

    expectAnswer("(12 - 2)", "10");
    expectAnswer("2 + 3 * 4", "14");
    expectAnswer("(2 + 3) * 4", "20");
    expectAnswer("2^3^2", "512");
    expectAnswer("-2^2", "-4");
    expectAnswer("2^-3", "0.125");
    expectAnswer("0.1 + 0.2", "0.3");
    expectAnswer("sqrt(81) + max(3, 7)", "16");
    expectAnswer("sin(pi / 2)", "1");
    expectAnswer("1e3 / 4", "250");
    expectAnswer("10 % 4", "2");

    expectError("", CalculatorError::EmptyExpression);
    expectError("12 / 0", CalculatorError::DivisionByZero);
    expectError("sqrt(-1)", CalculatorError::DomainError);
    expectError("2 +", CalculatorError::UnexpectedToken);
    expectError("(2 + 3", CalculatorError::MissingClosingParenthesis);
    expectError("unknown(2)", CalculatorError::UnknownIdentifier);
    expectError("min(1)", CalculatorError::InvalidArgumentCount);
    expectError("2 3", CalculatorError::UnexpectedToken);

    if (failures != 0) return 1;
    std::cout << "calculator tool tests passed\n";
    return 0;
}
