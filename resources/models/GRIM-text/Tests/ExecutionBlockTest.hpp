//======================================================//
//  ExecutionBlockTest.hpp
//  Test suite for execution-first numeric refactor
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace GRIM {
namespace Test {

struct ExecTestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

class ExecutionBlockTestSuite {
public:
    using TestFunc = std::function<bool(std::string&)>;

    void addTest(const std::string& name, TestFunc func) {
        tests_.push_back({name, func});
    }

    std::vector<ExecTestResult> runAll() {
        std::vector<ExecTestResult> results;
        results.reserve(tests_.size());

        std::cout << "\n========================================\n";
        std::cout << "  ExecutionBlock Test Suite\n";
        std::cout << "========================================\n\n";

        int passed = 0, failed = 0;

        for (const auto& [name, func] : tests_) {
            ExecTestResult result;
            result.name = name;
            auto start = std::chrono::high_resolution_clock::now();
            try {
                result.passed = func(result.message);
            } catch (const std::exception& e) {
                result.passed = false;
                result.message = std::string("Exception: ") + e.what();
            } catch (...) {
                result.passed = false;
                result.message = "Unknown exception";
            }
            auto end = std::chrono::high_resolution_clock::now();
            result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << (result.passed ? "[PASS]" : "[FAIL]") << " "
                      << std::setw(50) << std::left << result.name
                      << " (" << std::fixed << std::setprecision(2)
                      << result.duration_ms << " ms)";
            if (!result.passed && !result.message.empty())
                std::cout << "\n       " << result.message;
            std::cout << "\n";
            if (result.passed) ++passed; else ++failed;
            results.push_back(std::move(result));
        }

        std::cout << "\n----------------------------------------\n";
        std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
        std::cout << "========================================\n\n";
        return results;
    }

private:
    std::vector<std::pair<std::string, TestFunc>> tests_;
};

#define EB_ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        message = std::string(msg) + " (line " + std::to_string(__LINE__) + ")"; \
        return false; \
    }

#define EB_ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        message = std::string(msg) + ": expected " + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define EB_ASSERT_NEAR(a, b, eps, msg) \
    if (std::abs((a) - (b)) > (eps)) { \
        message = std::string(msg) + ": expected ~" + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

int runExecutionBlockTests();

} // namespace Test
} // namespace GRIM
