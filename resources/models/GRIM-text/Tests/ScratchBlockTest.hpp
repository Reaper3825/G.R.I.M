//======================================================//
//  ScratchBlockTest.hpp
//  Test suite for ScratchBlock reasoning layer
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

//======================================================//
//  Test Result
//======================================================//

struct ScratchBlockTestResult {
    std::string name;
    bool passed;
    std::string message;
    double duration_ms;
};

//======================================================//
//  Test Suite
//======================================================//

class ScratchBlockTestSuite {
public:
    using TestFunc = std::function<bool(std::string&)>;
    
    void addTest(const std::string& name, TestFunc func) {
        tests_.push_back({name, func});
    }
    
    std::vector<ScratchBlockTestResult> runAll() {
        std::vector<ScratchBlockTestResult> results;
        results.reserve(tests_.size());
        
        std::cout << "\n========================================\n";
        std::cout << "  ScratchBlock Layer Test Suite\n";
        std::cout << "========================================\n\n";
        
        int passed = 0;
        int failed = 0;
        
        for (const auto& [name, func] : tests_) {
            ScratchBlockTestResult result;
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
            
            // Print result
            std::cout << (result.passed ? "[PASS]" : "[FAIL]") << " "
                      << std::setw(45) << std::left << result.name
                      << " (" << std::fixed << std::setprecision(2) 
                      << result.duration_ms << " ms)";
            
            if (!result.passed && !result.message.empty()) {
                std::cout << "\n       " << result.message;
            }
            std::cout << "\n";
            
            if (result.passed) ++passed;
            else ++failed;
            
            results.push_back(std::move(result));
        }
        
        std::cout << "\n----------------------------------------\n";
        std::cout << "Results: " << passed << " passed, " << failed << " failed\n";
        std::cout << "========================================\n\n";
        
        return results;
    }
    
    bool allPassed() const {
        for (const auto& [name, func] : tests_) {
            std::string msg;
            if (!func(msg)) return false;
        }
        return true;
    }
    
private:
    std::vector<std::pair<std::string, TestFunc>> tests_;
};

//======================================================//
//  Test Assertions
//======================================================//

#define SB_ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        message = std::string(msg) + " (line " + std::to_string(__LINE__) + ")"; \
        return false; \
    }

#define SB_ASSERT_FALSE(cond, msg) \
    SB_ASSERT_TRUE(!(cond), msg)

#define SB_ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        message = std::string(msg) + ": expected " + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define SB_ASSERT_NEAR(a, b, eps, msg) \
    if (std::abs((a) - (b)) > (eps)) { \
        message = std::string(msg) + ": expected ~" + std::to_string(b) + ", got " + std::to_string(a); \
        return false; \
    }

#define SB_ASSERT_CUDA_SUCCESS(err, msg) \
    if ((err) != cudaSuccess) { \
        message = std::string(msg) + ": " + cudaGetErrorString(err); \
        return false; \
    }

//======================================================//
//  Entry Point Declaration
//======================================================//

// Run all ScratchBlock tests
int runScratchBlockTests();

} // namespace Test
} // namespace GRIM
