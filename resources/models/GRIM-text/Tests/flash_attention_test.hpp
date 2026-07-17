#pragma once
//======================================================//
//  flash_attention_test.hpp
//  Test harness for Flash Attention forward/backward
//======================================================//

#include <string>
#include <functional>
#include <vector>

namespace GRIM {
namespace Test {

// Test assertion macros
#define ATTN_ASSERT(cond, msg) \
    if (!(cond)) { message = std::string(msg); return false; }

#define ATTN_ASSERT_NEAR(a, b, tol, msg) \
    if (std::abs((a) - (b)) > (tol)) { \
        message = std::string(msg) + " (got " + std::to_string(a) + ", expected " + std::to_string(b) + ")"; \
        return false; \
    }

struct AttentionTestResult {
    bool passed;
    std::string name;
    std::string message;
    double elapsed_ms;
};

// Test function signature
using AttentionTestFn = std::function<bool(std::string&)>;

// Run all attention tests
std::vector<AttentionTestResult> runAllAttentionTests();

// Individual test declarations
bool testFlashForwardBasic(std::string& message);
bool testFlashForwardCausalMask(std::string& message);
bool testFlashForwardCustomScale(std::string& message);
bool testFlashBackwardGradientFlow(std::string& message);
bool testFlashBackwardNumericalGradient(std::string& message);
bool testFlashBackwardProductionGQA(std::string& message);
bool testFlashForwardVsNaive(std::string& message);
bool testFlashBackwardVsNaive(std::string& message);
bool testFlashLargeSequence(std::string& message);
bool testFlashMultiBatch(std::string& message);
bool testFlashMultiHead(std::string& message);
bool testFlashGradientMagnitude(std::string& message);

} // namespace Test
} // namespace GRIM
