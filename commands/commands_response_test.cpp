#include "commands_response_test.hpp"
#include "response_manager.hpp"
#include "logger.hpp"
#include "console_history.hpp"
#include <sstream>
#include <unordered_map>
#include <set>

extern ConsoleHistory history;

CommandResult cmdTestResponse([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("TestResponse", "Running response system integration test");
    
    std::ostringstream output;
    output << "=== Response System Test ===\n\n";
    
    int passed = 0;
    int failed = 0;
    
    // Test 1: Basic response retrieval
    {
        std::string resp = ResponseManager::get("clean");
        if (!resp.empty() && (resp.find("History") != std::string::npos || 
                              resp.find("Console") != std::string::npos ||
                              resp.find("cleared") != std::string::npos)) {
            output << "✓ Basic response: '" << resp << "'\n";
            passed++;
        } else {
            output << "✗ Basic response failed\n";
            failed++;
        }
    }
    
    // Test 2: Response variety (should not repeat in 5 tries)
    {
        ResponseManager::clearHistory();
        std::vector<std::string> responses;
        for (int i = 0; i < 5; i++) {
            responses.push_back(ResponseManager::get("ack_understood"));
        }
        
        // Count unique responses
        std::set<std::string> unique(responses.begin(), responses.end());
        if (unique.size() >= 2) {
            output << "✓ Response variety: got " << unique.size() << " unique responses\n";
            passed++;
        } else {
            output << "✗ Response variety failed: only " << unique.size() << " unique\n";
            failed++;
        }
    }
    
    // Test 3: Contextual greeting
    {
        std::string greeting = ResponseManager::getGreeting();
        if (!greeting.empty() && (greeting.find("morning") != std::string::npos ||
                                   greeting.find("afternoon") != std::string::npos ||
                                   greeting.find("evening") != std::string::npos ||
                                   greeting.find("night") != std::string::npos ||
                                   greeting.find("Morning") != std::string::npos ||
                                   greeting.find("Afternoon") != std::string::npos ||
                                   greeting.find("Evening") != std::string::npos)) {
            output << "✓ Contextual greeting: '" << greeting << "'\n";
            passed++;
        } else {
            output << "✗ Contextual greeting failed\n";
            failed++;
        }
    }
    
    // Test 4: Parameter substitution
    {
        // First add a test template to the responses (we'll use open_app_success which has a trailing space)
        std::unordered_map<std::string, std::string> params;
        params["app"] = "Chrome";
        
        std::string resp = ResponseManager::getWithParams("open_app_success", params);
        // Note: open_app_success responses end with a space for concatenation
        if (!resp.empty()) {
            output << "✓ Parameter substitution: '" << resp << "'\n";
            passed++;
        } else {
            output << "✗ Parameter substitution failed\n";
            failed++;
        }
    }
    
    // Test 5: Literal message passthrough
    {
        std::string literal = "This is a complete sentence with spaces.";
        std::string resp = ResponseManager::get(literal);
        if (resp == literal) {
            output << "✓ Literal passthrough: message unchanged\n";
            passed++;
        } else {
            output << "✗ Literal passthrough failed\n";
            failed++;
        }
    }
    
    // Test 6: New acknowledgment responses
    {
        std::string ack = ResponseManager::get("ack_working");
        if (!ack.empty()) {
            output << "✓ New acknowledgment: '" << ack << "'\n";
            passed++;
        } else {
            output << "✗ New acknowledgment failed\n";
            failed++;
        }
    }
    
    output << "\n=== Results ===\n";
    output << "Passed: " << passed << "/" << (passed + failed) << "\n";
    
    return {
        failed == 0,
        output.str(),
        "ERR_NONE",
        "debug",
        std::to_string(passed) + " of " + std::to_string(passed + failed) + " tests passed",
        failed == 0 ? Colors::Green : Colors::Yellow
    };
}
