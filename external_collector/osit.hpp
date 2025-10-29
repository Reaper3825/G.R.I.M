#pragma once
#include <string>
#include <vector>
#include <optional>
#include <chrono>

struct OSINTFinding {
    std::string platform;
    std::string url;
    bool found;
    std::string statusCode;  // HTTP status or error info
    std::chrono::milliseconds responseTime{0};
};

struct OSINTReport {
    bool success;
    std::string username;
    std::vector<OSINTFinding> findings;
    std::string rawJson;
    std::string error;
    std::chrono::system_clock::time_point timestamp;
    int totalChecked{0};
    int totalFound{0};
    
    // Summary statistics
    std::string getSummary() const;
    std::string getDetailedReport() const;
};

struct OSINTConfig {
    bool useCache{true};
    int timeoutSeconds{30};
    bool verboseOutput{false};
    std::string pythonPath;  // Optional custom python path
};

// Main OSINT functions
OSINTReport runSelfAudit(const std::string& username, const OSINTConfig& config = OSINTConfig{});

// Utility functions
std::optional<OSINTReport> getCachedReport(const std::string& username);
void cacheReport(const OSINTReport& report);
void clearOSINTCache();
std::string getPythonExecutable();
bool isSherlockAvailable();
