#pragma once
#include <string>
#include <vector>

// Forward declarations
struct OSINTReport;
struct OSINTConfig;

/**
 * @brief Scan discovered URLs for sensitive data exposure
 * @param report The OSINT report containing discovered accounts
 * @param config Configuration for the scan
 * @return Path to the generated sensitive data log file
 */
std::string runSensitiveDataScan(const OSINTReport& report, const OSINTConfig& config);

/**
 * @brief Parse sensitive data scan results
 * @param logPath Path to the JSONL log file
 * @return Summary of findings organized by severity
 */
struct SensitiveFindingSummary {
    int totalFindings{0};
    int criticalFindings{0};
    int highFindings{0};
    int mediumFindings{0};
    int lowFindings{0};
    std::vector<std::string> affectedDomains;
    std::string detailedReport;
};

SensitiveFindingSummary parseSensitiveScanLog(const std::string& logPath);
