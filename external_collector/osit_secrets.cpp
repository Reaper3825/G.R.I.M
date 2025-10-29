#include "osit_secrets.hpp"
#include "osit.hpp"
#include "logger.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <map>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

std::string runSensitiveDataScan(const OSINTReport& report, const OSINTConfig& config) {
    try {
        fs::path root = fs::current_path();
        fs::path scannerScript = root / "resources" / "python" / "sherlock_sensitive_scanner.py";
        fs::path tempJsonFile = root / "resources" / "python" / ("temp_sherlock_" + report.username + ".json");
        fs::path outputLog = root / "cache" / "osint" / ("sensitive_" + report.username + ".jsonl");
        
        // Create cache directory if it doesn't exist
        fs::create_directories(outputLog.parent_path());
        
        // Check if scanner script exists
        if (!fs::exists(scannerScript)) {
            LOG_ERROR("OSINT", "Sensitive scanner script not found at: " + scannerScript.string());
            return "";
        }
        
        // Write Sherlock results to temp JSON file in format expected by scanner
        json sherlockData;
        for (const auto& finding : report.findings) {
            if (finding.found) {
                sherlockData[finding.platform] = {
                    {"url", finding.url},
                    {"status", "Claimed"},
                    {"http_status", finding.statusCode}
                };
            }
        }
        
        std::ofstream tempOut(tempJsonFile);
        tempOut << sherlockData.dump(2);
        tempOut.close();
        
        // Build command to run scanner
        std::string pythonCmd = config.pythonPath.empty() ? getPythonExecutable() : config.pythonPath;
        std::ostringstream cmd;
        
#ifdef _WIN32
        cmd << "\"\"" << pythonCmd << "\" \"" << scannerScript.string() << "\" "
            << "\"" << tempJsonFile.string() << "\" "
            << "--out \"" << outputLog.string() << "\"\"";
#else
        cmd << pythonCmd << " \"" << scannerScript.string() << "\" "
            << "\"" << tempJsonFile.string() << "\" "
            << "--out \"" << outputLog.string() << "\"";
#endif
        
        LOG_DEBUG("OSINT", "Running sensitive data scanner: " + cmd.str());
        
        // Execute scanner
        int code = std::system(cmd.str().c_str());
        
        // Cleanup temp file
        try {
            fs::remove(tempJsonFile);
        } catch (...) {}
        
        if (code != 0) {
            LOG_ERROR("OSINT", "Sensitive scanner returned code: " + std::to_string(code));
            return "";
        }
        
        if (!fs::exists(outputLog)) {
            LOG_ERROR("OSINT", "Scanner did not create output file");
            return "";
        }
        
        LOG_DEBUG("OSINT", "Sensitive scan complete: " + outputLog.string());
        return outputLog.string();
        
    } catch (const std::exception& e) {
        LOG_ERROR("OSINT", std::string("Sensitive scan failed: ") + e.what());
        return "";
    }
}

SensitiveFindingSummary parseSensitiveScanLog(const std::string& logPath) {
    SensitiveFindingSummary summary;
    
    if (!fs::exists(logPath)) {
        return summary;
    }
    
    try {
        std::ifstream logFile(logPath);
        std::string line;
        std::map<std::string, int> findingsByType;
        std::set<std::string> domains;
        
        while (std::getline(logFile, line)) {
            if (line.empty()) continue;
            
            try {
                json entry = json::parse(line);
                auto findings = entry.value("findings", json::array());
                
                for (const auto& finding : findings) {
                    int severity = finding.value("severity", 0);
                    std::string tag = finding.value("tag", "unknown");
                    
                    summary.totalFindings++;
                    findingsByType[tag]++;
                    
                    if (severity >= 9) {
                        summary.criticalFindings++;
                    } else if (severity >= 7) {
                        summary.highFindings++;
                    } else if (severity >= 5) {
                        summary.mediumFindings++;
                    } else {
                        summary.lowFindings++;
                    }
                }
                
                std::string domain = entry.value("domain", "");
                if (!domain.empty() && !findings.empty()) {
                    domains.insert(domain);
                }
                
            } catch (const json::exception&) {
                continue;
            }
        }
        
        summary.affectedDomains = std::vector<std::string>(domains.begin(), domains.end());
        
        // Build detailed report
        std::ostringstream report;
        report << "\n=== Sensitive Data Exposure Analysis ===\n\n";
        report << "Total Findings: " << summary.totalFindings << "\n";
        report << "  CRITICAL (9-10): " << summary.criticalFindings << " findings\n";
        report << "  HIGH (7-8):      " << summary.highFindings << " findings\n";
        report << "  MEDIUM (5-6):    " << summary.mediumFindings << " findings\n";
        report << "  LOW (0-4):       " << summary.lowFindings << " findings\n\n";
        
        report << "Affected Domains: " << summary.affectedDomains.size() << "\n";
        for (const auto& domain : summary.affectedDomains) {
            report << "  • " << domain << "\n";
        }
        
        if (!findingsByType.empty()) {
            report << "\nFinding Types:\n";
            for (const auto& [type, count] : findingsByType) {
                report << "  " << type << ": " << count << "\n";
            }
        }
        
        report << "\n? RECOMMENDATIONS:\n";
        if (summary.criticalFindings > 0) {
            report << "  [!] URGENT: Critical secrets detected (API keys, private keys, tokens)\n";
            report << "      ? Rotate/revoke these credentials immediately\n";
            report << "      ? Remove from public profiles\n";
        }
        if (summary.highFindings > 0) {
            report << "  [!] HIGH PRIORITY: PII or sensitive data exposed\n";
            report << "      ? Review and remove from public accounts\n";
            report << "      ? Enable privacy settings on affected platforms\n";
        }
        if (summary.totalFindings > 0) {
            report << "  • Review full log for details: " << logPath << "\n";
            report << "  • Request account deletion/deactivation where appropriate\n";
            report << "  • Monitor for data breaches (use HaveIBeenPwned)\n";
        }
        
        summary.detailedReport = report.str();
        
    } catch (const std::exception& e) {
        LOG_ERROR("OSINT", std::string("Failed to parse sensitive scan log: ") + e.what());
    }
    
    return summary;
}
