// commands/commands_osint.cpp - OSINT Self-Audit Commands
#include "pch.hpp"
#include "commands_osint.hpp"
#include "commands_core.hpp"
#include "logger.hpp"
#include "helpers/color.hpp"
#include "../external_collector/osit.hpp"
#include <sstream>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <map>
#include <mutex>
#include <atomic>

// Global cache for async operations
struct AsyncScan {
    std::future<OSINTReport> future;
    std::atomic<bool> completed{false};
    OSINTReport result;
    std::chrono::steady_clock::time_point startTime;
};

static std::map<std::string, std::shared_ptr<AsyncScan>> g_activeScans;
static std::mutex g_scanMutex;

// Profile Self - Searches for user's digital footprint
CommandResult cmdProfileSelf(const std::string& args) {
    LOG_DEBUG("OSINT", "Profile self command executed");
    
    std::ostringstream oss;
    oss << "=== Digital Footprint Analysis ===\n\n";
    
    if (args.empty()) {
        oss << "Usage: profile_self <username> [--no-cache] [--verbose]\n";
        oss << "Example: profile_self john_doe\n\n";
        oss << "This command searches for your digital presence across 300+ platforms.\n";
        oss << "Uses the Sherlock OSINT tool for real username enumeration.\n\n";
        oss << "Options:\n";
        oss << "  --no-cache    Skip cache and force fresh scan\n";
        oss << "  --verbose     Show detailed output\n\n";
        oss << "Platforms checked include:\n";
        oss << "  • Social media (Twitter, Instagram, Facebook, etc.)\n";
        oss << "  • Professional networks (LinkedIn, GitHub, GitLab)\n";
        oss << "  • Gaming platforms (Steam, Xbox, PlayStation)\n";
        oss << "  • Developer sites (Dev.to, Medium, HackerOne)\n";
        oss << "  • And 300+ more platforms\n";
        
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Cyan};
    }
    
    // Parse arguments
    std::istringstream iss(args);
    std::string username;
    iss >> username;
    
    OSINTConfig config;
    std::string token;
    while (iss >> token) {
        if (token == "--no-cache") {
            config.useCache = false;
        } else if (token == "--verbose") {
            config.verboseOutput = true;
        }
    }
    
    // Check if Sherlock is available
    if (!isSherlockAvailable()) {
        oss << "⚠ Sherlock bridge not found!\n\n";
        oss << "Please ensure the Sherlock tool is installed:\n";
        oss << "  1. Install Sherlock: pip install sherlock-project\n";
        oss << "  2. Verify bridge exists at: resources/python/osit_bridge.py\n\n";
        oss << "Falling back to basic URL listing...\n\n";
        
        // Fallback to simple URL listing
        oss << "Checking platforms for: " << username << "\n\n";
        oss << "  [?] GitHub: https://github.com/" << username << "\n";
        oss << "  [?] Twitter: https://twitter.com/" << username << "\n";
        oss << "  [?] LinkedIn: https://linkedin.com/in/" << username << "\n";
        oss << "  [?] Reddit: https://reddit.com/user/" << username << "\n";
        oss << "  [?] Instagram: https://instagram.com/" << username << "\n";
        oss << "  [?] YouTube: https://youtube.com/@" << username << "\n";
        
        return CommandResult{true, oss.str(), "", "osint", 
                            "Basic profile URLs generated for " + username, Colors::Yellow};
    }
    
    oss << "🔍 Scanning digital footprint for: " << username << "\n";
    if (!config.useCache) {
        oss << "Cache disabled - performing fresh scan\n";
    }
    oss << "This will take 1-5 minutes to scan 400+ platforms...\n";
    oss << "Scan running in background - you can continue using G.R.I.M\n\n";
    
    // Run scan on separate thread
    auto scanPtr = std::make_shared<AsyncScan>();
    scanPtr->startTime = std::chrono::steady_clock::now();
    
    scanPtr->future = std::async(std::launch::async, [username, config, scanPtr]() {
        OSINTReport report = runSelfAudit(username, config);
        scanPtr->result = report;
        scanPtr->completed = true;
        return report;
    });
    
    {
        std::lock_guard<std::mutex> lock(g_scanMutex);
        g_activeScans[username] = scanPtr;
    }
    
    oss << "✓ Background scan started\n";
    oss << "Use 'osint_status " << username << "' to check progress\n";
    oss << "Use 'osint_report " << username << "' to see results when complete\n";
    
    return CommandResult{true, oss.str(), "", "osint", 
                        "Background scan started for " + username, Colors::Cyan};
}

// Sherlock Sweep - Comprehensive username search across platforms
CommandResult cmdSherlockSweep(const std::string& args) {
    LOG_DEBUG("OSINT", "Sherlock sweep command executed");
    
    std::ostringstream oss;
    oss << "=== Sherlock Username Sweep ===\n\n";
    
    if (args.empty()) {
        oss << "Usage: sherlock_sweep <username> [--async]\n";
        oss << "Example: sherlock_sweep alice_crypto\n\n";
        oss << "Performs comprehensive username enumeration across 300+ platforms.\n";
        oss << "Powered by the Sherlock OSINT project.\n\n";
        oss << "Options:\n";
        oss << "  --async    Run scan in background and return immediately\n\n";
        oss << "Use 'osint_status <username>' to check async scan progress.\n";
        
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Magenta};
    }
    
    std::istringstream iss(args);
    std::string username;
    iss >> username;
    
    bool async = false;
    std::string token;
    while (iss >> token) {
        if (token == "--async") {
            async = true;
        }
    }
    
    if (async) {
        // Start async scan
        std::lock_guard<std::mutex> lock(g_scanMutex);
        
        if (g_activeScans.find(username) != g_activeScans.end() && 
            !g_activeScans[username]->completed) {
            oss << "Scan already in progress for: " << username << "\n";
            oss << "Use 'osint_status " << username << "' to check progress.\n";
            return CommandResult{true, oss.str(), "", "osint", "", Colors::Yellow};
        }
        
        auto scanPtr = std::make_shared<AsyncScan>();
        scanPtr->startTime = std::chrono::steady_clock::now();
        
        scanPtr->future = std::async(std::launch::async, [username, scanPtr]() {
            OSINTConfig config;
            OSINTReport report = runSelfAudit(username, config);
            scanPtr->result = report;
            scanPtr->completed = true;
            return report;
        });
        
        g_activeScans[username] = scanPtr;
        
        oss << "Started background scan for: " << username << "\n";
        oss << "Use 'osint_status " << username << "' to check results.\n";
        
        return CommandResult{true, oss.str(), "", "osint", 
                            "Background scan started for " + username, Colors::Magenta};
    }
    
    // Synchronous scan (still on separate thread but we wait for it)
    oss << "🔍 Sweeping for username: " << username << "\n";
    oss << "Scanning 400+ platforms... This will take 1-5 minutes.\n";
    oss << "Running in background thread...\n\n";
    
    OSINTConfig config;
    auto future = std::async(std::launch::async, [username, config]() {
        return runSelfAudit(username, config);
    });
    
    // Wait for completion
    OSINTReport report = future.get();
    
    if (!report.success) {
        oss << "⚠ Sweep failed: " << report.error << "\n";
        return CommandResult{false, oss.str(), report.error, "osint", "", Colors::Red};
    }
    
    oss << report.getSummary() << "\n";
    
    if (report.totalFound > 0) {
        oss << "\nFound accounts:\n";
        int shown = 0;
        for (const auto& finding : report.findings) {
            if (finding.found && shown < 20) {
                oss << "  [✓] " << finding.platform << " - " << finding.url << "\n";
                shown++;
            }
        }
        if (report.totalFound > 20) {
            oss << "  ... and " << (report.totalFound - 20) << " more\n";
        }
    }
    
    oss << "\nUse 'osint_report " << username << "' for detailed analysis.\n";
    
    return CommandResult{true, oss.str(), "", "osint",
                        "Sweep complete: " + std::to_string(report.totalFound) + " / " + 
                        std::to_string(report.totalChecked) + " accounts found", 
                        Colors::Magenta};
}

// OSINT Report - Generate comprehensive OSINT report
CommandResult cmdOsintReport(const std::string& args) {
    LOG_DEBUG("OSINT", "OSINT report command executed");
    
    std::ostringstream oss;
    oss << "=== OSINT Self-Audit Report ===\n\n";
    
    if (args.empty()) {
        oss << "Usage: osint_report <username> [--export <file>]\n";
        oss << "Example: osint_report username@email.com\n";
        oss << "         osint_report john_doe --export report.json\n\n";
        oss << "Generates a comprehensive OSINT report including:\n";
        oss << "  • Username enumeration across 300+ platforms\n";
        oss << "  • Account discovery statistics\n";
        oss << "  • Privacy recommendations\n";
        oss << "  • Exportable JSON data\n\n";
        oss << "Options:\n";
        oss << "  --export <file>   Export full report to JSON file\n";
        
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Yellow};
    }
    
    std::istringstream iss(args);
    std::string target;
    iss >> target;
    
    std::string exportFile;
    std::string token;
    while (iss >> token) {
        if (token == "--export") {
            iss >> exportFile;
        }
    }
    
    oss << "Generating OSINT report for: " << target << "\n\n";
    
    OSINTConfig config;
    OSINTReport report = runSelfAudit(target, config);
    
    if (!report.success) {
        oss << "⚠ Report generation failed: " << report.error << "\n";
        return CommandResult{false, oss.str(), report.error, "osint", "", Colors::Red};
    }
    
    // Full detailed report
    oss << report.getDetailedReport();
    
    oss << "\n--- Privacy Recommendations ---\n";
    if (report.totalFound > 10) {
        oss << "  [!] HIGH footprint detected (" << report.totalFound << " accounts)\n";
        oss << "  [!] Consider reviewing and deactivating unused accounts\n";
    } else if (report.totalFound > 5) {
        oss << "  [!] MODERATE footprint detected (" << report.totalFound << " accounts)\n";
    } else {
        oss << "  [✓] LOW footprint detected (" << report.totalFound << " accounts)\n";
    }
    
    oss << "  • Enable 2FA on all active accounts\n";
    oss << "  • Review privacy settings on social media platforms\n";
    oss << "  • Use unique passwords for each service\n";
    oss << "  • Monitor for data breaches (use HaveIBeenPwned)\n";
    oss << "  • Consider using alias emails for different services\n";
    oss << "  • Regularly audit and remove old/unused accounts\n\n";
    
    oss << "--- OSINT Tools Referenced ---\n";
    oss << "  • Sherlock - Username enumeration (github.com/sherlock-project/sherlock)\n";
    oss << "  • HaveIBeenPwned - Breach detection (haveibeenpwned.com)\n";
    oss << "  • Holehe - Email to account finder (github.com/megadose/holehe)\n";
    oss << "  • Social-Analyzer - Social media OSINT (github.com/qeeqbox/social-analyzer)\n\n";
    
    // Export if requested
    if (!exportFile.empty()) {
        try {
            std::ofstream out(exportFile);
            out << report.rawJson;
            oss << "Report exported to: " << exportFile << "\n";
        } catch (const std::exception& e) {
            oss << "⚠ Export failed: " << e.what() << "\n";
        }
    }
    
    return CommandResult{true, oss.str(), "", "osint",
                        "Report complete: " + std::to_string(report.totalFound) + " accounts found", 
                        Colors::Yellow};
}

// OSINT Status - Check status of async scans
CommandResult cmdOsintStatus(const std::string& args) {
    LOG_DEBUG("OSINT", "OSINT status command executed");
    
    std::ostringstream oss;
    oss << "=== OSINT Scan Status ===\n\n";
    
    std::lock_guard<std::mutex> lock(g_scanMutex);
    
    if (g_activeScans.empty()) {
        oss << "No active scans in progress.\n";
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Cyan};
    }
    
    if (args.empty()) {
        oss << "Active scans:\n";
        for (auto& [username, scanPtr] : g_activeScans) {
            if (scanPtr->completed) {
                oss << "  [✓] " << username << " - Complete\n";
            } else {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - scanPtr->startTime);
                oss << "  [⏳] " << username << " - Running (" << elapsed.count() << "s elapsed)\n";
            }
        }
        oss << "\nUse 'osint_status <username>' to get results.\n";
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Cyan};
    }
    
    // Check specific username
    std::string username = args;
    auto it = g_activeScans.find(username);
    
    if (it == g_activeScans.end()) {
        oss << "No scan found for: " << username << "\n";
        oss << "Use 'profile_self " << username << "' to start a scan.\n";
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Yellow};
    }
    
    auto scanPtr = it->second;
    
    if (!scanPtr->completed) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - scanPtr->startTime);
        oss << "Scan still in progress for: " << username << "\n";
        oss << "Elapsed time: " << elapsed.count() << " seconds\n";
        oss << "Please wait... (typical scan takes 1-5 minutes)\n";
        return CommandResult{true, oss.str(), "", "osint", "", Colors::Yellow};
    }
    
    // Get results
    OSINTReport report = scanPtr->result;
    
    if (!report.success) {
        oss << "Scan failed: " << report.error << "\n";
        g_activeScans.erase(it); // Remove failed scan
        return CommandResult{false, oss.str(), report.error, "osint", "", Colors::Red};
    }
    
    oss << report.getDetailedReport();
    
    g_activeScans.erase(it); // Remove completed scan
    
    return CommandResult{true, oss.str(), "", "osint",
                        "Scan complete: " + std::to_string(report.totalFound) + " accounts found",
                        Colors::Green};
}

// Clear OSINT cache
CommandResult cmdOsintClearCache(const std::string& args) {
    LOG_DEBUG("OSINT", "Clear cache command executed");
    
    std::ostringstream oss;
    oss << "=== Clear OSINT Cache ===\n\n";
    
    try {
        clearOSINTCache();
        oss << "✓ OSINT cache cleared successfully\n";
        oss << "All cached scan results have been removed.\n";
        oss << "Next scans will perform fresh queries.\n";
        
        return CommandResult{true, oss.str(), "", "osint", "Cache cleared", Colors::Green};
    } catch (const std::exception& e) {
        oss << "⚠ Failed to clear cache: " << e.what() << "\n";
        return CommandResult{false, oss.str(), e.what(), "osint", "", Colors::Red};
    }
}
