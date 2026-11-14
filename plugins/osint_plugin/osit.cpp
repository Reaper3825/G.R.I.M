#include "osit.hpp"
#include "provider_paths.hpp"
#include "process.hpp"
#include <mutex>
#include "logger.hpp"
#ifdef GRIM_BUILD_PLUGIN
#include "plugins/plugin_logging.hpp"
#endif
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#include "core/grim_platform.h"
#endif

using json = nlohmann::json;
namespace fs = std::filesystem;

#define CACHE_DIR osintProvider::cache()
static const std::chrono::hours CACHE_EXPIRY{24}; // 24 hours

std::string getPythonExecutable() {
#ifdef _WIN32
    return "python.exe";
#else
    return "python3";
#endif
}

bool isSherlockAvailable() {
    fs::path root = osintProvider::root;
    
    // Check for Python bridge
    fs::path bridge = osintProvider::scripts() / "osit_bridge.py";
    if (!fs::exists(bridge)) {
        return false;
    }
    
    // Check for Sherlock installation in osint/sherlock folder
    fs::path sherlockDir = root / "osint" / "sherlock";
    if (fs::exists(sherlockDir)) {
        // Check for newer sherlock_project structure
        fs::path sherlock1 = sherlockDir / "sherlock_project" / "sherlock.py";
        // Check for older sherlock/sherlock.py structure
        fs::path sherlock2 = sherlockDir / "sherlock" / "sherlock.py";
        // Check for sherlock.py in root
        fs::path sherlock3 = sherlockDir / "sherlock.py";
        
        if (fs::exists(sherlock1) || fs::exists(sherlock2) || fs::exists(sherlock3)) {
            return true;
        }
    }
    
    // Could also be installed as Python module - we'll let the Python bridge check that
    return true;
}

std::string OSINTReport::getSummary() const {
    std::ostringstream oss;
    oss << "Username: " << username << "\n";
    oss << "Total Platforms Checked: " << totalChecked << "\n";
    oss << "Accounts Found: " << totalFound << "\n";
    if (!findings.empty() && totalChecked > 0) {
        oss << "Success Rate: " << std::fixed << std::setprecision(1) 
            << (100.0 * totalFound / totalChecked) << "%\n";
    }
    return oss.str();
}

std::string OSINTReport::getDetailedReport() const {
    std::ostringstream oss;
    oss << "=== OSINT Report for '" << username << "' ===\n\n";
    
    auto t = std::chrono::system_clock::to_time_t(timestamp);
    oss << "Generated: " << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S") << "\n\n";
    
    if (!success) {
        oss << "Error: " << error << "\n";
        return oss.str();
    }
    
    oss << "Summary:\n";
    oss << getSummary() << "\n";
    
    if (totalFound > 0) {
        oss << "\n--- Accounts Found ---\n";
        for (const auto& finding : findings) {
            if (finding.found) {
                oss << "  [?] " << finding.platform << "\n";
                oss << "      URL: " << finding.url << "\n";
                if (finding.responseTime.count() > 0) {
                    oss << "      Response: " << finding.responseTime.count() << "ms\n";
                }
            }
        }
    }
    
    int notFound = totalChecked - totalFound;
    if (notFound > 0) {
        oss << "\n--- Not Found (" << notFound << " platforms) ---\n";
        int shown = 0;
        for (const auto& finding : findings) {
            if (!finding.found && shown < 5) {
                oss << "  [?] " << finding.platform << "\n";
                shown++;
            }
        }
        if (notFound > 5) {
            oss << "  ... and " << (notFound - 5) << " more\n";
        }
    }
    
    return oss.str();
}

std::optional<OSINTReport> getCachedReport(const std::string& username) {
    osintProvider::validateTarget(username);
    std::lock_guard<std::recursive_mutex> storageLock(osintProvider::storageMutex);
    try {
        fs::path cacheFile = CACHE_DIR / (username + ".json");
        if (!fs::exists(cacheFile)) {
            return std::nullopt;
        }
        
        // Check cache age
        auto fileTime = fs::last_write_time(cacheFile);
        auto now = fs::file_time_type::clock::now();
        auto age = std::chrono::duration_cast<std::chrono::hours>(now - fileTime);
        
        if (age > CACHE_EXPIRY) {
            fs::remove(cacheFile); // Remove stale cache
            return std::nullopt;
        }
        
        std::ifstream in(cacheFile);
        json j;
        in >> j;
        
        OSINTReport report;
        report.success = j.value("success", false);
        report.username = username; // Cache metadata cannot redirect provider file paths.
        report.error = j.value("error", "");
        report.rawJson = j.value("rawJson", "");
        report.totalChecked = j.value("totalChecked", 0);
        report.totalFound = j.value("totalFound", 0);
        
        // Deserialize findings
        if (j.contains("findings") && j["findings"].is_array()) {
            for (const auto& f : j["findings"]) {
                OSINTFinding finding;
                finding.platform = f.value("platform", "");
                finding.url = f.value("url", "");
                finding.found = f.value("found", false);
                finding.statusCode = f.value("statusCode", "");
                report.findings.push_back(finding);
            }
        }
        
        report.timestamp = std::chrono::system_clock::now();
        return report;
        
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

void cacheReport(const OSINTReport& report) {
    osintProvider::validateTarget(report.username);
    std::lock_guard<std::recursive_mutex> storageLock(osintProvider::storageMutex);
    try {
        fs::create_directories(CACHE_DIR);
        fs::path cacheFile = CACHE_DIR / (report.username + ".json");
        
        json j;
        j["success"] = report.success;
        j["username"] = report.username;
        j["error"] = report.error;
        j["rawJson"] = report.rawJson;
        j["totalChecked"] = report.totalChecked;
        j["totalFound"] = report.totalFound;
        
        json findings = json::array();
        for (const auto& f : report.findings) {
            json jf;
            jf["platform"] = f.platform;
            jf["url"] = f.url;
            jf["found"] = f.found;
            jf["statusCode"] = f.statusCode;
            findings.push_back(jf);
        }
        j["findings"] = findings;
        
        std::ofstream out(cacheFile);
        out << j.dump(2);
        
    } catch (const std::exception&) {
        // Silent fail on cache write
    }
}

void clearOSINTCache() {
    std::lock_guard<std::recursive_mutex> storageLock(osintProvider::storageMutex);
    try {
        if (fs::exists(CACHE_DIR)) {
            fs::remove_all(CACHE_DIR);
            fs::create_directories(CACHE_DIR);
        }
    } catch (const std::exception&) {
        // Silent fail
    }
}

OSINTReport runSelfAudit(const std::string& username, const OSINTConfig& config) {
    std::lock_guard<std::recursive_mutex> storageLock(osintProvider::storageMutex);
    osintProvider::validateTarget(username);
    OSINTReport report;
    report.username = username;
    report.timestamp = std::chrono::system_clock::now();
    
    // Check cache first
    if (config.useCache) {
        auto cached = getCachedReport(username);
        if (cached.has_value()) {
            return cached.value();
        }
    }
    
    try {
        fs::path root = osintProvider::root;
        fs::path resourcesDir = osintProvider::scripts();
        fs::path bridge = resourcesDir / "osit_bridge.py";
        
        // Provider-owned unique temporary output, removed on every exit.
        osintProvider::TemporaryFile temporary("sherlock-");
        const auto& outFile = temporary.path;
        
        if (!fs::exists(bridge)) {
            report.success = false;
            report.error = std::string("OSINT bridge not found at: ") + bridge.string() + 
                          std::string("\nExpected location: plugins/osint_plugin/python/osit_bridge.py");
            return report;
        }
        
        // Build command
        std::string pythonCmd = config.pythonPath.empty() ? getPythonExecutable() : config.pythonPath;
        LOG_DEBUG("OSINT", "Running provider-owned Sherlock bridge");

        // Execute
        auto startTime = std::chrono::steady_clock::now();
        int code = osintProvider::runProcess({pythonCmd, bridge.string(), username, outFile.string(), "--timeout", std::to_string(config.timeoutSeconds)}, std::max(300, config.timeoutSeconds * 15) + 30);
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
        
        LOG_DEBUG("OSINT", "Command returned code: " + std::to_string(code));
        
        if (code != 0) {
            report.success = false;
            report.error = std::string("Sherlock bridge returned code ") + std::to_string(code) + 
                          std::string(" (execution time: ") + std::to_string(duration.count()) + std::string("s)\n") +
                          std::string("Make sure Sherlock is installed in osint/sherlock/ folder or via: pip install sherlock-project");
            return report;
        }
        
        if (!fs::exists(outFile)) {
            report.success = false;
            report.error = std::string("No output JSON generated by Sherlock\n") +
                          std::string("Output file expected at: ") + outFile.string() + std::string("\n") +
                          std::string("Check that Sherlock is properly installed in osint/sherlock/");
            return report;
        }
        
        // Parse results
        std::ifstream in(outFile);
        json j;
        in >> j;
        report.rawJson = j.dump(2);
        
        // Check if the result contains an error from the bridge
        if (j.contains("error")) {
            report.success = false;
            report.error = j.value("message", j.value("error", "Unknown error from OSINT bridge"));
            
            // Cleanup temp file even on error
            try {
                fs::remove(outFile);
            } catch (...) {}
            
            return report;
        }
        
        report.success = true;
        
        for (auto& [site, entry] : j.items()) {
            OSINTFinding f;
            f.platform = site;
            f.url = entry.value("url_main", entry.value("url", ""));
            
            std::string status = entry.value("status", "");
            f.found = (status == "Claimed" || status == "FOUND");
            f.statusCode = entry.value("http_status", status);
            
            if (entry.contains("query_time")) {
                f.responseTime = std::chrono::milliseconds(
                    static_cast<int>(entry["query_time"].get<double>() * 1000)
                );
            }
            
            report.findings.push_back(f);
            report.totalChecked++;
            if (f.found) {
                report.totalFound++;
            }
        }
        
        // Cache successful results
        if (config.useCache && report.success) {
            cacheReport(report);
        }
        
        // Cleanup temp file
        try {
            fs::remove(outFile);
        } catch (...) {}
        
    } catch (const std::exception& e) {
        report.success = false;
        report.error = std::string("Exception: ") + e.what();
    }
    
    return report;
}
