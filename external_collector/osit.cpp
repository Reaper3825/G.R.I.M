#include "osit.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static std::string getPythonCmd() {
#ifdef _WIN32
    return "python";
#else
    return "python3";
#endif
}

OSINTReport runSelfAudit(const std::string& username) {
    OSINTReport report;
    report.username = username;
    try {
        std::filesystem::path root = std::filesystem::current_path();
        std::filesystem::path ositDir = root / "osit";
        std::filesystem::path bridge = ositDir / "osit_bridge.py";
        std::filesystem::path outFile = ositDir / "sherlock_results.json";

        std::ostringstream cmd;
        cmd << getPythonCmd() << " \"" << bridge.string() << "\" "
            << "\"" << username << "\" "
            << "\"" << outFile.string() << "\"";

        int code = std::system(cmd.str().c_str());
        if (code != 0) {
            report.success = false;
            report.error = "Sherlock bridge returned code " + std::to_string(code);
            return report;
        }

        if (!std::filesystem::exists(outFile)) {
            report.success = false;
            report.error = "No output JSON generated";
            return report;
        }

        std::ifstream in(outFile);
        json j;
        in >> j;
        report.rawJson = j.dump(2);
        report.success = true;

        for (auto& [site, entry] : j.items()) {
            OSINTFinding f;
            f.platform = site;
            f.url = entry.value("url_main", "");
            f.found = entry.value("status", "") == "FOUND";
            report.findings.push_back(f);
        }

        // optional cleanup
        std::filesystem::remove(outFile);
    } catch (const std::exception& e) {
        report.success = false;
        report.error = e.what();
    }
    return report;
}
