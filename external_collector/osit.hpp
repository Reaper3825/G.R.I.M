#pragma once
#include <string>
#include <vector>

struct OSINTFinding {
    std::string platform;
    std::string url;
    bool found;
};

struct OSINTReport {
    bool success;
    std::string username;
    std::vector<OSINTFinding> findings;
    std::string rawJson;
    std::string error;
};

OSINTReport runSelfAudit(const std::string& username);
