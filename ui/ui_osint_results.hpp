#pragma once
#include "ui_panel.hpp"
#include <vector>
#include <string>

// Include plugin.hpp for GRIM_HOST_API macro
#include "core/plugin.hpp"

// Structure to hold a single sensitive finding
struct SensitiveFinding {
    std::string severity;      // "CRITICAL", "HIGH", "MEDIUM", "LOW"
    int severityScore = 0;     // Numeric severity (0-10)
    std::string tag;           // Type: "api_key", "email", "phone", etc.
    std::string match;         // The actual found data
    std::string context;       // Surrounding text
    std::string domain;        // Where it was found
    std::string url;           // Full URL
    double entropy = 0.0;      // Entropy score
};

// Structure for OSINT scan summary
struct OSINTScanSummary {
    std::string username;
    int totalFindings = 0;
    int criticalFindings = 0;
    int highFindings = 0;
    int mediumFindings = 0;
    int lowFindings = 0;
    std::vector<std::string> affectedDomains;
};

class GRIM_HOST_API UIOsintResults : public UIPanel {
public:
    UIOsintResults();
    
    void drawOverlay(OverlayRenderer& renderer) override;
    void update(const InputState& input, float dt) override;
    
    // Load findings from JSONL log file
    bool loadFindings(const std::string& username);
    
    // Set filter
    void setFilter(const std::string& severity); // "all", "critical", "high", "medium", "low"
    void setSearchFilter(const std::string& search);
    
    // Getters
    const OSINTScanSummary& getSummary() const { return summary; }
    const std::vector<SensitiveFinding>& getFindings() const { return filteredFindings; }
    
private:
    void applyFilters();
    uint32_t getSeverityColor(int severity) const;
    void drawDetailPanel(OverlayRenderer& renderer);
    
    OSINTScanSummary summary;
    std::vector<SensitiveFinding> allFindings;
    std::vector<SensitiveFinding> filteredFindings;
    
    std::string currentSeverityFilter = "all";
    std::string currentSearchFilter = "";
    
    // UI state
    float scrollOffset = 0.0f;
    int hoveredRow = -1;
    int selectedRow = -1;
    
    // Detail panel state
    bool showDetailPanel = false;
    SensitiveFinding selectedFinding;
    
    // Column widths (percentages)
    float colWidthSeverity = 0.12f;   // 12%
    float colWidthType = 0.15f;       // 15%
    float colWidthMatch = 0.25f;      // 25%
    float colWidthDomain = 0.20f;     // 20%
    float colWidthContext = 0.28f;    // 28%
    
    float rowHeight = 30.0f;
    float headerHeight = 40.0f;
    
    // Filter buttons state
    bool filterCritical = true;
    bool filterHigh = true;
    bool filterMedium = true;
    bool filterLow = true;
};
