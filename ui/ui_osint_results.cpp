#include "ui_osint_results.hpp"
#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "core/input_parser.hpp"  // For InputState
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace fs = std::filesystem;
using json = nlohmann::json;

UIOsintResults::UIOsintResults()
    : UIPanel("OSINT Sensitive Findings", true)
{
    // Set initial size
    size.x = 1200.0f;
    size.y = 700.0f;
    
    // Use simple window-relative coordinates (like other panels)
    position = { 300, 200 };
    
    setResizable(true);
    setDraggable(true);
    
    LOG_DEBUG("OSINT-UI", "Panel positioned at (" + std::to_string(position.x) + ", " + 
              std::to_string(position.y) + ")");
}

bool UIOsintResults::loadFindings(const std::string& username) {
    summary.username = username;
    allFindings.clear();
    filteredFindings.clear();
    
    // Find the log file
    fs::path logPath = fs::current_path() / "cache" / "osint" / ("sensitive_" + username + ".jsonl");
    
    if (!fs::exists(logPath)) {
        LOG_ERROR("OSINT-UI", "No scan results found for: " + username);
        return false;
    }
    
    try {
        std::ifstream logFile(logPath);
        std::string line;
        std::set<std::string> domains;
        
        while (std::getline(logFile, line)) {
            if (line.empty()) continue;
            
            try {
                json entry = json::parse(line);
                std::string url = entry.value("source_url", "");
                std::string domain = entry.value("domain", "");
                auto findings = entry.value("findings", json::array());
                
                if (!domain.empty()) {
                    domains.insert(domain);
                }
                
                for (const auto& finding : findings) {
                    SensitiveFinding f;
                    f.severityScore = finding.value("severity", 0);
                    f.tag = finding.value("tag", "unknown");
                    f.match = finding.value("match", "");
                    f.context = finding.value("context", "");
                    f.domain = domain;
                    f.url = url;
                    f.entropy = finding.value("entropy", 0.0);
                    
                    // Set severity label
                    if (f.severityScore >= 9) {
                        f.severity = "CRITICAL";
                        summary.criticalFindings++;
                    } else if (f.severityScore >= 7) {
                        f.severity = "HIGH";
                        summary.highFindings++;
                    } else if (f.severityScore >= 5) {
                        f.severity = "MEDIUM";
                        summary.mediumFindings++;
                    } else {
                        f.severity = "LOW";
                        summary.lowFindings++;
                    }
                    
                    allFindings.push_back(f);
                    summary.totalFindings++;
                }
                
            } catch (const json::exception& e) {
                LOG_ERROR("OSINT-UI", std::string("Failed to parse line: ") + e.what());
                continue;
            }
        }
        
        summary.affectedDomains = std::vector<std::string>(domains.begin(), domains.end());
        
        // Sort by severity (highest first)
        std::sort(allFindings.begin(), allFindings.end(),
                 [](const SensitiveFinding& a, const SensitiveFinding& b) {
                     return a.severityScore > b.severityScore;
                 });
        
        applyFilters();
        
        LOG_DEBUG("OSINT-UI", "Loaded " + std::to_string(summary.totalFindings) + 
                  " findings for " + username);
        
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("OSINT-UI", std::string("Failed to load findings: ") + e.what());
        return false;
    }
}

void UIOsintResults::setFilter(const std::string& severity) {
    currentSeverityFilter = severity;
    applyFilters();
}

void UIOsintResults::setSearchFilter(const std::string& search) {
    currentSearchFilter = search;
    applyFilters();
}

void UIOsintResults::applyFilters() {
    filteredFindings.clear();
    
    int minSeverity = 0;
    if (currentSeverityFilter == "critical") {
        minSeverity = 9;
    } else if (currentSeverityFilter == "high") {
        minSeverity = 7;
    } else if (currentSeverityFilter == "medium") {
        minSeverity = 5;
    }
    
    for (const auto& finding : allFindings) {
        // Apply severity filter
        if (finding.severityScore < minSeverity) {
            continue;
        }
        
        // Apply search filter
        if (!currentSearchFilter.empty()) {
            std::string searchLower = currentSearchFilter;
            std::transform(searchLower.begin(), searchLower.end(), searchLower.begin(), ::tolower);
            
            std::string matchLower = finding.match;
            std::transform(matchLower.begin(), matchLower.end(), matchLower.begin(), ::tolower);
            
            std::string tagLower = finding.tag;
            std::transform(tagLower.begin(), tagLower.end(), tagLower.begin(), ::tolower);
            
            std::string domainLower = finding.domain;
            std::transform(domainLower.begin(), domainLower.end(), domainLower.begin(), ::tolower);
            
            if (matchLower.find(searchLower) == std::string::npos &&
                tagLower.find(searchLower) == std::string::npos &&
                domainLower.find(searchLower) == std::string::npos) {
                continue;
            }
        }
        
        filteredFindings.push_back(finding);
    }
}

uint32_t UIOsintResults::getSeverityColor(int severity) const {
    if (severity >= 9) {
        return 0xFFFF3030; // Red (CRITICAL)
    } else if (severity >= 7) {
        return 0xFFFFA500; // Orange (HIGH)
    } else if (severity >= 5) {
        return 0xFFFFFF00; // Yellow (MEDIUM)
    } else {
        return 0xFF00FF00; // Green (LOW)
    }
}

void UIOsintResults::update(const InputState& input, float dt) {
    // Call base panel update FIRST to handle drag/resize
    UIPanel::update(input, dt);
    
    // Calculate the actual content area Y position (after summary + filters + header)
    float summaryHeight = 80.0f;
    float filterHeight = 40.0f;
    float contentStartY = position.y + titleBarHeight + summaryHeight + filterHeight + headerHeight;
    float contentHeight = size.y - (titleBarHeight + summaryHeight + filterHeight + headerHeight);
    
    // Handle mouse wheel scrolling
    if (hovered && input.mouseWheelDelta != 0.0f) {
        // Scroll by 3 rows per wheel notch (120 is one notch)
        float scrollSpeed = rowHeight * 3.0f;
        scrollOffset -= input.mouseWheelDelta * scrollSpeed / 120.0f;
        
        // FIX: Better clamp calculation to prevent crashes
        if (!filteredFindings.empty()) {
            float maxScroll = std::max(0.0f, filteredFindings.size() * rowHeight - contentHeight);
            scrollOffset = std::clamp(scrollOffset, 0.0f, maxScroll);
        } else {
            scrollOffset = 0.0f;
        }
    }
    
    // Handle keyboard scrolling (fallback)
    if (hovered) {
        if (input.keysDown.count(VK_DOWN) && input.keyPressed.count(VK_DOWN)) {
            scrollOffset += rowHeight;
        }
        if (input.keysDown.count(VK_UP) && input.keyPressed.count(VK_UP)) {
            scrollOffset -= rowHeight;
        }
        
        // FIX: Better clamp calculation to prevent crashes
        if (!filteredFindings.empty()) {
            float maxScroll = std::max(0.0f, filteredFindings.size() * rowHeight - contentHeight);
            scrollOffset = std::clamp(scrollOffset, 0.0f, maxScroll);
        } else {
            scrollOffset = 0.0f;
        }
    }
    
    // *** FIX: Don't process row hover/selection if detail panel is showing ***
    // This prevents the main table from stealing focus/hover from the detail panel
    if (!showDetailPanel) {
        // Detect row hover - use screen coordinates directly since overlay spans full screen
        Vec2 mousePos{static_cast<float>(input.mousePos.x), static_cast<float>(input.mousePos.y)};
        
        // Check if mouse is in the table content area
        if (hovered && mousePos.y >= contentStartY && mousePos.y < position.y + size.y) {
            // Calculate which row the mouse is over, accounting for scroll offset
            float relY = (mousePos.y - contentStartY) + scrollOffset;
            hoveredRow = static_cast<int>(relY / rowHeight);
            
            // FIX: Proper bounds checking
            if (hoveredRow < 0 || hoveredRow >= static_cast<int>(filteredFindings.size())) {
                hoveredRow = -1;
            }
        } else {
            hoveredRow = -1;
        }
        
        // Click to select
        // FIX: Add extra validation before accessing filteredFindings
        if (hoveredRow >= 0 && hoveredRow < static_cast<int>(filteredFindings.size()) && input.mousePressed[0]) {
            selectedRow = hoveredRow;
            selectedFinding = filteredFindings[selectedRow];
            showDetailPanel = true;
            LOG_DEBUG("OSINT-UI", "Selected finding: " + filteredFindings[selectedRow].tag);
        }
    } else {
        // *** FIX: Detail panel is showing - handle ESC and click-outside to close ***
        hoveredRow = -1;  // Clear hover while detail panel is open
        
        // Close detail panel with ESC
        if (input.keysDown.count(VK_ESCAPE) && input.keyPressed.count(VK_ESCAPE)) {
            showDetailPanel = false;
            LOG_DEBUG("OSINT-UI", "Detail panel closed via ESC");
        }
        
        // *** FIX: Close detail panel by clicking OUTSIDE the detail panel area ***
        if (input.mousePressed[0]) {
            Vec2 mousePos{static_cast<float>(input.mousePos.x), static_cast<float>(input.mousePos.y)};
            
            // Calculate detail panel bounds
            float panelWidth = 500.0f;
            float panelHeight = size.y * 0.8f;
            float panelX = position.x + size.x - panelWidth - 20.0f;
            float panelY = position.y + titleBarHeight + 10.0f;
            
            // Check if click is OUTSIDE detail panel
            bool clickedOutside = (mousePos.x < panelX || mousePos.x > panelX + panelWidth ||
                                  mousePos.y < panelY || mousePos.y > panelY + panelHeight);
            
            if (clickedOutside) {
                showDetailPanel = false;
                LOG_DEBUG("OSINT-UI", "Detail panel closed by clicking outside");
            }
        }
    }
    
    // FIX: If filteredFindings changes size, validate selectedRow
    if (selectedRow >= static_cast<int>(filteredFindings.size())) {
        selectedRow = -1;
        showDetailPanel = false;
    }
}

bool UIOsintResults::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;
    
    float contentY = position.y + titleBarHeight;
    
    // Draw summary header
    float summaryHeight = 80.0f;
    renderer.drawRect({position.x, contentY}, {size.x, summaryHeight}, 0xFF2A2A2A);
    
    float summaryX = position.x + 10.0f;
    float summaryY = contentY + 10.0f;
    
    renderer.drawText({summaryX, summaryY}, "Username: " + summary.username, 0xFFFFFFFF);
    summaryY += 25.0f;
    
    std::string summaryText = "Total: " + std::to_string(summary.totalFindings) + 
                             " | CRITICAL: " + std::to_string(summary.criticalFindings) +
                             " | HIGH: " + std::to_string(summary.highFindings) +
                             " | MEDIUM: " + std::to_string(summary.mediumFindings) +
                             " | LOW: " + std::to_string(summary.lowFindings);
    
    renderer.drawText({summaryX, summaryY}, summaryText, 0xFFCCCCCC);
    summaryY += 25.0f;
    
    renderer.drawText({summaryX, summaryY}, "Affected Domains: " + std::to_string(summary.affectedDomains.size()), 0xFF999999);
    
    contentY += summaryHeight;
    
    // Draw filter buttons
    float filterHeight = 40.0f;
    renderer.drawRect({position.x, contentY}, {size.x, filterHeight}, 0xFF1A1A1A);
    
    float filterX = position.x + 10.0f;
    float filterY = contentY + 8.0f;
    float buttonWidth = 90.0f;
    float buttonHeight = 24.0f;
    float buttonSpacing = 10.0f;
    
    // Filter button: ALL
    uint32_t allColor = (currentSeverityFilter == "all") ? 0xFF4080FF : 0xFF303030;
    renderer.drawRect({filterX, filterY}, {buttonWidth, buttonHeight}, allColor);
    renderer.drawText({filterX + 30.0f, filterY + 5.0f}, "ALL", 0xFFFFFFFF);
    filterX += buttonWidth + buttonSpacing;
    
    // Filter button: CRITICAL
    uint32_t critColor = (currentSeverityFilter == "critical") ? 0xFFFF3030 : 0xFF303030;
    renderer.drawRect({filterX, filterY}, {buttonWidth, buttonHeight}, critColor);
    renderer.drawText({filterX + 10.0f, filterY + 5.0f}, "CRITICAL", 0xFFFFFFFF);
    filterX += buttonWidth + buttonSpacing;
    
    // Filter button: HIGH
    uint32_t highColor = (currentSeverityFilter == "high") ? 0xFFFFA500 : 0xFF303030;
    renderer.drawRect({filterX, filterY}, {buttonWidth, buttonHeight}, highColor);
    renderer.drawText({filterX + 25.0f, filterY + 5.0f}, "HIGH", 0xFFFFFFFF);
    filterX += buttonWidth + buttonSpacing;
    
    // Filter button: MEDIUM
    uint32_t medColor = (currentSeverityFilter == "medium") ? 0xFFFFFF00 : 0xFF303030;
    renderer.drawRect({filterX, filterY}, {buttonWidth, buttonHeight}, medColor);
    renderer.drawText({filterX + 15.0f, filterY + 5.0f}, "MEDIUM", 0xFFFFFFFF);
    filterX += buttonWidth + buttonSpacing;
    
    // Filter button: LOW
    uint32_t lowColor = (currentSeverityFilter == "low") ? 0xFF00FF00 : 0xFF303030;
    renderer.drawRect({filterX, filterY}, {buttonWidth, buttonHeight}, lowColor);
    renderer.drawText({filterX + 30.0f, filterY + 5.0f}, "LOW", 0xFFFFFFFF);
    
    contentY += filterHeight;
    
    // Draw table header
    renderer.drawRect({position.x, contentY}, {size.x, headerHeight}, 0xFF404040);
    
    float colX = position.x + 5.0f;
    float headerY = contentY + 10.0f;
    
    renderer.drawText({colX, headerY}, "Severity", 0xFFFFFFFF);
    colX += size.x * colWidthSeverity;
    
    renderer.drawText({colX, headerY}, "Type", 0xFFFFFFFF);
    colX += size.x * colWidthType;
    
    renderer.drawText({colX, headerY}, "Match", 0xFFFFFFFF);
    colX += size.x * colWidthMatch;
    
    renderer.drawText({colX, headerY}, "Domain", 0xFFFFFFFF);
    colX += size.x * colWidthDomain;
    
    renderer.drawText({colX, headerY}, "Context", 0xFFFFFFFF);
    
    contentY += headerHeight;
    
    // FIX: Early exit if no findings to prevent crashes
    if (filteredFindings.empty()) {
        renderer.drawText({position.x + size.x / 2 - 100, contentY + 50}, 
                         "No findings match current filter", 0xFF888888);
        renderer.popClipRect();
        return true;
    }
    
    // Draw table rows
    float tableHeight = size.y - (contentY - position.y);
    int visibleRows = static_cast<int>(tableHeight / rowHeight) + 1;
    int startRow = static_cast<int>(scrollOffset / rowHeight);
    
    // FIX: Clamp startRow to valid range
    startRow = std::max(0, std::min(startRow, static_cast<int>(filteredFindings.size()) - 1));
    
    for (int i = startRow; i < std::min(startRow + visibleRows, static_cast<int>(filteredFindings.size())); ++i) {
        // FIX: Extra bounds check
        if (i < 0 || i >= static_cast<int>(filteredFindings.size())) {
            continue;
        }
        
        const auto& finding = filteredFindings[i];
        
        float rowY = contentY + (i - startRow) * rowHeight - (scrollOffset - startRow * rowHeight);
        
        // Row background
        uint32_t rowColor = (i == selectedRow) ? 0xFF404060 : 
                           (i == hoveredRow) ? 0xFF353535 :
                           (i % 2 == 0) ? 0xFF2A2A2A : 0xFF252525;
        
        renderer.drawRect({position.x, rowY}, {size.x, rowHeight}, rowColor);
        
        // Draw columns
        colX = position.x + 5.0f;
        float textY = rowY + 8.0f;
        
        // Severity (colored)
        uint32_t sevColor = getSeverityColor(finding.severityScore);
        renderer.drawText({colX, textY}, finding.severity, sevColor);
        colX += size.x * colWidthSeverity;
        
        // Type
        renderer.drawText({colX, textY}, finding.tag, 0xFFCCCCCC);
        colX += size.x * colWidthType;
        
        // Match (truncate if too long)
        std::string match = finding.match;
        if (match.length() > 30) {
            match = match.substr(0, 27) + "...";
        }
        renderer.drawText({colX, textY}, match, 0xFFFFFFFF);
        colX += size.x * colWidthMatch;
        
        // Domain
        renderer.drawText({colX, textY}, finding.domain, 0xFF8888FF);
        colX += size.x * colWidthDomain;
        
        // Context (truncate)
        std::string context = finding.context;
        if (context.length() > 40) {
            context = context.substr(0, 37) + "...";
        }
        renderer.drawText({colX, textY}, context, 0xFF999999);
    }
    
    // Draw scrollbar if needed
    // FIX: Prevent division by zero
    float maxScroll = std::max(0.0f, filteredFindings.size() * rowHeight - tableHeight);
    if (maxScroll > 0 && !filteredFindings.empty()) {
        float scrollbarX = position.x + size.x - 10.0f;
        float scrollbarHeight = tableHeight;
        float totalContentHeight = filteredFindings.size() * rowHeight;
        
        // FIX: Prevent division by zero
        if (totalContentHeight > 0) {
            float thumbHeight = (tableHeight / totalContentHeight) * scrollbarHeight;
            thumbHeight = std::max(20.0f, thumbHeight); // Minimum thumb height
            
            float thumbY = contentY + (scrollOffset / maxScroll) * (scrollbarHeight - thumbHeight);
            renderer.drawRect({scrollbarX, thumbY}, {8.0f, thumbHeight}, 0xFF606060);
        }
    }
    
    if (showDetailPanel && selectedRow >= 0 && selectedRow < static_cast<int>(filteredFindings.size())) {
        drawDetailPanel(renderer);
    }
    
    renderer.popClipRect();
    return true;
}

void UIOsintResults::drawDetailPanel(OverlayRenderer& renderer) {
    // FIX: Safety check - ensure selectedRow is valid
    if (selectedRow < 0 || selectedRow >= static_cast<int>(filteredFindings.size())) {
        showDetailPanel = false;
        return;
    }
    
    // Detail panel dimensions (overlay on right side)
    float panelWidth = 500.0f;
    float panelHeight = size.y * 0.8f;
    float panelX = position.x + size.x - panelWidth - 20.0f;
    float panelY = position.y + titleBarHeight + 10.0f;
    
    // Background with shadow
    renderer.drawRect({panelX + 5, panelY + 5}, {panelWidth, panelHeight}, 0x80000000); // Shadow
    renderer.drawRect({panelX, panelY}, {panelWidth, panelHeight}, 0xF0252525); // Background
    
    // Border
    renderer.drawRect({panelX, panelY}, {panelWidth, 2}, 0xFF00DDFF);
    renderer.drawRect({panelX, panelY + panelHeight - 2}, {panelWidth, 2}, 0xFF00DDFF);
    renderer.drawRect({panelX, panelY}, {2, panelHeight}, 0xFF00DDFF);
    renderer.drawRect({panelX + panelWidth - 2, panelY}, {2, panelHeight}, 0xFF00DDFF);
    
    // Title bar
    renderer.drawRect({panelX, panelY}, {panelWidth, 30.0f}, 0xFF303030);
    renderer.drawText({panelX + 10, panelY + 8}, "Finding Details", 0xFFFFFFFF);
    
    // Close button hint
    renderer.drawText({panelX + panelWidth - 110, panelY + 8}, "[ESC to close]", 0xFF888888);
    
    float yPos = panelY + 40.0f;
    float xPad = panelX + 15.0f;
    float lineHeight = 25.0f;
    
    // Severity (color-coded)
    renderer.drawText({xPad, yPos}, "Severity:", 0xFFAAAAAA);
    uint32_t sevColor = getSeverityColor(selectedFinding.severityScore);
    renderer.drawText({xPad + 100, yPos}, selectedFinding.severity + " (" + std::to_string(selectedFinding.severityScore) + "/10)", sevColor);
    yPos += lineHeight;
    
    // Type
    renderer.drawText({xPad, yPos}, "Type:", 0xFFAAAAAA);
    renderer.drawText({xPad + 100, yPos}, selectedFinding.tag, 0xFFFFFFFF);
    yPos += lineHeight;
    
    // Domain
    renderer.drawText({xPad, yPos}, "Domain:", 0xFFAAAAAA);
    renderer.drawText({xPad + 100, yPos}, selectedFinding.domain, 0xFF8888FF);
    yPos += lineHeight;
    
    // Entropy (if available)
    if (selectedFinding.entropy > 0) {
        renderer.drawText({xPad, yPos}, "Entropy:", 0xFFAAAAAA);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << selectedFinding.entropy;
        renderer.drawText({xPad + 100, yPos}, oss.str(), 0xFFFFFFFF);
        yPos += lineHeight;
    }
    
    yPos += 10; // Spacing
    
    // Full URL (wrapped if needed)
    renderer.drawText({xPad, yPos}, "URL:", 0xFFFFAA00);
    yPos += lineHeight;
    
    // Word wrap the URL
    std::string url = selectedFinding.url;
    float maxWidth = panelWidth - 30.0f;
    size_t pos = 0;
    size_t charsPerLine = 60; // Approximate
    
    while (pos < url.length()) {
        std::string line = url.substr(pos, std::min(charsPerLine, url.length() - pos));
        renderer.drawText({xPad + 10, yPos}, line, 0xFFCCCCCC);
        yPos += 20;
        pos += charsPerLine;
    }
    
    yPos += 10;
    
    // ===== ACTUAL MATCH VALUE (FULL, UNTRUNCATED) =====
    renderer.drawText({xPad, yPos}, "MATCH (Full Value):", 0xFFFF4444);
    yPos += lineHeight;
    
    // Draw the FULL match value in a box
    float matchBoxHeight = 60.0f;
    renderer.drawRect({xPad, yPos}, {panelWidth - 30, matchBoxHeight}, 0xFF1A1A1A);
    renderer.drawRect({xPad, yPos}, {panelWidth - 30, 2}, 0xFFFF4444); // Top border
    
    // Word wrap the match value
    std::string match = selectedFinding.match;
    pos = 0;
    float matchY = yPos + 5;
    
    while (pos < match.length() && matchY < yPos + matchBoxHeight - 20) {
        std::string line = match.substr(pos, std::min(charsPerLine, match.length() - pos));
        renderer.drawText({xPad + 10, matchY}, line, 0xFFFFFF00); // Yellow for visibility
        matchY += 18;
        pos += charsPerLine;
    }
    
    yPos += matchBoxHeight + 15;
    
    // ===== FULL CONTEXT (SURROUNDING TEXT) =====
    renderer.drawText({xPad, yPos}, "CONTEXT (Full Text):", 0xFF00FF00);
    yPos += lineHeight;
    
    // Draw the FULL context in a box
    float contextBoxHeight = 80.0f;
    renderer.drawRect({xPad, yPos}, {panelWidth - 30, contextBoxHeight}, 0xFF1A1A1A);
    renderer.drawRect({xPad, yPos}, {panelWidth - 30, 2}, 0xFF00FF00); // Top border
    
    // Word wrap the context
    std::string context = selectedFinding.context;
    pos = 0;
    float contextY = yPos + 5;
    
    while (pos < context.length() && contextY < yPos + contextBoxHeight - 20) {
        std::string line = context.substr(pos, std::min(charsPerLine, context.length() - pos));
        renderer.drawText({xPad + 10, contextY}, line, 0xFFCCCCCC);
        contextY += 18;
        pos += charsPerLine;
    }
    
    yPos += contextBoxHeight + 15;
    
    // Warning for critical findings
    if (selectedFinding.severityScore >= 9) {
        yPos += 10;
        renderer.drawRect({xPad, yPos}, {panelWidth - 30, 50}, 0xFFAA2020);
        renderer.drawText({xPad + 10, yPos + 5}, "! CRITICAL: This is sensitive data!", 0xFFFFFFFF);
        renderer.drawText({xPad + 10, yPos + 25}, "Action required:", 0xFFFFFFFF);
        renderer.drawText({xPad + 10, yPos + 40}, "- Rotate/revoke this credential immediately", 0xFFFFFFFF);
    }
}
