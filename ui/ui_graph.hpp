#pragma once
#include "helpers/widget.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>

// Forward declarations
class OverlayRenderer;
struct InputState;

// ============================================================
// Graph Types
// ============================================================
enum class GraphType {
    Line,           // Line graph with optional fill
    Bar,            // Vertical bar chart
    HorizontalBar,  // Horizontal bar chart
    Scatter,        // Scatter plot
    Area,           // Area chart (filled line graph)
    Pie,            // Pie chart
    Donut,          // Donut chart (pie with center hole)
    StackedBar,     // Stacked bar chart
    MultiLine       // Multiple line series on same graph
};

// ============================================================
// Data Point Structure
// ============================================================
struct DataPoint {
    float value;
    std::string label;
    uint32_t color;  // For per-point coloring (optional)
    
    DataPoint() : value(0.0f), label(""), color(0xFFFFFFFF) {}
    DataPoint(float v) : value(v), label(""), color(0xFFFFFFFF) {}
    DataPoint(float v, const std::string& lbl) : value(v), label(lbl), color(0xFFFFFFFF) {}
    DataPoint(float v, const std::string& lbl, uint32_t col) : value(v), label(lbl), color(col) {}
};

// ============================================================
// Data Series (for multi-line graphs)
// ============================================================
struct DataSeries {
    std::string name;
    std::vector<DataPoint> data;
    uint32_t color;
    bool visible;
    
    DataSeries() : name(""), color(0xFFEAEAEA), visible(true) {}
    DataSeries(const std::string& n, uint32_t col = 0xFFEAEAEA) 
        : name(n), color(col), visible(true) {}
};

// ============================================================
// Graph Configuration
// ============================================================
struct GraphConfig {
    // Display options
    bool showGrid = true;
    bool showAxes = true;
    bool showLabels = true;
    bool showLegend = false;
    bool showValues = false;
    bool animated = false;
    
    // Axis options
    bool autoScale = true;
    float minValue = 0.0f;
    float maxValue = 100.0f;
    int gridLines = 5;
    
    // Visual options
    float lineThickness = 2.0f;
    float pointRadius = 4.0f;
    float barWidth = 0.8f;  // As fraction of available space (0.0-1.0)
    float donutThickness = 0.4f;  // As fraction of radius (0.0-1.0)
    
    // Colors  [GLASS_PHASE5]
    uint32_t backgroundColor = 0xF01A1A1A;
    uint32_t gridColor = 0x10FFFFFF;
    uint32_t axisColor = 0x15FFFFFF;
    uint32_t textColor = 0xFF909090;
    uint32_t primaryColor = 0xFF6B8CFF;  // Default data color
    
    // Padding
    float paddingLeft = 50.0f;
    float paddingRight = 20.0f;
    float paddingTop = 30.0f;
    float paddingBottom = 40.0f;
    
    // Performance options
    int maxDataPoints = 500;  // Automatic downsampling if exceeded
    bool useDownsampling = true;
};

// ============================================================
// UIGraph - Modular Graph Widget
// ============================================================
class UIGraph : public Widget {
public:
    UIGraph(const std::string& title = "Graph", GraphType type = GraphType::Line);
    virtual ~UIGraph() = default;

    // ========================================
    // Data Management
    // ========================================
    
    // Single series graphs (Line, Bar, Area, Scatter, Pie, Donut)
    void setData(const std::vector<DataPoint>& data);
    void addDataPoint(const DataPoint& point);
    void addDataPoint(float value, const std::string& label = "");
    void clearData();
    
    // Multi-series graphs (MultiLine, StackedBar)
    void addSeries(const DataSeries& series);
    void addSeries(const std::string& name, const std::vector<DataPoint>& data, uint32_t color);
    void clearSeries();
    void setSeriesVisible(const std::string& name, bool visible);
    
    // ========================================
    // Configuration
    // ========================================
    
    void setGraphType(GraphType type);
    GraphType getGraphType() const { return m_type; }
    
    void setTitle(const std::string& title) { m_title = title; }
    std::string getTitle() const { return m_title; }
    
    void setConfig(const GraphConfig& config) { m_config = config; }
    GraphConfig& getConfig() { return m_config; }
    
    void setAxisRange(float minVal, float maxVal);
    void enableAutoScale(bool enable) { m_config.autoScale = enable; }
    
    // ========================================
    // Widget Interface
    // ========================================
    
    void update(const InputState& input, float dt) override;
    void draw(class UIRenderer& renderer) override { /* unused */ }
    void drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) override;

    // ========================================
    // Interaction
    // ========================================
    
    void setOnPointHover(std::function<void(int index, const DataPoint& point)> callback);
    void setOnPointClick(std::function<void(int index, const DataPoint& point)> callback);
    
private:
    // ========================================
    // Core Data
    // ========================================
    
    std::string m_title;
    GraphType m_type;
    GraphConfig m_config;
    
    std::vector<DataPoint> m_data;          // Single series data
    std::vector<DataSeries> m_series;       // Multi-series data
    
    // ========================================
    // Interaction State
    // ========================================
    
    int m_hoveredIndex = -1;
    float m_animationProgress = 0.0f;
    bool m_isDragging = false;
    Vec2 m_dragStart{0, 0};
    
    std::function<void(int, const DataPoint&)> m_onPointHover;
    std::function<void(int, const DataPoint&)> m_onPointClick;
    
    // ========================================
    // Rendering Helpers
    // ========================================
    
    void drawBackground(OverlayRenderer& renderer);
    void drawTitle(OverlayRenderer& renderer);
    void drawAxes(OverlayRenderer& renderer);
    void drawGrid(OverlayRenderer& renderer);
    void drawLegend(OverlayRenderer& renderer);
    
    // Graph type specific rendering
    void drawLineGraph(OverlayRenderer& renderer);
    void drawBarGraph(OverlayRenderer& renderer);
    void drawHorizontalBarGraph(OverlayRenderer& renderer);
    void drawScatterGraph(OverlayRenderer& renderer);
    void drawAreaGraph(OverlayRenderer& renderer);
    void drawPieChart(OverlayRenderer& renderer);
    void drawDonutChart(OverlayRenderer& renderer);
    void drawStackedBarGraph(OverlayRenderer& renderer);
    void drawMultiLineGraph(OverlayRenderer& renderer);
    
    // ========================================
    // Utility Functions
    // ========================================
    
    Vec2 getGraphArea() const;
    Vec2 getGraphOrigin() const;
    float mapValueToY(float value) const;
    float mapValueToX(float index, int totalPoints) const;
    void calculateAutoScale();
    std::vector<DataPoint> downsampleData(const std::vector<DataPoint>& data, int maxPoints) const;
    bool isPointInCircle(const Vec2& point, const Vec2& center, float radius) const;
    int findHoveredPoint(const Vec2& mousePos) const;
    
    // Color generation for multi-series
    uint32_t generateSeriesColor(int index) const;
};
