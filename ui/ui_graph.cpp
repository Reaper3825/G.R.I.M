#include "ui_graph.hpp"
#include "overlay_renderer.hpp"
#include "core/input_parser.hpp"
#include "helpers/mouse.hpp"
#include "logger.hpp"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Constructor
// ============================================================

UIGraph::UIGraph(const std::string& title, GraphType type)
    : m_title(title), m_type(type)
{
    size = {400.0f, 300.0f};  // Default size
    
    LOG_DEBUG("UIGraph", "Created graph: " + title + " of type " + std::to_string(static_cast<int>(type)));
}

// ============================================================
// Data Management - Single Series
// ============================================================

void UIGraph::setData(const std::vector<DataPoint>& data) {
    m_data = data;
    
    // Apply downsampling if needed
    if (m_config.useDownsampling && m_data.size() > static_cast<size_t>(m_config.maxDataPoints)) {
        m_data = downsampleData(m_data, m_config.maxDataPoints);
        LOG_DEBUG("UIGraph", "Downsampled data from " + std::to_string(data.size()) + 
                  " to " + std::to_string(m_data.size()) + " points");
    }
    
    if (m_config.autoScale) {
        calculateAutoScale();
    }
}

void UIGraph::addDataPoint(const DataPoint& point) {
    m_data.push_back(point);
    
    // Apply downsampling if exceeded limit
    if (m_config.useDownsampling && m_data.size() > static_cast<size_t>(m_config.maxDataPoints)) {
        m_data = downsampleData(m_data, m_config.maxDataPoints);
    }
    
    if (m_config.autoScale) {
        calculateAutoScale();
    }
}

void UIGraph::addDataPoint(float value, const std::string& label) {
    addDataPoint(DataPoint(value, label));
}

void UIGraph::clearData() {
    m_data.clear();
    m_hoveredIndex = -1;
}

// ============================================================
// Data Management - Multi Series
// ============================================================

void UIGraph::addSeries(const DataSeries& series) {
    m_series.push_back(series);
    
    // Apply downsampling to series if needed
    if (m_config.useDownsampling && m_series.back().data.size() > static_cast<size_t>(m_config.maxDataPoints)) {
        m_series.back().data = downsampleData(m_series.back().data, m_config.maxDataPoints);
    }
    
    if (m_config.autoScale) {
        calculateAutoScale();
    }
}

void UIGraph::addSeries(const std::string& name, const std::vector<DataPoint>& data, uint32_t color) {
    DataSeries series;
    series.name = name;
    series.data = data;
    series.color = color;
    series.visible = true;
    addSeries(series);
}

void UIGraph::clearSeries() {
    m_series.clear();
    m_hoveredIndex = -1;
}

void UIGraph::setSeriesVisible(const std::string& name, bool visible) {
    for (auto& series : m_series) {
        if (series.name == name) {
            series.visible = visible;
            break;
        }
    }
}

// ============================================================
// Configuration
// ============================================================

void UIGraph::setGraphType(GraphType type) {
    m_type = type;
    m_hoveredIndex = -1;  // Reset hover state
}

void UIGraph::setAxisRange(float minVal, float maxVal) {
    m_config.minValue = minVal;
    m_config.maxValue = maxVal;
    m_config.autoScale = false;
}

void UIGraph::setOnPointHover(std::function<void(int, const DataPoint&)> callback) {
    m_onPointHover = std::move(callback);
}

void UIGraph::setOnPointClick(std::function<void(int, const DataPoint&)> callback) {
    m_onPointClick = std::move(callback);
}

// ============================================================
// Update
// ============================================================

void UIGraph::update(const InputState& input, float dt) {
    if (!visible) return;
    
    Vec2 m = input.mousePos;
    
    // Check if mouse is over graph area
    bool inBounds = (m.x >= position.x && m.x <= position.x + size.x &&
                     m.y >= position.y && m.y <= position.y + size.y);
    
    if (inBounds) {
        // Update hover state
        int oldHovered = m_hoveredIndex;
        m_hoveredIndex = findHoveredPoint(m);
        
        // Trigger hover callback if changed
        if (m_hoveredIndex != oldHovered && m_hoveredIndex >= 0 && m_onPointHover) {
            if (m_type == GraphType::MultiLine && !m_series.empty()) {
                // For multi-line, we'd need more complex logic - simplified here
                if (m_hoveredIndex < static_cast<int>(m_series[0].data.size())) {
                    m_onPointHover(m_hoveredIndex, m_series[0].data[m_hoveredIndex]);
                }
            } else if (m_hoveredIndex < static_cast<int>(m_data.size())) {
                m_onPointHover(m_hoveredIndex, m_data[m_hoveredIndex]);
            }
        }
        
        // Handle click
        if (Mouse::wasPressed(MouseButton::Left) && m_hoveredIndex >= 0 && m_onPointClick) {
            if (m_type == GraphType::MultiLine && !m_series.empty()) {
                if (m_hoveredIndex < static_cast<int>(m_series[0].data.size())) {
                    m_onPointClick(m_hoveredIndex, m_series[0].data[m_hoveredIndex]);
                }
            } else if (m_hoveredIndex < static_cast<int>(m_data.size())) {
                m_onPointClick(m_hoveredIndex, m_data[m_hoveredIndex]);
            }
        }
    } else {
        m_hoveredIndex = -1;
    }
    
    // Update animation
    if (m_config.animated && m_animationProgress < 1.0f) {
        m_animationProgress += dt * 2.0f;  // 0.5 second animation
        if (m_animationProgress > 1.0f) m_animationProgress = 1.0f;
    }
}

// ============================================================
// Main Draw Function
// ============================================================

void UIGraph::drawOverlay(OverlayRenderer& renderer, const Vec2& panelPos) {
    if (!visible) return;
    
    drawBackground(renderer);
    drawTitle(renderer);
    
    if (m_config.showGrid) {
        drawGrid(renderer);
    }
    
    if (m_config.showAxes) {
        drawAxes(renderer);
    }
    
    // Draw graph based on type
    switch (m_type) {
        case GraphType::Line:
            drawLineGraph(renderer);
            break;
        case GraphType::Bar:
            drawBarGraph(renderer);
            break;
        case GraphType::HorizontalBar:
            drawHorizontalBarGraph(renderer);
            break;
        case GraphType::Scatter:
            drawScatterGraph(renderer);
            break;
        case GraphType::Area:
            drawAreaGraph(renderer);
            break;
        case GraphType::Pie:
            drawPieChart(renderer);
            break;
        case GraphType::Donut:
            drawDonutChart(renderer);
            break;
        case GraphType::StackedBar:
            drawStackedBarGraph(renderer);
            break;
        case GraphType::MultiLine:
            drawMultiLineGraph(renderer);
            break;
    }
    
    if (m_config.showLegend && (m_type == GraphType::MultiLine || m_type == GraphType::Area || m_type == GraphType::Pie || m_type == GraphType::Donut)) {
        drawLegend(renderer);
    }
}

// ============================================================
// Background & Title
// ============================================================

void UIGraph::drawBackground(OverlayRenderer& renderer) {
    renderer.drawRect(position, size, m_config.backgroundColor);
    
    // Border
    renderer.drawRect(position, {size.x, 2}, m_config.axisColor);
    renderer.drawRect(position, {2, size.y}, m_config.axisColor);
    renderer.drawRect({position.x, position.y + size.y - 2}, {size.x, 2}, m_config.axisColor);
    renderer.drawRect({position.x + size.x - 2, position.y}, {2, size.y}, m_config.axisColor);
}

void UIGraph::drawTitle(OverlayRenderer& renderer) {
    if (!m_title.empty()) {
        Vec2 titlePos = {position.x + size.x / 2.0f - (m_title.length() * 4.0f), position.y + 8};
        renderer.drawText(titlePos, m_title, m_config.textColor);
    }
}

// ============================================================
// Axes & Grid
// ============================================================

void UIGraph::drawAxes(OverlayRenderer& renderer) {
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    // Y-axis
    renderer.drawRect({origin.x, origin.y}, {2, graphArea.y}, m_config.axisColor);
    
    // X-axis
    renderer.drawRect({origin.x, origin.y + graphArea.y}, {graphArea.x, 2}, m_config.axisColor);
}

void UIGraph::drawGrid(OverlayRenderer& renderer) {
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    // Horizontal grid lines (with Y-axis values)
    for (int i = 0; i <= m_config.gridLines; ++i) {
        float y = origin.y + (graphArea.y / m_config.gridLines) * i;
        renderer.drawRect({origin.x, y}, {graphArea.x, 1}, m_config.gridColor);
        
        // Y-axis labels
        if (m_config.showLabels) {
            float value = m_config.maxValue - ((m_config.maxValue - m_config.minValue) / m_config.gridLines) * i;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << value;
            renderer.drawText({position.x + 5, y - 6}, oss.str(), m_config.textColor);
        }
    }
    
    // Vertical grid lines (with X-axis labels)
    int xGridLines = 5;  // Show 5 vertical lines
    for (int i = 0; i <= xGridLines; ++i) {
        float x = origin.x + (graphArea.x / xGridLines) * i;
        renderer.drawRect({x, origin.y}, {1, graphArea.y}, m_config.gridColor);
        
        // X-axis labels (show time or index)
        if (m_config.showLabels && i > 0) {
            std::ostringstream oss;
            if (!m_series.empty() && !m_series[0].data.empty()) {
                size_t dataSize = m_series[0].data.size();
                size_t index = (dataSize * i) / xGridLines;
                if (index < dataSize && !m_series[0].data[index].label.empty()) {
                    oss << m_series[0].data[index].label;
                } else {
                    oss << index;
                }
            } else if (!m_data.empty()) {
                size_t index = (m_data.size() * i) / xGridLines;
                if (index < m_data.size() && !m_data[index].label.empty()) {
                    oss << m_data[index].label;
                } else {
                    oss << index;
                }
            } else {
                oss << i;
            }
            renderer.drawText({x - 10, origin.y + graphArea.y + 5}, oss.str(), m_config.textColor);
        }
    }
}

void UIGraph::drawLegend(OverlayRenderer& renderer) {
    if (m_series.empty() && m_data.empty()) return;
    
    // Position legend horizontally at bottom-left, below the graph area
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    Vec2 legendPos = {origin.x, origin.y + graphArea.y + 10};
    float entryWidth = 100.0f;  // Width per entry
    float entryHeight = 20.0f;
    
    // Calculate background size for horizontal layout
    size_t numEntries = m_series.empty() ? m_data.size() : m_series.size();
    float legendWidth = numEntries * entryWidth + 10;
    
    // Background
    renderer.drawRect(legendPos, {legendWidth, entryHeight + 10}, 0xCC000000);
    
    // Entries (horizontal layout)
    if (m_type == GraphType::MultiLine || m_type == GraphType::StackedBar || m_type == GraphType::Area) {
        for (size_t i = 0; i < m_series.size(); ++i) {
            Vec2 entryPos = {legendPos.x + 5 + i * entryWidth, legendPos.y + 5};
            
            // Color box
            renderer.drawRect(entryPos, {12, 12}, m_series[i].color);
            
            // Label
            renderer.drawText({entryPos.x + 18, entryPos.y}, m_series[i].name, m_config.textColor);
        }
    } else if (m_type == GraphType::Pie || m_type == GraphType::Donut) {
        for (size_t i = 0; i < m_data.size() && i < 10; ++i) {  // Limit to 10 entries
            Vec2 entryPos = {legendPos.x + 5, legendPos.y + 5 + i * entryHeight};
            
            // Color box
            uint32_t color = m_data[i].color != 0xFFFFFFFF ? m_data[i].color : generateSeriesColor(i);
            renderer.drawRect(entryPos, {12, 12}, color);
            
            // Label
            std::string label = m_data[i].label.empty() ? ("Item " + std::to_string(i + 1)) : m_data[i].label;
            renderer.drawText({entryPos.x + 18, entryPos.y}, label, m_config.textColor);
        }
    }
}

// ============================================================
// Line Graph
// ============================================================

void UIGraph::drawLineGraph(OverlayRenderer& renderer) {
    if (m_data.empty()) return;
    
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    // Draw lines between points
    for (size_t i = 1; i < m_data.size(); ++i) {
        float x1 = mapValueToX(i - 1, m_data.size());
        float y1 = mapValueToY(m_data[i - 1].value);
        float x2 = mapValueToX(i, m_data.size());
        float y2 = mapValueToY(m_data[i].value);
        
        // Simple line rendering (draw as thick rect)
        float thickness = m_config.lineThickness;
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len = std::sqrt(dx * dx + dy * dy);
        
        if (len > 0.01f) {
            // Draw line as series of small segments
            int segments = static_cast<int>(len / 2.0f) + 1;
            for (int s = 0; s < segments; ++s) {
                float t = static_cast<float>(s) / segments;
                float x = x1 + dx * t;
                float y = y1 + dy * t;
                renderer.drawRect({x - thickness / 2, y - thickness / 2}, 
                                  {thickness, thickness}, m_config.primaryColor);
            }
        }
    }
    
    // Draw points
    for (size_t i = 0; i < m_data.size(); ++i) {
        float x = mapValueToX(i, m_data.size());
        float y = mapValueToY(m_data[i].value);
        
        float radius = (m_hoveredIndex == static_cast<int>(i)) ? m_config.pointRadius * 1.5f : m_config.pointRadius;
        uint32_t pointColor = (m_hoveredIndex == static_cast<int>(i)) ? 0xFFFFFF00 : 0xFFFFFFFF;
        
        // Draw point as small square
        renderer.drawRect({x - radius, y - radius}, {radius * 2, radius * 2}, pointColor);
        
        // Show value on hover
        if (m_hoveredIndex == static_cast<int>(i) && m_config.showValues) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << m_data[i].value;
            renderer.drawText({x + 10, y - 10}, oss.str(), 0xFFFFFF00);
        }
    }
}

// ============================================================
// Bar Graph
// ============================================================

void UIGraph::drawBarGraph(OverlayRenderer& renderer) {
    if (m_data.empty()) return;
    
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    float barSpacing = graphArea.x / m_data.size();
    float barWidth = barSpacing * m_config.barWidth;
    
    for (size_t i = 0; i < m_data.size(); ++i) {
        float x = origin.x + barSpacing * i + (barSpacing - barWidth) / 2;
        float barHeight = (m_data[i].value - m_config.minValue) / (m_config.maxValue - m_config.minValue) * graphArea.y;
        float y = origin.y + graphArea.y - barHeight;
        
        // Apply animation
        if (m_config.animated) {
            barHeight *= m_animationProgress;
            y = origin.y + graphArea.y - barHeight;
        }
        
        uint32_t barColor = (m_hoveredIndex == static_cast<int>(i)) ? 0xFFFFAA00 : m_config.primaryColor;
        renderer.drawRect({x, y}, {barWidth, barHeight}, barColor);
        
        // Draw label
        if (m_config.showLabels && !m_data[i].label.empty()) {
            renderer.drawText({x, origin.y + graphArea.y + 5}, m_data[i].label, m_config.textColor);
        }
        
        // Show value
        if (m_config.showValues) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << m_data[i].value;
            renderer.drawText({x, y - 15}, oss.str(), m_config.textColor);
        }
    }
}

// ============================================================
// Horizontal Bar Graph
// ============================================================

void UIGraph::drawHorizontalBarGraph(OverlayRenderer& renderer) {
    if (m_data.empty()) return;
    
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    float barSpacing = graphArea.y / m_data.size();
    float barHeight = barSpacing * m_config.barWidth;
    
    for (size_t i = 0; i < m_data.size(); ++i) {
        float y = origin.y + barSpacing * i + (barSpacing - barHeight) / 2;
        float barWidth = (m_data[i].value - m_config.minValue) / (m_config.maxValue - m_config.minValue) * graphArea.x;
        float x = origin.x;
        
        // Apply animation
        if (m_config.animated) {
            barWidth *= m_animationProgress;
        }
        
        uint32_t barColor = (m_hoveredIndex == static_cast<int>(i)) ? 0xFFFFAA00 : m_config.primaryColor;
        renderer.drawRect({x, y}, {barWidth, barHeight}, barColor);
        
        // Draw label
        if (m_config.showLabels && !m_data[i].label.empty()) {
            renderer.drawText({position.x + 5, y + barHeight / 2 - 6}, m_data[i].label, m_config.textColor);
        }
        
        // Show value
        if (m_config.showValues) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << m_data[i].value;
            renderer.drawText({x + barWidth + 5, y + barHeight / 2 - 6}, oss.str(), m_config.textColor);
        }
    }
}

// ============================================================
// Scatter Graph
// ============================================================

void UIGraph::drawScatterGraph(OverlayRenderer& renderer) {
    if (m_data.empty()) return;
    
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    for (size_t i = 0; i < m_data.size(); ++i) {
        float x = mapValueToX(i, m_data.size());
        float y = mapValueToY(m_data[i].value);
        
        float radius = (m_hoveredIndex == static_cast<int>(i)) ? m_config.pointRadius * 1.5f : m_config.pointRadius;
        uint32_t color = m_data[i].color != 0xFFFFFFFF ? m_data[i].color : m_config.primaryColor;
        if (m_hoveredIndex == static_cast<int>(i)) color = 0xFFFFFF00;
        
        // Draw point
        renderer.drawRect({x - radius, y - radius}, {radius * 2, radius * 2}, color);
        
        // Show value on hover
        if (m_hoveredIndex == static_cast<int>(i) && m_config.showValues) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << m_data[i].value;
            if (!m_data[i].label.empty()) {
                oss << " (" << m_data[i].label << ")";
            }
            renderer.drawText({x + 10, y - 10}, oss.str(), 0xFFFFFF00);
        }
    }
}

// ============================================================
// Area Graph (Filled Line)
// ============================================================

void UIGraph::drawAreaGraph(OverlayRenderer& renderer) {
    // Handle multi-series area chart (stacked)
    if (!m_series.empty()) {
        Vec2 origin = getGraphOrigin();
        Vec2 graphArea = getGraphArea();
        
        // Find max data length across all series
        size_t maxLen = 0;
        for (const auto& series : m_series) {
            maxLen = std::max(maxLen, series.data.size());
        }
        
        if (maxLen == 0) return;
        
        // Draw each series as filled area (reverse order for proper stacking)
        for (int s = static_cast<int>(m_series.size()) - 1; s >= 0; --s) {
            const auto& series = m_series[s];
            if (series.data.empty()) continue;
            
            // Draw filled area
            for (size_t i = 0; i < series.data.size(); ++i) {
                float x = mapValueToX(static_cast<float>(i), static_cast<float>(maxLen));
                float y = mapValueToY(series.data[i].value);
                float baseY = origin.y + graphArea.y;
                float fillHeight = baseY - y;
                
                // Semi-transparent fill with series color
                uint32_t fillColor = (series.color & 0x00FFFFFF) | 0x80000000;  // More opaque
                renderer.drawRect({x - 1, y}, {3, fillHeight}, fillColor);
            }
            
            // Draw line on top of area
            for (size_t i = 0; i < series.data.size() - 1; ++i) {
                float x1 = mapValueToX(static_cast<float>(i), static_cast<float>(maxLen));
                float y1 = mapValueToY(series.data[i].value);
                float x2 = mapValueToX(static_cast<float>(i + 1), static_cast<float>(maxLen));
                float y2 = mapValueToY(series.data[i + 1].value);
                
                renderer.drawLine({x1, y1}, {x2, y2}, series.color, 2.0f);
            }
        }
        
        // Draw legend for multi-series
        drawLegend(renderer);
        return;
    }
    
    // Fallback to single-series
    if (m_data.empty()) return;
    
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    // Draw filled area under line
    for (size_t i = 0; i < m_data.size(); ++i) {
        float x = mapValueToX(i, m_data.size());
        float y = mapValueToY(m_data[i].value);
        float fillHeight = origin.y + graphArea.y - y;
        
        // Semi-transparent fill
        uint32_t fillColor = (m_config.primaryColor & 0x00FFFFFF) | 0x40000000;
        renderer.drawRect({x, y}, {2, fillHeight}, fillColor);
    }
    
    // Draw line on top
    drawLineGraph(renderer);
}

// ============================================================
// Pie Chart
// ============================================================

void UIGraph::drawPieChart(OverlayRenderer& renderer) {
    if (m_data.empty()) return;
    
    Vec2 center = {position.x + size.x / 2, position.y + size.y / 2};
    float radius = std::min(size.x, size.y) * 0.35f;
    
    // Calculate total
    float total = 0.0f;
    for (const auto& point : m_data) {
        total += point.value;
    }
    
    if (total <= 0.0f) return;
    
    // Draw slices
    float currentAngle = -M_PI / 2;  // Start at top
    
    for (size_t i = 0; i < m_data.size(); ++i) {
        float sliceAngle = (m_data[i].value / total) * 2 * M_PI;
        
        uint32_t color = m_data[i].color != 0xFFFFFFFF ? m_data[i].color : generateSeriesColor(i);
        if (m_hoveredIndex == static_cast<int>(i)) {
            color = 0xFFFFFF00;  // Highlight hovered
        }
        
        // Draw slice as filled triangle fan (approximated with rectangles)
        int segments = std::max(2, static_cast<int>(sliceAngle / (M_PI / 32)));
        for (int s = 0; s < segments; ++s) {
            float angle1 = currentAngle + (sliceAngle / segments) * s;
            float angle2 = currentAngle + (sliceAngle / segments) * (s + 1);
            
            float x1 = center.x + std::cos(angle1) * radius;
            float y1 = center.y + std::sin(angle1) * radius;
            float x2 = center.x + std::cos(angle2) * radius;
            float y2 = center.y + std::sin(angle2) * radius;
            
            // Draw line from center to edge
            float steps = radius / 2;
            for (int r = 0; r < steps; ++r) {
                float t = r / steps;
                float xa = center.x + (x1 - center.x) * t;
                float ya = center.y + (y1 - center.y) * t;
                float xb = center.x + (x2 - center.x) * t;
                float yb = center.y + (y2 - center.y) * t;
                
                renderer.drawRect({xa, ya}, {xb - xa + 1, yb - ya + 1}, color);
            }
        }
        
        // Show percentage on hover
        if (m_hoveredIndex == static_cast<int>(i) && m_config.showValues) {
            float midAngle = currentAngle + sliceAngle / 2;
            float labelX = center.x + std::cos(midAngle) * radius * 0.7f;
            float labelY = center.y + std::sin(midAngle) * radius * 0.7f;
            
            float percentage = (m_data[i].value / total) * 100.0f;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << percentage << "%";
            renderer.drawText({labelX - 15, labelY - 6}, oss.str(), 0xFF000000);
        }
        
        currentAngle += sliceAngle;
    }
}

// ============================================================
// Donut Chart
// ============================================================

void UIGraph::drawDonutChart(OverlayRenderer& renderer) {
    if (m_data.empty()) return;
    
    // Draw pie chart first
    drawPieChart(renderer);
    
    // Draw center hole
    Vec2 center = {position.x + size.x / 2, position.y + size.y / 2};
    float outerRadius = std::min(size.x, size.y) * 0.35f;
    float innerRadius = outerRadius * (1.0f - m_config.donutThickness);
    
    // Fill center with background color
    for (float r = 0; r < innerRadius; r += 1.0f) {
        int segments = std::max(8, static_cast<int>(r * 2 * M_PI / 4));
        for (int i = 0; i < segments; ++i) {
            float angle = (2 * M_PI / segments) * i;
            float x = center.x + std::cos(angle) * r;
            float y = center.y + std::sin(angle) * r;
            renderer.drawRect({x, y}, {2, 2}, m_config.backgroundColor);
        }
    }
}

// ============================================================
// Stacked Bar Graph
// ============================================================

void UIGraph::drawStackedBarGraph(OverlayRenderer& renderer) {
    if (m_series.empty()) return;
    
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    // Find max data points across all series
    size_t maxPoints = 0;
    for (const auto& series : m_series) {
        maxPoints = std::max(maxPoints, series.data.size());
    }
    
    float barSpacing = graphArea.x / maxPoints;
    float barWidth = barSpacing * m_config.barWidth;
    
    // Calculate totals for scaling
    std::vector<float> totals(maxPoints, 0.0f);
    for (const auto& series : m_series) {
        if (!series.visible) continue;
        for (size_t i = 0; i < series.data.size(); ++i) {
            totals[i] += series.data[i].value;
        }
    }
    
    // Draw stacked bars
    for (size_t i = 0; i < maxPoints; ++i) {
        float x = origin.x + barSpacing * i + (barSpacing - barWidth) / 2;
        float cumulativeHeight = 0.0f;
        
        for (const auto& series : m_series) {
            if (!series.visible || i >= series.data.size()) continue;
            
            float barHeight = (series.data[i].value / (m_config.maxValue - m_config.minValue)) * graphArea.y;
            float y = origin.y + graphArea.y - cumulativeHeight - barHeight;
            
            renderer.drawRect({x, y}, {barWidth, barHeight}, series.color);
            
            cumulativeHeight += barHeight;
        }
    }
}

// ============================================================
// Multi-Line Graph
// ============================================================

void UIGraph::drawMultiLineGraph(OverlayRenderer& renderer) {
    if (m_series.empty()) return;
    
    for (const auto& series : m_series) {
        if (!series.visible || series.data.empty()) continue;
        
        // Draw lines
        for (size_t i = 1; i < series.data.size(); ++i) {
            float x1 = mapValueToX(i - 1, series.data.size());
            float y1 = mapValueToY(series.data[i - 1].value);
            float x2 = mapValueToX(i, series.data.size());
            float y2 = mapValueToY(series.data[i].value);
            
            float thickness = m_config.lineThickness;
            float dx = x2 - x1;
            float dy = y2 - y1;
            float len = std::sqrt(dx * dx + dy * dy);
            
            if (len > 0.01f) {
                int segments = static_cast<int>(len / 2.0f) + 1;
                for (int s = 0; s < segments; ++s) {
                    float t = static_cast<float>(s) / segments;
                    float x = x1 + dx * t;
                    float y = y1 + dy * t;
                    renderer.drawRect({x - thickness / 2, y - thickness / 2}, 
                                      {thickness, thickness}, series.color);
                }
            }
        }
        
        // Draw points
        for (size_t i = 0; i < series.data.size(); ++i) {
            float x = mapValueToX(i, series.data.size());
            float y = mapValueToY(series.data[i].value);
            
            float radius = m_config.pointRadius;
            renderer.drawRect({x - radius, y - radius}, {radius * 2, radius * 2}, series.color);
        }
    }
}

// ============================================================
// Utility Functions
// ============================================================

Vec2 UIGraph::getGraphArea() const {
    return {
        size.x - m_config.paddingLeft - m_config.paddingRight,
        size.y - m_config.paddingTop - m_config.paddingBottom
    };
}

Vec2 UIGraph::getGraphOrigin() const {
    return {
        position.x + m_config.paddingLeft,
        position.y + m_config.paddingTop
    };
}

float UIGraph::mapValueToY(float value) const {
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    float normalized = (value - m_config.minValue) / (m_config.maxValue - m_config.minValue);
    return origin.y + graphArea.y - (normalized * graphArea.y);
}

float UIGraph::mapValueToX(float index, int totalPoints) const {
    Vec2 origin = getGraphOrigin();
    Vec2 graphArea = getGraphArea();
    
    if (totalPoints <= 1) return origin.x;
    
    return origin.x + (index / (totalPoints - 1)) * graphArea.x;
}

void UIGraph::calculateAutoScale() {
    if (m_data.empty() && m_series.empty()) {
        m_config.minValue = 0.0f;
        m_config.maxValue = 100.0f;
        return;
    }
    
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    
    // Check single series data
    for (const auto& point : m_data) {
        minVal = std::min(minVal, point.value);
        maxVal = std::max(maxVal, point.value);
    }
    
    // Check multi-series data
    for (const auto& series : m_series) {
        for (const auto& point : series.data) {
            minVal = std::min(minVal, point.value);
            maxVal = std::max(maxVal, point.value);
        }
    }
    
    // Add 10% padding
    float range = maxVal - minVal;
    if (range < 0.001f) range = 1.0f;  // Avoid division by zero
    
    m_config.minValue = minVal - range * 0.1f;
    m_config.maxValue = maxVal + range * 0.1f;
}

std::vector<DataPoint> UIGraph::downsampleData(const std::vector<DataPoint>& data, int maxPoints) const {
    if (data.size() <= static_cast<size_t>(maxPoints)) {
        return data;
    }
    
    std::vector<DataPoint> downsampled;
    downsampled.reserve(maxPoints);
    
    // Simple decimation - take every Nth point
    int step = data.size() / maxPoints;
    for (size_t i = 0; i < data.size(); i += step) {
        downsampled.push_back(data[i]);
    }
    
    return downsampled;
}

bool UIGraph::isPointInCircle(const Vec2& point, const Vec2& center, float radius) const {
    float dx = point.x - center.x;
    float dy = point.y - center.y;
    return (dx * dx + dy * dy) <= (radius * radius);
}

int UIGraph::findHoveredPoint(const Vec2& mousePos) const {
    // Different logic for different graph types
    if (m_type == GraphType::Pie || m_type == GraphType::Donut) {
        Vec2 center = {position.x + size.x / 2, position.y + size.y / 2};
        float dx = mousePos.x - center.x;
        float dy = mousePos.y - center.y;
        float dist = std::sqrt(dx * dx + dy * dy);
        float radius = std::min(size.x, size.y) * 0.35f;
        
        if (dist > radius) return -1;
        
        // Calculate angle
        float angle = std::atan2(dy, dx) + M_PI / 2;
        if (angle < 0) angle += 2 * M_PI;
        
        // Find which slice
        float total = 0.0f;
        for (const auto& point : m_data) {
            total += point.value;
        }
        
        float currentAngle = 0.0f;
        for (size_t i = 0; i < m_data.size(); ++i) {
            float sliceAngle = (m_data[i].value / total) * 2 * M_PI;
            if (angle >= currentAngle && angle < currentAngle + sliceAngle) {
                return static_cast<int>(i);
            }
            currentAngle += sliceAngle;
        }
        
        return -1;
    } else {
        // For other types, check proximity to data points
        const float hitRadius = m_config.pointRadius * 3.0f;
        
        for (size_t i = 0; i < m_data.size(); ++i) {
            float x = mapValueToX(i, m_data.size());
            float y = mapValueToY(m_data[i].value);
            
            if (isPointInCircle(mousePos, {x, y}, hitRadius)) {
                return static_cast<int>(i);
            }
        }
        
        return -1;
    }
}

uint32_t UIGraph::generateSeriesColor(int index) const {
    // Predefined color palette
    static const uint32_t colors[] = {
        0xFF00AAFF,  // Cyan
        0xFFFF6600,  // Orange
        0xFF00FF00,  // Green
        0xFFFF00FF,  // Magenta
        0xFFFFFF00,  // Yellow
        0xFFFF0000,  // Red
        0xFF00FFFF,  // Light cyan
        0xFFAA00FF,  // Purple
        0xFF00AA00,  // Dark green
        0xFFFFAA00,  // Gold
    };
    
    return colors[index % 10];
}
