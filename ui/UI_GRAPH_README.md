# UIGraph - Modular Graph Widget

A highly modular, resource-efficient graph UI element for the GRIM UI system. Supports multiple graph types, interactive features, and performance optimizations.

## Features

✅ **Multiple Graph Types**
- Line Graph
- Bar Chart (Vertical & Horizontal)
- Scatter Plot
- Area Chart
- Pie Chart
- Donut Chart
- Stacked Bar Chart
- Multi-Line Graph

✅ **Performance Optimized**
- Automatic downsampling for large datasets
- Configurable data point limits
- Efficient rendering using GDI primitives
- Minimal memory footprint

✅ **Interactive**
- Hover detection on data points
- Click callbacks
- Real-time data updates
- Smooth animations

✅ **Highly Configurable**
- Extensive appearance options
- Customizable colors, grids, axes
- Flexible padding and sizing
- Legend and label controls

## Quick Start

### Basic Line Graph

```cpp
#include "ui_graph.hpp"

// Create graph
auto graph = std::make_shared<UIGraph>("Training Loss", GraphType::Line);
graph->setPosition(50, 100);
graph->setSize(600, 300);

// Add data
std::vector<DataPoint> data;
for (int i = 0; i < 100; ++i) {
    data.push_back(DataPoint(i * 0.5f, "Step " + std::to_string(i)));
}
graph->setData(data);

// Add to panel
panel->addChild(graph);
```

### Bar Chart

```cpp
auto graph = std::make_shared<UIGraph>("Monthly Sales", GraphType::Bar);
graph->setData({
    DataPoint(85.0f, "Jan"),
    DataPoint(92.0f, "Feb"),
    DataPoint(78.0f, "Mar"),
    DataPoint(88.0f, "Apr")
});
graph->getConfig().showLabels = true;
graph->getConfig().showValues = true;
```

### Pie Chart

```cpp
auto graph = std::make_shared<UIGraph>("Resource Usage", GraphType::Pie);
graph->setData({
    DataPoint(45.0f, "Training", 0xFF00AAFF),
    DataPoint(25.0f, "Inference", 0xFFFF6600),
    DataPoint(20.0f, "Data", 0xFF00FF00),
    DataPoint(10.0f, "Other", 0xFFAA00FF)
});
graph->getConfig().showLegend = true;
```

### Multi-Line Graph

```cpp
auto graph = std::make_shared<UIGraph>("Metrics", GraphType::MultiLine);

// Add multiple series
graph->addSeries("Train Loss", trainData, 0xFF00AAFF);
graph->addSeries("Val Loss", valData, 0xFFFF6600);
graph->addSeries("Accuracy", accuracyData, 0xFF00FF00);

graph->getConfig().showLegend = true;
```

## Graph Types

### GraphType::Line
Standard line graph with optional point markers.
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::Line);
graph->getConfig().lineThickness = 2.0f;
graph->getConfig().pointRadius = 4.0f;
```

### GraphType::Bar
Vertical bar chart.
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::Bar);
graph->getConfig().barWidth = 0.8f;  // 80% of available space
```

### GraphType::HorizontalBar
Horizontal bar chart (useful for rankings).
```cpp
auto graph = std::make_shared<UIGraph>("Rankings", GraphType::HorizontalBar);
```

### GraphType::Scatter
Scatter plot for correlation visualization.
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::Scatter);
graph->getConfig().pointRadius = 6.0f;
```

### GraphType::Area
Filled line graph (area under curve).
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::Area);
```

### GraphType::Pie
Traditional pie chart.
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::Pie);
```

### GraphType::Donut
Donut chart (pie with center hole).
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::Donut);
graph->getConfig().donutThickness = 0.4f;  // 40% thickness
```

### GraphType::StackedBar
Stacked bar chart for multiple series.
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::StackedBar);
graph->addSeries("Series1", data1, 0xFF00AAFF);
graph->addSeries("Series2", data2, 0xFFFF6600);
```

### GraphType::MultiLine
Multiple line series on same graph.
```cpp
auto graph = std::make_shared<UIGraph>("Title", GraphType::MultiLine);
```

## Configuration Options

### GraphConfig Structure

```cpp
GraphConfig config;

// Display options
config.showGrid = true;           // Show grid lines
config.showAxes = true;           // Show X/Y axes
config.showLabels = true;         // Show data labels
config.showLegend = false;        // Show legend (multi-series)
config.showValues = false;        // Show values on data points
config.animated = false;          // Enable animation

// Axis options
config.autoScale = true;          // Auto-calculate min/max
config.minValue = 0.0f;           // Manual min value
config.maxValue = 100.0f;         // Manual max value
config.gridLines = 5;             // Number of grid lines

// Visual options
config.lineThickness = 2.0f;      // Line width (pixels)
config.pointRadius = 4.0f;        // Point size (pixels)
config.barWidth = 0.8f;           // Bar width (0.0-1.0)
config.donutThickness = 0.4f;     // Donut thickness (0.0-1.0)

// Colors (ARGB format)
config.backgroundColor = 0xFF1A1A1A;
config.gridColor = 0xFF303030;
config.axisColor = 0xFF505050;
config.textColor = 0xFFCCCCCC;
config.primaryColor = 0xFF00AAFF;

// Padding (pixels)
config.paddingLeft = 50.0f;
config.paddingRight = 20.0f;
config.paddingTop = 30.0f;
config.paddingBottom = 40.0f;

// Performance
config.maxDataPoints = 500;       // Max points before downsampling
config.useDownsampling = true;    // Enable downsampling
```

### Applying Configuration

```cpp
// Method 1: Modify config directly
graph->getConfig().showGrid = true;
graph->getConfig().primaryColor = 0xFF00AAFF;

// Method 2: Create and set config
GraphConfig config;
config.showGrid = true;
config.animated = true;
graph->setConfig(config);
```

## Data Management

### DataPoint Structure

```cpp
struct DataPoint {
    float value;          // Numeric value
    std::string label;    // Optional label
    uint32_t color;       // Optional custom color
    
    // Constructors
    DataPoint(float v);
    DataPoint(float v, const std::string& lbl);
    DataPoint(float v, const std::string& lbl, uint32_t col);
};
```

### Single Series Data

```cpp
// Set all data at once
std::vector<DataPoint> data = {
    DataPoint(10.0f, "A"),
    DataPoint(20.0f, "B"),
    DataPoint(15.0f, "C")
};
graph->setData(data);

// Add points incrementally
graph->addDataPoint(DataPoint(25.0f, "D"));
graph->addDataPoint(30.0f, "E");  // Shorthand

// Clear data
graph->clearData();
```

### Multi-Series Data

```cpp
// DataSeries structure
DataSeries series;
series.name = "Series 1";
series.data = {...};
series.color = 0xFF00AAFF;
series.visible = true;

// Add series
graph->addSeries(series);

// Or use convenience method
graph->addSeries("Series Name", dataVector, 0xFF00AAFF);

// Toggle visibility
graph->setSeriesVisible("Series Name", false);

// Clear all series
graph->clearSeries();
```

## Interactive Features

### Hover Detection

```cpp
graph->setOnPointHover([](int index, const DataPoint& point) {
    std::cout << "Hovering over point " << index 
              << " with value " << point.value << std::endl;
});
```

### Click Callbacks

```cpp
graph->setOnPointClick([](int index, const DataPoint& point) {
    std::cout << "Clicked: " << point.label 
              << " = " << point.value << std::endl;
});
```

## Real-Time Updates

```cpp
class RealtimePanel : public UIPanel {
public:
    RealtimePanel() : UIPanel("Monitoring", true) {
        graph = std::make_shared<UIGraph>("CPU Usage", GraphType::Line);
        graph->getConfig().maxDataPoints = 100;
        addChild(graph);
    }
    
    void update(const InputState& input, float dt) override {
        UIPanel::update(input, dt);
        
        updateTimer += dt;
        if (updateTimer >= 0.1f) {  // Update every 100ms
            updateTimer = 0.0f;
            float cpuUsage = getCurrentCPU();
            graph->addDataPoint(cpuUsage);
        }
    }
    
private:
    std::shared_ptr<UIGraph> graph;
    float updateTimer = 0.0f;
};
```

## Performance Optimization

### Automatic Downsampling

When data exceeds `maxDataPoints`, the graph automatically downsamples:

```cpp
// Configure downsampling
graph->getConfig().maxDataPoints = 500;
graph->getConfig().useDownsampling = true;

// Add large dataset (10,000 points)
std::vector<DataPoint> largeData;
for (int i = 0; i < 10000; ++i) {
    largeData.push_back(DataPoint(getValue(i)));
}

// Automatically downsampled to 500 points
graph->setData(largeData);
```

### Disable Expensive Features

```cpp
// For maximum performance
graph->getConfig().animated = false;
graph->getConfig().showLabels = false;
graph->getConfig().showValues = false;
graph->getConfig().showGrid = false;
```

### Memory Usage

- **DataPoint**: 20 bytes (float + string + uint32_t)
- **Overhead per graph**: ~200 bytes
- **500 points**: ~10 KB
- **Downsampled from 10,000**: ~10 KB (not 200 KB)

## Advanced Usage

### Custom Colors per Point

```cpp
std::vector<DataPoint> data;
data.push_back(DataPoint(85.0f, "Good", 0xFF00AA00));    // Green
data.push_back(DataPoint(65.0f, "Medium", 0xFFFFAA00));  // Yellow
data.push_back(DataPoint(45.0f, "Poor", 0xFFFF0000));    // Red
graph->setData(data);
```

### Dynamic Graph Type Switching

```cpp
// Change graph type at runtime
graph->setGraphType(GraphType::Bar);

// Update configuration for new type
graph->getConfig().barWidth = 0.7f;
```

### Manual Axis Range

```cpp
// Disable auto-scaling
graph->setAxisRange(0.0f, 100.0f);

// Or
graph->getConfig().autoScale = false;
graph->getConfig().minValue = 0.0f;
graph->getConfig().maxValue = 100.0f;
```

### Integration with Training Panel

```cpp
// In UITrainingPanel
class UITrainingPanel : public UIPanel {
    std::shared_ptr<UIGraph> lossGraph;
    
    void initGraphs() {
        lossGraph = std::make_shared<UIGraph>("Training Loss", GraphType::Line);
        lossGraph->setPosition(20, 100);
        lossGraph->setSize(600, 250);
        lossGraph->getConfig().primaryColor = 0xFF00AAFF;
        lossGraph->enableAutoScale(true);
        addChild(lossGraph);
    }
    
    void updateTrainingStats(float loss) {
        lossGraph->addDataPoint(loss);
    }
};
```

## Best Practices

1. **Choose the Right Graph Type**
   - Line/Area: Time series, trends
   - Bar: Comparisons, categories
   - Scatter: Correlations, distributions
   - Pie/Donut: Proportions, percentages

2. **Optimize for Performance**
   - Enable downsampling for >500 points
   - Disable animations for real-time data
   - Use appropriate update intervals

3. **Color Selection**
   - Use high contrast for readability
   - Consistent color schemes across graphs
   - Consider colorblind-friendly palettes

4. **Data Management**
   - Clear old data when no longer needed
   - Use `setData()` for bulk updates
   - Use `addDataPoint()` for incremental updates

5. **Configuration**
   - Set configuration before adding data
   - Use auto-scale unless specific range needed
   - Adjust padding based on label lengths

## Troubleshooting

### Graph not visible
```cpp
// Ensure visibility is set
graph->setVisible(true);

// Check position/size
graph->setPosition(50, 100);
graph->setSize(600, 400);

// Verify parent panel is visible
panel->setVisible(true);
```

### Data not showing
```cpp
// Check data is added
if (graph->getData().empty()) {
    // Add data
}

// Verify axis range
graph->enableAutoScale(true);
```

### Performance issues
```cpp
// Enable downsampling
graph->getConfig().maxDataPoints = 500;
graph->getConfig().useDownsampling = true;

// Disable expensive features
graph->getConfig().animated = false;
```

## API Reference

### Core Methods

```cpp
// Construction
UIGraph(const std::string& title, GraphType type);

// Data (Single Series)
void setData(const std::vector<DataPoint>& data);
void addDataPoint(const DataPoint& point);
void addDataPoint(float value, const std::string& label = "");
void clearData();

// Data (Multi-Series)
void addSeries(const DataSeries& series);
void addSeries(const std::string& name, const std::vector<DataPoint>& data, uint32_t color);
void clearSeries();
void setSeriesVisible(const std::string& name, bool visible);

// Configuration
void setGraphType(GraphType type);
void setTitle(const std::string& title);
void setConfig(const GraphConfig& config);
GraphConfig& getConfig();
void setAxisRange(float minVal, float maxVal);
void enableAutoScale(bool enable);

// Callbacks
void setOnPointHover(std::function<void(int, const DataPoint&)> callback);
void setOnPointClick(std::function<void(int, const DataPoint&)> callback);
```

## License

Part of the GRIM project. See main project LICENSE file.

## Contributing

When adding new graph types:
1. Add enum value to `GraphType`
2. Implement `draw[Type]Graph()` method
3. Add to switch statement in `drawOverlay()`
4. Update documentation and examples
5. Test with various data sizes

## Version History

- **v1.0** - Initial release
  - 9 graph types
  - Performance optimization
  - Interactive features
  - Comprehensive configuration
