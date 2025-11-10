/*
 * Quick Integration Guide for UIGraph in UITrainingPanel
 * 
 * This shows how to add graphs to your existing training panel
 */

// ============================================================
// Step 1: Add to ui_training_panel.hpp
// ============================================================

#include "ui_graph.hpp"  // Add this include

class UITrainingPanel : public UIPanel {
    // ... existing members ...
    
    // Add graph members
    std::shared_ptr<UIGraph> lossGraph;
    std::shared_ptr<UIGraph> accuracyGraph;
    std::shared_ptr<UIGraph> learningRateGraph;
    
    // Optional: Combined metrics graph
    std::shared_ptr<UIGraph> metricsGraph;
};

// ============================================================
// Step 2: Initialize in Constructor
// ============================================================

UITrainingPanel::UITrainingPanel()
    : UIPanel("GRIM-text Training Control", true)
{
    // ... existing initialization ...
    
    // Initialize loss graph
    lossGraph = std::make_shared<UIGraph>("Training Loss", GraphType::Line);
    lossGraph->setPosition(20, 450);  // Position below sliders
    lossGraph->setSize(580, 200);
    lossGraph->getConfig().primaryColor = 0xFF00AAFF;
    lossGraph->getConfig().showGrid = true;
    lossGraph->getConfig().animated = false;  // Disable for real-time
    lossGraph->getConfig().maxDataPoints = 500;
    lossGraph->enableAutoScale(true);
    
    // Initialize combined metrics graph (multi-line)
    metricsGraph = std::make_shared<UIGraph>("Training Metrics", GraphType::MultiLine);
    metricsGraph->setPosition(620, 450);
    metricsGraph->setSize(580, 200);
    metricsGraph->getConfig().showLegend = true;
    metricsGraph->getConfig().showGrid = true;
    metricsGraph->getConfig().maxDataPoints = 500;
    
    // Series will be added when data arrives
    
    // Don't forget to add to children!
    // Note: They'll be rendered in drawOverlay(), no need to manually add
}

// ============================================================
// Step 3: Update with Training Data
// ============================================================

void UITrainingPanel::pollServer() {
    // ... existing polling code ...
    
    // After receiving training stats
    if (currentState == Control::TrainingState_Training) {
        // Update loss graph
        if (currentStats.currentLoss > 0.0f) {
            lossGraph->addDataPoint(currentStats.currentLoss);
        }
        
        // Update multi-line metrics graph
        // Only add series once, then update data
        static bool seriesInitialized = false;
        if (!seriesInitialized && currentStats.epoch > 0) {
            // Initialize series (do this once)
            metricsGraph->addSeries("Train Loss", {}, 0xFF00AAFF);
            metricsGraph->addSeries("Val Loss", {}, 0xFFFF6600);
            metricsGraph->addSeries("Accuracy", {}, 0xFF00FF00);
            seriesInitialized = true;
        }
        
        // Add data points to series
        // Note: For multi-series, you'd need to modify addSeries to support updates
        // Or track data separately and call setData() periodically
    }
}

// ============================================================
// Step 4: Render in drawOverlay
// ============================================================

void UITrainingPanel::drawOverlay(OverlayRenderer& renderer) {
    // ... existing drawing code ...
    
    // Draw loss graph
    if (lossGraph) {
        lossGraph->drawOverlay(renderer, position);
    }
    
    // Draw metrics graph
    if (metricsGraph) {
        metricsGraph->drawOverlay(renderer, position);
    }
    
    // Optional: Draw graph title/info
    renderer.drawText({position.x + 20, position.y + 430}, 
                     "Training Progress Visualization", 0xFFCCCCCC);
}

// ============================================================
// Alternative: Simpler Single-Graph Integration
// ============================================================

// In constructor:
void UITrainingPanel::initLossGraph() {
    lossGraph = std::make_shared<UIGraph>("Loss", GraphType::Line);
    lossGraph->setPosition(20, 500);
    lossGraph->setSize(1200, 250);
    lossGraph->getConfig().primaryColor = 0xFF00AAFF;
    lossGraph->getConfig().maxDataPoints = 500;
    lossGraph->enableAutoScale(true);
    
    // Interactive callbacks
    lossGraph->setOnPointHover([this](int index, const DataPoint& point) {
        // Show tooltip or status
        addLog("Epoch " + std::to_string(index) + ": Loss = " + 
               std::to_string(point.value), 0);
    });
}

// In update loop:
void UITrainingPanel::updateGraphs(float dt) {
    // Update graph with widget's update method
    if (lossGraph && isVisible()) {
        lossGraph->update(input, dt);
    }
}

// ============================================================
// Performance Tips for Training Panel Integration
// ============================================================

/*
1. Data Update Strategy:
   - Don't update every frame
   - Batch updates (e.g., every 100ms)
   - Use addDataPoint() for incremental updates
   
2. Memory Management:
   - Set maxDataPoints to reasonable value (500-1000)
   - Enable downsampling: config.useDownsampling = true
   - Clear old data when starting new session
   
3. Visual Optimization:
   - Disable animations for real-time data
   - Hide labels if updating frequently
   - Use simple graph types (Line, Area)
   
4. Layout Considerations:
   - Place graphs below controls (don't overlap)
   - Use reasonable sizes (400-800px wide)
   - Consider scrolling panel if many graphs
*/

// ============================================================
// Example: Complete Integration with Existing Code
// ============================================================

class UITrainingPanel : public UIPanel {
public:
    UITrainingPanel();
    
    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;
    
private:
    // Existing members
    std::shared_ptr<UISlider> epochsSlider;
    std::shared_ptr<UIButton> startButton;
    std::shared_ptr<UIProgressBar> trainingProgressBar;
    
    // NEW: Graph members
    std::shared_ptr<UIGraph> lossGraph;
    
    // Graph data tracking
    std::vector<float> lossHistory;
    float lastGraphUpdate = 0.0f;
    float graphUpdateInterval = 0.5f;  // Update every 500ms
    
    void initGraphs();
    void updateGraphData(float dt);
};

// Constructor
UITrainingPanel::UITrainingPanel() : UIPanel("Training", true) {
    // ... existing initialization ...
    
    initGraphs();
}

// Initialize graphs
void UITrainingPanel::initGraphs() {
    lossGraph = std::make_shared<UIGraph>("Training Loss", GraphType::Line);
    lossGraph->setPosition(20, 800);  // Below other controls
    lossGraph->setSize(1200, 180);
    lossGraph->getConfig().primaryColor = 0xFF00AAFF;
    lossGraph->getConfig().showGrid = true;
    lossGraph->getConfig().maxDataPoints = 200;
    lossGraph->getConfig().animated = false;
    lossGraph->enableAutoScale(true);
}

// Update
void UITrainingPanel::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    
    if (lossGraph) {
        lossGraph->update(input, dt);
    }
    
    updateGraphData(dt);
}

// Update graph data periodically
void UITrainingPanel::updateGraphData(float dt) {
    lastGraphUpdate += dt;
    
    if (lastGraphUpdate >= graphUpdateInterval) {
        lastGraphUpdate = 0.0f;
        
        // Add latest loss to graph
        if (currentState == Control::TrainingState_Training && 
            currentStats.currentLoss > 0.0f) {
            
            std::string label = "E" + std::to_string(currentStats.epoch) + 
                               "S" + std::to_string(currentStats.step);
            
            lossGraph->addDataPoint(currentStats.currentLoss, label);
            
            // Track for statistics
            lossHistory.push_back(currentStats.currentLoss);
            if (lossHistory.size() > 200) {
                lossHistory.erase(lossHistory.begin());
            }
        }
    }
}

// Draw
void UITrainingPanel::drawOverlay(OverlayRenderer& renderer) {
    UIPanel::drawOverlay(renderer);
    
    if (lossGraph) {
        lossGraph->drawOverlay(renderer, position);
    }
}

// ============================================================
// Reset on New Training Session
// ============================================================

void UITrainingPanel::startTrainingSession() {
    // ... existing start logic ...
    
    // Clear previous graph data
    if (lossGraph) {
        lossGraph->clearData();
    }
    
    lossHistory.clear();
    lastGraphUpdate = 0.0f;
}

// ============================================================
// Save/Load Graph Data (Optional)
// ============================================================

void UITrainingPanel::saveGraphData(const std::string& filename) {
    // Export loss history
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Step,Loss\n";
        for (size_t i = 0; i < lossHistory.size(); ++i) {
            file << i << "," << lossHistory[i] << "\n";
        }
        file.close();
    }
}

// ============================================================
// That's it! The graph is now integrated and will:
// - Display training loss in real-time
// - Automatically downsample if needed
// - Handle hover/click interactions
// - Scale axes automatically
// - Render efficiently without frame drops
// ============================================================
