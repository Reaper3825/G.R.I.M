
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <Windows.h>
#include "ui_training_panel.hpp"
#include "ui_slider.hpp"
#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "system_detect.hpp"
#include "../ai/grim_text_server_manager.hpp"

using namespace GRIMText;

extern SystemInfo g_systemInfo;

UITrainingPanel::UITrainingPanel()
    : UIPanel("GRIM-text Training Control", true),
      currentState(Control::TrainingState_Idle),
      serverConnected(false),
      pollTimer(0.0f),
      pollInterval(1.5f),
      maxLogEntries(1000),
      logScrollPosition(0.0f),
      autoScrollLogs(true),
      leftPanelScrollPosition(0.0f),
      leftPanelContentHeight(0.0f),
      leftPanelScrolling(false),
      maxLossHistory(500)
{
    position = { 250, 500 };  // Increased from 150 to 200 to ensure title bar is grabbable
    size = { 1000, 700 };
    setVisible(false);
    setBackground(0xE0181818);
    
    // Ensure title bar is accessible (minimum 50px from top of screen)
    if (position.y < 50.0f) {
        position.y = 50.0f;
    }
    
    // Initialize state
    resetState();
    
    // Load config from JSON
    loadConfigFromJSON();
    
    // Initialize client with config from JSON
    std::string host = TrainingConfigManager::getServerHost();
    int port = TrainingConfigManager::getServerPort();
    try {
        client = std::make_unique<TrainingControlClient>(host, port);
        LOG_DEBUG("UITrainingPanel", "Training control client created");
    } catch (const std::exception& e) {
        LOG_ERROR("UITrainingPanel", std::string("Failed to create client: ") + e.what());
        lastError = std::string("Client init failed: ") + e.what();
    }
    
    // Initialize configuration sliders
    epochsSlider = std::make_shared<UISlider>("Epochs", 1.0f, 50.0f, 
        static_cast<float>(currentConfig.epochs),
        [this](float val) { 
            currentConfig.epochs = static_cast<int>(val);
            calculateTrainingEstimate();
        });
    
    batchSizeSlider = std::make_shared<UISlider>("Batch Size", 1.0f, 128.0f,
        static_cast<float>(currentConfig.batchSize),
        [this](float val) {
            currentConfig.batchSize = static_cast<int>(val);
            calculateTrainingEstimate();
        });
    
    learningRateSlider = std::make_shared<UISlider>("Learning Rate", 0.00001f, 0.01f,
        currentConfig.learningRate,
        [this](float val) {
            currentConfig.learningRate = val;
            calculateTrainingEstimate();
        });
    
    maxSeqLenSlider = std::make_shared<UISlider>("Max Seq Length", 512.0f, 16384.0f,
        static_cast<float>(currentConfig.maxSeqLen),
        [this](float val) {
            currentConfig.maxSeqLen = static_cast<int>(val);
            calculateTrainingEstimate();
        });
    
    warmupStepsSlider = std::make_shared<UISlider>("Warmup Steps", 0.0f, 5000.0f,
        static_cast<float>(currentConfig.warmupSteps),
        [this](float val) {
            currentConfig.warmupSteps = static_cast<int>(val);
            calculateTrainingEstimate();
        });
    
    // Initialize widgets - simplified
    startButton = std::make_unique<UIButton>("Start Training", [this]() { 
        // Start training control server (required for training)
        addLog("Starting training control server...", 0);
        if (!GRIM::startTrainingControlServer()) {
            addLog("WARNING: Failed to start training control server!", 0xFF888800);
            addLog("Server may already be running - attempting to connect...", 0);
        } else {
            addLog("Training control server started!", 0);
        }
        
        // Wait for training control server to be ready (max 5 seconds)
        bool trainingServerReady = false;
        for (int i = 0; i < 10; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (GRIM::isTrainingControlServerRunning()) {
                trainingServerReady = true;
                addLog("Training control server ready!", 0);
                break;
            }
            
            // Progress indicator every 2 seconds
            if ((i + 1) % 4 == 0) {
                addLog("Waiting for training control server... (" + std::to_string((i + 1) / 2) + "s)", 0);
            }
        }
        
        if (!trainingServerReady) {
            addLog("ERROR: Training control server not responding!", 0xFF0000FF);
            addLog("Please check that port 11436 is not in use.", 0xFF0000FF);
            return;
        }
        
        // Give server a moment to fully initialize after health check passes
        addLog("Server ready, initializing...", 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Update connection status
        pollServer();
        
        // Now start training
        addLog("Sending training start command...", 0);
        startTrainingSession(); 
    });
    stopButton = std::make_unique<UIButton>("Stop Training", [this]() { 
        // Safe stop - ensure training stops gracefully
        if (currentState == Control::TrainingState_Training || currentState == Control::TrainingState_Paused) {
            stopTrainingSession();
            addLog("Stopping training session...", 0);
        }
    });
    pauseResumeButton = std::make_unique<UIButton>("Pause", [this]() { 
        if (currentState == Control::TrainingState_Training) {
            pauseTrainingSession();
            addLog("Pausing training...", 0);
        } else if (currentState == Control::TrainingState_Paused) {
            resumeTrainingSession();
            addLog("Resuming training...", 0);
        }
    });
    saveConfigButton = std::make_unique<UIButton>("Save Config", [this]() { saveConfigToJSON(); });
    shutdownServerButton = std::make_unique<UIButton>("Close Panel", [this]() { 
        // Hide the panel instead of shutting down server
        setVisible(false);
        addLog("Training panel closed", 0);
    });
    
    // Source URL input and add button
    sourceUrlInput = std::make_shared<UIInputBox>(&sourceUrlBuffer);
    sourceUrlInput->setPlaceholder("Enter data source URL...");
    
    addSourceButton = std::make_unique<UIButton>("Add Source", [this]() {
        if (!sourceUrlBuffer.empty()) {
            addDataSource(sourceUrlBuffer);
            sourceUrlBuffer.clear();
        }
    });
    
    // Verification button
    runVerificationButton = std::make_unique<UIButton>("Run Verification", [this]() {
        runDataVerification();
    });
    
    // Create vertical layout box for the 4 main buttons (Start, Stop, Pause, Close)
    buttonVBox = std::make_shared<UIVBox>(LayoutDirection::Vertical, 10.0f);
    
    // Initialize progress bar
    trainingProgressBar = std::make_shared<UIProgressBar>("Training Progress");
    trainingProgressBar->setFillColor(0xFF00AA00);
    trainingProgressBar->setBackgroundColor(0xFF1A1A1A);
    
    // Update hardware info and estimate
    updateHardwareInfo();
    calculateTrainingEstimate();
    
    addLog("Training panel initialized", 0);
}

UITrainingPanel::~UITrainingPanel() {}

void UITrainingPanel::resetState() {
    currentState = Control::TrainingState_Idle;
    memset(&currentStats, 0, sizeof(currentStats));
    memset(&currentConfig, 0, sizeof(currentConfig));
    
    currentConfig.epochs = 5;
    currentConfig.batchSize = 16;
    currentConfig.learningRate = 0.0002f;
}

void UITrainingPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    
    UIPanel::update(input, dt);
    
    // Update sliders
    if (epochsSlider) epochsSlider->update(input, dt);
    if (batchSizeSlider) batchSizeSlider->update(input, dt);
    if (learningRateSlider) learningRateSlider->update(input, dt);
    if (maxSeqLenSlider) maxSeqLenSlider->update(input, dt);
    if (warmupStepsSlider) warmupStepsSlider->update(input, dt);
    
    // Update buttons
    if (startButton) startButton->update(input, dt);
    if (stopButton) stopButton->update(input, dt);
    if (pauseResumeButton) pauseResumeButton->update(input, dt);
    if (shutdownServerButton) shutdownServerButton->update(input, dt);
    if (saveConfigButton) saveConfigButton->update(input, dt);
    if (addSourceButton) addSourceButton->update(input, dt);
    if (runVerificationButton) runVerificationButton->update(input, dt);
    
    // Update source input
    if (sourceUrlInput) sourceUrlInput->update(input, dt);
    
    // Poll server periodically
    pollTimer += dt;
    if (pollTimer >= pollInterval) {
        pollTimer = 0.0f;
        pollServer();
    }
}

void UITrainingPanel::pollServer() {
    if (!client) return;
    
    // Check connection
    serverConnected = client->isServerRunning();
    
    if (serverConnected) {
        // Get status
        Control::TrainingState state;
        TrainingStats stats;
        TrainingConfig config;
        
        if (client->getStatus(state, stats, config)) {
            // Detect state change to completed
            bool justCompleted = (currentState == Control::TrainingState_Training && 
                                 state == Control::TrainingState_Completed);
            
            currentState = state;
            currentStats = stats;
            currentConfig = config;
            
            // Log completion but keep server running for potential new training sessions
            if (justCompleted) {
                addLog("Training completed!", 0);
            }
        }
    }
}

void UITrainingPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!isVisible()) return;
    
    UIPanel::drawOverlay(renderer);
    
    float panelX = position.x + 10;
    float panelY = position.y + 40;
    float panelWidth = size.x - 20;
    float panelHeight = size.y - 50;
    
    // Split into two columns
    float leftPanelWidth = panelWidth * 0.35f;  // 35% for config
    float rightPanelWidth = panelWidth * 0.65f; // 65% for verbose/stats
    float columnGap = 10;
    
    // ============================================================
    // LEFT PANEL - Configuration with scrolling
    // ============================================================
    float leftX = panelX;
    float leftY = panelY;
    
    // Server status header (fixed at top)
    std::string serverStatus = serverConnected ? "🟢 Server Online" : "🔴 Server Offline";
    uint32_t serverStatusColor = serverConnected ? 0xFF00FF00 : 0xFFFF0000;
    renderer.drawText({leftX, leftY}, serverStatus, serverStatusColor);
    leftY += 25;
    
    // Draw separator line
    renderer.drawRect({leftX, leftY}, {leftPanelWidth, 2}, 0xFF404040);
    leftY += 10;
    
    // Scrollable area for configuration
    float scrollAreaY = leftY;
    float scrollAreaHeight = panelHeight - (leftY - panelY) - 10;
    
    // Draw scroll area background
    renderer.drawRect({leftX, scrollAreaY}, {leftPanelWidth, scrollAreaHeight}, 0xFF0A0A0A);
    
    // Calculate content height
    float contentY = 0;
    float sliderHeight = 35;
    float btnHeight = 35;
    float btnWidth = 150;
    
    leftPanelContentHeight = 30 + // "Training Configuration" header
                            (sliderHeight + 5) * 5 + // 5 sliders
                            15 + // spacing
                            btnHeight + 20 + // Save button + spacing
                            25 + // "Add Data Source" header
                            30 + 5 + // Input box + spacing
                            btnHeight + 20 + // Add Source button + spacing
                            25 + // "Data Verification" header
                            btnHeight + 10 + // Run Verification button
                            60; // Stats display
    
    // Clip rendering to scroll area
    float offsetY = -leftPanelScrollPosition;
    float renderY = scrollAreaY + offsetY;
    
    // Configuration header
    if (renderY >= scrollAreaY - 30 && renderY <= scrollAreaY + scrollAreaHeight) {
        renderer.drawText({leftX + 10, renderY + 10}, "Training Configuration", 0xFF00FFFF);
    }
    renderY += 40;
    
    // Draw sliders (only if visible in scroll area)
    float sliderWidth = leftPanelWidth - 20;
    
    if (epochsSlider) {
        if (renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
            epochsSlider->setPosition(leftX + 10, renderY);
            epochsSlider->setSize(sliderWidth, sliderHeight);
            epochsSlider->drawOverlay(renderer, position);
        }
        renderY += sliderHeight + 5;
    }
    
    if (batchSizeSlider) {
        if (renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
            batchSizeSlider->setPosition(leftX + 10, renderY);
            batchSizeSlider->setSize(sliderWidth, sliderHeight);
            batchSizeSlider->drawOverlay(renderer, position);
        }
        renderY += sliderHeight + 5;
    }
    
    if (learningRateSlider) {
        if (renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
            learningRateSlider->setPosition(leftX + 10, renderY);
            learningRateSlider->setSize(sliderWidth, sliderHeight);
            learningRateSlider->drawOverlay(renderer, position);
        }
        renderY += sliderHeight + 5;
    }
    
    if (maxSeqLenSlider) {
        if (renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
            maxSeqLenSlider->setPosition(leftX + 10, renderY);
            maxSeqLenSlider->setSize(sliderWidth, sliderHeight);
            maxSeqLenSlider->drawOverlay(renderer, position);
        }
        renderY += sliderHeight + 5;
    }
    
    if (warmupStepsSlider) {
        if (renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
            warmupStepsSlider->setPosition(leftX + 10, renderY);
            warmupStepsSlider->setSize(sliderWidth, sliderHeight);
            warmupStepsSlider->drawOverlay(renderer, position);
        }
        renderY += sliderHeight + 15;
    }
    
    // Save Config button
    if (saveConfigButton && renderY >= scrollAreaY - btnHeight && renderY <= scrollAreaY + scrollAreaHeight) {
        saveConfigButton->setPosition(leftX + 10, renderY);
        saveConfigButton->setSize(sliderWidth, btnHeight);
        saveConfigButton->drawOverlay(renderer, position);
    }
    
    renderY += btnHeight + 15;
    
    // Data Source Input Section
    if (renderY >= scrollAreaY - 80 && renderY <= scrollAreaY + scrollAreaHeight) {
        // Section header
        renderer.drawText({leftX + 10, renderY}, "Add Data Source", 0xFF00FFFF);
        renderY += 25;
        
        // Input box
        if (sourceUrlInput) {
            sourceUrlInput->setPosition(leftX + 10, renderY);
            sourceUrlInput->setSize(sliderWidth, 30);
            
            // Draw input box background
            renderer.drawRect({leftX + 10, renderY}, {sliderWidth, 30}, 0xFF1A1A1A);
            renderer.drawRect({leftX + 10, renderY}, {sliderWidth, 2}, 0xFF00AAFF);
            renderer.drawRect({leftX + 10, renderY}, {2, 30}, 0xFF00AAFF);
            renderer.drawRect({leftX + 10, renderY + 28}, {sliderWidth, 2}, 0xFF00AAFF);
            renderer.drawRect({leftX + 10 + sliderWidth - 2, renderY}, {2, 30}, 0xFF00AAFF);
            
            // Draw text
            std::string displayText = sourceUrlBuffer.empty() ? "Enter data source URL..." : sourceUrlBuffer;
            uint32_t textColor = sourceUrlBuffer.empty() ? 0xFF606060 : 0xFFFFFFFF;
            renderer.drawText({leftX + 15, renderY + 8}, displayText, textColor);
            renderY += 35;
        }
        
        // Add Source button
        if (addSourceButton) {
            addSourceButton->setPosition(leftX + 10, renderY);
            addSourceButton->setSize(sliderWidth, btnHeight);
            addSourceButton->drawOverlay(renderer, position);
        }
    }
    
    renderY += btnHeight + 15;
    
    // Data Verification Section
    if (renderY >= scrollAreaY - 80 && renderY <= scrollAreaY + scrollAreaHeight) {
        // Section header
        renderer.drawText({leftX + 10, renderY}, "Data Verification", 0xFF00FFFF);
        renderY += 25;
        
        // Run Verification button
        if (runVerificationButton) {
            runVerificationButton->setPosition(leftX + 10, renderY);
            runVerificationButton->setSize(sliderWidth, btnHeight);
            runVerificationButton->drawOverlay(renderer, position);
            renderY += btnHeight + 10;
        }
        
        // Verification stats
        if (!verificationStats.empty()) {
            renderer.drawText({leftX + 10, renderY}, verificationStats, 0xFF808080);
        }
    }
    
    // Draw scroll bar if content overflows
    if (leftPanelContentHeight > scrollAreaHeight) {
        float scrollBarX = leftX + leftPanelWidth - 10;
        float scrollBarWidth = 8;
        float scrollBarHeight = (scrollAreaHeight / leftPanelContentHeight) * scrollAreaHeight;
        float scrollBarY = scrollAreaY + (leftPanelScrollPosition / leftPanelContentHeight) * scrollAreaHeight;
        
        renderer.drawRect({scrollBarX, scrollAreaY}, {scrollBarWidth, scrollAreaHeight}, 0xFF202020);
        renderer.drawRect({scrollBarX, scrollBarY}, {scrollBarWidth, scrollBarHeight}, 0xFF00FFFF);
    }
    
    // Draw scroll area border
    renderer.drawRect({leftX, scrollAreaY}, {leftPanelWidth, 2}, 0xFF303030);
    renderer.drawRect({leftX, scrollAreaY}, {2, scrollAreaHeight}, 0xFF303030);
    renderer.drawRect({leftX, scrollAreaY + scrollAreaHeight - 2}, {leftPanelWidth, 2}, 0xFF303030);
    renderer.drawRect({leftX + leftPanelWidth - 2, scrollAreaY}, {2, scrollAreaHeight}, 0xFF303030);
    
    // ============================================================
    // RIGHT PANEL - Stats, Controls, and Verbose Output
    // ============================================================
    float rightX = panelX + leftPanelWidth + columnGap;
    float rightY = panelY;
    
    // Connection status
    std::string connStatus = serverConnected ? "✓ Connected" : "✗ Disconnected";
    uint32_t connColor = serverConnected ? 0xFF00FF00 : 0xFFFF0000;
    renderer.drawText({rightX, rightY}, connStatus, connColor);
    rightY += 25;
    
    // Training state
    std::string stateText = getStateString(currentState);
    uint32_t stateColor = getStateColor(currentState);
    renderer.drawText({rightX, rightY}, "State: " + stateText, stateColor);
    rightY += 30;
    
    // Stats
    if (serverConnected) {
        char buf[256];
        
        snprintf(buf, sizeof(buf), "Epoch: %d/%d", currentStats.currentEpoch, currentStats.totalEpochs);
        renderer.drawText({rightX, rightY}, buf, 0xFFCCCCCC);
        rightY += 25;
        
        snprintf(buf, sizeof(buf), "Batch: %d/%d", currentStats.currentBatch, currentStats.totalBatches);
        renderer.drawText({rightX, rightY}, buf, 0xFFCCCCCC);
        rightY += 25;
        
        snprintf(buf, sizeof(buf), "Loss: %.4f", currentStats.currentLoss);
        renderer.drawText({rightX, rightY}, buf, 0xFFCCCCCC);
        rightY += 25;
        
        snprintf(buf, sizeof(buf), "Perplexity: %.2f", currentStats.perplexity);
        renderer.drawText({rightX, rightY}, buf, 0xFFCCCCCC);
        rightY += 35;
    }
    
    // Hardware info
    renderer.drawText({rightX, rightY}, "Hardware:", 0xFF00FFFF);
    rightY += 20;
    
    std::istringstream hwStream(hardwareInfo);
    std::string hwLine;
    while (std::getline(hwStream, hwLine)) {
        renderer.drawText({rightX + 10, rightY}, hwLine, 0xFFAAAAAA);
        rightY += 18;
    }
    rightY += 15;
    
    // Training time estimate
    renderer.drawText({rightX, rightY}, "Estimated Time:", 0xFF00FFFF);
    rightY += 20;
    renderer.drawText({rightX + 10, rightY}, estimatedTimeStr, 0xFFFFAA00);
    rightY += 35;
    
    // Progress bar
    if (trainingProgressBar) {
        float progressBarWidth = rightPanelWidth - 20;
        float progressBarHeight = 30;
        
        // Use training progress directly from server (0-100%)
        // train_gpu.exe calculates this on every batch update
        float progress = currentStats.trainingProgress / 100.0f;  // Convert to 0.0-1.0 range
        
        trainingProgressBar->setValue(progress);
        
        trainingProgressBar->setPosition({rightX, rightY});
        trainingProgressBar->setSize({progressBarWidth, progressBarHeight});
        trainingProgressBar->drawOverlay(renderer, position);
        rightY += progressBarHeight + 20;
    }
    
    // ============================================================
    // BUTTONS AT BOTTOM LEFT (Using VBox for layout)
    // ============================================================
    float bottomBtnHeight = 35;
    float bottomBtnWidth = 140;
    float bottomY = position.y + size.y - (bottomBtnHeight * 4 + 10 * 3) - 15; // 4 buttons + 3 spacings + 15px margin
    float bottomBtnX = leftX;
    
    // Set up VBox position
    if (buttonVBox) {
        buttonVBox->setPosition(bottomBtnX, bottomY);
        buttonVBox->clearWidgets();
        
        // Add buttons to VBox with consistent sizing
        if (startButton) {
            startButton->setSize(bottomBtnWidth, bottomBtnHeight);
            buttonVBox->addWidget(startButton);
        }
        if (stopButton) {
            stopButton->setSize(bottomBtnWidth, bottomBtnHeight);
            buttonVBox->addWidget(stopButton);
        }
        if (pauseResumeButton) {
            pauseResumeButton->setSize(bottomBtnWidth, bottomBtnHeight);
            buttonVBox->addWidget(pauseResumeButton);
        }
        if (shutdownServerButton) {
            shutdownServerButton->setSize(bottomBtnWidth, bottomBtnHeight);
            buttonVBox->addWidget(shutdownServerButton);
        }
        
        // Layout and draw
        buttonVBox->layout();
    }
    
    // Draw buttons using VBox (handles layout and positioning)
    if (buttonVBox) {
        buttonVBox->drawOverlay(renderer, position);
    }
    
    // Draw individual buttons (they handle their own input in update())
    if (startButton) {
        startButton->drawOverlay(renderer, position);
    }
    if (stopButton) {
        stopButton->drawOverlay(renderer, position);
    }
    if (pauseResumeButton) {
        pauseResumeButton->drawOverlay(renderer, position);
    }
    if (shutdownServerButton) {
        shutdownServerButton->drawOverlay(renderer, position);
    }
    
    rightY += btnHeight + btnHeight + 50;  // Account for two rows of buttons
    
    // Verbose Calculation Area (placeholder for future use)
    renderer.drawText({rightX, rightY}, "Verbose Output / Calculations", 0xFF00FFFF);
    rightY += 30;
    
    float verboseHeight = panelHeight - (rightY - panelY) - 250; // Leave room for logs
    float verboseWidth = rightPanelWidth;
    
    // Draw verbose area background
    renderer.drawRect({rightX, rightY}, {verboseWidth, verboseHeight}, 0xFF0A0A0A);
    
    // Placeholder text
    renderer.drawText({rightX + 10, rightY + 10}, "[Reserved for verbose training calculations]", 0xFF808080);
    renderer.drawText({rightX + 10, rightY + 30}, "• GPU utilization graphs", 0xFF606060);
    renderer.drawText({rightX + 10, rightY + 50}, "• Memory usage tracking", 0xFF606060);
    renderer.drawText({rightX + 10, rightY + 70}, "• Token throughput metrics", 0xFF606060);
    renderer.drawText({rightX + 10, rightY + 90}, "• Gradient statistics", 0xFF606060);
    
    // Draw verbose area border
    renderer.drawRect({rightX, rightY}, {verboseWidth, 2}, 0xFF303030);
    renderer.drawRect({rightX, rightY}, {2, verboseHeight}, 0xFF303030);
    renderer.drawRect({rightX, rightY + verboseHeight - 2}, {verboseWidth, 2}, 0xFF303030);
    renderer.drawRect({rightX + verboseWidth - 2, rightY}, {2, verboseHeight}, 0xFF303030);
    
    rightY += verboseHeight + 20;
    
    // Logs section
    renderer.drawText({rightX, rightY}, "Training Logs", 0xFF00FFFF);
    rightY += 25;
    
    float logHeight = panelHeight - (rightY - panelY);
    float logWidth = rightPanelWidth;
    
    renderer.drawRect({rightX, rightY}, {logWidth, logHeight}, 0xFF0A0A0A);
    renderer.drawRect({rightX, rightY}, {logWidth, 1}, 0xFF303030);
    renderer.drawRect({rightX, rightY}, {1, logHeight}, 0xFF303030);
    renderer.drawRect({rightX, rightY + logHeight - 1}, {logWidth, 1}, 0xFF303030);
    renderer.drawRect({rightX + logWidth - 1, rightY}, {1, logHeight}, 0xFF303030);
    
    // Draw log entries
    std::lock_guard<std::mutex> lock(logMutex);
    
    float logY = rightY + 5;
    int visibleLogs = static_cast<int>(logHeight / 18);
    int startIdx = std::max(0, static_cast<int>(logEntries.size()) - visibleLogs);
    
    for (size_t i = startIdx; i < logEntries.size(); i++) {
        const auto& entry = logEntries[i];
        
        uint32_t color = 0xFF00FF00;
        if (entry.level == 1) color = 0xFFFFAA00;
        else if (entry.level == 2) color = 0xFFFF0000;
        
        renderer.drawText({rightX + 5, logY}, entry.timestamp + " " + entry.message, color);
        
        logY += 18;
        if (logY > rightY + logHeight) break;
    }
}

// ============================================================
// Public Control Functions
// ============================================================

void UITrainingPanel::startTrainingSession() {
    if (!client) {
        addLog("Cannot start: client not initialized", 2);
        return;
    }
    
    // Check server connection, retry if needed
    if (!serverConnected) {
        addLog("Server not connected, checking connection...", 0);
        pollServer(); // Try to connect
        
        if (!serverConnected) {
            addLog("Cannot start: server not connected. Please ensure server is running.", 2);
            return;
        }
    }
    
    // Check if already training
    if (currentState == Control::TrainingState_Training) {
        addLog("Training session already in progress", 1);
        return;
    }
    
    // Save current config to ensure consistency
    updateConfigFromSliders();
    
    addLog("Starting training session...", 0);
    
    if (client->startTraining(&currentConfig)) {
        addLog("Training session started successfully", 0);
        currentState = Control::TrainingState_Training;
    } else {
        std::string error = "Failed to start training session";
        if (!client->getLastError().empty()) {
            error += ": " + client->getLastError();
        }
        addLog(error, 2);
    }
}

void UITrainingPanel::stopTrainingSession() {
    if (!client || !serverConnected) {
        addLog("Cannot stop: server not connected", 2);
        return;
    }
    
    // Check if training is actually running
    if (currentState != Control::TrainingState_Training && currentState != Control::TrainingState_Paused) {
        addLog("No active training session to stop", 1);
        return;
    }
    
    addLog("Stopping training session gracefully...", 0);
    
    if (client->stopTraining()) {
        addLog("Training session stopped successfully", 0);
        currentState = Control::TrainingState_Idle;
        
        // Reset progress bar
        if (trainingProgressBar) {
            trainingProgressBar->setValue(0.0f);
        }
    } else {
        addLog("Failed to stop training session", 2);
    }
}

void UITrainingPanel::pauseTrainingSession() {
    if (!client || !serverConnected) {
        addLog("Cannot pause: server not connected", 2);
        return;
    }
    
    // Check if training is running
    if (currentState != Control::TrainingState_Training) {
        addLog("Cannot pause: training is not running", 1);
        return;
    }
    
    // TODO: Implement pause functionality in training server
    addLog("Pausing training (functionality pending on server)", 1);
    currentState = Control::TrainingState_Paused;
}

void UITrainingPanel::resumeTrainingSession() {
    if (!client || !serverConnected) {
        addLog("Cannot resume: server not connected", 2);
        return;
    }
    
    // Check if training is paused
    if (currentState != Control::TrainingState_Paused) {
        addLog("Cannot resume: training is not paused", 1);
        return;
    }
    
    // TODO: Implement resume functionality in training server
    addLog("Resuming training (functionality pending on server)", 1);
    currentState = Control::TrainingState_Training;
}

void UITrainingPanel::shutdownTrainingServer() {
    if (!client) {
        addLog("Cannot shutdown: client not initialized", 2);
        return;
    }
    
    // First stop training if running
    if (serverConnected && currentState == Control::TrainingState_Training) {
        stopTrainingSession();
    }
    
    // TODO: Add proper server shutdown endpoint to training_control_server
    addLog("Server shutdown requested (implementation pending)", 1);
    serverConnected = false;
    currentState = Control::TrainingState_Idle;
}

void UITrainingPanel::handleStartTraining() {
    startTrainingSession();
}

void UITrainingPanel::handleStopTraining() {
    stopTrainingSession();
}

void UITrainingPanel::handlePauseResume() {
    if (currentState == Control::TrainingState_Training) {
        pauseTrainingSession();
    } else if (currentState == Control::TrainingState_Paused) {
        resumeTrainingSession();
    }
}

void UITrainingPanel::addLog(const std::string& message, int level) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    char timeBuf[32];
    std::strftime(timeBuf, sizeof(timeBuf), "%H:%M:%S", &tm);
    
    LogEntry entry;
    entry.timestamp = timeBuf;
    entry.message = message;
    entry.level = level;
    
    logEntries.push_back(entry);
    
    if (logEntries.size() > maxLogEntries) {
        logEntries.erase(logEntries.begin());
    }
}

std::string UITrainingPanel::getStateString(Control::TrainingState state) const {
    switch (state) {
        case Control::TrainingState_Idle: return "Idle";
        case Control::TrainingState_Collecting: return "Collecting";
        case Control::TrainingState_Verifying: return "Verifying";
        case Control::TrainingState_Training: return "⚡ Training";
        case Control::TrainingState_Paused: return "⏸ Paused";
        case Control::TrainingState_Completed: return "✓ Completed";
        case Control::TrainingState_Error: return "✗ Error";
        default: return "Unknown";
    }
}

uint32_t UITrainingPanel::getStateColor(Control::TrainingState state) const {
    switch (state) {
        case Control::TrainingState_Idle: return 0xFF808080;
        case Control::TrainingState_Collecting: return 0xFF00AAFF;
        case Control::TrainingState_Verifying: return 0xFF00AAFF;
        case Control::TrainingState_Training: return 0xFF00FF00;
        case Control::TrainingState_Paused: return 0xFFFFAA00;
        case Control::TrainingState_Completed: return 0xFF00FFFF;
        case Control::TrainingState_Error: return 0xFFFF0000;
        default: return 0xFFFFFFFF;
    }
}

void UITrainingPanel::loadConfigFromJSON() {
    currentConfig = TrainingConfigManager::loadFromJSON();
    updateSlidersFromConfig();
    addLog("Configuration loaded from ai_config.json", 0);
}

void UITrainingPanel::saveConfigToJSON() {
    updateConfigFromSliders();
    
    if (TrainingConfigManager::saveToJSON(currentConfig)) {
        addLog("Configuration saved to ai_config.json", 0);
    } else {
        addLog("Failed to save configuration", 2);
    }
}

void UITrainingPanel::updateConfigFromSliders() {
    // Values are already updated via slider callbacks, but ensure consistency
    if (epochsSlider) currentConfig.epochs = static_cast<int>(epochsSlider->getValue());
    if (batchSizeSlider) currentConfig.batchSize = static_cast<int>(batchSizeSlider->getValue());
    if (learningRateSlider) currentConfig.learningRate = learningRateSlider->getValue();
    if (maxSeqLenSlider) currentConfig.maxSeqLen = static_cast<int>(maxSeqLenSlider->getValue());
    if (warmupStepsSlider) currentConfig.warmupSteps = static_cast<int>(warmupStepsSlider->getValue());
}

void UITrainingPanel::updateSlidersFromConfig() {
    if (epochsSlider) epochsSlider->setValue(static_cast<float>(currentConfig.epochs));
    if (batchSizeSlider) batchSizeSlider->setValue(static_cast<float>(currentConfig.batchSize));
    if (learningRateSlider) learningRateSlider->setValue(currentConfig.learningRate);
    if (maxSeqLenSlider) maxSeqLenSlider->setValue(static_cast<float>(currentConfig.maxSeqLen));
    if (warmupStepsSlider) warmupStepsSlider->setValue(static_cast<float>(currentConfig.warmupSteps));
}

void UITrainingPanel::updateHardwareInfo() {
    std::stringstream ss;
    
    // GPU info
    if (g_systemInfo.hasGPU && g_systemInfo.hasCUDA) {
        ss << "GPU: " << g_systemInfo.gpuName << " (" << g_systemInfo.gpuVRAM_MB << " MB VRAM)\n";
        ss << "CUDA: Available\n";
    } else if (g_systemInfo.hasGPU) {
        ss << "GPU: " << g_systemInfo.gpuName << " (No CUDA)\n";
    } else {
        ss << "GPU: None (CPU training only)\n";
    }
    
    // CPU/RAM
    ss << "CPU: " << g_systemInfo.cpuCores << " cores\n";
    ss << "RAM: " << g_systemInfo.ramMB << " MB";
    
    hardwareInfo = ss.str();
}

void UITrainingPanel::calculateTrainingEstimate() {
    // Rough estimation based on hardware and config
    // Formula: (epochs * dataset_size / batch_size) * time_per_batch
    
    float timePerBatch = 1.0f; // Base time in seconds per batch
    
    // Adjust based on GPU availability
    if (g_systemInfo.hasGPU && g_systemInfo.hasCUDA) {
        // CUDA GPU - much faster than CPU
        // RTX 3080 Ti class: ~0.1-0.3s per batch for typical transformer training
        timePerBatch = 0.15f;
        
        // Adjust for VRAM (more VRAM = can handle larger batches/models efficiently)
        if (g_systemInfo.gpuVRAM_MB >= 12000) {
            timePerBatch *= 0.8f; // High-end GPU (RTX 3080 Ti, 4080, etc)
        } else if (g_systemInfo.gpuVRAM_MB >= 8000) {
            timePerBatch *= 0.9f; // Mid-high GPU
        } else if (g_systemInfo.gpuVRAM_MB >= 4000) {
            timePerBatch *= 1.1f; // Mid-range GPU (slower)
        }
    } else {
        // CPU training is dramatically slower
        timePerBatch = 8.0f;
        
        // Adjust for CPU cores
        if (g_systemInfo.cpuCores >= 16) {
            timePerBatch *= 0.6f;
        } else if (g_systemInfo.cpuCores >= 8) {
            timePerBatch *= 0.75f;
        }
    }
    
    // Adjust for sequence length (longer sequences = quadratic compute cost)
    float seqLenMultiplier = (currentConfig.maxSeqLen / 512.0f);
    timePerBatch *= seqLenMultiplier;
    
    // Adjust for batch size (larger batches are slightly more efficient per sample)
    // But total time per batch increases with batch size
    if (currentConfig.batchSize >= 32) {
        timePerBatch *= 1.5f;
    } else if (currentConfig.batchSize >= 16) {
        timePerBatch *= 1.2f;
    } else if (currentConfig.batchSize >= 8) {
        timePerBatch *= 1.0f;
    } else if (currentConfig.batchSize >= 4) {
        timePerBatch *= 0.7f;
    } else {
        timePerBatch *= 0.5f; // Small batches process faster per batch
    }
    
    // Realistic dataset size based on web collector and enabled sources
    int estimatedDatasetSize = 500;  // Default baseline
    
    // Count enabled sources from source_data.json
    std::string sourcePath = "resources/models/GRIM-text/training/source_data.json";
    try {
        if (std::filesystem::exists(sourcePath)) {
            std::ifstream sourceFile(sourcePath);
            nlohmann::json sourceData;
            sourceFile >> sourceData;
            sourceFile.close();
            
            int enabledSourceCount = 0;
            int totalFetchLimit = 0;
            
            if (sourceData.contains("data_sources")) {
                for (const auto& source : sourceData["data_sources"]) {
                    if (source.value("enabled", false)) {
                        enabledSourceCount++;
                        // Add fetch limits if specified
                        int fetchLimit = source.value("fetch_limit", 100);
                        totalFetchLimit += fetchLimit;
                    }
                }
            }
            
            // Use actual fetch limits if available, otherwise estimate
            if (totalFetchLimit > 0) {
                // Assume 60% verification pass rate
                estimatedDatasetSize = static_cast<int>(totalFetchLimit * 0.6f);
            } else if (enabledSourceCount > 0) {
                // Fallback: estimate 200 samples per enabled source
                estimatedDatasetSize = enabledSourceCount * 200;
            }
            
            // Log the calculation
            if (enabledSourceCount > 0) {
                std::stringstream logMsg;
                logMsg << "Dataset estimate: " << enabledSourceCount 
                       << " sources, ~" << estimatedDatasetSize << " samples";
                addLog(logMsg.str(), 0);
            }
        }
    } catch (const std::exception& e) {
        std::stringstream errMsg;
        errMsg << "Warning: Could not read source_data.json for estimation: " << e.what();
        addLog(errMsg.str(), 1);
        // Fall back to default estimate
    }
    
    int batchesPerEpoch = estimatedDatasetSize / std::max(1, currentConfig.batchSize);
    int totalBatches = batchesPerEpoch * currentConfig.epochs;
    
    estimatedTrainingTimeSeconds = totalBatches * timePerBatch;
    
    // Add warmup steps overhead
    // Warmup steps are additional batches at the start with slower learning rate
    // They typically take the same time per batch as regular training
    if (currentConfig.warmupSteps > 0) {
        float warmupTime = currentConfig.warmupSteps * timePerBatch;
        estimatedTrainingTimeSeconds += warmupTime;
    }
    
    // Format as human-readable string
    std::stringstream ss;
    int hours = static_cast<int>(estimatedTrainingTimeSeconds / 3600);
    int minutes = static_cast<int>((estimatedTrainingTimeSeconds - hours * 3600) / 60);
    int seconds = static_cast<int>(estimatedTrainingTimeSeconds) % 60;
    
    if (hours > 0) {
        ss << hours << "h " << minutes << "m " << seconds << "s";
    } else if (minutes > 0) {
        ss << minutes << "m " << seconds << "s";
    } else {
        ss << seconds << "s";
    }
    
    estimatedTimeStr = ss.str();
}

void UITrainingPanel::addDataSource(const std::string& url) {
    if (url.empty()) {
        addLog("Cannot add empty URL", 1);
        return;
    }
    
    // Basic URL validation
    if (url.find("http://") != 0 && url.find("https://") != 0) {
        addLog("Invalid URL - must start with http:// or https://", 2);
        return;
    }
    
    addLog("Adding data source: " + url, 0);
    saveSourceToJSON(url);
}

void UITrainingPanel::saveSourceToJSON(const std::string& url) {
    std::string sourcePath = "resources/models/GRIM-text/training/source_data.json";
    
    try {
        nlohmann::json sourceData;
        
        // Load existing data
        std::ifstream inFile(sourcePath);
        if (inFile.good()) {
            inFile >> sourceData;
            inFile.close();
        } else {
            // Create new structure if file doesn't exist
            sourceData = {
                {"version", "1.0.0"},
                {"description", "GRIM Web Data Collection Configuration"},
                {"data_sources", nlohmann::json::array()}
            };
        }
        
        // Create new source entry
        nlohmann::json newSource = {
            {"name", "Custom Source"},
            {"url", url},
            {"source_type", "custom"},
            {"enabled", true},
            {"priority", 5},
            {"requires_auth", false}
        };
        
        // Add to data_sources array
        if (!sourceData.contains("data_sources")) {
            sourceData["data_sources"] = nlohmann::json::array();
        }
        sourceData["data_sources"].push_back(newSource);
        
        // Save back to file
        std::ofstream outFile(sourcePath);
        if (outFile.is_open()) {
            outFile << sourceData.dump(2);
            outFile.close();
            addLog("Data source added successfully", 0);
        } else {
            addLog("Failed to write source_data.json", 2);
        }
        
    } catch (const std::exception& e) {
        addLog(std::string("Error saving source: ") + e.what(), 2);
    }
}

void UITrainingPanel::runDataVerification() {
    addLog("Running data verification...", 0);
    
    // Build path to verifier executable
    std::string verifierPath = "resources/models/GRIM-text/training/build_vs_cuda/Release/verifier.exe";
    
    // Check if verifier exists
    if (!std::filesystem::exists(verifierPath)) {
        addLog("Verifier not found. Please build the training tools first.", 2);
        verificationStats = "Verifier not built";
        return;
    }
    
    // Run verifier asynchronously using proper process management
    std::thread([this, verifierPath]() {
        try {
            addLog("Executing verifier...", 0);
            verificationStats = "Running...";
            
            // Convert to absolute path
            std::filesystem::path absPath = std::filesystem::absolute(verifierPath);
            std::wstring wPath = absPath.wstring();
            
            // Setup process structures
            STARTUPINFOW si = {};
            si.cb = sizeof(si);
            si.dwFlags = STARTF_USESHOWWINDOW;
            si.wShowWindow = SW_HIDE; // Hide console window
            
            PROCESS_INFORMATION pi = {};
            
            // Create the process
            BOOL success = CreateProcessW(
                wPath.c_str(),           // Application name
                nullptr,                 // Command line
                nullptr,                 // Process security attributes
                nullptr,                 // Thread security attributes
                FALSE,                   // Inherit handles
                CREATE_NO_WINDOW,        // Creation flags - no console window
                nullptr,                 // Environment
                nullptr,                 // Current directory
                &si,                     // Startup info
                &pi                      // Process info
            );
            
            if (!success) {
                DWORD error = GetLastError();
                addLog("Failed to start verifier. Error code: " + std::to_string(error), 2);
                verificationStats = "Failed to start";
                return;
            }
            
            addLog("Verifier process started (PID: " + std::to_string(pi.dwProcessId) + ")", 0);
            
            // Wait for process to complete (with timeout)
            DWORD waitResult = WaitForSingleObject(pi.hProcess, 60000); // 60 second timeout
            
            if (waitResult == WAIT_TIMEOUT) {
                addLog("Verification timed out after 60 seconds", 2);
                TerminateProcess(pi.hProcess, 1);
                verificationStats = "Timed out";
            } else if (waitResult == WAIT_OBJECT_0) {
                // Process completed, get exit code
                DWORD exitCode = 0;
                GetExitCodeProcess(pi.hProcess, &exitCode);
                
                if (exitCode == 0) {
                    addLog("Verification completed successfully", 0);
                    updateVerificationStats();
                } else {
                    addLog("Verification failed with exit code: " + std::to_string(exitCode), 2);
                    verificationStats = "Failed (code " + std::to_string(exitCode) + ")";
                }
            } else {
                addLog("Error waiting for verifier process", 2);
                verificationStats = "Process error";
            }
            
            // Clean up handles
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);
            
        } catch (const std::exception& e) {
            addLog(std::string("Verification error: ") + e.what(), 2);
            verificationStats = "Error during verification";
        }
    }).detach();
}

void UITrainingPanel::updateVerificationStats() {
    // Read verification stats from output file
    std::string statsPath = "resources/models/GRIM-text/training/data/verification_stats.json";
    
    try {
        if (std::filesystem::exists(statsPath)) {
            std::ifstream statsFile(statsPath);
            nlohmann::json statsData;
            statsFile >> statsData;
            statsFile.close();
            
            // Format stats for display
            std::stringstream ss;
            if (statsData.contains("total_processed")) {
                int total = statsData["total_processed"];
                int passed = statsData.value("passed_verification", 0);
                int failed = statsData.value("failed_verification", 0);
                
                ss << "Processed: " << total << "\n";
                ss << "Passed: " << passed << "\n";
                ss << "Failed: " << failed;
                
                verificationStats = ss.str();
                addLog("Verification stats updated", 0);
            }
        }
    } catch (const std::exception& e) {
        addLog(std::string("Error reading stats: ") + e.what(), 2);
    }
}

bool UITrainingPanel::isAnySliderEditing() const {
    if (epochsSlider && epochsSlider->isEditing()) return true;
    if (batchSizeSlider && batchSizeSlider->isEditing()) return true;
    if (learningRateSlider && learningRateSlider->isEditing()) return true;
    if (maxSeqLenSlider && maxSeqLenSlider->isEditing()) return true;
    if (warmupStepsSlider && warmupStepsSlider->isEditing()) return true;
    return false;
}
