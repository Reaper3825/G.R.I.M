
#include <sstream>
#include <iomanip>
#include <thread>
#include <future>
#include <chrono>
#include <ctime>
#include <fstream>
#include <nlohmann/json.hpp>
#include <Windows.h>
#include "ui_training_panel.hpp"
#include "ui_slider.hpp"
#include "overlay_renderer.hpp"
#include "../logger.hpp"
#include "../system_detect.hpp"
#include "../ai/training_server_manager.hpp"
#include "../control/data_collection_server.hpp"
#include "../resources.hpp"

using namespace GRIMText;

extern SystemInfo g_systemInfo;

UITrainingPanel::UITrainingPanel()
    : UIPanel("GRIM-text Training Control", true),
      currentState(Control::TrainingState_Idle),
      serverConnected(false),
      serverStarting(false),  // Initialize server starting flag
      dataCollectionServerConnected(false),  // Initialize data collection server connection
      dataCollectionActive(false),  // Initialize data collection flag
      dataCollectionCompleted(false),  // Initialize completion flag
      pipelineRequestPending(false),  // Initialize pipeline request flag
      firstPollDone(false),  // Initialize first poll flag
      collectionStuckTimer(0.0f),  // Initialize stuck timer
      lastCollectionProgress(0.0f),  // Initialize last progress
      pollTimer(0.0f),
      pollInterval(0.2f),  // Poll every 200ms for fast training updates
      dataCollectionPollTimer(0.0f),  // Initialize data collection poll timer
      dataCollectionPollInterval(1.0f),  // Poll data collection server every 1 second (reduce load)
      dataCollectionPollInProgress(false),  // Initialize async poll flag
      maxLogEntries(1000),
      logScrollPosition(0.0f),
      autoScrollLogs(true),
      leftPanelScrollPosition(0.0f),
      leftPanelContentHeight(0.0f),
      leftPanelScrolling(false),
      maxLossHistory(500),
      collectionAnimTime(0.0f)  // Initialize collection animation timer

    //Panel initialization
{
    position = { 250, 500 };  //default position
    size = { 1250, 1000 };   //default size
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
        // Prevent duplicate server starts
        if (serverStarting) {
            addLog("Server startup already in progress, please wait...", 1);
            return;
        }
        
        // Launch server startup in background thread to avoid blocking UI
        addLog("=== Starting Training Session ===", 0x00FF00FF);
        serverStarting = true;  // Set flag to prevent duplicate starts
        
        std::thread([this]() {
            // Check if training control server is already running
            addLog("Checking for training control server...", 0);
            bool trainingServerReady = GRIM::isTrainingServerRunning();
            
            if (trainingServerReady) {
                addLog("Training control server already running and healthy!", 0x00FF00FF);
            } else {
                // Server not running, try to start it
                addLog("Training control server not detected, starting new instance...", 0);
                if (!GRIM::startTrainingServer()) {
                    addLog("WARNING: Failed to start training control server!", 0xFF888800);
                    addLog("Attempting to connect anyway...", 0);
                } else {
                    addLog("Training control server launch initiated!", 0x00FF00FF);
                }
                
                // Wait for training control server to be ready (max 5 seconds)
                addLog("Waiting for server to respond...", 0);
                for (int i = 0; i < 10; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    if (GRIM::isTrainingServerRunning()) {
                        trainingServerReady = true;
                        addLog("Training control server ready and responding!", 0x00FF00FF);
                        break;
                    }
                    
                    // Progress indicator every 2 seconds
                    if ((i + 1) % 4 == 0) {
                        addLog("Still waiting... (" + std::to_string((i + 1) / 2) + "s)", 0);
                    }
                }
                
                if (!trainingServerReady) {
                    addLog("ERROR: Training control server not responding!", 0xFF0000FF);
                    addLog("Please check that port 11436 is not in use.", 0xFF0000FF);
                    addLog("You may need to manually kill any existing server process.", 0xFF0000FF);
                    serverStarting = false;  // Reset flag on error
                    return;
                }
            }
            
            // Give server a moment to fully initialize after health check passes
            addLog("Server ready, initializing...", 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            // Update connection status
            pollServer();
            
            // Now start training
            addLog("Sending training start command...", 0);
            startTrainingSession();
            
            serverStarting = false;  // Reset flag after startup completes
        }).detach(); // Run in background, don't block UI
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
    
    // Reset status button to clear stale training data
    resetStatusButton = std::make_unique<UIButton>("Reset Status", [this]() {
        resetState();
        checkpointMergeStatus = "";
        addLog("Training status reset", 0);
        
        // Clear the status file if it exists
        std::string statusFilePath = getResourcePath() + "/models/GRIM-text/training/training_status.fb";
        if (std::filesystem::exists(statusFilePath)) {
            std::filesystem::remove(statusFilePath);
            addLog("Cleared stale status file", 0);
        }
    });
    
    // Source URL input and add button
    sourceUrlInput = std::make_shared<UIInputBox>(&sourceUrlBuffer);
    sourceUrlInput->setPlaceholder("Enter data source URL...");
    
    addSourceButton = std::make_shared<UIButton>("Add Source", [this]() {
        if (!sourceUrlBuffer.empty()) {
            addDataSource(sourceUrlBuffer);
            sourceUrlBuffer.clear();
        }
    });
    
    // Unified data pipeline button (replaces separate collect & verify buttons)
    collectDataButton = std::make_shared<UIButton>("Run Data Pipeline", [this]() {
        startDataCollection();
    });
    
    // Create vertical layout box for the 4 main buttons (Start, Stop, Pause, Close)
    buttonVBox = std::make_shared<UIVBox>(LayoutDirection::Vertical, 10.0f);

    
    // Initialize progress bars with max value 1.0 (we pass 0.0-1.0 range)
    trainingProgressBar = std::make_shared<UIProgressBar>("Training Progress", 1.0f);
    trainingProgressBar->setFillColor(0xFF00AA00);
    trainingProgressBar->setBackgroundColor(0xFF1A1A1A);
    
    collectionProgressBar = std::make_shared<UIProgressBar>("Data Collection Progress", 1.0f);
    collectionProgressBar->setFillColor(0xFF00AAFF);  // Cyan color for collection
    collectionProgressBar->setBackgroundColor(0xFF1A1A1A);
    
    // Update hardware info and estimate
    updateHardwareInfo();
    calculateTrainingEstimate();
    updateDatasetSize();
    
    addLog("Training panel initialized", 0);
}

UITrainingPanel::~UITrainingPanel() {
    // Shutdown data collection server if connected
    if (dataCollectionClient && dataCollectionServerConnected) {
        addLog("[DataCollection] Shutting down data collection server...", 0);
        dataCollectionClient->shutdownServer();
    }
}

void UITrainingPanel::resetState() {
    currentState = Control::TrainingState_Idle;
    serverStarting = false;  // Reset server starting flag
    dataCollectionServerConnected = false;  // Reset data collection server connection
    dataCollectionActive = false;  // Reset data collection flag
    dataCollectionCompleted = false;  // Reset completion flag
    pipelineRequestPending = false;  // Reset pipeline request flag
    memset(&currentStats, 0, sizeof(currentStats));
    memset(&currentConfig, 0, sizeof(currentConfig));
    
    currentConfig.epochs = 5;
    currentConfig.batchSize = 16;
    currentConfig.learningRate = 0.0002f;
}

void UITrainingPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    
    UIPanel::update(input, dt);
    
    // Update collection animation timer
    if (currentState == Control::TrainingState_Collecting) {
        collectionAnimTime += dt * 2.0f;  // Speed multiplier for animation
    }
    
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
    if (resetStatusButton) resetStatusButton->update(input, dt);
    if (addSourceButton) addSourceButton->update(input, dt);
    if (collectDataButton) collectDataButton->update(input, dt);
    
    // Update source input
    if (sourceUrlInput) sourceUrlInput->update(input, dt);
    
    // Poll training server periodically
    pollTimer += dt;
    if (pollTimer >= pollInterval) {
        pollTimer = 0.0f;
        pollServer();
    }
    
    // Poll data collection server more frequently for real-time progress
    dataCollectionPollTimer += dt;
    if (dataCollectionPollTimer >= dataCollectionPollInterval) {
        dataCollectionPollTimer = 0.0f;
        pollDataCollectionServerAsync();  // Use async version to avoid blocking
    }
}

void UITrainingPanel::pollServer() {
    if (!client) return;
    
    // ✅ FIX: Skip if already polling (prevents stacking async requests)
    static std::atomic<bool> isPolling{false};
    if (isPolling.exchange(true)) {
        return; // Already polling, skip this frame
    }
    
    // ✅ FIX: Run network I/O on background thread to avoid blocking UI
    std::thread([this]() {
        // Check connection
        bool previouslyConnected = serverConnected;
        serverConnected = client->isServerRunning();
        
        // Reset first poll flag when we reconnect
        if (!previouslyConnected && serverConnected) {
            firstPollDone = false;
        }
        
        // Detect server disconnect during training
        if (previouslyConnected && !serverConnected) {
            if (currentState == Control::TrainingState_Training || 
                currentState == Control::TrainingState_Collecting ||
                currentState == Control::TrainingState_Verifying) {
                addLog("Server disconnected during operation - training may have crashed", 2);
                currentState = Control::TrainingState_Error;
                currentStats.lastError = "Server disconnected unexpectedly";
                checkpointMergeStatus = "";
            }
        }
    
    if (serverConnected) {
        // Get status
        Control::TrainingState state;
        TrainingStats stats;
        TrainingConfig config;
        
        if (client->getStatus(state, stats, config)) {
            Control::TrainingState previousState = currentState;
            
            // Detect stale collection/verification states on first poll after connection
            // (server crashed or was killed mid-operation, leaving stale status file)
            if (!firstPollDone && (state == Control::TrainingState_Collecting || state == Control::TrainingState_Verifying)) {
                // If we're in collecting/verifying state but there's no active progress, it's stale
                addLog("Detected stale '" + getStateString(state) + "' state from previous session, resetting...", 1);
                state = Control::TrainingState_Idle;
                stats = TrainingStats();  // Clear stats
            }
            firstPollDone = true;
            
            // Detect state transitions
            if (previousState != state) {
                // Training -> Completed
                if (previousState == Control::TrainingState_Training && 
                    state == Control::TrainingState_Completed) {
                    addLog("Training completed successfully!", 0);
                    checkpointMergeStatus = "";
                }
                // Training -> Error
                else if (previousState == Control::TrainingState_Training && 
                         state == Control::TrainingState_Error) {
                    addLog("Training encountered an error: " + stats.lastError, 2);
                    checkpointMergeStatus = "";
                }
                // Training -> Paused
                else if (previousState == Control::TrainingState_Training && 
                         state == Control::TrainingState_Paused) {
                    addLog("Training paused", 1);
                }
                // Paused -> Training
                else if (previousState == Control::TrainingState_Paused && 
                         state == Control::TrainingState_Training) {
                    addLog("Training resumed", 0);
                }
                // Any -> Idle (unexpected stop)
                else if (previousState == Control::TrainingState_Training && 
                         state == Control::TrainingState_Idle) {
                    addLog("Training stopped unexpectedly - process may have crashed", 2);
                    stats.lastError = "Training process terminated";
                    checkpointMergeStatus = "";
                }
            }
            
            // Detect training crash by checking if process_running flag is false during training
            if (state == Control::TrainingState_Training) {
                // The StatusResponse has a process_running field we should check
                // If state says Training but process isn't running, it crashed
                static int crashCheckCounter = 0;
                crashCheckCounter++;
                
                // Check every 5 polls (~1 second) to avoid false positives
                if (crashCheckCounter >= 5) {
                    crashCheckCounter = 0;
                    
                    // If we haven't received any stat updates in a while, something is wrong
                    static float lastLoss = -1.0f;
                    static int lastBatch = -1;
                    
                    if (stats.currentLoss == lastLoss && stats.currentBatch == lastBatch) {
                        static int stalledCounter = 0;
                        stalledCounter++;
                        
                        // If stats haven't changed for 25 polls (~5 seconds), assume crash
                        if (stalledCounter > 25) {
                            addLog("Training appears to have stalled or crashed - no progress detected", 2);
                            currentState = Control::TrainingState_Error;
                            stats.lastError = "Training process stalled - no progress";
                            stalledCounter = 0;
                            checkpointMergeStatus = "";
                        }
                    } else {
                        static int stalledCounter = 0;
                        stalledCounter = 0; // Reset if we see progress
                        lastLoss = stats.currentLoss;
                        lastBatch = stats.currentBatch;
                    }
                }
            }
            
            currentState = state;
            currentStats = stats;
            currentConfig = config;
            
            // Update data collection active flag based on server state
            dataCollectionActive = (state == Control::TrainingState_Collecting || 
                                   state == Control::TrainingState_Verifying);
            
            // Log phase changes during data collection
            static std::string lastPhase = "";
            if (state == Control::TrainingState_Collecting && 
                !stats.currentPhase.empty() && 
                stats.currentPhase != lastPhase) {
                addLog("  → " + stats.currentPhase, 0);
                lastPhase = stats.currentPhase;
            } else if (state != Control::TrainingState_Collecting) {
                lastPhase = "";  // Reset when not collecting
            }
            
        } else {
            // Failed to get status from server
            if (currentState == Control::TrainingState_Training) {
                addLog("Failed to communicate with training server", 2);
                // Don't immediately switch to error, might be temporary
            }
        }
    } else {
        // Server disconnected - reset collection flag
        dataCollectionActive = false;
    }
    
    isPolling = false; // ✅ FIX: Release lock so next poll can run
    }).detach(); // ✅ FIX: Run async, don't wait
}

void UITrainingPanel::pollDataCollectionServerAsync() {
    // Check if previous poll is still running
    if (dataCollectionPollInProgress.exchange(true)) {
        return; // Skip if already polling
    }
    
    // ✅ FIX: Launch DIRECT async task (avoid double-threading with pollDataCollectionServer)
    dataCollectionPollFuture = std::async(std::launch::async, [this]() {
        if (!dataCollectionClient) {
            // Initialize client on first poll
            dataCollectionClient = std::make_unique<GRIM::DataCollection::DataCollectionClient>("localhost", 11437);
            addLog("[DataCollection] Client initialized for localhost:11437", 0);
        }
        
        // Check if data collection server is running (FAST timeout)
        bool previouslyConnected = dataCollectionServerConnected;
        dataCollectionServerConnected = dataCollectionClient->isServerRunning();
        
        if (!previouslyConnected && dataCollectionServerConnected) {
            addLog("[DataCollection] Server connected!", 0);
        } else if (previouslyConnected && !dataCollectionServerConnected) {
            addLog("[DataCollection] Server disconnected!", 1);
        }
        
        // If server is not connected, nothing more to poll
        if (!dataCollectionServerConnected) {
            dataCollectionPollInProgress = false;
            return;
        }
        
        // Get current collection status (FAST timeout)
        auto status = dataCollectionClient->getStatus();
        
        // Only update state if we got a valid response (prevents flickering on timeout)
        if (!status.valid) {
            dataCollectionPollInProgress = false;
            return;
        }
        
        // Update dataCollectionActive flag based on server status
        bool wasCollecting = dataCollectionActive;
        
        // Don't update to inactive if we've already marked as completed (prevents flickering)
        if (!dataCollectionCompleted || status.isCollecting) {
            dataCollectionActive = status.isCollecting;
        }
        
        // Detect collection start
        if (!wasCollecting && dataCollectionActive) {
            addLog("[DataCollection] Collection started on server", 0);
            lastCollectionProgress = 0.0f;
            collectionStuckTimer = 0.0f;
            dataCollectionCompleted = false;  // Reset completion flag on new collection
        }
        
        // Detect collection completion (only log once)
        if (wasCollecting && !dataCollectionActive) {
            if (!dataCollectionCompleted) {
                addLog("[DataCollection] Collection completed/stopped on server", 0);
                dataCollectionCompleted = true;  // Mark as completed to prevent re-logging
                pipelineRequestPending = false;
                // Keep final progress/phase values to show completion state
                
                if (!status.error.empty()) {
                    addLog("[DataCollection] ERROR: " + status.error, 2);
                }
            }
        }
        
        // Update collection progress and phase (only during active collection, not after completion)
        if (dataCollectionActive && !dataCollectionCompleted) {
            // Only update progress if it's non-zero or we're just starting (prevents false resets)
            if (status.progress > 0.0f || currentStats.collectionProgress < 1.0f) {
                // Log significant progress changes to debug oscillation
                if (std::abs(status.progress - currentStats.collectionProgress) > 5.0f) {
                    addLog("[DataCollection] Progress jump: " + 
                           std::to_string((int)currentStats.collectionProgress) + "% -> " + 
                           std::to_string((int)status.progress) + "%", 0);
                }
                currentStats.collectionProgress = status.progress;
            }
            
            // Only update phase if not empty (prevents clearing on idle status)
            if (!status.phase.empty()) {
                currentStats.currentPhase = status.phase;
            }
            
            // Detect stuck collection (progress not changing)
            if (status.progress == lastCollectionProgress) {
                collectionStuckTimer += dataCollectionPollInterval;
                
                // If stuck for more than 30 seconds, log warning
                if (collectionStuckTimer > 30.0f) {
                    static float lastWarningTime = 0.0f;
                    if (collectionStuckTimer - lastWarningTime > 10.0f) {  // Warn every 10 seconds
                        addLog("[DataCollection] WARNING: Progress stuck at " + 
                               std::to_string((int)status.progress) + "% for " + 
                               std::to_string((int)collectionStuckTimer) + " seconds", 1);
                        lastWarningTime = collectionStuckTimer;
                    }
                }
            } else {
                collectionStuckTimer = 0.0f;  // Reset timer when progress changes
                lastCollectionProgress = status.progress;
            }
        }
        
        dataCollectionPollInProgress = false;
    });
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
    std::string trainingServerStatus = serverConnected ? "[ONLINE] Training" : "[OFFLINE] Training";
    uint32_t trainingServerColor = serverConnected ? 0xFF00FF00 : 0xFFFF0000;
    renderer.drawText({leftX, leftY}, trainingServerStatus, trainingServerColor);
    
    // Data collection server status (right side)
    std::string collectionServerStatus = dataCollectionServerConnected ? "[ONLINE] Collection" : "[OFFLINE] Collection";
    uint32_t collectionServerColor = dataCollectionServerConnected ? 0xFF00FF00 : 0xFFFF0000;
    renderer.drawText({leftX + 200, leftY}, collectionServerStatus, collectionServerColor);
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
                            btnHeight + 10 + // Collect Data button
                            60 + // Stats display
                            25; // Dataset size info
    
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
        renderer.drawText({leftX + 10, renderY}, "Data Pipeline", 0xFF00FFFF);
        renderY += 25;
        
        // Data collection status indicator
        std::string collectionStatusIndicator;
        uint32_t collectionIndicatorColor;
        
        if (dataCollectionActive) {
            collectionStatusIndicator = "[ACTIVE] Collecting";
            collectionIndicatorColor = 0xFF00AAFF;  // Cyan
        } else if (dataCollectionCompleted && currentStats.collectionProgress >= 99.0f) {
            collectionStatusIndicator = "[COMPLETED]";
            collectionIndicatorColor = 0xFF00FF00;  // Green
        } else {
            collectionStatusIndicator = "[IDLE] Ready";
            collectionIndicatorColor = 0xFF808080;  // Grey
        }
        
        renderer.drawText({leftX + 10, renderY}, collectionStatusIndicator, collectionIndicatorColor);
        
        // Show progress if collecting or completed
        if ((dataCollectionActive || dataCollectionCompleted) && currentStats.collectionProgress > 0.0f) {
            std::string progressText = " (" + std::to_string((int)currentStats.collectionProgress) + "%)";
            renderer.drawText({leftX + 120, renderY}, progressText, 
                            dataCollectionActive ? 0xFF00AAFF : 0xFF00FF00);
        }
        renderY += 25;
        
        // Unified Data Pipeline button
        if (collectDataButton) {
            collectDataButton->setPosition(leftX + 10, renderY);
            collectDataButton->setSize(sliderWidth, btnHeight);
            collectDataButton->drawOverlay(renderer, position);
            renderY += btnHeight + 10;
        }
        
        // Verification stats display
        if (!verificationStats.empty()) {
            renderer.drawText({leftX + 10, renderY}, verificationStats, 0xFF808080);
            renderY += 45;
        }
        
        // Dataset size info
        if (!datasetSizeInfo.empty()) {
            renderer.drawText({leftX + 10, renderY}, datasetSizeInfo, 0xFF00FFAA);
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
    
    // Data collection status indicator
    std::string collectionStatus;
    uint32_t collectionColor;
    
    if (dataCollectionActive) {
        collectionStatus = "[ACTIVE] Data Collection";
        collectionColor = 0xFF00AAFF;  // Cyan
    } else if (dataCollectionCompleted && currentStats.collectionProgress >= 99.0f) {
        collectionStatus = "[COMPLETED] Data Collection";
        collectionColor = 0xFF00FF00;  // Green
    } else {
        collectionStatus = "[IDLE] Data Collection";
        collectionColor = 0xFF606060;  // Dark grey
    }
    
    renderer.drawText({rightX, rightY}, collectionStatus, collectionColor);
    rightY += 25;
    
    // Training state
    std::string stateText = getStateString(currentState);
    uint32_t stateColor = getStateColor(currentState);
    renderer.drawText({rightX, rightY}, "State: " + stateText, stateColor);
    rightY += 30;
    
    // Show checkpoint merge status if applicable
    if (!checkpointMergeStatus.empty() && 
        (currentState == Control::TrainingState_Collecting || 
         currentState == Control::TrainingState_Verifying)) {
        renderer.drawText({rightX, rightY}, checkpointMergeStatus, 0xFFFFAA00);
        rightY += 25;
    }
    
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
    
    // Training progress bar
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
        rightY += progressBarHeight + 10;
    }
    
    // Data collection progress bar
    if (collectionProgressBar) {
        float progressBarWidth = rightPanelWidth - 20;
        float progressBarHeight = 30;
        
        // Use collection progress directly from server (0-100%)
        // Data pipeline updates this every second during collection
        float targetProgress = currentStats.collectionProgress / 100.0f;  // Convert to 0.0-1.0 range
        
        // Smooth progress updates to avoid visual glitches
        static float smoothedProgress = 0.0f;
        static bool wasActiveLastFrame = false;
        
        // Reset progress when collection starts fresh
        if (!wasActiveLastFrame && dataCollectionActive) {
            smoothedProgress = 0.0f;
        }
        wasActiveLastFrame = dataCollectionActive;
        
        if (dataCollectionActive && targetProgress > smoothedProgress) {
            // Gradually move towards target (prevents jumps)
            smoothedProgress = smoothedProgress * 0.7f + targetProgress * 0.3f;
        }
        // Progress never decreases or resets during active collection
        
        collectionProgressBar->setValue(smoothedProgress);
        
        collectionProgressBar->setPosition({rightX, rightY});
        collectionProgressBar->setSize({progressBarWidth, progressBarHeight});
        collectionProgressBar->drawOverlay(renderer, position);
        rightY += progressBarHeight + 5;
        
        // Show current collection phase if active
        if (dataCollectionActive && !currentStats.currentPhase.empty()) {
            renderer.drawText({rightX, rightY}, "Phase: " + std::string(currentStats.currentPhase), 0xFF00AAFF);
            rightY += 20;
        }
        
        rightY += 10;
    }
    
    // ============================================================
    // BUTTONS AT BOTTOM LEFT (Using VBox for layout)
    // ============================================================
    float bottomBtnHeight = 35;
    float bottomBtnWidth = 140;
    float bottomY = position.y + size.y - (bottomBtnHeight * 5 + 10 * 4) - 15; // 5 buttons + 4 spacings + 15px margin
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
        if (resetStatusButton) {
            resetStatusButton->setSize(bottomBtnWidth, bottomBtnHeight);
            buttonVBox->addWidget(resetStatusButton);
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
    if (resetStatusButton) {
        resetStatusButton->drawOverlay(renderer, position);
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
        // Double-check with server to see if training is ACTUALLY running
        Control::TrainingState serverState;
        TrainingStats serverStats;
        TrainingConfig serverConfig;
        
        if (client->getStatus(serverState, serverStats, serverConfig)) {
            if (serverState == Control::TrainingState_Training) {
                addLog("Training session already in progress on server (progress: " + 
                       std::to_string((int)serverStats.trainingProgress) + "%), please wait...", 1);
                return;
            } else {
                // Server says it's not training, but our UI thinks it is - fix the desync
                addLog("Detected state desync - server is " + getStateString(serverState) + 
                       ", resetting UI state...", 1);
                currentState = serverState;
                currentStats = serverStats;
                // Continue to start training
            }
        } else {
            addLog("WARNING: Cannot verify server state, assuming stale UI state, proceeding...", 1);
            currentState = Control::TrainingState_Idle;
            // Continue to start training
        }
    }
    
    // Save current config to ensure consistency
    updateConfigFromSliders();
    
    // Reset stats for new training session
    currentStats = TrainingStats();
    if (trainingProgressBar) {
        trainingProgressBar->setValue(0.0f);
    }
    
    // Check if training data exists (should be created by DataCollection pipeline)
    std::string trainBinPath = "resources/models/GRIM-text/training/data/tokenized/train.bin";
    std::string grmtPath = "resources/models/GRIM-text/training/data/training_data.grmt";
    
    std::ifstream checkBin(trainBinPath, std::ios::binary | std::ios::ate);
    std::ifstream checkGrmt(grmtPath, std::ios::binary | std::ios::ate);
    
    bool hasTrainBin = checkBin.is_open() && checkBin.tellg() > 0;
    bool hasGrmt = checkGrmt.is_open() && checkGrmt.tellg() > 0;
    
    checkBin.close();
    checkGrmt.close();
    
    if (!hasTrainBin && !hasGrmt) {
        addLog("ERROR: No training data found!", 2);
        addLog("Please run DataCollection pipeline first to generate training data.", 2);
        addLog("Expected files:", 1);
        addLog("  - " + trainBinPath, 1);
        addLog("  - " + grmtPath, 1);
        currentState = Control::TrainingState_Idle;
        return;
    }
    
    addLog("Training data found, starting training...", 0);
    // Prefer tokenized .bin files (native format for train_gpu.exe)
    if (hasTrainBin) {
        addLog("  Using tokenized binary: " + trainBinPath, 0);
        currentConfig.dataPath = "data/tokenized/train.bin";  // Relative to training/ dir
    } else if (hasGrmt) {
        addLog("  Using GRMT data: " + grmtPath, 0);
        currentConfig.dataPath = "data/training_data.grmt";  // Relative to training/ dir
    }
    
    // Start training with existing data
    addLog("Starting training session...", 0);
    checkpointMergeStatus = "";  // Clear status before training starts
    
    if (client->startTraining(&currentConfig)) {
        addLog("Training session started successfully", 0);
        currentState = Control::TrainingState_Training;
    } else {
        std::string error = "Failed to start training session";
        if (!client->getLastError().empty()) {
            error += ": " + client->getLastError();
            
            // If it's "already in progress", show current session details from cached stats
            if (client->getLastError().find("already in progress") != std::string::npos && currentStats.startTime > 0) {
                // Calculate when the session started using cached stats (no network call)
                auto now = std::chrono::system_clock::now();
                auto startTime = std::chrono::system_clock::from_time_t(currentStats.startTime);
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime);
                
                std::time_t startTimeT = currentStats.startTime;
                std::tm startTm;
#ifdef _WIN32
                localtime_s(&startTm, &startTimeT);  // Thread-safe version
#else
                localtime_r(&startTimeT, &startTm);  // POSIX thread-safe version
#endif
                char timeBuf[64];
                std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", &startTm);
                
                addLog("Current training session details:", 1);
                addLog("  Started: " + std::string(timeBuf), 1);
                addLog("  Elapsed: " + std::to_string(elapsed.count()) + " seconds", 1);
                addLog("  State: " + getStateString(currentState), 1);
                addLog("  Progress: " + std::to_string((int)currentStats.trainingProgress) + "%", 1);
            }
        }
        addLog(error, 2);
        currentState = Control::TrainingState_Idle;
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
    
    // Also write to grim.log file for TTS system
    if (level == 0) {
        LOG_DEBUG("TrainingPanel", message);
    } else if (level == 1) {
        LOG_TRACE("TrainingPanel", message);
    } else {
        LOG_ERROR("TrainingPanel", message);
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
    std::string sourcePath = getResourcePath() + "/models/GRIM-text/training/source_data.json";
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
    std::string sourcePath = getResourcePath() + "/models/GRIM-text/training/source_data.json";
    
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

void UITrainingPanel::updateVerificationStats() {
    // Read verification stats from output file
    std::string statsPath = getResourcePath() + "/models/GRIM-text/training/data/verification_stats.json";
    
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

void UITrainingPanel::startDataCollection() {
    addLog("=== DATA COLLECTION INITIATED ===", 0x00FFFF00);
    addLog("Starting unified data pipeline via dedicated collection server...", 0);
    addLog("DEBUG: Checking data collection server connection...", 0);
    
    // Initialize data collection client if not already done
    if (!dataCollectionClient) {
        dataCollectionClient = std::make_unique<GRIM::DataCollection::DataCollectionClient>("localhost", 11437);
        addLog("[DataCollection] Client initialized", 0);
    }
    
    // Check if data collection server is running
    if (!dataCollectionClient->isServerRunning()) {
        addLog("ERROR: Data collection server not running!", 2);
        addLog("Please start the data collection server on port 11437", 2);
        addLog("Attempting to start data collection server...", 0);
        
        // Try to start the data collection server
        if (!GRIM::DataCollection::startDataCollectionServer()) {
            addLog("CRITICAL: Failed to start data collection server!", 2);
            addLog("Please check logs and ensure port 11437 is available.", 2);
            return;
        }
        
        // Wait briefly for server to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        // Re-check connection
        if (!dataCollectionClient->isServerRunning()) {
            addLog("CRITICAL: Data collection server started but not responding!", 2);
            return;
        }
        
        addLog("Data collection server started successfully!", 0);
        dataCollectionServerConnected = true;
    }
    
    addLog("DEBUG: Checking pipelineRequestPending and dataCollectionActive flags...", 0);
    
    // Guard 1: Check if request is already pending
    if (pipelineRequestPending) {
        addLog("Pipeline request already in progress, please wait...", 1);
        return;
    }
    
    addLog("DEBUG: Getting FRESH data collection server status...", 0);
    
    // Guard 2: Check ACTUAL server status (not cached flag)
    auto status = dataCollectionClient->getStatus();
    if (status.isCollecting) {
        addLog("WARNING: Server reports collection already in progress!", 1);
        addLog("  Progress: " + std::to_string((int)status.progress) + "%", 0);
        addLog("  Phase: " + status.phase, 0);
        dataCollectionActive = true;  // Sync our state
        return;
    }
    
    // Sync flag with fresh server status
    dataCollectionActive = false;
    
    // Reset completion flag to allow logging completion of new collection
    dataCollectionCompleted = false;
    
    addLog("DEBUG: All guards passed, setting pipelineRequestPending = true", 0);
    
    // Set pending flag to prevent duplicate clicks
    pipelineRequestPending = true;
    
    addLog("DEBUG: Launching async thread for HTTP request to data collection server...", 0);
    
    // Run pipeline asynchronously via data collection server
    std::thread([this]() {
        try {
            addLog(">>> [UI_TRAINING_PANEL] Sending HTTP POST request", 0);
            addLog("    FROM: UI Thread (ui_training_panel.cpp:startDataCollection)", 0);
            addLog("    TO: http://localhost:11437/api/collection/start", 0);
            addLog("    METHOD: POST | BODY: mode='full'", 0);
            addLog(">>> Requesting data collection from dedicated collection server...", 0);
            
            // Store start time
            auto startTime = std::chrono::steady_clock::now();
            
            // Call the data collection client (returns quickly, collection runs async on server)
            addLog(">>> [DATA_COLLECTION_CLIENT] Calling DataCollectionClient::startCollection('full')", 0);
            auto result = dataCollectionClient->startCollection("full");
            
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - startTime
            ).count();
            
            addLog("<<< [DATA_COLLECTION_CLIENT] Got HTTP response after " + std::to_string(elapsed) + "ms", 0);
            addLog("    FROM: http://localhost:11437/api/collection/start", 0);
            addLog("    SUCCESS: " + std::string(result.success ? "true" : "false"), 0);
            addLog("    MESSAGE: " + result.message, 0);
            addLog("    ERROR: " + result.error, 0);
            
            if (result.success) {
                addLog("✓ Data collection request accepted by server!", 1);
                addLog("  " + result.message, 0);
                addLog("  Monitor progress in real-time via status indicator", 0);
                // Note: pipelineRequestPending will be cleared by pollDataCollectionServer when collection finishes
            } else if (!result.error.empty()) {
                addLog("✗ Data collection request failed: " + result.error, 2);
                pipelineRequestPending = false;  // Clear pending flag on error
            } else {
                addLog("✗ Data collection failed with no error message", 2);
                pipelineRequestPending = false;  // Clear pending flag on error
            }
            
            addLog("Data collection request processing complete", 0);
            
        } catch (const std::exception& e) {
            addLog("✗ Exception during data collection request: " + std::string(e.what()), 2);
            pipelineRequestPending = false;  // Clear pending flag on exception
            addLog("Pipeline request failed due to exception", 1);
        }
    }).detach();
}

void UITrainingPanel::updateDatasetSize() {
    // Check for training data file
    std::string grmtPath = getResourcePath() + "/models/GRIM-text/training/data/training_data.grmt";
    
    try {
        if (std::filesystem::exists(grmtPath)) {
            auto fileSize = std::filesystem::file_size(grmtPath);
            
            // Format size in human-readable format
            std::stringstream ss;
            if (fileSize >= 1024 * 1024 * 1024) {
                ss << std::fixed << std::setprecision(2) << (fileSize / (1024.0 * 1024.0 * 1024.0)) << " GB";
            } else if (fileSize >= 1024 * 1024) {
                ss << std::fixed << std::setprecision(2) << (fileSize / (1024.0 * 1024.0)) << " MB";
            } else if (fileSize >= 1024) {
                ss << std::fixed << std::setprecision(2) << (fileSize / 1024.0) << " KB";
            } else {
                ss << fileSize << " bytes";
            }
            
            datasetSizeInfo = "Dataset: " + ss.str();
            
            // Estimate number of samples based on file size
            // Rough estimate: ~2KB per training sample in .grmt format
            int estimatedSamples = static_cast<int>(fileSize / 2048);
            datasetSizeInfo += " (~" + std::to_string(estimatedSamples) + " samples)";
            
        } else {
            datasetSizeInfo = "Dataset: Not found";
        }
    } catch (const std::exception& e) {
        datasetSizeInfo = "Dataset: Error reading size";
        addLog(std::string("Error reading dataset size: ") + e.what(), 1);
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


