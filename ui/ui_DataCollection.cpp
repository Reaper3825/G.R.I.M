
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <Windows.h>
#include <filesystem>
#include "ui_DataCollection.hpp"
#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "resources.hpp"
#include "control/ai_config_paths.hpp"
#include "core/input_parser.hpp"
#include "ui_theme.hpp"
#include "ui_draw_helpers.hpp"

UIDataCollectionPanel::UIDataCollectionPanel()
    : UIPanel("DataCollection", true),
      collectionActive(false),
      collectionCompleted(false),
      pollTimer(0.0f),
      pollInterval(0.2f),
      statsUpdateTimer(0.0f),
      statsUpdateInterval(2.0f),
      currentProgress(0.0f),
      sourcesProcessed(0),
      totalSources(0),
      checkpointsCollected(0),
      fetchLimit(100),
      vocabSize(50000),
      verificationThreshold(0.7f),
      maxLogEntries(1000),
      logScrollPosition(0.0f),
      autoScrollLogs(true),
      leftPanelScrollPosition(0.0f),
      leftPanelContentHeight(0.0f),
      collectionAnimTime(0.0f),
      selectedHFDataset(-1),
      activeSourceFilter("all"),
      activeStatusFilter("all"),
      maxHFResults(4)
{

    // Panel dimensions
    position = { 250, 550 };
    size = { 1350, 900 };
    setVisible(false);
    setBackground(0xE0181818);




    
    // Initialize scroll box for left panel
    leftPanelScrollBox = std::make_shared<UIScrollBox>();
    leftPanelScrollBox->setChildSpacing(5.0f);
    
    // Initialize HF results scroll box
    hfResultsScrollBox = std::make_shared<UIScrollBox>();
    hfResultsScrollBox->setChildSpacing(5.0f);
    
    // Initialize data collection manager
    collectionManager = std::make_unique<GRIM::DataCollection::DataCollectionManager>();
    
    // Initialize Hugging Face webhook
    hfWebhook = std::make_unique<GRIM::DataCollection::HuggingFaceWebhook>();
    
    // Initialize buttons
    startFullButton = std::make_shared<UIButton>("Start Full Pipeline", [this]() {
        startFullCollection();
    });
    
    startCollectButton = std::make_shared<UIButton>("Collect Only", [this]() {
        startCollectOnly();
    });
    
    startVerifyButton = std::make_shared<UIButton>("Verify Only", [this]() {
        startVerifyOnly();
    });
    
    startMergeButton = std::make_shared<UIButton>("Merge Only", [this]() {
        startMergeOnly();
    });
    
    forceRebuildButton = std::make_shared<UIButton>("Force Rebuild", [this]() {
        startForceRebuild();
    });
    
    stopButton = std::make_shared<UIButton>("Stop Collection", [this]() {
        stopCollection();
    });
    
    addSourceButton = std::make_shared<UIButton>("Add Source", [this]() {
        if (!sourceUrlBuffer.empty()) {
            addDataSource(sourceUrlBuffer);
            sourceUrlBuffer.clear();
        }
    });
    
    refreshStatsButton = std::make_shared<UIButton>("Refresh Stats", [this]() {
        updateDatasetStats();
        updateSourceList();
    });
    
    // Initialize Hugging Face buttons
    searchHFButton = std::make_shared<UIButton>("Search HF Datasets", [this]() {
        searchHuggingFaceDatasets();
    });
    
    browseHFButton = std::make_shared<UIButton>("Browse Popular", [this]() {
        browseHuggingFaceDatasets();
    });
    
    // Initialize filter buttons
    filterWebButton = std::make_shared<UIButton>("Filter: Web", [this]() {
        setSourceFilter("web");
    });
    
    filterHFButton = std::make_shared<UIButton>("Filter: HF", [this]() {
        setSourceFilter("huggingface");
    });
    
    filterAllButton = std::make_shared<UIButton>("Show All", [this]() {
        setSourceFilter("all");
    });
    
    clearFiltersButton = std::make_shared<UIButton>("Clear Filters", [this]() {
        clearFilters();
    });
    
    // Initialize queue buttons
    processQueueButton = std::make_shared<UIButton>("Process Queue", [this]() {
        processDownloadQueue();
    });
    
    clearQueueButton = std::make_shared<UIButton>("Clear Queue", [this]() {
        clearDownloadQueue();
    });
    
    // Initialize input boxes
    sourceUrlInput = std::make_shared<UIInputBox>();
    sourceUrlInput->setText("");
    sourceUrlInput->setPlaceholder("Enter data source URL...");
    
    hfSearchInput = std::make_shared<UIInputBox>();
    hfSearchInput->setText("");
    hfSearchInput->setPlaceholder("Search Hugging Face datasets...");
    
    // Initialize category dropdown with common HF categories
    std::vector<std::string> categories = {
        "All Categories",
        "text-generation",
        "question-answering",
        "summarization",
        "translation",
        "text-classification",
        "conversational",
        "token-classification",
        "fill-mask",
        "text2text-generation",
        "sentence-similarity"
    };
    hfCategoryDropdown = std::make_shared<UIDropdown>("Category", categories, 0, 
        [this](int idx, const std::string& category) {
            if (idx > 0) {  // Skip "All Categories"
                searchHuggingFaceByCategory(category);
            }
        });
    hfCategoryDropdown->setMaxVisibleItems(8);
    
    hfTokenInput = std::make_shared<UIInputBox>();
    hfTokenInput->setPlaceholder("HF API Token (optional)...");
    
    // Load HF token from config
    loadHFTokenFromConfig();
    
    // Initialize sliders with save-on-change callbacks
    fetchLimitSlider = std::make_shared<UISlider>("Fetch Limit", 10.0f, 1000.0f, 
        static_cast<float>(fetchLimit),
        [this](float val) { 
            fetchLimit = static_cast<int>(val); 
            saveUIConfig();  // Persist change
        });
    
    vocabSizeSlider = std::make_shared<UISlider>("Vocab Size", 5000.0f, 100000.0f,
        static_cast<float>(vocabSize),
        [this](float val) { 
            vocabSize = static_cast<int>(val); 
            saveUIConfig();  // Persist change
        });
    
    verificationThresholdSlider = std::make_shared<UISlider>("Verify Threshold", 0.0f, 1.0f,
        verificationThreshold,
        [this](float val) { 
            verificationThreshold = val; 
            saveUIConfig();  // Persist change
        });
    
    maxHFResultsSlider = std::make_shared<UISlider>("Max HF Results", 1.0f, 20.0f,
        static_cast<float>(maxHFResults),
        [this](float val) { 
            maxHFResults = static_cast<int>(val); 
            saveUIConfig();  // Persist change
        });
    
    // Initialize progress bar with theme colors
    collectionProgressBar = std::make_shared<UIProgressBar>("Collection Progress", 1.0f);
    collectionProgressBar->setFillColor(UITheme::Colors::Info);
    collectionProgressBar->setBackgroundColor(UITheme::Colors::Background);
    
    // Initialize layout
    buttonVBox = std::make_shared<UIVBox>(LayoutDirection::Vertical, 10.0f);
    
    // Load persisted UI config (slider values, filters)
    loadUIConfig();
    
    // Load initial data
    loadSourcesFromJSON();
    updateDatasetStats();
    updateSourceList();
    
    // Load persisted download queue
    loadDownloadQueue();
    
    addLog("Data Collection Panel initialized", 0);
}

UIDataCollectionPanel::~UIDataCollectionPanel() {
    // Save UI config and queue before shutting down
    saveUIConfig();
    saveDownloadQueue();
    
    if (collectionManager) {
        collectionManager->shutdown();
    }
}

void UIDataCollectionPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
    
    UIPanel::update(input, dt);
    
    // Update poll timer
    pollTimer += dt;
    if (pollTimer >= pollInterval) {
        pollTimer = 0.0f;
        pollCollectionManager();
    }
    
    // Update stats periodically
    statsUpdateTimer += dt;
    if (statsUpdateTimer >= statsUpdateInterval) {
        statsUpdateTimer = 0.0f;
        updateDatasetStats();
    }
    
    // Update collection animation
    if (collectionActive) {
        collectionAnimTime += dt;
    }
    
    // Update search animation timer
    if (hfSearching.load()) {
        searchAnimTime += dt;
    }
    
    // Update widgets
    if (sourceUrlInput) {
        sourceUrlInput->update(input, dt);
        sourceUrlBuffer = sourceUrlInput->getText();
    }
    
    if (hfSearchInput) {
        hfSearchInput->update(input, dt);
        hfSearchBuffer = hfSearchInput->getText();
    }
    
    if (hfCategoryDropdown) {
        hfCategoryDropdown->update(input, dt);
    }
    
    if (hfTokenInput) {
        hfTokenInput->update(input, dt);
        std::string newToken = hfTokenInput->getText();
        if (newToken != hfTokenBuffer) {
            hfTokenBuffer = newToken;
            if (hfWebhook) {
                hfWebhook->setApiToken(hfTokenBuffer);
            }
        }
    }
    
    // Update sliders
    if (fetchLimitSlider) fetchLimitSlider->update(input, dt);
    if (vocabSizeSlider) vocabSizeSlider->update(input, dt);
    if (verificationThresholdSlider) verificationThresholdSlider->update(input, dt);
    if (maxHFResultsSlider) maxHFResultsSlider->update(input, dt);
    
    // Update buttons
    if (startFullButton) startFullButton->update(input, dt);
    if (startCollectButton) startCollectButton->update(input, dt);
    if (startVerifyButton) startVerifyButton->update(input, dt);
    if (startMergeButton) startMergeButton->update(input, dt);
    if (forceRebuildButton) forceRebuildButton->update(input, dt);
    if (stopButton) stopButton->update(input, dt);
    if (addSourceButton) addSourceButton->update(input, dt);
    if (refreshStatsButton) refreshStatsButton->update(input, dt);
    if (searchHFButton) searchHFButton->update(input, dt);
    if (searchHFCategoryButton) searchHFCategoryButton->update(input, dt);
    if (browseHFButton) browseHFButton->update(input, dt);
    if (filterWebButton) filterWebButton->update(input, dt);
    if (filterHFButton) filterHFButton->update(input, dt);
    if (filterAllButton) filterAllButton->update(input, dt);
    if (clearFiltersButton) clearFiltersButton->update(input, dt);
    if (processQueueButton) processQueueButton->update(input, dt);
    if (clearQueueButton) clearQueueButton->update(input, dt);






    // Update left panel scrollbox
    float panelX = position.x + 10;
    float panelY = position.y + 40;
    float panelWidth = size.x - 20;
    float leftPanelWidth = panelWidth * 0.35f;
    float sliderWidth = leftPanelWidth - 15;
    
    if (leftPanelScrollBox) {
        leftPanelScrollBox->setPosition(panelX, panelY + 35);
        leftPanelScrollBox->setSize(leftPanelWidth, size.y - 85);
        leftPanelScrollBox->update(input, dt);
    }
    
    // Update HF results scrollbox (handles its own interaction)
    if (hfResultsScrollBox && !hfSearchResults.empty()) {
        // Position will be set in drawOverlay, just update here
        hfResultsScrollBox->update(input, dt);
    }
    
    // Handle queue item interactions (hover, click to remove, right-click context)
    hoveredQueueItem = -1;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (!downloadQueue.empty()) {
            // Approximate queue box position based on layout
            // This should match the drawOverlay calculations
            float queueBoxX = panelX + 10;
            float queueBoxWidth = sliderWidth;
            float queueItemHeight = 42;
            float queueBoxY = panelY + 300;  // Approximate - after HF results + spacing
            float queueBoxHeight = std::min(180.0f, downloadQueue.size() * queueItemHeight + 10);
            
            bool mouseOverQueue = (input.mousePos.x >= queueBoxX && 
                                   input.mousePos.x <= queueBoxX + queueBoxWidth &&
                                   input.mousePos.y >= queueBoxY &&
                                   input.mousePos.y <= queueBoxY + queueBoxHeight);
            
            if (mouseOverQueue) {
                // Determine which item is hovered
                float itemY = queueBoxY + 5;
                for (size_t i = 0; i < downloadQueue.size() && itemY + queueItemHeight <= queueBoxY + queueBoxHeight; ++i) {
                    if (input.mousePos.y >= itemY && input.mousePos.y <= itemY + queueItemHeight) {
                        hoveredQueueItem = static_cast<int>(i);
                        
                        // Check for click on remove button (X) - right side of item
                        if (input.mousePressed[0] && downloadQueue[i].status != "downloading") {
                            float removeX = queueBoxX + queueBoxWidth - 25;
                            if (input.mousePos.x >= removeX && input.mousePos.x <= removeX + 18) {
                                // Remove this item (outside of lock to avoid deadlock)
                                // Mark for removal
                                int indexToRemove = static_cast<int>(i);
                                // We can't call removeFromQueue here due to lock, so do it directly
                                std::string name = downloadQueue[indexToRemove].displayName;
                                downloadQueue.erase(downloadQueue.begin() + indexToRemove);
                                addLog("Removed from queue: " + name, 0);
                                break;
                            }
                            
                            // Check for retry button click (for failed items)
                            if (downloadQueue[i].status == "failed") {
                                float retryX = removeX - 25;
                                if (input.mousePos.x >= retryX && input.mousePos.x <= retryX + 20) {
                                    downloadQueue[i].status = "pending";
                                    downloadQueue[i].retryCount++;
                                    downloadQueue[i].progress = 0.0f;
                                    addLog("Retrying: " + downloadQueue[i].displayName, 0);
                                    break;
                                }
                            }
                        }
                        break;
                    }
                    itemY += queueItemHeight;
                }
                
                // Right-click to clear completed items
                if (input.mousePressed[1]) {
                    clearCompletedFromQueue();
                }
            }
        }
    }
}

void UIDataCollectionPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!isVisible()) return;
    
    UIPanel::drawOverlay(renderer);
    
    float panelX = position.x + 10;
    float panelY = position.y + 40;
    float panelWidth = size.x - 20;
    float panelHeight = size.y - 50;
    
    // Split into two columns
    float leftPanelWidth = panelWidth * 0.35f;
    float rightPanelWidth = panelWidth * 0.65f;
    float columnGap = 10;
    
    // ============================================================
    // LEFT PANEL - Configuration with scrolling
    // ============================================================
    float leftX = panelX;
    float leftY = panelY;
    
    // Status header
    std::string statusText = collectionActive ? "[ACTIVE] Collection" : "[IDLE] Ready";
    uint32_t statusColor = collectionActive ? UITheme::Colors::Info : UITheme::Colors::TextDisabled;
    renderer.drawText({leftX, leftY}, statusText, statusColor);
    UIDrawHelpers::drawDivider(renderer, {leftX, leftY + 20}, leftPanelWidth);
    leftY += 25;
    
    // Scrollable area
    float scrollAreaY = leftY;
    float scrollAreaHeight = panelHeight - (leftY - panelY) - 10;
    renderer.drawRect({leftX, scrollAreaY}, {leftPanelWidth, scrollAreaHeight}, UITheme::Colors::Background);
    
    // Calculate content height
    leftPanelContentHeight = 30 + (40 * 3) + 60 + 35 + 40 + 60 + 200;
    
    float offsetY = -leftPanelScrollPosition;
    float renderY = scrollAreaY + offsetY;
    
    // Sliders
    float sliderWidth = leftPanelWidth - 20;
    float sliderHeight = 35;
    
    // Configuration section
    if (renderY >= scrollAreaY - 30 && renderY <= scrollAreaY + scrollAreaHeight) {
        UIDrawHelpers::drawSectionHeader(renderer, {leftX + 10, renderY + 10}, sliderWidth, "Collection Configuration");
    }
    renderY += 40;
    
    if (fetchLimitSlider && renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
        fetchLimitSlider->setPosition(leftX + 10, renderY);
        fetchLimitSlider->setSize(sliderWidth, sliderHeight);
        fetchLimitSlider->drawOverlay(renderer, position);
    }
    renderY += sliderHeight + 5;
    
    if (vocabSizeSlider && renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
        vocabSizeSlider->setPosition(leftX + 10, renderY);
        vocabSizeSlider->setSize(sliderWidth, sliderHeight);
        vocabSizeSlider->drawOverlay(renderer, position);
    }
    renderY += sliderHeight + 5;
    
    if (verificationThresholdSlider && renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
        verificationThresholdSlider->setPosition(leftX + 10, renderY);
        verificationThresholdSlider->setSize(sliderWidth, sliderHeight);
        verificationThresholdSlider->drawOverlay(renderer, position);
    }
    renderY += sliderHeight + 5;
    
    if (maxHFResultsSlider && renderY >= scrollAreaY - sliderHeight && renderY <= scrollAreaY + scrollAreaHeight) {
        maxHFResultsSlider->setPosition(leftX + 10, renderY);
        maxHFResultsSlider->setSize(sliderWidth, sliderHeight);
        maxHFResultsSlider->drawOverlay(renderer, position);
    }
    renderY += sliderHeight + 15;
    
    // Add data source section
    if (renderY >= scrollAreaY - 80 && renderY <= scrollAreaY + scrollAreaHeight) {
        UIDrawHelpers::drawSectionHeader(renderer, {leftX + 10, renderY}, sliderWidth, "Add Data Source");
        renderY += 25;
        
        if (sourceUrlInput) {
            sourceUrlInput->setPosition(leftX + 10, renderY);
            sourceUrlInput->setSize(sliderWidth, 30);
            sourceUrlInput->drawOverlay(renderer, position);
            renderY += 35;
        }
        
        if (addSourceButton) {
            addSourceButton->setPosition(leftX + 10, renderY);
            addSourceButton->setSize(sliderWidth, 35);
            addSourceButton->drawOverlay(renderer, position);
        }
    }
    renderY += 40;
    
    // Hugging Face section
    if (renderY >= scrollAreaY - 200 && renderY <= scrollAreaY + scrollAreaHeight) {
        UIDrawHelpers::drawSectionHeader(renderer, {leftX + 10, renderY}, sliderWidth, "Hugging Face Datasets");
        renderY += 20;
        renderer.drawText({leftX + 15, renderY}, "Select category or search by keyword", UITheme::Colors::TextSecondary);
        renderY += 20;
        
        // HF Token input
        if (hfTokenInput) {
            hfTokenInput->setPosition(leftX + 10, renderY);
            hfTokenInput->setSize(sliderWidth, 25);
            hfTokenInput->drawOverlay(renderer, position);
            renderY += 30;
        }
        
        // HF Search input
        if (hfSearchInput) {
            hfSearchInput->setPosition(leftX + 10, renderY);
            hfSearchInput->setSize(sliderWidth, 25);
            hfSearchInput->drawOverlay(renderer, position);
            renderY += 30;
        }
        
        // HF Category dropdown
        if (hfCategoryDropdown) {
            hfCategoryDropdown->setPosition(leftX + 10, renderY);
            hfCategoryDropdown->setSize(sliderWidth, 30);
            hfCategoryDropdown->drawOverlay(renderer, position);
            renderY += 35;
        }
        
        // HF Buttons
        float btnHeight = 30;
        float btnWidth = sliderWidth / 3 - 3.5f;
        
        if (searchHFButton) {
            searchHFButton->setPosition(leftX + 10, renderY);
            searchHFButton->setSize(btnWidth, btnHeight);
            searchHFButton->drawOverlay(renderer, position);
        }
        
        if (searchHFCategoryButton) {
            searchHFCategoryButton->setPosition(leftX + 10 + btnWidth + 5, renderY);
            searchHFCategoryButton->setSize(btnWidth, btnHeight);
            searchHFCategoryButton->drawOverlay(renderer, position);
        }
        
        if (browseHFButton) {
            browseHFButton->setPosition(leftX + 10 + (btnWidth + 5) * 2, renderY);
            browseHFButton->setSize(btnWidth, btnHeight);
            browseHFButton->drawOverlay(renderer, position);
        }
        renderY += btnHeight + 15;
        
        // HF search results - Using scrollbox with widgets
        if (!hfSearchResults.empty() && hfResultsScrollBox) {
            UIDrawHelpers::drawSectionHeader(renderer, {leftX + 10, renderY}, sliderWidth, "Search Results:");
            renderY += 20;
            
            float resultsBoxHeight = std::min(150.0f, maxHFResults * 75.0f);
            
            // Position and size the scrollbox FIRST
            hfResultsScrollBox->setPosition(leftX + 10, renderY);
            hfResultsScrollBox->setSize(leftPanelWidth - 20, resultsBoxHeight);
            
            // NOW populate if background thread signaled ready OR if scrollbox is empty but we have results
            if (hfResultsNeedsPopulate.load() || hfResultsScrollBox->getChildren().empty()) {
                populateHFResults(leftPanelWidth - 20);
                hfResultsNeedsPopulate.store(false);
            }
            
            // Draw the scrollbox with positioned children
            hfResultsScrollBox->drawOverlay(renderer, position);
            
            renderY += resultsBoxHeight + UITheme::Spacing::Medium;
        } else if (hfSearching.load()) {
            // Show loading indicator when searching
            renderer.drawText({leftX + 10, renderY}, "Search Results:", UITheme::Colors::Primary);
            renderY += 20;
            
            // Animated loading dots
            int dots = static_cast<int>(searchAnimTime * 3.0f) % 4;
            std::string loadingText = "Searching HuggingFace" + std::string(dots, '.');
            renderer.drawText({leftX + 15, renderY}, loadingText, UITheme::Colors::Warning);
            renderY += 20;
        } else if (!lastSearchError.empty()) {
            // Show error state
            renderer.drawText({leftX + 10, renderY}, "Search Results:", UITheme::Colors::Primary);
            renderY += 20;
            renderer.drawText({leftX + 15, renderY}, "⚠ " + lastSearchError, UITheme::Colors::Danger);
            renderY += 20;
        }
    }
    renderY += UITheme::Spacing::Large;
    
    // Download Queue section with improved UI using UITheme
    if (renderY >= scrollAreaY - 200 && renderY <= scrollAreaY + scrollAreaHeight) {
        // Header with queue count
        size_t queueCount = 0;
        size_t pendingCount = 0;
        size_t completedCount = 0;
        size_t failedCount = 0;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            queueCount = downloadQueue.size();
            for (const auto& item : downloadQueue) {
                if (item.status == "pending") pendingCount++;
                else if (item.status == "completed") completedCount++;
                else if (item.status == "failed") failedCount++;
            }
        }
        
        std::string queueHeader = "Download Queue";
        if (queueCount > 0) {
            queueHeader += " (" + std::to_string(queueCount) + ")";
        }
        if (queueProcessing.load()) {
            queueHeader += " [PROCESSING]";
        }
        renderer.drawText({leftX + 10, renderY}, queueHeader, UITheme::Colors::Warning);
        
        // Show stats summary using theme colors
        if (queueCount > 0) {
            std::string statsText = "";
            if (pendingCount > 0) statsText += std::to_string(pendingCount) + " pending ";
            if (completedCount > 0) statsText += std::to_string(completedCount) + " done ";
            if (failedCount > 0) statsText += std::to_string(failedCount) + " failed";
            if (!statsText.empty()) {
                renderer.drawText({leftX + sliderWidth - 120, renderY}, statsText, UITheme::Colors::TextDisabled);
            }
        }
        renderY += 25;
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            if (!downloadQueue.empty()) {
                float queueItemHeight = UITheme::Sizes::WidgetHeight;  // Use theme size
                float queueBoxHeight = std::min(180.0f, downloadQueue.size() * queueItemHeight + 10);
                
                // Draw queue container using theme
                renderer.drawRect({leftX + 10, renderY}, {sliderWidth, queueBoxHeight}, UITheme::Colors::ScrollboxBg);
                
                float itemY = renderY + 5;
                for (size_t i = 0; i < downloadQueue.size() && itemY + queueItemHeight <= renderY + queueBoxHeight; ++i) {
                    const auto& item = downloadQueue[i];
                    bool isHovered = (hoveredQueueItem == static_cast<int>(i));
                    bool isCurrentlyProcessing = (currentQueueIndex.load() == static_cast<int>(i));
                    
                    // Draw item using UIDrawHelpers
                    UIDrawHelpers::drawWidgetBackground(renderer,
                        {leftX + 12, itemY},
                        {sliderWidth - 4, queueItemHeight - 4},
                        isHovered, isCurrentlyProcessing, false);
                    
                    // Status indicator bar using theme colors
                    uint32_t statusColor = UITheme::Colors::TextDisabled;
                    if (item.status == "downloading") statusColor = UITheme::Colors::Info;
                    else if (item.status == "completed") statusColor = UITheme::Colors::Success;
                    else if (item.status == "failed") statusColor = UITheme::Colors::Danger;
                    else if (item.status == "pending") statusColor = UITheme::Colors::Warning;
                    
                    UIDrawHelpers::drawCategoryIndicator(renderer, {leftX + 12, itemY}, queueItemHeight - 4, statusColor);
                    
                    // Dataset name (truncated)
                    std::string displayName = item.displayName;
                    if (displayName.length() > 30) displayName = displayName.substr(0, 27) + "...";
                    renderer.drawText({leftX + 20, itemY + 3}, displayName, UITheme::Colors::TextPrimary);
                    
                    // Status line with progress or error
                    std::string statusLine = "[" + item.status + "]";
                    if (item.status == "downloading") {
                        int pct = static_cast<int>(item.progress * 100);
                        statusLine = "Downloading... " + std::to_string(pct) + "%";
                        
                        // Draw progress bar using theme colors
                        float progressBarWidth = sliderWidth - 80;
                        float progressBarHeight = UITheme::Sizes::SliderHeight * 0.4f;
                        float progressBarY = itemY + queueItemHeight - 12;
                        renderer.drawRect({leftX + 20, progressBarY}, {progressBarWidth, progressBarHeight}, UITheme::Colors::SliderTrack);
                        renderer.drawRect({leftX + 20, progressBarY}, {progressBarWidth * item.progress, progressBarHeight}, UITheme::Colors::Info);
                    } else if (item.status == "failed" && !item.errorMessage.empty()) {
                        std::string err = item.errorMessage;
                        if (err.length() > 35) err = err.substr(0, 32) + "...";
                        statusLine = "✗ " + err;
                    } else if (item.status == "completed") {
                        statusLine = "✓ Download complete";
                    } else if (item.status == "pending") {
                        if (item.retryCount > 0) {
                            statusLine = "Pending (retry #" + std::to_string(item.retryCount) + ")";
                        } else {
                            statusLine = "Waiting in queue...";
                        }
                    }
                    renderer.drawText({leftX + 20, itemY + 18}, statusLine, statusColor);
                    
                    // Hover actions: [X] remove button
                    if (isHovered && item.status != "downloading") {
                        float removeX = leftX + sliderWidth - 25;
                        UIDrawHelpers::drawWidgetBackground(renderer, {removeX, itemY + 8}, {18, 18}, true, false, false);
                        renderer.drawRect({removeX, itemY + 8}, {18, 18}, UITheme::Colors::Danger & 0x44FFFFFF);
                        renderer.drawText({removeX + 4, itemY + 8}, "X", UITheme::Colors::Danger);
                        
                        // Retry button for failed items
                        if (item.status == "failed") {
                            float retryX = removeX - 25;
                            UIDrawHelpers::drawWidgetBackground(renderer, {retryX, itemY + 8}, {20, 18}, true, false, false);
                            renderer.drawRect({retryX, itemY + 8}, {20, 18}, UITheme::Colors::Success & 0x44FFFFFF);
                            renderer.drawText({retryX + 3, itemY + 8}, "⟳", UITheme::Colors::Success);
                        }
                    }
                    
                    itemY += queueItemHeight;
                }
                renderY += queueBoxHeight + 5;
            } else {
                renderer.drawText({leftX + 15, renderY}, "Queue empty - click datasets to add", UITheme::Colors::TextDisabled);
                renderY += 20;
            }
        }
        renderY += 10;
        
        // Queue control buttons
        float btnHeight = UITheme::Sizes::ButtonHeight;
        float btnWidth = sliderWidth / 3 - 3.5f;
        
        // Process Queue button - show "Processing..." when active
        if (processQueueButton && renderY <= scrollAreaY + scrollAreaHeight) {
            if (queueProcessing.load()) {
                processQueueButton->setText("Processing...");
            } else {
                processQueueButton->setText("Process Queue");
            }
            processQueueButton->setPosition(leftX + 10, renderY);
            processQueueButton->setSize(btnWidth, btnHeight);
            processQueueButton->drawOverlay(renderer, position);
        }
        
        if (clearQueueButton && renderY <= scrollAreaY + scrollAreaHeight) {
            clearQueueButton->setPosition(leftX + 10 + btnWidth + 5, renderY);
            clearQueueButton->setSize(btnWidth, btnHeight);
            clearQueueButton->drawOverlay(renderer, position);
        }
        
        // Add "Clear Done" button - new
        // This would need a new button, but we can reuse clearFiltersButton position for now
        // or add text hint
        renderer.drawText({leftX + 10 + (btnWidth + 5) * 2 + 5, renderY + 7}, "Right-click: clear done", UITheme::Colors::TextDisabled);
        
        renderY += btnHeight + 15;
    }
    
    // Filter section
    if (renderY >= scrollAreaY - 100 && renderY <= scrollAreaY + scrollAreaHeight) {
        UIDrawHelpers::drawSectionHeader(renderer, {leftX + 10, renderY}, sliderWidth, "Filters");
        renderY += 25;
        
        std::string filterStatus = "Source: " + activeSourceFilter + " | Status: " + activeStatusFilter;
        renderer.drawText({leftX + 15, renderY}, filterStatus, UITheme::Colors::TextSecondary);
        renderY += 20;
        
        float btnHeight = UITheme::Sizes::ButtonHeight;
        float btnWidth = sliderWidth / 3 - 3.5f;
        
        if (filterAllButton && renderY <= scrollAreaY + scrollAreaHeight) {
            filterAllButton->setPosition(leftX + 10, renderY);
            filterAllButton->setSize(btnWidth, btnHeight);
            filterAllButton->drawOverlay(renderer, position);
        }
        
        if (filterWebButton && renderY <= scrollAreaY + scrollAreaHeight) {
            filterWebButton->setPosition(leftX + 10 + btnWidth + 5, renderY);
            filterWebButton->setSize(btnWidth, btnHeight);
            filterWebButton->drawOverlay(renderer, position);
        }
        
        if (filterHFButton && renderY <= scrollAreaY + scrollAreaHeight) {
            filterHFButton->setPosition(leftX + 10 + (btnWidth + 5) * 2, renderY);
            filterHFButton->setSize(btnWidth, btnHeight);
            filterHFButton->drawOverlay(renderer, position);
        }
        renderY += btnHeight + 5;
        
        if (clearFiltersButton && renderY <= scrollAreaY + scrollAreaHeight) {
            clearFiltersButton->setPosition(leftX + 10, renderY);
            clearFiltersButton->setSize(sliderWidth, btnHeight);
            clearFiltersButton->drawOverlay(renderer, position);
        }
        renderY += btnHeight + 15;
    }
    
    // Source list section
    if (renderY >= scrollAreaY - 200 && renderY <= scrollAreaY + scrollAreaHeight) {
        UIDrawHelpers::drawSectionHeader(renderer, {leftX + 10, renderY}, sliderWidth, "Data Sources");
        renderY += 25;
        
        if (!sourceListInfo.empty()) {
            std::istringstream stream(sourceListInfo);
            std::string line;
            while (std::getline(stream, line) && renderY <= scrollAreaY + scrollAreaHeight) {
                renderer.drawText({leftX + 15, renderY}, line, UITheme::Colors::TextSecondary);
                renderY += 18;
            }
        } else {
            renderer.drawText({leftX + 15, renderY}, "No sources configured", UITheme::Colors::TextDisabled);
        }
    }
    
    // Draw scroll bar using theme colors
    if (leftPanelContentHeight > scrollAreaHeight) {
        float scrollBarX = leftX + leftPanelWidth - 10;
        float scrollBarWidth = 8;
        float scrollBarHeight = (scrollAreaHeight / leftPanelContentHeight) * scrollAreaHeight;
        float scrollBarY = scrollAreaY + (leftPanelScrollPosition / leftPanelContentHeight) * scrollAreaHeight;
        
        renderer.drawRect({scrollBarX, scrollAreaY}, {scrollBarWidth, scrollAreaHeight}, UITheme::Colors::SliderTrack);
        renderer.drawRect({scrollBarX, scrollBarY}, {scrollBarWidth, scrollBarHeight}, UITheme::Colors::Primary);
    }
    
    // ============================================================
    // RIGHT PANEL - Stats, Controls, and Output
    // ============================================================
    float rightX = panelX + leftPanelWidth + columnGap;
    float rightY = panelY;
    
    // Collection status
    renderer.drawText({rightX, rightY}, "Phase: " + currentPhase, UITheme::Colors::Primary);
    rightY += 25;
    
    if (!collectionMessage.empty()) {
        renderer.drawText({rightX, rightY}, collectionMessage, UITheme::Colors::TextPrimary);
        rightY += 20;
    }
    
    // Stats
    if (!datasetSizeInfo.empty()) {
        renderer.drawText({rightX, rightY}, datasetSizeInfo, UITheme::Colors::Success);
        rightY += 20;
    }
    
    if (!checkpointStatsInfo.empty()) {
        renderer.drawText({rightX, rightY}, checkpointStatsInfo, UITheme::Colors::TextSecondary);
        rightY += 20;
    }
    
    if (!verificationStatsInfo.empty()) {
        renderer.drawText({rightX, rightY}, verificationStatsInfo, UITheme::Colors::TextSecondary);
        rightY += 25;
    }
    
    // Progress bar
    if (collectionProgressBar) {
        float progressBarWidth = rightPanelWidth - 20;
        float progressBarHeight = UITheme::Sizes::ProgressBarHeight;
        
        collectionProgressBar->setValue(currentProgress / 100.0f);
        collectionProgressBar->setPosition({rightX, rightY});
        collectionProgressBar->setSize({progressBarWidth, progressBarHeight});
        collectionProgressBar->drawOverlay(renderer, position);
        rightY += progressBarHeight + 20;
    }
    
    // Control buttons
    float btnHeight = UITheme::Sizes::ButtonHeight + 5;
    float btnWidth = 180;
    
    if (buttonVBox) {
        buttonVBox->setPosition(rightX, rightY);
        buttonVBox->clearWidgets();
        
        if (startFullButton) {
            startFullButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(startFullButton);
        }
        if (startCollectButton) {
            startCollectButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(startCollectButton);
        }
        if (startVerifyButton) {
            startVerifyButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(startVerifyButton);
        }
        if (startMergeButton) {
            startMergeButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(startMergeButton);
        }
        if (forceRebuildButton) {
            forceRebuildButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(forceRebuildButton);
        }
        if (stopButton) {
            stopButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(stopButton);
        }
        if (refreshStatsButton) {
            refreshStatsButton->setSize(btnWidth, btnHeight);
            buttonVBox->addWidget(refreshStatsButton);
        }
        
        buttonVBox->layout();
        buttonVBox->drawOverlay(renderer, position);
    }
    
    if (startFullButton) startFullButton->drawOverlay(renderer, position);
    if (startCollectButton) startCollectButton->drawOverlay(renderer, position);
    if (startVerifyButton) startVerifyButton->drawOverlay(renderer, position);
    if (startMergeButton) startMergeButton->drawOverlay(renderer, position);
    if (forceRebuildButton) forceRebuildButton->drawOverlay(renderer, position);
    if (stopButton) stopButton->drawOverlay(renderer, position);
    if (refreshStatsButton) refreshStatsButton->drawOverlay(renderer, position);
    
    rightY += (btnHeight + 10) * 6 + 20;
    
    // Logs section
    float logWidth = rightPanelWidth;
    UIDrawHelpers::drawSectionHeader(renderer, {rightX, rightY}, logWidth, "Collection Logs");
    rightY += 25;
    
    float logHeight = panelHeight - (rightY - panelY);
    
    renderer.drawRect({rightX, rightY}, {logWidth, logHeight}, UITheme::Colors::Background);
    UIDrawHelpers::drawDivider(renderer, {rightX, rightY}, logWidth);
    
    // Draw log entries
    {
        std::lock_guard<std::mutex> lock(logMutex);
        
        float logY = rightY + 5;
        int visibleLogs = static_cast<int>(logHeight / 18);
        int startIdx = std::max(0, static_cast<int>(logEntries.size()) - visibleLogs);
        
        for (size_t i = startIdx; i < logEntries.size(); i++) {
            const auto& entry = logEntries[i];
            
            uint32_t color = UITheme::Colors::Success;
            if (entry.level == 1) color = UITheme::Colors::Warning;
            else if (entry.level == 2) color = UITheme::Colors::Danger;
            
            renderer.drawText({rightX + 5, logY}, entry.timestamp + " " + entry.message, color);
            
            logY += 18;
            if (logY > rightY + logHeight) break;
        }
    }
    
    // Draw expanded dropdown on top of everything else
    if (hfCategoryDropdown && hfCategoryDropdown->isExpanded()) {
        hfCategoryDropdown->drawExpandedList(renderer, position);
    }
}

void UIDataCollectionPanel::startFullCollection() {
    if (collectionActive) {
        addLog("Collection already active", 1);
        return;
    }
    
    addLog("=== STARTING FULL COLLECTION PIPELINE ===", 0);
    collectionActive = true;
    collectionCompleted = false;
    
    if (collectionManager->startCollection("full")) {
        addLog("Full collection pipeline started", 0);
    } else {
        addLog("Failed to start collection", 2);
        collectionActive = false;
    }
}

void UIDataCollectionPanel::startCollectOnly() {
    if (collectionActive) {
        addLog("Collection already active", 1);
        return;
    }
    
    addLog("=== STARTING DATA COLLECTION ===", 0);
    collectionActive = true;
    collectionCompleted = false;
    
    if (collectionManager->startCollection("collect")) {
        addLog("Data collection started", 0);
    } else {
        addLog("Failed to start collection", 2);
        collectionActive = false;
    }
}

void UIDataCollectionPanel::startVerifyOnly() {
    if (collectionActive) {
        addLog("Collection already active", 1);
        return;
    }
    
    if (!collectionManager) {
        addLog("ERROR: Collection manager not initialized", 2);
        return;
    }
    
    addLog("=== STARTING DATA VERIFICATION ===", 0);
    collectionActive = true;
    collectionCompleted = false;
    
    if (collectionManager->startCollection("verify")) {
        addLog("Data verification started", 0);
    } else {
        addLog("Failed to start verification", 2);
        collectionActive = false;
    }
}

void UIDataCollectionPanel::startMergeOnly() {
    if (collectionActive) {
        addLog("Collection already active", 1);
        return;
    }
    
    addLog("=== STARTING DATA MERGE ===", 0);
    collectionActive = true;
    collectionCompleted = false;
    
    if (collectionManager->startCollection("merge")) {
        addLog("Data merge started", 0);
    } else {
        addLog("Failed to start merge", 2);
        collectionActive = false;
    }
}

void UIDataCollectionPanel::startForceRebuild() {
    if (collectionActive) {
        addLog("Collection already active", 1);
        return;
    }
    
    addLog("=== STARTING FORCE REBUILD (ignoring dedup state) ===", 0);
    collectionActive = true;
    collectionCompleted = false;
    
    if (collectionManager->startCollection("merge-rebuild")) {
        addLog("Force rebuild started - processing all verified data", 0);
    } else {
        addLog("Failed to start force rebuild", 2);
        collectionActive = false;
    }
}

void UIDataCollectionPanel::stopCollection() {
    if (!collectionActive) {
        addLog("No active collection to stop", 1);
        return;
    }
    
    addLog("Stopping collection...", 0);
    collectionManager->stopCollection();
    collectionActive = false;
}

void UIDataCollectionPanel::addDataSource(const std::string& url) {
    if (url.empty()) {
        addLog("Cannot add empty URL", 1);
        return;
    }
    
    if (url.find("http://") != 0 && url.find("https://") != 0) {
        addLog("Invalid URL - must start with http:// or https://", 2);
        return;
    }
    
    addLog("Adding data source: " + url, 0);
    saveSourceToJSON(url);
    updateSourceList();
}

void UIDataCollectionPanel::loadSourcesFromJSON() {
    std::string sourcePath = getResourcePath() + "/models/GRIM-text/training/source_data.json";
    
    try {
        if (std::filesystem::exists(sourcePath)) {
            std::ifstream sourceFile(sourcePath);
            nlohmann::json sourceData;
            sourceFile >> sourceData;
            sourceFile.close();
            
            if (sourceData.contains("data_sources")) {
                totalSources = 0;
                for (const auto& source : sourceData["data_sources"]) {
                    if (source.value("enabled", false)) {
                        totalSources++;
                    }
                }
                addLog("Loaded " + std::to_string(totalSources) + " enabled sources", 0);
            }
        }
    } catch (const std::exception& e) {
        addLog("Error loading sources: " + std::string(e.what()), 2);
    }
}

void UIDataCollectionPanel::loadHFTokenFromConfig() {
    std::string configPath = "ai_config.json";
    
    try {
        if (std::filesystem::exists(configPath)) {
            std::ifstream configFile(configPath);
            nlohmann::json config;
            configFile >> config;
            configFile.close();
            
            if (config.contains("api_keys") && config["api_keys"].contains("huggingface")) {
                std::string token = config["api_keys"]["huggingface"];
                if (!token.empty()) {
                    hfTokenInput->setText(token);
                    hfTokenBuffer = token;
                    if (hfWebhook) {
                        hfWebhook->setApiToken(token);
                    }
                    addLog("HuggingFace token loaded from config", 0);
                }
            }
        }
    } catch (const std::exception& e) {
        addLog("Error loading HF token from config: " + std::string(e.what()), 1);
    }
}

void UIDataCollectionPanel::saveSourceToJSON(const std::string& url) {
    std::string sourcePath = getResourcePath() + "/models/GRIM-text/training/source_data.json";
    
    try {
        nlohmann::json sourceData;
        
        std::ifstream inFile(sourcePath);
        if (inFile.good()) {
            inFile >> sourceData;
            inFile.close();
        } else {
            sourceData = {
                {"version", "1.0.0"},
                {"description", "GRIM Web Data Collection Configuration"},
                {"data_sources", nlohmann::json::array()}
            };
        }
        
        nlohmann::json newSource = {
            {"name", "Custom Source"},
            {"url", url},
            {"source_type", "custom"},
            {"enabled", true},
            {"priority", 5},
            {"fetch_limit", fetchLimit}
        };
        
        if (!sourceData.contains("data_sources")) {
            sourceData["data_sources"] = nlohmann::json::array();
        }
        sourceData["data_sources"].push_back(newSource);
        
        std::ofstream outFile(sourcePath);
        if (outFile.is_open()) {
            outFile << sourceData.dump(2);
            outFile.close();
            addLog("Data source added successfully", 0);
            loadSourcesFromJSON();
        }
    } catch (const std::exception& e) {
        addLog("Error saving source: " + std::string(e.what()), 2);
    }
}

void UIDataCollectionPanel::pollCollectionManager() {
    if (!collectionManager) return;
    
    auto status = collectionManager->getStatus();
    
    currentPhase = status.phase;
    currentProgress = status.progress;
    collectionMessage = status.message;
    sourcesProcessed = status.sourcesProcessed;
    checkpointsCollected = status.checkpointsCollected;
    
    if (status.phase == "complete" && collectionActive) {
        addLog("Collection completed successfully!", 0);
        collectionActive = false;
        collectionCompleted = true;
        updateDatasetStats();
    } else if (status.phase == "error" && collectionActive) {
        addLog("Collection error: " + status.message, 2);
        collectionActive = false;
    }
}

void UIDataCollectionPanel::updateDatasetStats() {
    GRIM::Config::GrimTextPaths paths;
    if (!GRIM::Config::loadGrimTextPaths(paths)) {
        datasetSizeInfo = "Dataset: Config error";
        return;
    }
    
    try {
        std::string grmtPath = paths.training_data;
        if (std::filesystem::exists(grmtPath)) {
            auto fileSize = std::filesystem::file_size(grmtPath);
            std::stringstream ss;
            ss << "Dataset: ";
            if (fileSize >= 1024ull * 1024ull * 1024ull) {
                ss << std::fixed << std::setprecision(2)
                   << (fileSize / (1024.0 * 1024.0 * 1024.0)) << " GB";
            } else if (fileSize >= 1024ull * 1024ull) {
                ss << std::fixed << std::setprecision(2)
                   << (fileSize / (1024.0 * 1024.0)) << " MB";
            } else {
                ss << (fileSize / 1024) << " KB";
            }
            datasetSizeInfo = ss.str();
        } else {
            datasetSizeInfo = "Dataset: Not found";
        }
        
        // Check checkpoint directory
        std::string checkpointDir = getResourcePath() + "/models/GRIM-text/training/checkpoints";
        if (std::filesystem::exists(checkpointDir)) {
            int checkpointCount = 0;
            for (const auto& entry : std::filesystem::directory_iterator(checkpointDir)) {
                if (entry.path().extension() == ".ckpt") {
                    checkpointCount++;
                }
            }
            checkpointStatsInfo = "Checkpoints: " + std::to_string(checkpointCount);
        }
    } catch (const std::exception& e) {
        datasetSizeInfo = "Dataset: Error";
    }
}

void UIDataCollectionPanel::updateSourceList() {
    std::string sourcePath = getResourcePath() + "/models/GRIM-text/training/source_data.json";
    
    try {
        if (std::filesystem::exists(sourcePath)) {
            std::ifstream sourceFile(sourcePath);
            nlohmann::json sourceData;
            sourceFile >> sourceData;
            sourceFile.close();
            
            std::stringstream ss;
            int count = 0;
            int totalCount = 0;
            if (sourceData.contains("data_sources")) {
                for (const auto& source : sourceData["data_sources"]) {
                    if (source.value("enabled", false)) {
                        totalCount++;
                        std::string name = source.value("name", "Unknown");
                        std::string url = source.value("url", "");
                        
                        // Apply source filter
                        if (activeSourceFilter != "all") {
                            bool matchesFilter = false;
                            if (activeSourceFilter == "web" && 
                                (url.find("http://") == 0 || url.find("https://") == 0) && 
                                url.find("huggingface") == std::string::npos) {
                                matchesFilter = true;
                            } else if (activeSourceFilter == "huggingface" && 
                                       url.find("huggingface://") == 0) {
                                matchesFilter = true;
                            }
                            
                            if (!matchesFilter) continue;
                        }
                        
                        // Format source display
                        if (name.length() > 30) name = name.substr(0, 27) + "...";
                        
                        // Add source type indicator
                        std::string typeIndicator = "🌐";
                        if (url.find("huggingface://") == 0) {
                            typeIndicator = "🤗";
                        }
                        
                        ss << typeIndicator << " " << name << "\n";
                        count++;
                        if (count >= 15) {
                            ss << "... (" << (totalCount - count) << " more)\n";
                            break;
                        }
                    }
                }
            }
            
            if (count == 0) {
                sourceListInfo = "No sources match filter";
            } else {
                sourceListInfo = ss.str();
            }
        } else {
            sourceListInfo = "No source file found";
        }
    } catch (const std::exception& e) {
        sourceListInfo = "Error loading sources";
    }
}

void UIDataCollectionPanel::addLog(const std::string& message, int level) {
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
    
    if (level >= 1) {
        LOG_DEBUG("DataCollectionPanel", message);
    }
}

// ============================================================
// Hugging Face Integration Functions
// ============================================================

void UIDataCollectionPanel::searchHuggingFaceDatasets() {
    if (hfSearchBuffer.empty()) {
        addLog("Please enter a search query", 1);
        lastSearchError = "Please enter a search query";
        return;
    }
    
    if (hfSearching.load()) {
        addLog("Search already in progress...", 1);
        return;
    }
    
    addLog("Searching Hugging Face for: " + hfSearchBuffer, 0);
    hfSearching.store(true);
    lastSearchError.clear();
    searchAnimTime = 0.0f;
    
    // Capture search query for thread
    std::string searchQuery = hfSearchBuffer;
    std::string token = hfTokenBuffer;
    
    // Run in background thread to avoid blocking UI
    std::thread([this, searchQuery, token]() {
        try {
            // Set API token if provided
            if (!token.empty() && hfWebhook) {
                hfWebhook->setApiToken(token);
            }
            
            // Search datasets
            auto results = hfWebhook->searchDatasets(searchQuery, 20, "task_categories:text-generation");
            
            // Update results on main thread context (atomic operations are thread-safe)
            hfSearchResults = std::move(results);
            
            if (hfSearchResults.empty()) {
                addLog("No datasets found for query: " + searchQuery, 1);
                lastSearchError = "No datasets found for: " + searchQuery;
                hfSearchResultsInfo = "No results found";
            } else {
                addLog("Found " + std::to_string(hfSearchResults.size()) + " datasets", 0);
                lastSearchError.clear();
                
                addLog("Setting hfResultsNeedsPopulate flag...", 0);
                
                // Format results for display
                std::stringstream ss;
                for (size_t i = 0; i < hfSearchResults.size() && i < 10; i++) {
                    const auto& ds = hfSearchResults[i];
                    ss << "[" << i << "] " << ds.id << "\n";
                    ss << "    Downloads: " << ds.downloads << " | Likes: " << ds.likes << "\n";
                }
                hfSearchResultsInfo = ss.str();
            }
        } catch (const std::exception& e) {
            std::string errorMsg = e.what();
            addLog("HF search error: " + errorMsg, 2);
            lastSearchError = "Search failed: " + errorMsg;
            hfSearchResultsInfo = "Search failed";
            hfSearchResults.clear();
        }
        
        hfSearching.store(false);

        // Signal main thread to populate the HF results (UI work must run on main thread)
        hfResultsNeedsPopulate.store(true);
    }).detach();
}

void UIDataCollectionPanel::searchHuggingFaceByCategory(const std::string& category) {
    if (category.empty() || category == "All Categories") {
        addLog("Please select a specific category", 1);
        lastSearchError = "Please select a category";
        return;
    }
    
    if (hfSearching.load()) {
        addLog("Search already in progress...", 1);
        return;
    }
    
    addLog("Searching HF by category: " + category, 0);
    hfSearching.store(true);
    lastSearchError.clear();
    searchAnimTime = 0.0f;
    
    // Capture values for thread
    std::string token = hfTokenBuffer;
    
    // Run in background thread to avoid blocking UI
    std::thread([this, category, token]() {
        try {
            // Set API token if provided
            if (!token.empty() && hfWebhook) {
                hfWebhook->setApiToken(token);
            }
            
            // Build filter string for category
            std::string filter = "task_categories:" + category;
            
            // Search datasets with category filter
            auto results = hfWebhook->searchDatasets("", 20, filter);
            
            hfSearchResults = std::move(results);
            
            if (hfSearchResults.empty()) {
                addLog("No datasets found for category: " + category, 1);
                lastSearchError = "No datasets in category: " + category;
                hfSearchResultsInfo = "No results found";
            } else {
                addLog("Found " + std::to_string(hfSearchResults.size()) + " datasets in category", 0);
                lastSearchError.clear();
                
                // Format results for display
                std::stringstream ss;
                for (size_t i = 0; i < hfSearchResults.size() && i < 10; i++) {
                    const auto& ds = hfSearchResults[i];
                    ss << "[" << i << "] " << ds.id << "\n";
                    ss << "    Downloads: " << ds.downloads << " | Likes: " << ds.likes << "\n";
                }
                hfSearchResultsInfo = ss.str();
            }
        } catch (const std::exception& e) {
            std::string errorMsg = e.what();
            addLog("HF category search error: " + errorMsg, 2);
            lastSearchError = "Category search failed: " + errorMsg;
            hfSearchResultsInfo = "Search failed";
            hfSearchResults.clear();
        }
        
        hfSearching.store(false);

        // Signal main thread to populate the HF results (UI work must run on main thread)
        hfResultsNeedsPopulate.store(true);
    }).detach();
}

void UIDataCollectionPanel::populateHFResults(float containerWidth) {
    if (!hfResultsScrollBox) {
        return;
    }
    
    // Clear existing widgets
    hfResultsScrollBox->clearChildren();
    
    // Create a button widget for each result
    for (size_t i = 0; i < hfSearchResults.size(); ++i) {
        const auto& dataset = hfSearchResults[i];
        
        // Create button with dataset name
        std::string buttonLabel = dataset.name + "\n " + dataset.author + 
                                  "\n⬇ " + std::to_string(dataset.downloads / 1000) + "k | ❤ " + std::to_string(dataset.likes);
        
        auto button = std::make_shared<UIButton>(buttonLabel, [this, datasetId = dataset.id, datasetName = dataset.name]() {
            addToDownloadQueue(datasetId, datasetName);
        });
        
        // Set button size with padding for scrollbar
        float btnWidth = containerWidth - 10.0f;
        button->setSize(btnWidth, 70.0f);
        
        // Add to scrollbox
        hfResultsScrollBox->addChild(button);
    }
    
    // Auto-layout the children
    hfResultsScrollBox->autoLayoutChildren();
}

void UIDataCollectionPanel::downloadHuggingFaceDataset(const std::string& datasetId) {
    addLog("=== DOWNLOADING HUGGING FACE DATASET ===" , 0);
    addLog("Dataset: " + datasetId, 0);
    
    // Determine output directory
    std::string outputDir = getResourcePath() + "/models/GRIM-text/training/data/huggingface/" + 
                            datasetId.substr(datasetId.find('/') + 1);
    
    // Run in background thread
    std::thread([this, datasetId, outputDir]() {
        try {
            hfDownloadStatus = "Downloading...";
            
            bool success = hfWebhook->downloadDataset(
                datasetId,
                outputDir,
                "train", // Download training split
                "",      // Default config
                [this](size_t downloaded, size_t total, const std::string& status) {
                    // Update progress
                    if (total > 0) {
                        currentProgress = (static_cast<float>(downloaded) / total) * 100.0f;
                    }
                    hfDownloadStatus = status + " (" + 
                        GRIM::DataCollection::HFUtils::formatSize(downloaded) + " / " +
                        GRIM::DataCollection::HFUtils::formatSize(total) + ")";
                }
            );
            
            if (success) {
                addLog("✓ Dataset downloaded successfully!", 0);
                addLog("Location: " + outputDir, 0);
                hfDownloadStatus = "Download complete";
                
                // Add to source_data.json for future processing
                saveSourceToJSON("huggingface://" + datasetId);
                updateDatasetStats();
            } else {
                std::string error = hfWebhook->getLastError();
                addLog("✗ Download failed: " + error, 2);
                hfDownloadStatus = "Download failed";
            }
        } catch (const std::exception& e) {
            addLog("HF download error: " + std::string(e.what()), 2);
            hfDownloadStatus = "Download error";
        }
    }).detach();
}

void UIDataCollectionPanel::browseHuggingFaceDatasets() {
    if (hfSearching.load()) {
        addLog("Search already in progress...", 1);
        return;
    }
    
    addLog("Browsing popular Hugging Face datasets...", 0);
    hfSearching.store(true);
    lastSearchError.clear();
    searchAnimTime = 0.0f;
    
    std::string token = hfTokenBuffer;
    
    std::thread([this, token]() {
        try {
            // Set API token if provided
            if (!token.empty() && hfWebhook) {
                hfWebhook->setApiToken(token);
            }
            
            // Search for popular text generation datasets
            auto results = hfWebhook->searchDatasets("", 20, "task_categories:text-generation");
            
            hfSearchResults = std::move(results);
            
            if (hfSearchResults.empty()) {
                addLog("Failed to load popular datasets", 1);
                lastSearchError = "Could not load popular datasets";
                hfSearchResultsInfo = "No results";
            } else {
                addLog("Loaded " + std::to_string(hfSearchResults.size()) + " popular datasets", 0);
                lastSearchError.clear();
                
                // Format results
                std::stringstream ss;
                for (size_t i = 0; i < hfSearchResults.size() && i < 10; i++) {
                    const auto& ds = hfSearchResults[i];
                    ss << "[" << i << "] " << ds.id << "\n";
                    ss << "    D: " << ds.downloads << " | L: " << ds.likes << "\n";
                }
                hfSearchResultsInfo = ss.str();
            }
        } catch (const std::exception& e) {
            std::string errorMsg = e.what();
            addLog("Browse error: " + errorMsg, 2);
            lastSearchError = "Browse failed: " + errorMsg;
            hfSearchResultsInfo = "Browse failed";
            hfSearchResults.clear();
        }
        
        hfSearching.store(false);
    }).detach();
}

// ============================================================
// Filter Functions
// ============================================================

void UIDataCollectionPanel::setSourceFilter(const std::string& filter) {
    activeSourceFilter = filter;
    addLog("Source filter set to: " + filter, 0);
    updateSourceList();
}

void UIDataCollectionPanel::setStatusFilter(const std::string& filter) {
    activeStatusFilter = filter;
    addLog("Status filter set to: " + filter, 0);
    updateSourceList();
}

void UIDataCollectionPanel::clearFilters() {
    activeSourceFilter = "all";
    activeStatusFilter = "all";
    addLog("All filters cleared", 0);
    updateSourceList();
}

// ============================================================
// Queue Functions
// ============================================================

void UIDataCollectionPanel::addToDownloadQueue(const std::string& datasetId, const std::string& name) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    // Check if already in queue
    for (const auto& item : downloadQueue) {
        if (item.datasetId == datasetId) {
            addLog("Dataset already in queue: " + name, 1);
            return;
        }
    }
    
    // Check if already downloaded (via HF webhook tracking)
    if (hfWebhook && hfWebhook->isDatasetDownloaded(datasetId)) {
        addLog("Dataset already downloaded previously: " + name, 1);
        return;
    }
    
    QueuedDownload item;
    item.datasetId = datasetId;
    item.displayName = name.empty() ? datasetId : name;
    item.status = "pending";
    item.progress = 0.0f;
    
    downloadQueue.push_back(item);
    addLog("Added to queue: " + item.displayName, 0);
    
    // Persist queue state
    saveDownloadQueue();
}

void UIDataCollectionPanel::processDownloadQueue() {
    if (queueProcessing.load()) {
        addLog("Queue is already being processed", 1);
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        if (downloadQueue.empty()) {
            addLog("Download queue is empty", 1);
            return;
        }
        
        // Check if there are any pending items
        bool hasPending = false;
        for (const auto& item : downloadQueue) {
            if (item.status == "pending") {
                hasPending = true;
                break;
            }
        }
        if (!hasPending) {
            addLog("No pending items in queue", 1);
            return;
        }
    }
    
    queueProcessing.store(true);
    currentQueueIndex.store(-1);
    addLog("=== PROCESSING DOWNLOAD QUEUE ===", 0);
    
    // Get queue size safely
    size_t queueSize = 0;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        queueSize = downloadQueue.size();
    }
    addLog("Items in queue: " + std::to_string(queueSize), 0);
    
    // Process queue in background thread
    std::thread([this]() {
        size_t processedCount = 0;
        size_t successCount = 0;
        size_t failCount = 0;
        size_t skippedCount = 0;  // Already downloaded in previous sessions
        
        // Get current queue size
        size_t queueSize = 0;
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            queueSize = downloadQueue.size();
        }
        
        for (size_t i = 0; i < queueSize; ++i) {
            // Check if queue was cleared or item was removed
            std::string datasetId;
            std::string displayName;
            bool shouldProcess = false;
            
            {
                std::lock_guard<std::mutex> lock(queueMutex);
                if (i >= downloadQueue.size()) break;  // Queue was modified
                
                auto& item = downloadQueue[i];
                if (item.status == "completed") continue;  // Already done
                if (item.status == "downloading") continue; // Another thread?
                if (item.status == "failed" && item.retryCount == 0) continue; // Failed and not retrying
                
                // Only process pending items
                if (item.status != "pending") continue;
                
                // Check if already downloaded in a previous session
                if (hfWebhook && hfWebhook->isDatasetDownloaded(item.datasetId)) {
                    item.status = "completed";
                    item.progress = 1.0f;
                    skippedCount++;
                    addLog("[SKIP] Already downloaded: " + item.displayName, 0);
                    continue;
                }
                
                item.status = "downloading";
                item.progress = 0.0f;
                datasetId = item.datasetId;
                displayName = item.displayName;
                shouldProcess = true;
            }
            
            if (!shouldProcess) continue;
            
            currentQueueIndex.store(static_cast<int>(i));
            processedCount++;
            
            addLog("Processing [" + std::to_string(processedCount) + "]: " + displayName, 0);
            
            // Extract dataset name from ID for output directory
            std::string datasetName = datasetId;
            size_t slashPos = datasetId.find('/');
            if (slashPos != std::string::npos) {
                datasetName = datasetId.substr(slashPos + 1);
            }
            
            std::string outputDir = getResourcePath() + "/models/GRIM-text/training/data/huggingface/" + datasetName;
            
            addLog("  Dataset ID: " + datasetId, 0);
            addLog("  Output directory: " + outputDir, 0);
            
            try {
                bool success = hfWebhook->downloadDataset(
                    datasetId,
                    outputDir,
                    "train",
                    "",
                    [this, i](size_t downloaded, size_t total, const std::string& status) {
                        std::lock_guard<std::mutex> lock(queueMutex);
                        if (i < downloadQueue.size() && total > 0) {
                            downloadQueue[i].progress = static_cast<float>(downloaded) / static_cast<float>(total);
                        }
                    }
                );
                
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    if (i < downloadQueue.size()) {
                        if (success) {
                            downloadQueue[i].status = "completed";
                            downloadQueue[i].progress = 1.0f;
                            downloadQueue[i].errorMessage.clear();
                            successCount++;
                            addLog("✓ Completed: " + displayName, 0);
                            
                            // Save to source_data.json outside of lock
                        } else {
                            downloadQueue[i].status = "failed";
                            downloadQueue[i].errorMessage = hfWebhook->getLastError();
                            addLog("  Error details: " + downloadQueue[i].errorMessage, 2);
                            failCount++;
                            addLog("✗ Failed: " + displayName + " - " + downloadQueue[i].errorMessage, 2);
                        }
                    }
                }
                
                // Save source outside of lock to avoid deadlock
                if (success) {
                    saveSourceToJSON("huggingface://" + datasetId);
                }
                
                // Persist queue state after each download completes/fails
                saveDownloadQueue();
                
            } catch (const std::exception& e) {
                std::string errorMsg = e.what();
                {
                    std::lock_guard<std::mutex> lock(queueMutex);
                    if (i < downloadQueue.size()) {
                        downloadQueue[i].status = "failed";
                        downloadQueue[i].errorMessage = errorMsg;
                        failCount++;
                    }
                }
                addLog("✗ Error downloading " + displayName + ": " + errorMsg, 2);
                
                // Persist queue state after failure
                saveDownloadQueue();
            }
        }
        
        currentQueueIndex.store(-1);
        queueProcessing.store(false);
        
        std::string summary = "=== QUEUE COMPLETE: " + std::to_string(successCount) + " succeeded, " +
                              std::to_string(failCount) + " failed";
        if (skippedCount > 0) {
            summary += ", " + std::to_string(skippedCount) + " skipped (already downloaded)";
        }
        summary += " ===";
        addLog(summary, 0);
        
        // Final save of queue state
        saveDownloadQueue();
        
        updateDatasetStats();
    }).detach();
}

void UIDataCollectionPanel::clearDownloadQueue() {
    if (queueProcessing.load()) {
        addLog("Cannot clear queue while processing", 1);
        return;
    }
    std::lock_guard<std::mutex> lock(queueMutex);
    downloadQueue.clear();
    currentQueueIndex.store(-1);
    addLog("Download queue cleared", 0);
    saveDownloadQueue();  // Persist cleared state
}

void UIDataCollectionPanel::removeFromQueue(int index) {
    if (queueProcessing.load() && currentQueueIndex.load() == index) {
        addLog("Cannot remove item currently being downloaded", 1);
        return;
    }
    std::lock_guard<std::mutex> lock(queueMutex);
    if (index >= 0 && index < static_cast<int>(downloadQueue.size())) {
        std::string name = downloadQueue[index].displayName;
        downloadQueue.erase(downloadQueue.begin() + index);
        addLog("Removed from queue: " + name, 0);
        saveDownloadQueue();  // Persist removal
    }
}

void UIDataCollectionPanel::retryQueueItem(int index) {
    std::lock_guard<std::mutex> lock(queueMutex);
    if (index >= 0 && index < static_cast<int>(downloadQueue.size())) {
        if (downloadQueue[index].status == "failed") {
            downloadQueue[index].status = "pending";
            downloadQueue[index].retryCount++;
            downloadQueue[index].progress = 0.0f;
            downloadQueue[index].errorMessage.clear();
            addLog("Retrying: " + downloadQueue[index].displayName + " (attempt " + 
                   std::to_string(downloadQueue[index].retryCount + 1) + ")", 0);
            saveDownloadQueue();  // Persist retry state
        }
    }
}

void UIDataCollectionPanel::clearCompletedFromQueue() {
    std::lock_guard<std::mutex> lock(queueMutex);
    size_t before = downloadQueue.size();
    downloadQueue.erase(
        std::remove_if(downloadQueue.begin(), downloadQueue.end(),
            [](const QueuedDownload& item) { return item.status == "completed"; }),
        downloadQueue.end()
    );
    size_t removed = before - downloadQueue.size();
    if (removed > 0) {
        addLog("Cleared " + std::to_string(removed) + " completed item(s) from queue", 0);
    }
}

void UIDataCollectionPanel::pauseQueueProcessing() {
    // This is a graceful pause - it doesn't interrupt the current download
    // but prevents the next item from starting
    if (queueProcessing.load()) {
        // We can't easily pause a detached thread, so this is more of a "stop after current"
        addLog("Queue will stop after current download completes", 0);
        // Set a flag that the processing thread checks
        // For now, just log the intent - proper implementation would need a cancel token
    }
}

//======================================================//
// UI Config Persistence
//======================================================//

void UIDataCollectionPanel::loadUIConfig() {
    std::string configPath = GRIM::Config::getCheckpointDir() + "/collection_state/ui_config.json";
    
    try {
        if (std::filesystem::exists(configPath)) {
            std::ifstream file(configPath);
            nlohmann::json config;
            file >> config;
            file.close();
            
            // Load slider values
            if (config.contains("fetchLimit")) {
                fetchLimit = config["fetchLimit"];
                if (fetchLimitSlider) {
                    fetchLimitSlider->setValue(static_cast<float>(fetchLimit));
                }
            }
            
            if (config.contains("vocabSize")) {
                vocabSize = config["vocabSize"];
                if (vocabSizeSlider) {
                    vocabSizeSlider->setValue(static_cast<float>(vocabSize));
                }
            }
            
            if (config.contains("verificationThreshold")) {
                verificationThreshold = config["verificationThreshold"];
                if (verificationThresholdSlider) {
                    verificationThresholdSlider->setValue(verificationThreshold);
                }
            }
            
            if (config.contains("maxHFResults")) {
                maxHFResults = config["maxHFResults"];
                if (maxHFResultsSlider) {
                    maxHFResultsSlider->setValue(static_cast<float>(maxHFResults));
                }
            }
            
            // Load filter states
            if (config.contains("activeSourceFilter")) {
                activeSourceFilter = config["activeSourceFilter"];
            }
            
            if (config.contains("activeStatusFilter")) {
                activeStatusFilter = config["activeStatusFilter"];
            }
            
            addLog("UI config loaded from: " + configPath, 0);
        }
    } catch (const std::exception& e) {
        addLog("Error loading UI config: " + std::string(e.what()), 1);
    }
}

void UIDataCollectionPanel::saveUIConfig() {
    std::string stateDir = GRIM::Config::getCheckpointDir() + "/collection_state";
    std::string configPath = stateDir + "/ui_config.json";
    
    try {
        // Ensure directory exists
        std::filesystem::create_directories(stateDir);
        
        nlohmann::json config;
        config["fetchLimit"] = fetchLimit;
        config["vocabSize"] = vocabSize;
        config["verificationThreshold"] = verificationThreshold;
        config["maxHFResults"] = maxHFResults;
        config["activeSourceFilter"] = activeSourceFilter;
        config["activeStatusFilter"] = activeStatusFilter;
        
        // Save timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        config["lastSaved"] = std::ctime(&time);
        
        std::ofstream file(configPath);
        file << std::setw(2) << config << std::endl;
        file.close();
        
    } catch (const std::exception& e) {
        addLog("Error saving UI config: " + std::string(e.what()), 1);
    }
}

void UIDataCollectionPanel::loadDownloadQueue() {
    std::string queuePath = GRIM::Config::getCheckpointDir() + "/collection_state/download_queue.json";
    
    try {
        if (std::filesystem::exists(queuePath)) {
            std::ifstream file(queuePath);
            nlohmann::json queueJson;
            file >> queueJson;
            file.close();
            
            std::lock_guard<std::mutex> lock(queueMutex);
            downloadQueue.clear();
            
            if (queueJson.contains("queue") && queueJson["queue"].is_array()) {
                for (const auto& item : queueJson["queue"]) {
                    QueuedDownload qd;
                    qd.datasetId = item.value("datasetId", "");
                    qd.displayName = item.value("displayName", "");
                    qd.status = item.value("status", "pending");
                    qd.progress = item.value("progress", 0.0f);
                    qd.retryCount = item.value("retryCount", 0);
                    qd.errorMessage = item.value("errorMessage", "");
                    
                    // Don't restore completed items or items that were downloading (reset to pending)
                    if (qd.status == "downloading") {
                        qd.status = "pending";
                        qd.progress = 0.0f;
                    }
                    
                    // Only restore pending or failed items
                    if (qd.status == "pending" || qd.status == "failed") {
                        downloadQueue.push_back(qd);
                    }
                }
                
                addLog("Restored " + std::to_string(downloadQueue.size()) + " items from download queue", 0);
            }
        }
    } catch (const std::exception& e) {
        addLog("Error loading download queue: " + std::string(e.what()), 1);
    }
}

void UIDataCollectionPanel::saveDownloadQueue() {
    std::string stateDir = GRIM::Config::getCheckpointDir() + "/collection_state";
    std::string queuePath = stateDir + "/download_queue.json";
    
    try {
        std::filesystem::create_directories(stateDir);
        
        nlohmann::json queueJson;
        nlohmann::json queueArray = nlohmann::json::array();
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            for (const auto& item : downloadQueue) {
                nlohmann::json itemJson;
                itemJson["datasetId"] = item.datasetId;
                itemJson["displayName"] = item.displayName;
                itemJson["status"] = item.status;
                itemJson["progress"] = item.progress;
                itemJson["retryCount"] = item.retryCount;
                itemJson["errorMessage"] = item.errorMessage;
                queueArray.push_back(itemJson);
            }
        }
        
        queueJson["queue"] = queueArray;
        
        // Save timestamp
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        queueJson["lastSaved"] = std::ctime(&time);
        
        std::ofstream file(queuePath);
        file << std::setw(2) << queueJson << std::endl;
        file.close();
        
    } catch (const std::exception& e) {
        addLog("Error saving download queue: " + std::string(e.what()), 1);
    }
}