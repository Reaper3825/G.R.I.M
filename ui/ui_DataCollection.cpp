
#include <sstream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <ctime>
#include <fstream>
#include <algorithm>
#include "logger.hpp"
#include <Windows.h>
#include <filesystem>
#include <string>
#include "overlay_renderer.hpp"
#include "ui_DataCollection.hpp"

std::string debugtext = "UIDataCollectionPanel.cpp loaded";

UIDataCollectionPanel::UIDataCollectionPanel()
    : UIPanel("DataCollection", true),  // Enable dragging

      DataCollectionRunButton(std::make_shared<UIButton>(" Data Collection ", []() {
        LOG_DEBUG(debugtext , "Data Collection button clicked");

      }))
{
    position = { 100, 300 };
    size = { 900, 500 };
    setBackground(0xE0101010);
    // ✅ Bind the OnTextSubmitted delegate to handle command execution

    
    LOG_DEBUG("DataCollectionPanel", "Console input box initialized with delegate binding");
    setBorder(0xFF00FF00);
    
    // ✅ Position buttons in top-right corner of console
    if (DataCollectionRunButton) {
        DataCollectionRunButton->setPosition(position.x + size.x - 110, position.y + 5);
        DataCollectionRunButton->setSize(100, 25);
    }
}
void UIDataCollectionPanel::update(const InputState& input, float dt) {
    if (!isVisible()) return;
}

// Destructor definition (was declared in header but not defined)
UIDataCollectionPanel::~UIDataCollectionPanel() {
    // Clean up any resources if needed
}

// Overlay drawing for layered window rendering


void UIDataCollectionPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!isVisible()) return;

    // Draw panel background
    renderer.drawRect(position, size, bgColor);

    // Draw title
    renderer.drawText({ position.x + 8, position.y + 6 }, getTitle(), 0xFFFFFFFF);

    // Draw child button overlay (pass panel position so children compute absolute positions)
    if (DataCollectionRunButton) {
        DataCollectionRunButton->drawOverlay(renderer, position);
    }
}