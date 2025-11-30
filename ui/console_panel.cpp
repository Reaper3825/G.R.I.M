#include "console_panel.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "overlay_renderer.hpp"
#include "ui_root.hpp" 
#include "ui_slider.hpp"  // For checking if slider is editing
#include "commands/commands_core.hpp"
#include "helpers/key.hpp"
#include <windows.h>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>

ConsolePanel::ConsolePanel()
    : UIPanel("Console", true),  // Enable dragging
      settingsButton(std::make_shared<UIButton>("Settings", []() {
          auto settingsPanel = UIRoot::get().getPanel("Settings");
          if (settingsPanel) {
              settingsPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened settings panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Settings panel not found - may not be initialized yet");
          }
      })),
      DCButton(std::make_shared<UIButton>(" Data Collection ", []() {
          // Open data collection panel
          auto dataCollectionPanel = UIRoot::get().getPanel("DataCollection");
          if (dataCollectionPanel) {
              dataCollectionPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened data collection panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Data collection panel not found - may not be initialized yet");
          }
      })),
      trainingButton(std::make_shared<UIButton>(" Training ", []() {
          // Open training panel
          auto trainingPanel = UIRoot::get().getPanel("GRIM-text Training Control");
          if (trainingPanel) {
              trainingPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened training panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Training panel not found - may not be initialized yet");
          }
      }))
{
    position = { 100, 300 };
    size = { 900, 500 };
    setBackground(0xE0101010);
    
    // Initialize console input box
    consoleInput = std::make_shared<UIInputBox>(&inputBuffer);
    consoleInput->setPlaceholder("Type command...");
    consoleInput->setVisible(true);
    
    // ✅ Bind the OnTextSubmitted delegate to handle command execution
    consoleInput->OnTextSubmitted.Bind([this](const std::string& submittedText) {
        if (!submittedText.empty()) {
            LOG_DEBUG("ConsolePanel", "Executing command via delegate: " + submittedText);
            
            auto& history = getConsoleHistory();
            
            // Add separator before command
            history.push("  -----------------------------------------------------------", 0xFF00FF00);
            
            // Add input to history with timestamp
            std::string timestamp = "[" + getCurrentTime() + "]";
            history.push("  " + timestamp + " > " + submittedText, 0xFF00FFFF);
            
            // Execute command (handleCommand adds results to history internally)
            handleCommand(submittedText);
            
            // Add separator after output
            history.push("  -----------------------------------------------------------", 0xFF444444);
            history.push("", 0xFF000000);
        }
    });
    
    LOG_DEBUG("ConsolePanel", "Console input box initialized with delegate binding");
    setBorder(0xFF00FF00);
    
    // ✅ Position buttons in top-right corner of console
    if (settingsButton) {
        settingsButton->setPosition(position.x + size.x - 110, position.y + 5);
        settingsButton->setSize(100, 25);
    }
    
    if (trainingButton) {
        trainingButton->setPosition(position.x + size.x - 220, position.y + 5);
        trainingButton->setSize(100, 25);
    }
    
    // Add stylized welcome message to GLOBAL history
    auto& history = getConsoleHistory();
    history.push("", 0xFF000000);
    history.push("  ===========================================================", 0xFF00FF00);
    history.push("           G.R.I.M - General Responsive Interface           ", 0xFF00FF00);
    history.push("              Machine Intelligence Console                   ", 0xFF00FFFF);
    history.push("  ===========================================================", 0xFF00FF00);
    history.push("", 0xFF000000);
    history.push("  System Status: ONLINE", 0xFF00FF00);
    history.push("  Type 'help' for available commands", 0xFF888888);
    history.push("  Press ESC to close | Click console to focus", 0xFF666666);
    history.push("", 0xFF000000);
    history.push("  -----------------------------------------------------------", 0xFF444444);
    history.push("", 0xFF000000);
}

void ConsolePanel::update(const InputState& input, float dt)
{
    // Call base panel update to handle drag/resize
    UIPanel::update(input, dt);

    // ✅ Update button positions to follow panel
    if (settingsButton) {
        settingsButton->setPosition(position.x + size.x - 110, position.y + 5);
        settingsButton->update(input, dt);
    }
    if (trainingButton) {
        trainingButton->setPosition(position.x + size.x - 220, position.y + 5);
        trainingButton->update(input, dt);
    }
        if (DCButton) {
        DCButton->setPosition(position.x + size.x - 330, position.y + 5);
        DCButton->update(input, dt);
    }

    if (!isVisible()) return;
    
    // ✅ Update console input box
    if (consoleInput) {
        // Position at bottom of console panel
        float inputY = position.y + size.y - 42;
        consoleInput->setPosition(position.x + 40, inputY);
        consoleInput->setSize(size.x - 50, 30);
        consoleInput->update(input, dt);
    }

    // Handle Escape - close console
    if (Key::wasPressed(KeyCode::Escape)) {
        setVisible(false);
        LOG_DEBUG("ConsolePanel", "Console closed via ESC key");
    }
}

void ConsolePanel::drawOverlay(OverlayRenderer& renderer)
{
    if (!isVisible()) return;
    
    // First, let the base panel draw its background, border, and title
    UIPanel::drawOverlay(renderer);
    
    // ✅ Draw buttons using their drawOverlay method
    if (settingsButton) {
        settingsButton->drawOverlay(renderer, position);
    }
    
    if (trainingButton) {
        trainingButton->drawOverlay(renderer, position);
    }

    if (DCButton) {
        DCButton->drawOverlay(renderer, position);
    }
    
    // Now draw console-specific content on top
    
    // Draw separator line under title
    renderer.drawRect({position.x + 10, position.y + titleBarHeight + 2}, 
                     {size.x - 20, 2}, 0xFF00FF00);

    // Draw console history - USE GLOBAL HISTORY
    auto& history = getConsoleHistory();
    
    // Calculate proper width for wrapping (account for padding)
    float maxTextWidth = size.x - 60.0f;
    
    // IMPORTANT: Ensure wrapped lines are generated with correct width!
    history.ensureWrapped(maxTextWidth);
    
    float y = position.y + titleBarHeight + 12; // Start below separator
    auto lines = history.wrapped();
    
    // Calculate scrollable area (leave room for input box)
    float scrollAreaHeight = size.y - titleBarHeight - 70;
    int maxLines = static_cast<int>(scrollAreaHeight / 20.0f);
    int startIdx = std::max(0, static_cast<int>(lines.size()) - maxLines);
    
    // Draw history lines
    for (int i = startIdx; i < static_cast<int>(lines.size()); ++i)
    {
        if (y >= position.y + size.y - 70) break; // Don't draw over input area
        renderer.drawText({position.x + 15, y}, lines[i].text, lines[i].color);
        y += 20.0f;
    }

    // ✅ Draw input area using UIInputBox
    float inputY = position.y + size.y - 50;
    
    // Input area background with border
    renderer.drawRect({position.x + 8, inputY - 2}, {size.x - 16, 42}, 0xFF1A1A1A);
    renderer.drawRect({position.x + 8, inputY - 3}, {size.x - 16, 1}, 0xFF00FF00);
    
    // Draw input prompt
    renderer.drawText({position.x + 18, inputY + 10}, ">", 0xFF00FF00);
    
    // ✅ Draw the UIInputBox
    if (consoleInput) {
        consoleInput->drawOverlay(renderer, position);
    }
    
    // Draw help text
    renderer.drawText({position.x + 15, inputY + 28}, "ESC: Close", 0xFF666666);
    renderer.drawText({position.x + size.x - 200, inputY + 28}, "Enter: Execute", 0xFF666666);
}

void ConsolePanel::executeCommand(const std::string& cmd)
{
    if (cmd.empty()) return;
    
    LOG_DEBUG("ConsolePanel", "Executing: " + cmd);
    
    auto& history = getConsoleHistory();
    
    // Add to history display
    history.push("> " + cmd, 0xFF00FF00);
    
    // Execute via command system (adds results to history internally)
    handleCommand(cmd);
}

std::string ConsolePanel::getCurrentTime() const
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &time);
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << tm.tm_hour << ":"
        << std::setw(2) << tm.tm_min << ":"
        << std::setw(2) << tm.tm_sec;
    return oss.str();
}
