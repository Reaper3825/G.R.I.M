#include "console_panel.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "ui_root.hpp" 
#include "ui_slider.hpp"  // For checking if slider is editing
#include "commands/commands_core.hpp"
#include "helpers/key.hpp"
#include "core/grim_platform.h"
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
      })),
      modelsButton(std::make_shared<UIButton>(" Models ", []() {
          auto modelPanel = UIRoot::get().getPanel("Model Registry");
          if (modelPanel) {
              modelPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened model registry panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Model registry panel not found - may not be initialized yet");
          }
      }))
{
    position = { 100, 300 };
    size = { 900, 500 };
    setBackground(UITheme::Colors::PanelBg);
    
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
            history.push("  -----------------------------------------------------------", UITheme::Colors::BorderPrimary);
            
            // Add input to history with timestamp
            std::string timestamp = "[" + getCurrentTime() + "]";
            history.push("  " + timestamp + " > " + submittedText, UITheme::Colors::PrimaryLight);
            
            // Execute command (handleCommand adds results to history internally)
            handleCommand(submittedText);
            
            // Add separator after output
            history.push("  -----------------------------------------------------------", UITheme::Colors::BorderMedium);
            history.push("", UITheme::Colors::Background);
        }
    });
    
    LOG_DEBUG("ConsolePanel", "Console input box initialized with delegate binding");
    setBorder(UITheme::Colors::DividerLine);
    
    // Build toolbar layout box with all buttons
    float btnW = 110.0f;
    float btnH = 25.0f;
    if (settingsButton) settingsButton->setSize(btnW, btnH);
    if (DCButton)       DCButton->setSize(btnW, btnH);
    if (trainingButton) trainingButton->setSize(btnW, btnH);
    if (modelsButton)   modelsButton->setSize(btnW, btnH);
    
    toolbarBox = std::make_shared<UIHBox>(LayoutDirection::Horizontal, 8.0f);
    toolbarBox->addWidget(modelsButton);
    toolbarBox->addWidget(DCButton);
    toolbarBox->addWidget(trainingButton);
    toolbarBox->addWidget(settingsButton);
    toolbarBox->layout();
    
    // Add stylized welcome message to GLOBAL history
    auto& history = getConsoleHistory();
    history.push("", UITheme::Colors::Background);
    history.push("  ===========================================================", UITheme::Colors::BorderDecorative);
    history.push("           G.R.I.M - General Responsive Interface           ", UITheme::Colors::TextHeader);
    history.push("              Machine Intelligence Console                   ", UITheme::Colors::PrimaryLight);
    history.push("  ===========================================================", UITheme::Colors::BorderDecorative);
    history.push("", UITheme::Colors::Background);
    history.push("  System Status: ONLINE", UITheme::Colors::Success);
    history.push("  Type 'help' for available commands", UITheme::Colors::TextSecondary);
    history.push("  Press ESC to close | Click console to focus", UITheme::Colors::TextDisabled);
    history.push("", UITheme::Colors::Background);
    history.push("  -----------------------------------------------------------", UITheme::Colors::BorderMedium);
    history.push("", UITheme::Colors::Background);
}

void ConsolePanel::update(const InputState& input, float dt)
{
    // Call base panel update to handle drag/resize
    UIPanel::update(input, dt);

    // Update toolbar layout box position (right-aligned, left of chrome buttons)
    if (toolbarBox) {
        toolbarBox->layout();
        float chromeWidth = 75.0f; // space reserved for close/max/min buttons
        float toolbarW = toolbarBox->getSize().x;
        float toolbarX = position.x + size.x - toolbarW - chromeWidth;
        float toolbarY = position.y + 5.0f;
        toolbarBox->setPosition(toolbarX, toolbarY);
        toolbarBox->layout();
        toolbarBox->update(input, dt);
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

bool ConsolePanel::drawOverlay(OverlayRenderer& renderer)
{
    if (!UIPanel::drawOverlay(renderer)) return false;
    
    // Draw separator line under title
    renderer.drawRect({position.x + 10, position.y + titleBarHeight + 2}, 
                     {size.x - 20, 1}, UITheme::Colors::DividerFaint);

    auto& history = getConsoleHistory();
    
    float maxTextWidth = size.x - 60.0f;
    history.ensureWrapped(maxTextWidth);
    
    float y = position.y + titleBarHeight + 12;
    auto lines = history.wrapped();
    
    float scrollAreaHeight = size.y - titleBarHeight - 70;
    int maxLines = static_cast<int>(scrollAreaHeight / 20.0f);
    int startIdx = std::max(0, static_cast<int>(lines.size()) - maxLines);
    
    for (int i = startIdx; i < static_cast<int>(lines.size()); ++i)
    {
        if (y >= position.y + size.y - 70) break;
        renderer.drawText({position.x + 15, y}, lines[i].text, lines[i].color);
        y += 20.0f;
    }

    float inputY = position.y + size.y - 50;
    
    renderer.drawRoundedRect({position.x + 8, inputY - 2}, {size.x - 16, 42}, UITheme::Colors::ContentAreaBg, UITheme::Sizes::WidgetRadius + 2.0f);
    renderer.drawRoundedBorder({position.x + 8, inputY - 2}, {size.x - 16, 42}, UITheme::Colors::BorderPrimary, UITheme::Sizes::WidgetRadius + 2.0f);
    
    renderer.drawText({position.x + 18, inputY + 10}, ">", UITheme::Colors::Primary);
    
    if (consoleInput) {
        consoleInput->drawOverlay(renderer, position);
    }
    
    renderer.drawText({position.x + 15, inputY + 28}, "ESC: Close", UITheme::Colors::TextDisabled);
    renderer.drawText({position.x + size.x - 200, inputY + 28}, "Enter: Execute", UITheme::Colors::TextDisabled);
    
    renderer.popClipRect();
    
    // Draw toolbar outside clip rect so it renders over the title bar
    if (toolbarBox) {
        toolbarBox->drawOverlay(renderer, position);
    }
    
    return true;
}

void ConsolePanel::executeCommand(const std::string& cmd)
{
    if (cmd.empty()) return;
    
    LOG_DEBUG("ConsolePanel", "Executing: " + cmd);
    
    auto& history = getConsoleHistory();
    
    // Add to history display
    history.push("> " + cmd, UITheme::Colors::PrimaryLight);
    
    // Execute via command system (adds results to history internally)
    handleCommand(cmd);
}

std::string ConsolePanel::getCurrentTime() const
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << tm.tm_hour << ":"
        << std::setw(2) << tm.tm_min << ":"
        << std::setw(2) << tm.tm_sec;
    return oss.str();
}
