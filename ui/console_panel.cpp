#include "console_panel.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "ui_root.hpp" 
#include "primitives/ui_slider.hpp"  // For checking if slider is editing
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
      DCButton(std::make_shared<UIButton>(" Data Hub ", []() {
          auto dataHubPanel = UIRoot::get().getPanel("DataHub");
          if (dataHubPanel) {
              dataHubPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened DataHub panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "DataHub panel not found");
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
      storageButton(std::make_shared<UIButton>(" Storage ", []() {
          auto storagePanel = UIRoot::get().getPanel("Shared Storage");
          if (storagePanel) {
              storagePanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened storage panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Storage panel not found");
          }
      })),
      cameraButton(std::make_shared<UIButton>(" Camera ", []() {
          auto physicalEnvPanel = UIRoot::get().getPanel("Physical Environment");
          if (physicalEnvPanel) {
              physicalEnvPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened Physical Environment panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Physical Environment panel not found");
          }
      })),
      digitalButton(std::make_shared<UIButton>(" Digital ", []() {
          auto digitalEnvPanel = UIRoot::get().getPanel("Digital Environment");
          if (digitalEnvPanel) {
              digitalEnvPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened Digital Environment panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "Digital Environment panel not found");
          }
      })),
      geoSpatialButton(std::make_shared<UIButton>(" GeoSpatial ", []() {
          auto geoSpatialPanel = UIRoot::get().getPanel("GeoSpatial");
          if (geoSpatialPanel) {
              geoSpatialPanel->setVisible(true);
              LOG_DEBUG("ConsolePanel", "Opened GeoSpatial panel via button");
          } else {
              LOG_DEBUG("ConsolePanel", "GeoSpatial panel not found");
          }
      }))
{
    position = { 100, 300 };
    size = { 900, 500 };
    setBackground(UITheme::Colors::PanelBg);
    
    // Initialize console input box
    consoleInput = std::make_shared<UIInputBox>(&inputBuffer);
    consoleInput->setPlaceholder("Type command...");
    consoleInput->setClearOnSubmit(true);
    consoleInput->setVisible(true);
    
    // ✅ Bind the OnTextSubmitted delegate to handle command execution
    consoleInput->OnTextSubmitted.Bind([this](const std::string& submittedText) {
        if (!submittedText.empty()) {
            LOG_DEBUG("ConsolePanel", "Executing command via delegate: " + submittedText);
            
            auto& history = getConsoleHistory();
            
            std::string timestamp = "[" + getCurrentTime() + "]";
            history.push("", UITheme::Colors::Background);
            history.push("  " + timestamp + " > " + submittedText, UITheme::Colors::PrimaryLight);
            
            handleCommand(submittedText);
            
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
    if (storageButton)  storageButton->setSize(btnW, btnH);
    if (cameraButton)   cameraButton->setSize(btnW, btnH);
    if (digitalButton)  digitalButton->setSize(btnW, btnH);
    if (geoSpatialButton) geoSpatialButton->setSize(120.0f, btnH);
    
    toolbarBox = std::make_shared<UIHBox>(LayoutDirection::Horizontal, 8.0f);
    toolbarBox->addWidget(DCButton);
    toolbarBox->addWidget(trainingButton);
    toolbarBox->addWidget(storageButton);
    toolbarBox->addWidget(cameraButton);
    toolbarBox->addWidget(digitalButton);
    toolbarBox->addWidget(geoSpatialButton);
    toolbarBox->addWidget(settingsButton);
    toolbarBox->layout();
    
    auto& history = getConsoleHistory();
    history.push("", UITheme::Colors::Background);
    history.push("  G.R.I.M - General Responsive Interface", UITheme::Colors::TextHeader);
    history.push("  Machine Intelligence Console", UITheme::Colors::PrimaryLight);
    history.push("", UITheme::Colors::Background);
    history.push("  System Status: ONLINE", UITheme::Colors::Success);
    history.push("  Type 'help' for available commands", UITheme::Colors::TextSecondary);
    history.push("  Press ESC to close | Click console to focus", UITheme::Colors::TextDisabled);
    history.push("", UITheme::Colors::Background);
}

void ConsolePanel::update(const InputState& input, float dt)
{
    // Call base panel update to handle drag/resize
    UIPanel::update(input, dt);

    if (toolbarBox) {
        toolbarBox->layout();
        float toolbarW = toolbarBox->getSize().x;
        float toolbarX = position.x + size.x - toolbarW - 12.0f;
        float toolbarY = position.y + (titleBarHeight - 25.0f) * 0.5f;
        toolbarBox->setPosition(toolbarX, toolbarY);
        toolbarBox->layout();
        toolbarBox->update(input, dt);
    }

    if (!isVisible()) return;
    
    if (consoleInput) {
        float inputY = position.y + size.y - 52;
        consoleInput->setPosition(position.x + 38, inputY);
        consoleInput->setSize(size.x - 56, 30);
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
    
    // Subtle separator under title bar
    renderer.drawRect({position.x + 16, position.y + titleBarHeight + 1}, 
                     {size.x - 32, 1}, UITheme::Colors::DividerFaint);

    auto& history = getConsoleHistory();
    
    float maxTextWidth = size.x - 60.0f;
    history.ensureWrapped(maxTextWidth);
    
    float y = position.y + titleBarHeight + 14;
    auto lines = history.wrapped();
    
    float scrollAreaHeight = size.y - titleBarHeight - 76;
    int maxLines = static_cast<int>(scrollAreaHeight / 20.0f);
    int startIdx = std::max(0, static_cast<int>(lines.size()) - maxLines);
    
    for (int i = startIdx; i < static_cast<int>(lines.size()); ++i)
    {
        if (y >= position.y + size.y - 76) break;
        renderer.drawText({position.x + 18, y}, lines[i].text, lines[i].color);
        y += 20.0f;
    }

    // Glass input area at bottom
    float inputY = position.y + size.y - 56;
    float inputRadius = UITheme::Sizes::WidgetRadius + 4.0f;
    
    renderer.drawRoundedRect({position.x + 12, inputY}, {size.x - 24, 40}, UITheme::Colors::ScrollboxBg, inputRadius);
    renderer.drawRoundedBorder({position.x + 12, inputY}, {size.x - 24, 40}, UITheme::Colors::BorderSubtle, inputRadius);
    
    renderer.drawText({position.x + 22, inputY + 11}, ">", UITheme::Colors::Primary);
    
    if (consoleInput) {
        consoleInput->drawOverlay(renderer, position);
    }
    
    renderer.drawText({position.x + 18, inputY + 30}, "ESC Close", UITheme::Colors::TextDisabled);
    renderer.drawText({position.x + size.x - 175, inputY + 30}, "Enter Execute", UITheme::Colors::TextDisabled);
    
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
