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
#include "MMO/Core/SessionContextManager.hpp"
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
constexpr float kSessionRowHeight = 30.0f;
constexpr float kSessionRowGap = 6.0f;

float sessionSidebarWidth(float panelWidth)
{
    return std::clamp(panelWidth * 0.28f, 170.0f, 210.0f);
}
}

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
      })),
      addSessionButton(std::make_shared<UIButton>("+", [this]() {
          addTemporarySession();
      })),
      sessionScrollBox(std::make_shared<UIScrollBox>())
{
    position = { 100, 300 };
    size = { 900, 500 };
    setBackground(UITheme::Colors::PanelBg);
    
    // Initialize console input box
    consoleInput = std::make_shared<UIInputBox>(&inputBuffer);
    consoleInput->setPlaceholder("Type command...");
    consoleInput->setClearOnSubmit(true);
    consoleInput->setVisible(true);
    
    consoleInput->OnTextSubmitted.Bind([this](const std::string& submittedText) {
        if (!submittedText.empty()) {
            LOG_DEBUG("ConsolePanel", "Executing command via delegate: " + submittedText);

            auto& session = activeSession();
            const std::string turnId = std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count());
            GRIM::MMO::SessionContextManager::instance().beginTurn(
                session.id, turnId, submittedText, submittedText);
            if (!session.committed) {
                session.committed = true;
            }
            
            auto& history = getConsoleHistory();
            
            std::string timestamp = "[" + getCurrentTime() + "]";
            history.push("", UITheme::Colors::Background);
            history.push("  " + timestamp + " > " + submittedText, UITheme::Colors::PrimaryLight);
            
            handleCommand(submittedText, session.id);
            
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
    addSessionButton->setSize(26.0f, 26.0f);
    
    toolbarBox = std::make_shared<UIHBox>(LayoutDirection::Horizontal, 8.0f);
    toolbarBox->addWidget(DCButton);
    toolbarBox->addWidget(trainingButton);
    toolbarBox->addWidget(storageButton);
    toolbarBox->addWidget(cameraButton);
    toolbarBox->addWidget(digitalButton);
    toolbarBox->addWidget(geoSpatialButton);
    toolbarBox->addWidget(settingsButton);
    toolbarBox->layout();

    addTemporarySession();
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

    const float sidebarX = position.x + 12.0f;
    const float sidebarWidth = sessionSidebarWidth(size.x);
    const float sidebarTop = position.y + titleBarHeight + 10.0f;
    const float inputY = position.y + size.y - 56.0f;
    const float sessionListTop = sidebarTop + 32.0f;
    const float sessionListHeight = std::max(40.0f, inputY - sessionListTop - 8.0f);

    addSessionButton->setPosition(
        sidebarX + sidebarWidth - addSessionButton->getSize().x,
        sidebarTop - 4.0f);
    addSessionButton->update(input, dt);

    sessionScrollBox->setPosition(sidebarX, sessionListTop);
    sessionScrollBox->setSize(sidebarWidth, sessionListHeight);
    for (size_t index = 0; index < sessionSelectButtons.size(); ++index) {
        const float rowY = 6.0f + static_cast<float>(index) *
            (kSessionRowHeight + kSessionRowGap);
        sessionSelectButtons[index]->setPosition(6.0f, rowY);
        sessionSelectButtons[index]->setSize(
            sidebarWidth - 48.0f, kSessionRowHeight);
        sessionDeleteButtons[index]->setPosition(
            sidebarWidth - 36.0f, rowY);
        sessionDeleteButtons[index]->setSize(28.0f, kSessionRowHeight);
    }
    sessionScrollBox->setContentHeight(
        12.0f + static_cast<float>(sessions.size()) *
            (kSessionRowHeight + kSessionRowGap));
    sessionScrollBox->update(input, dt);

    if (!pendingSessionSelection.empty()) {
        activeSessionId = std::move(pendingSessionSelection);
        pendingSessionSelection.clear();
        activeSession();
        rebuildSessionWidgets();
    }
    if (!pendingSessionDeletion.empty()) {
        const std::string sessionId = std::move(pendingSessionDeletion);
        pendingSessionDeletion.clear();
        deleteSession(sessionId);
    }
    
    if (consoleInput) {
        const float mainContentX = sidebarX + sidebarWidth + 16.0f;
        consoleInput->setPosition(mainContentX + 26.0f, inputY + 4.0f);
        consoleInput->setSize(
            std::max(20.0f, position.x + size.x - 16.0f - (mainContentX + 26.0f)),
            30.0f);
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

    const float sidebarX = position.x + 12.0f;
    const float sidebarWidth = sessionSidebarWidth(size.x);
    const float sidebarTop = position.y + titleBarHeight + 10.0f;
    const float mainContentX = sidebarX + sidebarWidth + 16.0f;
    const float mainContentWidth = position.x + size.x - 12.0f - mainContentX;

    renderer.drawText(
        {sidebarX + 4.0f, sidebarTop},
        "Session History",
        UITheme::Colors::TextHeader);
    addSessionButton->drawOverlay(renderer, position);
    sessionScrollBox->drawOverlay(renderer, position);
    renderer.drawRect(
        {mainContentX - 8.0f, position.y + titleBarHeight + 10.0f},
        {1.0f, size.y - titleBarHeight - 76.0f},
        UITheme::Colors::DividerLine);

    auto& history = getConsoleHistory();
    
    float maxTextWidth = mainContentWidth - 28.0f;
    history.ensureWrapped(maxTextWidth);
    
    float y = position.y + titleBarHeight + 14;
    auto lines = history.wrapped();

    if (history.rawCount() == 0) {
        const std::string title = "G.R.I.M";
        const std::string readyMessage = "Ready to begin.";
        constexpr float titleScale = 2.5f;
        constexpr float accentWidth = 72.0f;
        const float contentCenterX = mainContentX + mainContentWidth * 0.5f;
        const float contentCenterY = position.y + titleBarHeight
            + (size.y - titleBarHeight - 56.0f) * 0.5f;
        const float titleWidth = renderer.measureTextWidth(title) * titleScale;

        renderer.drawTextScaled(
            {contentCenterX - titleWidth * 0.5f,
             contentCenterY - 52.0f},
            title,
            UITheme::Colors::TextHeader,
            titleScale);
        renderer.drawRect(
            {contentCenterX - accentWidth * 0.5f, contentCenterY + 2.0f},
            {accentWidth, 2.0f},
            UITheme::Colors::Primary);
        renderer.drawText(
            {contentCenterX - renderer.measureTextWidth(readyMessage) * 0.5f,
             contentCenterY + 18.0f},
            readyMessage,
            UITheme::Colors::TextSecondary);
    } else {
        float scrollAreaHeight = size.y - titleBarHeight - 76;
        int maxLines = static_cast<int>(scrollAreaHeight / 20.0f);
        int startIdx = std::max(0, static_cast<int>(lines.size()) - maxLines);

        for (int i = startIdx; i < static_cast<int>(lines.size()); ++i)
        {
            if (y >= position.y + size.y - 76) break;
            renderer.drawText({mainContentX, y}, lines[i].text, lines[i].color);
            y += 20.0f;
        }
    }

    // Glass input area at bottom
    float inputY = position.y + size.y - 56;
    float inputRadius = UITheme::Sizes::WidgetRadius + 4.0f;
    
    renderer.drawRoundedRect({mainContentX - 6.0f, inputY}, {mainContentWidth + 6.0f, 40}, UITheme::Colors::ScrollboxBg, inputRadius);
    renderer.drawRoundedBorder({mainContentX - 6.0f, inputY}, {mainContentWidth + 6.0f, 40}, UITheme::Colors::BorderSubtle, inputRadius);
    
    renderer.drawText({mainContentX + 4.0f, inputY + 11}, ">", UITheme::Colors::Primary);
    
    if (consoleInput) {
        consoleInput->drawOverlay(renderer, position);
    }
    
    renderer.drawText({mainContentX, inputY + 30}, "ESC Close", UITheme::Colors::TextDisabled);
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
    auto& session = activeSession();
    const std::string turnId = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    GRIM::MMO::SessionContextManager::instance().beginTurn(
        session.id, turnId, cmd, cmd);
    session.committed = true;
    handleCommand(cmd, session.id);
}

void ConsolePanel::addTemporarySession()
{
    const std::string name = generateSessionName();
    SessionEntry session;
    session.id = "session-" + name;
    session.name = name;
    sessions.push_back(std::move(session));
    activeSessionId = sessions.back().id;
    rebuildSessionWidgets();
}

void ConsolePanel::deleteSession(const std::string& sessionId)
{
    const auto it = std::find_if(
        sessions.begin(), sessions.end(),
        [&](const SessionEntry& session) { return session.id == sessionId; });
    if (it == sessions.end()) {
        throw std::runtime_error("ConsolePanel cannot delete unknown session: " + sessionId);
    }

    const bool wasActive = activeSessionId == sessionId;
    if (it->committed) {
        GRIM::MMO::SessionContextManager::instance().destroySession(sessionId);
    }
    sessions.erase(it);

    if (sessions.empty()) {
        addTemporarySession();
        return;
    }
    if (wasActive) {
        activeSessionId = sessions.back().id;
    }
    rebuildSessionWidgets();
}

void ConsolePanel::rebuildSessionWidgets()
{
    sessionScrollBox->clearChildren();
    sessionSelectButtons.clear();
    sessionDeleteButtons.clear();

    for (const auto& session : sessions) {
        const std::string sessionId = session.id;
        auto selectButton = std::make_shared<UIButton>(
            session.name,
            [this, sessionId]() { pendingSessionSelection = sessionId; });
        auto deleteButton = std::make_shared<UIButton>(
            "x",
            [this, sessionId]() { pendingSessionDeletion = sessionId; });
        if (session.id == activeSessionId) {
            selectButton->setColors(
                UITheme::Colors::RowSelected,
                UITheme::Colors::RowHover,
                UITheme::Colors::WidgetBgActive);
        }
        deleteButton->setColors(
            UITheme::Colors::WidgetBg,
            UITheme::Colors::DangerBg,
            UITheme::Colors::DangerBright);
        sessionScrollBox->addChild(selectButton);
        sessionScrollBox->addChild(deleteButton);
        sessionSelectButtons.push_back(std::move(selectButton));
        sessionDeleteButtons.push_back(std::move(deleteButton));
    }
}

ConsolePanel::SessionEntry& ConsolePanel::activeSession()
{
    const auto it = std::find_if(
        sessions.begin(), sessions.end(),
        [&](const SessionEntry& session) { return session.id == activeSessionId; });
    if (it == sessions.end()) {
        throw std::runtime_error("ConsolePanel active session is missing: " + activeSessionId);
    }
    return *it;
}

std::string ConsolePanel::generateSessionName() const
{
    static std::mt19937 generator(std::random_device{}());
    static std::uniform_int_distribution<int> distribution(100000, 999999);

    for (int attempt = 0; attempt < 100; ++attempt) {
        const std::string candidate = std::to_string(distribution(generator));
        const bool duplicate = std::any_of(
            sessions.begin(), sessions.end(),
            [&](const SessionEntry& session) { return session.name == candidate; });
        if (!duplicate) {
            return candidate;
        }
    }
    throw std::runtime_error("ConsolePanel failed to generate a unique session name");
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
