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
            submitPrompt(submittedText);
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
    processCompletedRequest();

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

    auto& history = GRIM::MMO::SessionContextManager::instance()
        .displayHistory(activeSessionId);
    
    constexpr float bubblePadX = 9.0f;
    constexpr float bubblePadY = 5.0f;
    constexpr float bubbleRadius = 7.0f;
    constexpr float lineHeight = 20.0f;
    const float maxBubbleWidth = mainContentWidth - 20.0f;
    const float maxTextWidth = maxBubbleWidth - bubblePadX * 2.0f;
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
        const float historyBottom = position.y + size.y - 76.0f;
        const float scrollAreaHeight = historyBottom - y;
        int startIdx = static_cast<int>(lines.size());
        float usedHeight = 0.0f;
        while (startIdx > 0) {
            const int groupEnd = startIdx;
            int groupStart = groupEnd - 1;
            float groupHeight = lineHeight;
            if (!lines[groupStart].text.empty()) {
                while (groupStart > 0 &&
                       lines[groupStart - 1].message_id == lines[groupStart].message_id) {
                    --groupStart;
                }
                groupHeight = static_cast<float>(groupEnd - groupStart) * lineHeight
                    + bubblePadY * 3.0f;
            }

            if (usedHeight + groupHeight > scrollAreaHeight) {
                if (usedHeight == 0.0f && !lines[groupStart].text.empty()) {
                    const int fittingLines = std::max(
                        1,
                        static_cast<int>(
                            (scrollAreaHeight - bubblePadY * 3.0f) / lineHeight));
                    startIdx = std::max(groupStart, groupEnd - fittingLines);
                }
                break;
            }

            usedHeight += groupHeight;
            startIdx = groupStart;
        }

        for (int i = startIdx; i < static_cast<int>(lines.size());) {
            if (y >= historyBottom) break;

            if (lines[i].text.empty()) {
                y += lineHeight;
                ++i;
                continue;
            }

            int groupEnd = i + 1;
            float widestText = renderer.measureTextWidth(lines[i].text);
            while (groupEnd < static_cast<int>(lines.size()) &&
                   lines[groupEnd].message_id == lines[i].message_id) {
                widestText = std::max(
                    widestText,
                    renderer.measureTextWidth(lines[groupEnd].text));
                ++groupEnd;
            }

            const int visibleGroupEnd = std::min(
                groupEnd,
                i + static_cast<int>((historyBottom - y) / lineHeight));
            if (visibleGroupEnd <= i) break;

            const float bubbleWidth = std::min(
                maxBubbleWidth, widestText + bubblePadX * 2.0f);
            const float bubbleHeight =
                static_cast<float>(visibleGroupEnd - i) * lineHeight + bubblePadY * 2.0f;
            const bool isUser =
                lines[i].alignment == ConsoleHistory::Alignment::Right;
            const float bubbleX = isUser
                ? mainContentX + maxBubbleWidth - bubbleWidth
                : mainContentX;
            const uint32_t bubbleColor = isUser
                ? UITheme::Colors::RowSelected
                : UITheme::Colors::WidgetBg;

            renderer.drawRoundedRect(
                {bubbleX, y - bubblePadY},
                {bubbleWidth, bubbleHeight},
                bubbleColor,
                bubbleRadius);
            renderer.drawRoundedBorder(
                {bubbleX, y - bubblePadY},
                {bubbleWidth, bubbleHeight},
                UITheme::Colors::BorderSubtle,
                bubbleRadius);

            for (int lineIndex = i; lineIndex < visibleGroupEnd; ++lineIndex) {
                renderer.drawText(
                    {bubbleX + bubblePadX,
                     y + static_cast<float>(lineIndex - i) * lineHeight},
                    lines[lineIndex].text,
                    lines[lineIndex].color);
            }

            y += bubbleHeight + bubblePadY;
            i = groupEnd;
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
    submitPrompt(cmd);
}

void ConsolePanel::setReasoningState(std::optional<GRIM::ReasoningState> state)
{
    GRIM::MMO::SessionContextManager::instance().setReasoningState(activeSession().id, std::move(state));
}

void ConsolePanel::submitPrompt(const std::string& prompt)
{
    auto& session = activeSession();
    auto& sessionManager = GRIM::MMO::SessionContextManager::instance();
    auto& history = sessionManager.displayHistory(session.id);

    const std::string timestamp = "[" + getCurrentTime() + "]";
    history.push("", UITheme::Colors::Background);
    history.push(
        timestamp + "  " + prompt,
        UITheme::Colors::PrimaryLight,
        ConsoleHistory::Alignment::Right);

    // Display-only note: use the same state snapshot as the queued request.
    // These placeholders never enter conversation history or model input.
    const auto reasoningState = sessionManager.getReasoningState(session.id);
    const auto hasText = [](const std::string& text) {
        return text.find_first_not_of(" \t\r\n") != std::string::npos;
    };
    const auto entrySummary = [&hasText](const std::vector<std::string>& entries) {
        const auto count = std::count_if(entries.begin(), entries.end(), hasText);
        return count == 0 ? std::string("(empty)")
                          : std::to_string(count) + " populated";
    };
    const GRIM::ReasoningState emptyState;
    const auto& state = reasoningState ? *reasoningState : emptyState;
    const GRIM::ConceptBlockGoal emptyGoal;
    const auto& goal = state.goal ? *state.goal : emptyGoal;
    const auto criteriaCount = std::count_if(
        goal.success_criteria.begin(), goal.success_criteria.end(),
        [&hasText](const auto& entry) { return hasText(entry.criterion); });
    std::string stateNote = reasoningState ? "[State]" : "[State: not supplied]";
    stateNote += " | Knowns: " + entrySummary(state.knowns);
    stateNote += " | Unknowns: " + entrySummary(state.unknowns);
    stateNote += std::string(" | Goal target state: ") +
        (hasText(goal.target_state) ? "populated" : "(empty)");
    stateNote += " | Goal success criteria: " +
        (criteriaCount == 0 ? std::string("(empty)") : std::to_string(criteriaCount) + " populated");
    stateNote += " | Goal constraints: " + entrySummary(goal.constraints);
    history.push(stateNote, UITheme::Colors::TextSecondary, ConsoleHistory::Alignment::Right);

    const std::string turnId = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    sessionManager.beginTurn(session.id, turnId, prompt, prompt);
    session.committed = true;

    pendingRequests.push_back({session.id, prompt, reasoningState});
    startNextRequest();
}

void ConsolePanel::startNextRequest()
{
    if (activeRequest.has_value() || pendingRequests.empty()) return;

    PendingRequest request = std::move(pendingRequests.front());
    pendingRequests.pop_front();
    const std::string sessionId = request.sessionId;
    const std::string prompt = request.prompt;
    const auto reasoningState = request.reasoningState;

    activeRequest.emplace(ActiveRequest{
        sessionId,
        std::async(std::launch::async, [sessionId, prompt, reasoningState]() {
            return handleCommand(prompt, sessionId, reasoningState);
        }),
        false
    });
}

void ConsolePanel::processCompletedRequest()
{
    if (!activeRequest.has_value()) {
        startNextRequest();
        return;
    }

    if (activeRequest->result.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready) {
        return;
    }

    const CommandResult result = activeRequest->result.get();
    const std::string sessionId = activeRequest->sessionId;
    const bool destroySession = activeRequest->destroySessionWhenDone;
    activeRequest.reset();

    auto& sessionManager = GRIM::MMO::SessionContextManager::instance();
    if (destroySession) {
        sessionManager.destroySession(sessionId);
    } else {
        auto& history = sessionManager.displayHistory(sessionId);
        history.push(
            result.message,
            (result.color.a << 24) | (result.color.b << 16) |
            (result.color.g << 8) | result.color.r);
        history.push("", UITheme::Colors::Background);
    }

    startNextRequest();
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
    pendingRequests.erase(
        std::remove_if(
            pendingRequests.begin(), pendingRequests.end(),
            [&](const PendingRequest& request) {
                return request.sessionId == sessionId;
            }),
        pendingRequests.end());

    const bool requestActive = activeRequest.has_value() &&
        activeRequest->sessionId == sessionId;
    if (requestActive) {
        activeRequest->destroySessionWhenDone = true;
    } else {
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

ConsoleHistory& ConsolePanel::getHistory()
{
    return GRIM::MMO::SessionContextManager::instance()
        .displayHistory(activeSessionId);
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
