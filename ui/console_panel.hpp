#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_inputbox.hpp"
#include "console_history.hpp"
#include "commands/commands_core.hpp"
#include <string>
#include <memory>

class OverlayRenderer;

class ConsolePanel : public UIPanel {
public:
    ConsolePanel();

    void update(const InputState& input, float dt) override;
    void drawOverlay(OverlayRenderer& renderer) override;
    
    // Execute a command programmatically
    void executeCommand(const std::string& cmd);
    
    // Get console history for external use (returns the global history)
    ConsoleHistory& getHistory() { return getConsoleHistory(); }

private:
    std::string inputBuffer;  // Buffer for UIInputBox to bind to
    std::shared_ptr<UIInputBox> consoleInput;  // input widget
    bool caretVisible = true;
    uint64_t lastCaretToggle = 0;
    
    std::shared_ptr<UIButton> settingsButton;  // Settings button
    std::shared_ptr<UIButton> trainingButton;  // Training control button
    std::shared_ptr<UIButton> DCButton;        // Data Collection button
    std::shared_ptr<UIButton> modelsButton;    // Model Registry button
    
    // Helper to get current time string
    std::string getCurrentTime() const;
};
