#pragma once
#include "ui_panel.hpp"
#include "ui_button.hpp"  // ? NEW: For settings button
#include "console_history.hpp"
#include "commands/commands_core.hpp"
#include <string>

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
    std::string inputBuffer;
    bool caretVisible = true;
    uint64_t lastCaretToggle = 0;
    
    UIButton settingsButton;  // ? NEW: Settings button
    
    // Helper to get current time string
    std::string getCurrentTime() const;
};
