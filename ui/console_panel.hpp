#pragma once
#include "ui_panel.hpp"
#include "console_history.hpp"
#include "commands/commands_core.hpp"
#include <string>

class OverlayRenderer;  // Forward declaration

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
    // Don't store local history - use global getConsoleHistory() instead
    
    // Helper to get current time string
    std::string getCurrentTime() const;
};
