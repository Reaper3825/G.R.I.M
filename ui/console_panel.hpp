#pragma once
#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_layout_box.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "console_history.hpp"
#include "commands/commands_core.hpp"
#include <deque>
#include <future>
#include <string>
#include <memory>
#include <optional>
#include <vector>

class OverlayRenderer;

class ConsolePanel : public UIPanel {
public:
    ConsolePanel();

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;
    
    // Execute a command programmatically
    void executeCommand(const std::string& cmd);
    
    // Get the active session's console history.
    ConsoleHistory& getHistory();

private:
    struct SessionEntry {
        std::string id;
        std::string name;
        bool committed = false;
    };

    std::string inputBuffer;  // Buffer for UIInputBox to bind to
    std::shared_ptr<UIInputBox> consoleInput;  // input widget
    bool caretVisible = true;
    uint64_t lastCaretToggle = 0;
    
    std::shared_ptr<UIButton> settingsButton;  // Settings button
    std::shared_ptr<UIButton> trainingButton;  // Training control button
    std::shared_ptr<UIButton> DCButton;        // Data Collection button
    std::shared_ptr<UIButton> storageButton;   // Shared Storage button
    std::shared_ptr<UIButton> cameraButton;    // Physical Environment / IP camera button
    std::shared_ptr<UIButton> digitalButton;   // Digital Environment / screen capture button
    std::shared_ptr<UIButton> geoSpatialButton; // GeoSpatial / Cesium viewport target button
    std::shared_ptr<UIHBox> toolbarBox;         // Horizontal layout for toolbar buttons

    std::vector<SessionEntry> sessions;
    std::string activeSessionId;
    std::string pendingSessionSelection;
    std::string pendingSessionDeletion;
    std::shared_ptr<UIButton> addSessionButton;
    std::shared_ptr<UIScrollBox> sessionScrollBox;
    std::vector<std::shared_ptr<UIButton>> sessionSelectButtons;
    std::vector<std::shared_ptr<UIButton>> sessionDeleteButtons;

    struct PendingRequest {
        std::string sessionId;
        std::string prompt;
    };

    struct ActiveRequest {
        std::string sessionId;
        std::future<CommandResult> result;
        bool destroySessionWhenDone = false;
    };

    std::deque<PendingRequest> pendingRequests;
    std::optional<ActiveRequest> activeRequest;

    void submitPrompt(const std::string& prompt);
    void startNextRequest();
    void processCompletedRequest();
    void addTemporarySession();
    void deleteSession(const std::string& sessionId);
    void rebuildSessionWidgets();
    SessionEntry& activeSession();
    std::string generateSessionName() const;
    
    // Helper to get current time string
    std::string getCurrentTime() const;
};
