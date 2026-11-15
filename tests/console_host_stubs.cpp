// Only linked by the native panel test host; no command/model runtime starts.
#include "MMO/Core/SessionContextManager.hpp"
#include "commands/commands_core.hpp"
#include "helpers/key.hpp"
#include "ui/overlay_renderer.hpp"
#include <atomic>
#include <thread>

std::atomic<int> consoleTestCommands{0};
std::atomic<bool> consoleTestFinished{false};
std::string consoleTestSession;
namespace {
std::unordered_map<std::string, std::unique_ptr<ConsoleHistory>> histories;
std::unordered_map<std::string, std::optional<GRIM::ReasoningState>> states;
}
namespace GRIM::MMO {
SessionContextManager& SessionContextManager::instance() { static SessionContextManager value; return value; }
void SessionContextManager::setReasoningState(const std::string& id, std::optional<GRIM::ReasoningState> state) { states[id] = std::move(state); }
std::optional<GRIM::ReasoningState> SessionContextManager::getReasoningState(const std::string& id) const { return states[id]; }
void SessionContextManager::beginTurn(const std::string& id, const std::string&, const std::string&, const std::string&) { consoleTestSession = id; }
void SessionContextManager::destroySession(const std::string& id) { histories.erase(id); states.erase(id); }
ConsoleHistory& SessionContextManager::displayHistory(const std::string& id) {
    auto& history = histories[id];
    if (!history) history = std::make_unique<ConsoleHistory>();
    return *history;
}
}
CommandResult handleCommand(const std::string&, const std::string&, std::optional<GRIM::ReasoningState>) {
    ++consoleTestCommands;
    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    consoleTestFinished = true;
    CommandResult result; result.message = "console fixture response"; return result;
}
bool Key::wasPressed(KeyCode) { return false; }
void OverlayRenderer::drawTextScaled(const Vec2&, const std::string&, uint32_t, float) {}
