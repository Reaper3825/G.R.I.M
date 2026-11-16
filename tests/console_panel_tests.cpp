#include "tests/provider_panel_fixture.hpp"
#include "core/input_parser.hpp"
#include "ui/ui_root.hpp"
#include "ui/console_notification_target.hpp"
#include "ui/input/ui_input_router.hpp"
#include "MMO/Core/Registries/ProviderRegistry.hpp"
#include "MMO/Core/SessionContextManager.hpp"
#include <atomic>
#include <filesystem>
#include <thread>

extern std::atomic<int> consoleTestCommands;
extern std::atomic<bool> consoleTestFinished;
extern std::string consoleTestSession;
namespace {
void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void frame(UIPanel& panel, InputState input = {}) {
    auto& router = GRIM::UI::InputRouter::get();
    router.beginFrame(); router.currentPanel = panel.getPanelID();
    panel.update(input, 0.016f); router.endFrame();
}
struct Module {
    const char* const providerName = "Console Plugin";
    HMODULE handle;
    GrimPluginAPI api{};
    std::shared_ptr<GRIM::UI::ProviderPanelLease> lease = std::make_shared<GRIM::UI::ProviderPanelLease>();
    void (*shutdown)();
    Module(const std::filesystem::path& path, const std::string& generation) {
        lease->owner = generation;
        api.api_version = GRIM_API_VERSION;
        api.log = [](const GrimPluginAPI*, GrimLogLevel, const char*) {};
        handle = LoadLibraryW(path.c_str());
        require(handle != nullptr, "Console DLL failed to load against host exports");
        shutdown = reinterpret_cast<void(*)()>(GetProcAddress(handle, "grim_plugin_shutdown"));
        require(shutdown != nullptr, "Missing Console shutdown");
    }
    GrimResult init(bool permission = true) {
        auto init = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*)>(GetProcAddress(handle, "grim_plugin_init"));
        require(init != nullptr, "Missing Console init");
        prepareProviderPanelFixture(api, lease, permission);
        return init(&api);
    }
    void publish(const std::string& replacing = {}) {
        require(GRIM::MMO::ProviderRegistry::instance().publish(
            {GRIM::MMO::ToolProviderType::Plugin, providerName, lease->owner, {}}, {}, replacing), "Console provider publish failed");
        GRIM::UI::activateProviderPanels(lease);
    }
    void close() {
        if (!handle) return;
        GRIM::UI::retireProviderPanels(lease); shutdown();
        require(lease->nativePanels.empty(), "Console shutdown retained native objects");
        GRIM::MMO::ProviderRegistry::instance().retire(providerName, lease->owner);
        FreeLibrary(handle); handle = nullptr;
    }
    ~Module() { close(); }
};
}
void runNativeConsoleDllTests(const char* dllPath) {
    auto& root = UIRoot::get();
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-console-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    try {
        for (const auto* name : {"first.dll", "second.dll", "cancelled.dll", "denied.dll"})
            std::filesystem::copy_file(dllPath, directory / name);
        {
            Module first(directory / "first.dll", "console-first");
            require(first.init() == GRIM_OK, "Console init failed");
            require(!root.getPanelByKey("console"), "Console attached before commit");
            first.publish(); root.processPendingTasks();
            auto* panel = root.getPanelByKey("console");
            require(panel && !panel->isVisible(), "Console identity/default visibility changed");
            const auto runtime = panel->getPanelID();
            auto* target = dynamic_cast<ConsoleNotificationTarget*>(panel);
            require(target != nullptr, "Host notification interface missing across DLL boundary");
            target->appendNotification("native notification");
            panel->setVisible(true); frame(*panel);
            OverlayRenderer renderer; require(panel->drawOverlay(renderer), "Console DLL draw failed");
            InputState input; input.mousePos = {400, 760}; input.mousePressed[0] = input.mouseDown[0] = true; frame(*panel, input);
            input.mousePressed[0] = input.mouseDown[0] = false; input.mouseReleased[0] = true; frame(*panel, input);
            input = {}; input.textInput = "test prompt"; frame(*panel, input);
            input = {}; input.keyPressed[VK_RETURN] = true; frame(*panel, input);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!consoleTestCommands && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            require(consoleTestCommands == 1, "Console input did not reach host command handler");
            Module second(directory / "second.dll", "console-second"); require(second.init() == GRIM_OK, "Replacement init failed");
            require(!GRIM::UI::validateProviderPanels(second.lease), "Foreign collision accepted");
            require(GRIM::UI::validateProviderPanels(second.lease, first.lease), "Console replacement rejected");
            second.publish(first.lease->owner);
            first.close();
            require(consoleTestFinished && !root.getPanelByKey("console"), "Console unload did not drain/detach");
            auto& history = GRIM::MMO::SessionContextManager::instance().displayHistory(consoleTestSession);
            history.ensureWrapped(1000);
            bool delivered = false;
            for (const auto& line : history.wrapped()) if (line.text == "console fixture response") delivered = true;
            require(delivered, "Shutdown lost active command result");
            root.processPendingTasks();
            require(root.getPanelByKey("console") && root.getPanelByKey("console")->getPanelID() != runtime, "Console replacement missing");
            GRIM::UI::retireProviderPanels(first.lease);
            require(root.getPanelByKey("console"), "Old retirement removed replacement");
        }
        require(!root.getPanelByKey("console"), "Unloaded Console remained attached");
        {
            Module cancelled(directory / "cancelled.dll", "console-cancelled");
            require(cancelled.init() == GRIM_OK, "Cancelled Console init failed");
            cancelled.publish(); cancelled.close(); root.processPendingTasks();
            require(!root.getPanelByKey("console"), "Queued activation reached unloaded Console");
        }
        {
            Module denied(directory / "denied.dll", "console-denied");
            require(denied.init(false) != GRIM_OK, "Console bypassed UI permission");
        }
    } catch (...) { std::filesystem::remove_all(directory); throw; }
    std::filesystem::remove_all(directory);
}
