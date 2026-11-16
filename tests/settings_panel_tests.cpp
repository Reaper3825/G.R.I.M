#include "tests/provider_panel_fixture.hpp"
#include "core/input_parser.hpp"
#include "ui/ui_root.hpp"
#include "core/input/InputBindings.hpp"
#include "ui/input/ui_input_router.hpp"
#include "MMO/Core/Registries/ProviderRegistry.hpp"

#include <atomic>
#include <iostream>
#include <filesystem>
#include <thread>

extern int settingsTestSaves, settingsTestApplies;
extern bool settingsTestLoadFailure;
namespace {
void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void frame(UIPanel& panel, InputState input = {}) {
    auto& router = GRIM::UI::InputRouter::get();
    router.beginFrame(); router.currentPanel = panel.getPanelID();
    panel.update(input, 0.016f); router.endFrame();
}
struct Module {
    HMODULE handle;
    GrimPluginAPI api{};
    std::shared_ptr<GRIM::UI::ProviderPanelLease> lease = std::make_shared<GRIM::UI::ProviderPanelLease>();
    void (*shutdown)();
    Module(const std::filesystem::path& path, const std::string& generation) {
        lease->providerName = "Settings Plugin"; lease->owner = generation;
        api.api_version = GRIM_API_VERSION;
        api.log = [](const GrimPluginAPI*, GrimLogLevel, const char*) {};
        handle = LoadLibraryW(path.c_str());
        require(handle != nullptr, "Settings DLL failed to load against host exports");
        shutdown = reinterpret_cast<void(*)()>(GetProcAddress(handle, "grim_plugin_shutdown"));
        require(shutdown != nullptr, "Missing Settings shutdown");
    }
    GrimResult init(bool permission = true) {
        auto init = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*)>(GetProcAddress(handle, "grim_plugin_init"));
        require(init != nullptr, "Missing Settings init");
        prepareProviderPanelFixture(api, lease, permission);
        return init(&api);
    }
    void publish(const std::string& replacing = {}) {
        require(GRIM::MMO::ProviderRegistry::instance().publish(
            {GRIM::MMO::ToolProviderType::Plugin, lease->providerName, lease->owner, {}}, {}, replacing), "Settings provider publish failed");
        GRIM::UI::activateProviderPanels(lease);
    }
    void close() {
        if (!handle) return;
        GRIM::UI::retireProviderPanels(lease); shutdown();
        require(lease->nativePanels.empty(), "Settings shutdown retained native objects");
        GRIM::MMO::ProviderRegistry::instance().retire(lease->providerName, lease->owner);
        FreeLibrary(handle); handle = nullptr;
    }
    ~Module() { close(); }
};
}
void runNativeSettingsDllTests(const char* dllPath) {
    auto& root = UIRoot::get();
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-settings-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    try {
        for (const auto* name : {"first.dll", "second.dll", "cancelled.dll", "denied.dll", "failed.dll"})
            std::filesystem::copy_file(dllPath, directory / name);
        {
            Module first(directory / "first.dll", "settings-first");
            require(first.init() == GRIM_OK, "Settings init failed");
            require(!root.getPanelByKey("settings"), "Settings attached before commit");
            first.publish(); root.processPendingTasks();
            auto* panel = root.getPanelByKey("settings");
            require(panel && !panel->isVisible(), "Settings identity/default visibility changed");
            const auto runtime = panel->getPanelID();
            panel->setVisible(true); frame(*panel); frame(*panel);
            auto click = [&](float x, float y) {
                InputState input; input.mousePos = {x,y}; input.mousePressed[0] = input.mouseDown[0] = true; frame(*panel,input);
                input.mousePressed[0] = input.mouseDown[0] = false; input.mouseReleased[0] = true; frame(*panel,input);
                frame(*panel); frame(*panel);
            };
            // Exercise every tab and imported widget type through native UI events.
            for (int tab = 1; tab < 10; ++tab) {
                click(240.0f + 60.0f * tab, 298);
                OverlayRenderer renderer; require(panel->drawOverlay(renderer), "Settings draw failed");
            }
            click(240,298); // General
            click(260,340); // Backend edit
            click(650,920); // Cancel
            require(!panel->isVisible() && settingsTestSaves == 0, "Cancel persisted settings");
            panel->setVisible(true); frame(*panel); frame(*panel);
            click(260,340); // Backend edit
            click(300,920); // Save & Close
            require(!panel->isVisible() && settingsTestSaves == 1 && settingsTestApplies == 1, "Save/apply boundary failed");
            panel->setVisible(true); frame(*panel); frame(*panel);
            click(720,298); // Controls
            click(700,350); // Begin shortcut capture
            require(GRIM::InputBindings::isCaptureActive(), "Controls capture did not start");
            Module second(directory / "second.dll", "settings-second"); require(second.init() == GRIM_OK, "Replacement init failed");
            require(!GRIM::UI::validateProviderPanels(second.lease), "Foreign collision accepted");
            require(GRIM::UI::validateProviderPanels(second.lease, first.lease), "Settings replacement rejected");
            second.publish(first.lease->owner);
            first.close();
            require(!root.getPanelByKey("settings") && !GRIM::InputBindings::isCaptureActive(), "Settings unload retained panel/capture");
            require(settingsTestSaves == 1, "Unload saved pending settings");
            root.processPendingTasks();
            require(root.getPanelByKey("settings") && root.getPanelByKey("settings")->getPanelID() != runtime, "Settings replacement missing");
            GRIM::UI::retireProviderPanels(first.lease);
            require(root.getPanelByKey("settings"), "Old retirement removed replacement");
        }
        require(!root.getPanelByKey("settings"), "Unloaded Settings remained attached");
        {
            Module cancelled(directory / "cancelled.dll", "settings-cancelled");
            require(cancelled.init() == GRIM_OK, "Cancelled Settings init failed");
            cancelled.publish(); cancelled.close(); root.processPendingTasks();
            require(!root.getPanelByKey("settings"), "Queued activation reached unloaded Settings");
        }
        {
            Module denied(directory / "denied.dll", "settings-denied");
            require(denied.init(false) != GRIM_OK, "Settings bypassed UI permission");
        }
        {
            Module failed(directory / "failed.dll", "settings-failed");
            settingsTestLoadFailure = true;
            require(failed.init() != GRIM_OK, "Configuration load failure was accepted");
            settingsTestLoadFailure = false;
            require(failed.lease->nativePanels.empty(), "Failed construction leaked staged panel");
        }
    } catch (...) { std::filesystem::remove_all(directory); throw; }
    std::filesystem::remove_all(directory);
    std::cout << "Settings native DLL tabs, Save/Cancel, capture and lifecycle tests passed\n";
}
