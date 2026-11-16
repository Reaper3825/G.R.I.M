#include "tests/provider_panel_fixture.hpp"
#include "core/input_parser.hpp"
#include "ui/ui_root.hpp"

#include "ui/input/ui_input_router.hpp"
#include "MMO/Core/Registries/ProviderRegistry.hpp"

#include <atomic>
#include <iostream>
#include <filesystem>
#include <thread>

extern std::filesystem::path modelWorkspaceTestRoot;
extern bool modelWorkspaceTestFail;
namespace {
void require(bool ok, const char* error) { if (!ok) throw std::runtime_error(error); }
void frame(UIPanel& panel, InputState input = {}) {
    auto& router = GRIM::UI::InputRouter::get();
    router.beginFrame(); router.currentPanel = panel.getPanelID();
    panel.update(input, 0.016f); router.endFrame();
}
struct Module {
    const char* const providerName = "Model Workspace Plugin";
    HMODULE handle;
    GrimPluginAPI api{};
    std::shared_ptr<GRIM::UI::ProviderPanelLease> lease = std::make_shared<GRIM::UI::ProviderPanelLease>();
    void (*shutdown)();
    Module(const std::filesystem::path& path, const std::string& generation) {
        lease->owner = generation;
        api.api_version = GRIM_API_VERSION;
        api.log = [](const GrimPluginAPI*, GrimLogLevel, const char*) {};
        handle = LoadLibraryW(path.c_str());
        require(handle != nullptr, "ModelWorkspace DLL failed to load against host exports");
        shutdown = reinterpret_cast<void(*)()>(GetProcAddress(handle, "grim_plugin_shutdown"));
        require(shutdown != nullptr, "Missing ModelWorkspace shutdown");
    }
    GrimResult init(bool permission = true) {
        auto init = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*)>(GetProcAddress(handle, "grim_plugin_init"));
        require(init != nullptr, "Missing ModelWorkspace init");
        prepareProviderPanelFixture(api, lease, permission);
        return init(&api);
    }
    void publish(const std::string& replacing = {}) {
        require(GRIM::MMO::ProviderRegistry::instance().publish(
            {GRIM::MMO::ToolProviderType::Plugin, providerName, lease->owner, {}}, {}, replacing), "ModelWorkspace provider publish failed");
        GRIM::UI::activateProviderPanels(lease);
    }
    void close() {
        if (!handle) return;
        GRIM::UI::retireProviderPanels(lease); shutdown();
        require(lease->nativePanels.empty(), "ModelWorkspace shutdown retained native objects");
        GRIM::MMO::ProviderRegistry::instance().retire(providerName, lease->owner);
        FreeLibrary(handle); handle = nullptr;
    }
    ~Module() { close(); }
};
}
void runNativeModelWorkspaceDllTests(const char* dllPath) {
    auto& root = UIRoot::get();
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-workspace-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    modelWorkspaceTestRoot = directory;
    try {
        for (const auto* name : {"first.dll", "second.dll", "cancelled.dll", "denied.dll", "failed.dll"})
            std::filesystem::copy_file(dllPath, directory / name);
        {
            Module first(directory / "first.dll", "workspace-first");
            require(first.init() == GRIM_OK, "ModelWorkspace init failed");
            require(!root.getPanelByKey("training_control"), "ModelWorkspace attached before commit");
            first.publish(); root.processPendingTasks();
            auto* panel = root.getPanelByKey("training_control");
            require(panel && !panel->isVisible(), "ModelWorkspace identity/default visibility changed");
            const auto runtime = panel->getPanelID();
            panel->setVisible(true); frame(*panel);
            OverlayRenderer renderer; require(panel->drawOverlay(renderer), "Workspace draw failed");
            // Tokenizer requires a server; exercise the other eight tabs only.
            for (float offset : {95.0f,220.0f,315.0f,530.0f,650.0f,755.0f,840.0f,0.0f}) {
                InputState input; input.mousePos={230+offset,125}; input.mousePressed[0]=input.mouseDown[0]=true; frame(*panel,input);
                input.mousePressed[0]=input.mouseDown[0]=false; input.mouseReleased[0]=true; frame(*panel,input);
                frame(*panel); require(panel->drawOverlay(renderer), "Workspace tab draw failed");
            }
            Module second(directory / "second.dll", "workspace-second"); require(second.init() == GRIM_OK, "Replacement init failed");
            require(!GRIM::UI::validateProviderPanels(second.lease), "Foreign collision accepted");
            require(GRIM::UI::validateProviderPanels(second.lease, first.lease), "ModelWorkspace replacement rejected");
            second.publish(first.lease->owner);
            first.close();
            require(!root.getPanelByKey("training_control"), "ModelWorkspace unload retained panel/capture");
            root.processPendingTasks();
            require(root.getPanelByKey("training_control") && root.getPanelByKey("training_control")->getPanelID() != runtime, "ModelWorkspace replacement missing");
            GRIM::UI::retireProviderPanels(first.lease);
            require(root.getPanelByKey("training_control"), "Old retirement removed replacement");
        }
        require(!root.getPanelByKey("training_control"), "Unloaded ModelWorkspace remained attached");
        {
            Module cancelled(directory / "cancelled.dll", "workspace-cancelled");
            require(cancelled.init() == GRIM_OK, "Cancelled ModelWorkspace init failed");
            cancelled.publish(); cancelled.close(); root.processPendingTasks();
            require(!root.getPanelByKey("training_control"), "Queued activation reached unloaded ModelWorkspace");
        }
        {
            Module denied(directory / "denied.dll", "workspace-denied");
            require(denied.init(false) != GRIM_OK, "ModelWorkspace bypassed UI permission");
        }
        {
            Module failed(directory / "failed.dll", "workspace-failed");
            modelWorkspaceTestFail = true;
            require(failed.init() != GRIM_OK, "Configuration load failure was accepted");
            modelWorkspaceTestFail = false;
            require(failed.lease->nativePanels.empty(), "Failed construction leaked staged panel");
        }
    } catch (...) { std::filesystem::remove_all(directory); throw; }
    std::filesystem::remove_all(directory);
    std::cout << "ModelWorkspace native DLL tabs and lifecycle tests passed\n";
}
