#include "ui/data_hub_services.hpp"
#include "ui/ui_panel_service.hpp"
#include "ui/ui_root.hpp"
#include "ui/input/ui_input_router.hpp"
#include "core/input_parser.hpp"
#include "MMO/Core/Registries/ProviderRegistry.hpp"

#include <filesystem>
#include <iostream>

extern bool dataHubTestFail;
extern std::string renderedTestText;
void startDataHubFixture(const std::filesystem::path&);
void stopDataHubFixture();
size_t pendingDataHubFixtureQueries();
void completeDataHubFixtureQueries();
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void frame(UIPanel& panel, InputState input = {}) {
    auto& router = GRIM::UI::InputRouter::get();
    router.beginFrame(); router.currentPanel = panel.getPanelID();
    panel.update(input, 0.016f); router.endFrame();
}
void draw(UIPanel& panel) {
    OverlayRenderer renderer; renderedTestText.clear();
    require(panel.drawOverlay(renderer), "DataHub native draw failed");
}
void click(UIPanel& panel, Vec2 point) {
    InputState input; input.mousePos = point;
    input.mousePressed[0] = input.mouseDown[0] = true; frame(panel, input);
    input.mousePressed[0] = input.mouseDown[0] = false;
    input.mouseReleased[0] = true; frame(panel, input); draw(panel);
}
struct Module {
    HMODULE handle;
    GrimPluginAPI api{};
    std::shared_ptr<GRIM::UI::ProviderPanelLease> lease = std::make_shared<GRIM::UI::ProviderPanelLease>();
    GRIM::UI::ProviderInit initialize;
    void (*shutdown)();
    Module(const std::filesystem::path& path, const std::string& generation) {
        lease->providerName = "DataHub Plugin"; lease->owner = generation;
        api.api_version = GRIM_API_VERSION;
        api.log = [](const GrimPluginAPI*, GrimLogLevel, const char*) {};
        handle = LoadLibraryW(path.c_str());
        require(handle != nullptr, "DataHub DLL could not import host services");
        initialize = reinterpret_cast<GRIM::UI::ProviderInit>(GetProcAddress(handle, "grim_plugin_init"));
        shutdown = reinterpret_cast<void(*)()>(GetProcAddress(handle, "grim_plugin_shutdown"));
        if (!initialize || !shutdown) {
            FreeLibrary(handle);
            throw std::runtime_error("Missing DataHub lifecycle exports");
        }
    }
    GrimResult init(bool permission = true) {
        return GRIM::UI::initializeProviderPanels(lease, permission, initialize, &api);
    }
    void publish(const std::string& replacing = {}) {
        require(GRIM::MMO::ProviderRegistry::instance().publish(
            {GRIM::MMO::ToolProviderType::Plugin, lease->providerName, lease->owner, {}}, {}, replacing),
            "DataHub provider publication failed");
        GRIM::UI::activateProviderPanels(lease, {});
    }
    void close() {
        if (!handle) return;
        GRIM::UI::retireProviderPanels(lease); shutdown();
        require(lease->nativePanels.empty(), "DataHub shutdown retained native objects");
        GRIM::MMO::ProviderRegistry::instance().retire(lease->providerName, lease->owner);
        require(FreeLibrary(handle) != 0, "DataHub DLL unload failed");
        handle = nullptr;
    }
    ~Module() { close(); }
};
}
void runNativeDataHubDllTests(const char* dllPath) {
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-datahub-panel-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    startDataHubFixture(directory);
    try {
        namespace Service = GRIM::UI::DataHub;
        const auto source = nlohmann::json{{"name","Original"},{"url","https://example.invalid"},
            {"source_type","html_crawl"},{"priority",5},{"enabled",true},{"crawl_depth",2},
            {"fetch_limit",100},{"requires_auth",false}};
        const auto baseline = Service::saveSources(nlohmann::json::array({source}), nlohmann::json::array());
        Service::addDownloadedSource("fixture/download", "Downloaded");
        auto edited = baseline;
        edited[0]["name"] = "Edited";
        const auto merged = Service::saveSources(edited, baseline);
        require(merged.size() == 2 && merged[0]["name"] == "Edited", "Host download source was lost by draft save");
        bool rejected = false;
        try { Service::saveSources(baseline, baseline); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Stale source edits overwrote host state");
        const auto persisted = Service::sources();
        for (const auto* name : {"first.dll","second.dll","failed.dll","cancelled.dll","denied.dll"})
            std::filesystem::copy_file(dllPath, directory / name);
        auto& root = UIRoot::get();
        {
            Module first(directory / "first.dll", "datahub-first");
            require(first.init() == GRIM_OK && !root.getPanelByKey("data_hub"), "DataHub attached before commit");
            require(Service::loadSettings().empty() && Service::sources() == persisted,
                    "Candidate construction persisted panel state");
            first.publish(); root.processPendingTasks();
            auto* panel = root.getPanelByKey("data_hub");
            require(panel && !panel->isVisible(), "DataHub initial visibility changed");
            const auto runtime = panel->getPanelID();
            panel->setVisible(true); frame(*panel); draw(*panel);
            const auto position = panel->getPosition();
            for (const float offset : {105.0f,200.0f,315.0f,420.0f,525.0f,10.0f})
                click(*panel, {position.x + offset + 20, position.y + 48});
            click(*panel, {position.x + 220, position.y + 48});
            auto content = panel->getContentRect();
            const float searchY = content.origin.y + 33 + 10 + UITheme::Sizes::HeaderHeight + 5 + 30;
            const float browseX = content.origin.x + 15 + (content.size.x - 30) * 0.55f + 10 + 95 + 50;
            click(*panel, {browseX, searchY + 14});
            require(pendingDataHubFixtureQueries() == 1, "HF search did not submit a host request");
            completeDataHubFixtureQueries(); frame(*panel); draw(*panel);
            require(pendingDataHubFixtureQueries() == 0 &&
                    renderedTestText.find("Fixture dataset") != std::string::npos,
                    "HF completion did not update native widgets on the UI thread");
            click(*panel, {browseX, searchY + 14});
            require(pendingDataHubFixtureQueries() == 1, "Second HF request missing");
            {
                Module failed(directory / "failed.dll", "datahub-failed");
                dataHubTestFail = true;
                const auto result = failed.init();
                dataHubTestFail = false;
                require(result != GRIM_OK && failed.lease->nativePanels.empty(),
                        "Failed DataHub candidate leaked registration");
                require(root.getPanelByKey("data_hub") == panel && Service::sources() == persisted,
                        "Failed candidate changed live panel or persisted sources");
            }
            Module second(directory / "second.dll", "datahub-second");
            require(second.init() == GRIM_OK, "DataHub replacement construction failed");
            require(!GRIM::UI::validateProviderPanels(second.lease), "Foreign panel collision accepted");
            require(GRIM::UI::validateProviderPanels(second.lease, first.lease), "DataHub replacement rejected");
            second.publish(first.lease->owner); first.close();
            require(pendingDataHubFixtureQueries() == 0 && !root.getPanelByKey("data_hub"),
                    "DataHub retirement retained requests or drawing pointers");
            root.processPendingTasks();
            auto* replacement = root.getPanelByKey("data_hub");
            require(replacement && replacement->getPanelID() != runtime && !replacement->isVisible(),
                    "DataHub replacement identity/default visibility failed");
            GRIM::UI::retireProviderPanels(first.lease);
            require(root.getPanelByKey("data_hub") == replacement, "Old retirement removed replacement");
            bool owned = false;
            for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
                if (group.provider.name == "DataHub Plugin")
                    owned = group.provider.instance == second.lease->owner && group.tools.empty() &&
                        group.resources.size() == 1 && group.resources[0].resource_id == "panel:data_hub";
            require(owned, "DataHub resource ownership missing");
        }
        require(!root.getPanelByKey("data_hub"), "Unloaded DataHub remained attached");
        {
            Module cancelled(directory / "cancelled.dll", "datahub-cancelled");
            require(cancelled.init() == GRIM_OK, "Cancelled candidate init failed");
            cancelled.publish(); cancelled.close(); root.processPendingTasks();
            require(!root.getPanelByKey("data_hub"), "Activation reached unloaded DataHub");
        }
        {
            Module denied(directory / "denied.dll", "datahub-denied");
            require(denied.init(false) != GRIM_OK && denied.lease->nativePanels.empty(),
                    "DataHub bypassed UI permission");
        }
        stopDataHubFixture();
    } catch (...) {
        dataHubTestFail = false; stopDataHubFixture();
        std::filesystem::remove_all(directory); throw;
    }
    std::filesystem::remove_all(directory);
    std::cout << "DataHub native DLL six tabs, host persistence, requests and lifecycle tests passed\n";
}
