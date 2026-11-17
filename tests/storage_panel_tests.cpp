#include "ui/storage_panel_services.hpp"
#include "tests/provider_panel_fixture.hpp"
#include "ui/ui_root.hpp"
#include "ui/input/ui_input_router.hpp"
#include "core/input_parser.hpp"
#include "MMO/Core/Registries/ProviderRegistry.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

extern std::string renderedTestText;
namespace {
bool serviceAvailable = true;
GRIM::UI::Storage::Status state{4096, "1234-5678", false, {}, 0};
int regenerated = 0, connected = 0, disconnected = 0;
std::string linkedCode, pluginError;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void requireService() {
    UIRoot::get().requireUIThread();
    if (!serviceAvailable) throw std::runtime_error("Storage fixture service is unavailable");
}
}

// Only linked by the headless test host; no device server, filesystem or network work.
namespace GRIM::UI::Storage {
Status status() { requireService(); return state; }
std::vector<DirectoryEntry> listDirectory(const std::string& path) {
    requireService();
    require(path.empty(), "Storage initial listing is not the root");
    return {{"fixture.txt", false, 4096, "2026-11-15", "fixture-file", "fixture-device"}};
}
std::vector<DeviceSnapshot> listDevices() {
    requireService();
    DeviceSnapshot device{};
    device.device_id = "fixture-device"; device.device_name = "Fixture laptop";
    device.platform = Platform::Windows; device.pairing_state = PairingState::Paired;
    device.is_online = true;
    return {device};
}
std::vector<ActiveTransfer> listTransfers() {
    requireService();
    ActiveTransfer transfer;
    transfer.transfer_id = "fixture-transfer"; transfer.relative_path = "folder/upload.bin";
    transfer.total_size = 100; transfer.received_bytes = 50;
    return {transfer};
}
void regenerateCode() { requireService(); ++regenerated; state.localCode = "8765-4321"; }
bool linkDevice(const std::string& code) { requireService(); linkedCode = code; return true; }
HubResult connectHub(const std::string& host, uint16_t port) {
    requireService(); ++connected;
    state.connected = true; state.hubHost = host; state.hubPort = port;
    return {true, "Fixture connected"};
}
void disconnectHub() { requireService(); ++disconnected; state.connected = false; }
}

namespace {
void frame(UIPanel& panel, InputState input = {}, float dt = 0.016f) {
    auto& router = GRIM::UI::InputRouter::get();
    router.beginFrame(); router.currentPanel = panel.getPanelID();
    panel.update(input, dt); router.endFrame();
}
void draw(UIPanel& panel) {
    renderedTestText.clear();
    OverlayRenderer renderer;
    require(panel.drawOverlay(renderer), "Storage native draw failed");
}
void click(UIPanel& panel, Vec2 point) {
    InputState input; input.mousePos = point;
    input.mousePressed[0] = input.mouseDown[0] = true; frame(panel, input);
    input.mousePressed[0] = input.mouseDown[0] = false;
    input.mouseReleased[0] = true; frame(panel, input);
    draw(panel);
}
void edit(UIPanel& panel, Vec2 point, const std::string& text) {
    click(panel, point);
    InputState select; select.ctrl = true; select.keyPressed['A'] = true; frame(panel, select);
    InputState input; input.textInput = text; frame(panel, input);
}
void checkResource(const std::string& generation, uint64_t runtime) {
    const auto snapshot = GRIM::MMO::ProviderRegistry::instance().snapshot();
    for (const auto& group : snapshot.providers) {
        if (group.provider.name != "Storage Plugin") continue;
        require(group.provider.instance == generation && group.tools.empty(),
                "Storage provider generation/tools changed");
        require(group.resources.size() == 1 &&
                group.resources[0].resource_id == "panel:storage" &&
                group.resources[0].kind == "ui.panel" &&
                group.resources[0].runtime_id == runtime,
                "Storage panel resource ownership is missing or stale");
        return;
    }
    throw std::runtime_error("Storage provider resource missing");
}
struct Module {
    const char* const providerName = "Storage Plugin";
    HMODULE handle;
    GrimPluginAPI api{};
    std::shared_ptr<GRIM::UI::ProviderPanelLease> lease = std::make_shared<GRIM::UI::ProviderPanelLease>();
    GrimResult (*initialize)(const GrimPluginAPI*, GrimPluginHandle);
    void (*shutdown)();
    Module(const std::filesystem::path& path, const std::string& generation) {
        lease->owner = generation;
        api.api_version = GRIM_API_VERSION;
        api.log = [](GrimPluginHandle, GrimLogLevel, const char* message) { pluginError = message; };
        handle = LoadLibraryW(path.c_str());
        require(handle != nullptr, "Storage DLL failed to load against host exports");
        initialize = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*, GrimPluginHandle)>(GetProcAddress(handle, "grim_plugin_init"));
        shutdown = reinterpret_cast<void(*)()>(GetProcAddress(handle, "grim_plugin_shutdown"));
        if (!initialize || !shutdown) {
            FreeLibrary(handle);
            throw std::runtime_error("Storage DLL lifecycle exports missing");
        }
    }
    GrimResult init(bool permission = true) {
        const auto instance = prepareProviderPanelFixture(api, lease, permission);
        publish();
        return initialize(&api, instance);
    }
    void publish() {
        require(GRIM::MMO::ProviderRegistry::instance().publish(
            {GRIM::MMO::ToolProviderType::Plugin, providerName, lease->owner, {}}, {}),
            "Storage provider publication failed");
    }
    void close() {
        if (!handle) return;
        GRIM::MMO::ProviderRegistry::instance().retire(providerName, lease->owner);
        GRIM::UI::retireProviderPanels(lease); shutdown();
        require(lease->panels.empty(), "Storage shutdown retained native panels");

        require(FreeLibrary(handle) != 0, "Storage DLL unload failed");
        handle = nullptr;
    }
    ~Module() { close(); }
};
}

void runNativeStorageDllTests(const char* dllPath) {
    auto& root = UIRoot::get();
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-storage-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    try {
        for (const auto* name : {"first.dll", "second.dll", "failed.dll", "cancelled.dll", "denied.dll"})
            std::filesystem::copy_file(dllPath, directory / name);
        {
            Module first(directory / "first.dll", "storage-first");
            require(first.init() == GRIM_OK, "Initial panel load failed");
            root.processPendingTasks();
            auto* panel = root.getPanelByKey("storage");
            require(panel && !panel->isVisible() && panel->getTitle() == "Shared Storage",
                    "Storage identity/default visibility changed");
            const auto runtime = panel->getPanelID();
            checkResource(first.lease->owner, runtime);
            panel->setVisible(true); frame(*panel, {}, 2.1f); draw(*panel);
            require(renderedTestText.find("fixture.txt") != std::string::npos &&
                    renderedTestText.find("4 KB used") != std::string::npos &&
                    renderedTestText.find("upload.bin 50%") != std::string::npos,
                    "Storage listing, disk usage or transfer snapshot was lost");

            const auto position = panel->getPosition();
            click(*panel, {position.x + 135, position.y + 48});
            require(renderedTestText.find("Fixture laptop") != std::string::npos &&
                    renderedTestText.find("Online") != std::string::npos,
                    "Storage Devices tab lost snapshot data");
            const auto content = panel->getContentRect();
            const Vec2 row{content.origin.x + 4, content.origin.y + 39};
            click(*panel, {row.x + 240, row.y + 12});
            require(regenerated == 1 && renderedTestText.find("8765-4321") != std::string::npos,
                    "Regenerate did not reach the host service");
            edit(*panel, {row.x + 20, row.y + 46}, "PAIR-CODE");
            click(*panel, {row.x + 205, row.y + 46});
            require(linkedCode == "PAIR-CODE", "Device linking lost its host-service payload");
            for (const auto* address : {"hub:0", "hub:65536", "hub:12x", "hub:", ":11437"}) {
                edit(*panel, {row.x + 20, row.y + 80}, address);
                click(*panel, {row.x + 260, row.y + 80});
                require(connected == 0 && renderedTestText.find("Invalid hub address") != std::string::npos,
                        "Invalid hub address reached host networking or failed silently");
            }
            edit(*panel, {row.x + 20, row.y + 80}, "fixture-hub:12345");
            click(*panel, {row.x + 260, row.y + 80});
            require(connected == 1 && state.hubHost == "fixture-hub" && state.hubPort == 12345,
                    "Connect lost its host/port payload");
            const std::string label = "Connected to fixture-hub:12345";
            click(*panel, {row.x + static_cast<float>(label.size()) * 7.5f + 30, row.y + 80});
            require(disconnected == 1 && !state.connected, "Disconnect did not reach the host service");

            first.close();

            require(!root.getPanelByKey("storage"), "Storage retirement did not detach synchronously");
            {
                Module failed(directory / "failed.dll", "storage-failed");
                serviceAvailable = false; pluginError.clear();
                const auto result = failed.init();
                serviceAvailable = true;
                require(result != GRIM_OK && !pluginError.empty() && failed.lease->panels.empty(),
                        "Unavailable host service was accepted or leaked a panel");
                require(!root.getPanelByKey("storage"), "Failed init disturbed the live panel");
            }
            Module second(directory / "second.dll", "storage-second");
            require(second.init() == GRIM_OK, "Storage replacement init failed");
            root.processPendingTasks();
            auto* replacement = root.getPanelByKey("storage");
            require(replacement && !replacement->isVisible() && replacement->getPanelID() != runtime,
                    "Storage replacement identity/visibility changed");
            GRIM::UI::retireProviderPanels(first.lease);
            require(root.getPanelByKey("storage") == replacement, "Old retirement removed new Storage");
            checkResource(second.lease->owner, replacement->getPanelID());
        }
        require(!root.getPanelByKey("storage"), "Unloaded Storage panel remained attached");
        for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            require(group.provider.name != "Storage Plugin", "Unloaded Storage resources remained published");
        {
            Module cancelled(directory / "cancelled.dll", "storage-cancelled");
            require(cancelled.init() == GRIM_OK, "Cancelled Storage init failed");
            cancelled.close(); root.processPendingTasks();
            require(!root.getPanelByKey("storage"), "Queued activation reached unloaded Storage code");
        }
        {
            Module denied(directory / "denied.dll", "storage-denied");
            require(denied.init(false) != GRIM_OK, "Storage bypassed UI permission");
            require(denied.lease->panels.empty(), "Denied Storage initialization leaked a panel");
        }
    } catch (...) {
        serviceAvailable = true;
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
    std::cout << "Storage native DLL snapshots, actions, ownership and lifecycle tests passed\n";
}
