#include "plugins/schedule_plugin/calendar_panel.hpp"
#include "tests/provider_panel_fixture.hpp"
#include "ui/ui_root.hpp"
#include "ui/input/ui_input_router.hpp"
#include "MMO/Core/Registries/ProviderRegistry.hpp"
#include <filesystem>

namespace {
using namespace GRIM;
void require(bool value, const char* error) { if (!value) throw std::runtime_error(error); }
struct TestCalendar : grim::scheduling::CalendarPanel {
    using CalendarPanel::CalendarPanel;
    UIButton& button(const std::string& text) {
        for (auto& child : children)
            if (auto* button = dynamic_cast<UIButton*>(child.get()); button && button->getText() == text) return *button;
        throw std::runtime_error("Missing native calendar button: " + text);
    }
    UIInputBox& input(size_t index) {
        for (auto& child : children)
            if (auto* input = dynamic_cast<UIInputBox*>(child.get()); input && index-- == 0) return *input;
        throw std::runtime_error("Missing calendar input");
    }
    UIScrollBox& scroll() {
        for (auto& child : children) if (auto* scroll = dynamic_cast<UIScrollBox*>(child.get())) return *scroll;
        throw std::runtime_error("Missing scrollbox");
    }
};
struct Fixture {
    GrimPluginAPI api{};
    std::shared_ptr<UI::ProviderPanelLease> lease = std::make_shared<UI::ProviderPanelLease>();
    std::unique_ptr<grim::scheduling::Provider> provider;
    std::unique_ptr<TestCalendar> calendar;
    explicit Fixture(const std::filesystem::path& path, const std::string& generation) {
        lease->owner = generation; lease->providerName = "Native Calendar Fixture";
        api.api_version = GRIM_API_VERSION;
        api.replace_resources = [](const GrimPluginAPI*, const char*, const char*, size_t) { return GRIM_OK; };
        api.log = [](const GrimPluginAPI*, GrimLogLevel, const char*) {};
        provider = std::make_unique<grim::scheduling::Provider>(&api, path);
    }
    void initialize(bool permitted = true) {
        prepareProviderPanelFixture(api, lease, permitted);
        calendar = std::make_unique<TestCalendar>(lease->owner, &api, *provider);
    }
    void publish(const std::string& replacing = {}) {
        require(MMO::ProviderRegistry::instance().publish(
            {MMO::ToolProviderType::Plugin, lease->providerName, lease->owner, {}}, {}, replacing), "Owner publication failed");
    }
    ~Fixture() {
        UI::retireProviderPanels(lease);
        calendar.reset(); provider.reset();
        MMO::ProviderRegistry::instance().retire(lease->providerName, lease->owner);
    }
};
void frame(TestCalendar& panel, InputState input = {}) {
    auto& router = UI::InputRouter::get();
    router.beginFrame(); router.currentPanel = panel.getPanelID(); panel.update(input, 0.016f); router.endFrame();
}
void click(TestCalendar& panel, Widget& widget) {
    InputState input;
    const auto position = widget.getPosition(), size = widget.getSize();
    input.mousePos = {position.x + size.x / 2, position.y + size.y / 2};
    input.mousePressed[0] = input.mouseDown[0] = true; frame(panel, input);
    input.mousePressed[0] = input.mouseDown[0] = false; input.mouseReleased[0] = true; frame(panel, input);
}
}
void runCalendarPanelTests() {
    auto& root = UIRoot::get();
    UIPanel::setCanvasSize({1920,1080});
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-native-calendar-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        bool rejected = false;
        try { UIPanel unknown("Unknown", true, "test.unknown", "unknown-provider"); }
        catch (const std::logic_error&) { rejected = true; }
        require(rejected && !root.getPanelByKey("test.unknown"), "Unknown provider attached a panel");
    }
    {
        Fixture denied(directory / "denied.json", "denied");
        bool rejected = false;
        try { denied.initialize(false); } catch (const std::logic_error&) { rejected = true; }
        require(rejected && !UI::validateProviderPanels(denied.lease), "Native construction ignored UI permission");
        require(!root.getPanelByKey("calendar"), "Rejected constructor attached Calendar");
    }
    {
        Fixture first(directory / "events.json", "first"); first.initialize();
        require(!std::filesystem::exists(directory), "Initialization touched calendar storage");
        require(!root.getPanelByKey("calendar"), "Staged Calendar attached before commit");
        require(UI::validateProviderPanels(first.lease), "Valid Calendar rejected");
        first.publish(); UI::activateProviderPanels(first.lease); root.processPendingTasks();
        require(root.getPanelByKey("calendar") == first.calendar.get(), "Host did not attach the actual plugin panel");
        require(!first.calendar->isVisible(), "Calendar should start hidden");
        root.requestPanelOperation("calendar", UIPanelOperation::Show).get();
        auto& panel = *first.calendar; frame(panel);
        panel.input(0).setText("Native event");
        click(panel, panel.button("Add event"));
        auto result = first.provider->execute("list", {{"kind","event"}});
        require(result.at("total") == 1, "Native button did not save event");
        require(panel.input(0).getText().empty(), "Successful add did not clear title");
        panel.input(0).setText("Unfinished draft");
        click(panel, panel.button("Month >"));
        require(panel.input(0).getText() == "Unfinished draft", "Navigation erased draft");
        click(panel, panel.button("Today"));
        click(panel, panel.button("Delete event"));
        require(first.provider->execute("list", {{"kind","event"}}).at("total") == 0, "Native delete failed");
        panel.input(1).setText("invalid"); click(panel, panel.button("Add event"));
        require(panel.input(0).getText() == "Unfinished draft", "Failed add erased draft");
        panel.setSize(420, 300); frame(panel);
        require(panel.scroll().getContentHeight() > panel.getContentRect().size.y, "Narrow layout does not scroll");
        panel.scroll().scrollTo(500); frame(panel);
        require(panel.scroll().getScrollOffset() > 0, "Native scrolling failed");
        panel.setSize(1160,600); panel.scroll().scrollTo(0); frame(panel);
        {
            Fixture collision(directory / "collision.json", "collision"); collision.initialize();
            require(!UI::validateProviderPanels(collision.lease), "Foreign panel collision accepted");
        }
        require(root.getPanelByKey("calendar") == &panel, "Failed candidate displaced live Calendar");
        Fixture replacement(directory / "events.json", "replacement"); replacement.initialize();
        require(UI::validateProviderPanels(replacement.lease, first.lease), "Replacement panel rejected");
        replacement.publish(first.lease->owner);
        UI::retireProviderPanels(first.lease);
        require(!root.getPanelByKey("calendar"), "Retirement did not detach synchronously");
        first.calendar.reset();
        UI::activateProviderPanels(replacement.lease); root.processPendingTasks();
        UI::retireProviderPanels(first.lease);
        require(root.getPanelByKey("calendar") == replacement.calendar.get(), "Old retirement removed replacement");
    }
    require(!root.getPanelByKey("calendar"), "Provider destruction retained native panel");
    {
        Fixture cancelled(directory / "cancelled.json", "cancelled"); cancelled.initialize(); cancelled.publish();
        UI::activateProviderPanels(cancelled.lease);
        UI::retireProviderPanels(cancelled.lease); cancelled.calendar.reset();
        root.processPendingTasks();
        require(!root.getPanelByKey("calendar"), "Queued activation used a destroyed panel");
    }
    std::filesystem::remove_all(directory);
}

#ifdef _WIN32
void runNativeCalendarDllTests(const char* dllPath) {
    auto& root = UIRoot::get();
    const auto directory = std::filesystem::temp_directory_path() /
        ("grim-calendar-dll-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    struct Module {
        HMODULE handle = nullptr;
        GrimPluginAPI api{};
        std::shared_ptr<UI::ProviderPanelLease> lease = std::make_shared<UI::ProviderPanelLease>();
        void (*shutdown)() = nullptr;
        Module(const std::filesystem::path& path, const std::string& generation) {
            lease->providerName = "Schedule DLL Fixture"; lease->owner = generation;
            api.api_version = GRIM_API_VERSION;
            api.register_tools = [](const GrimPluginAPI*, const GrimToolRegistration*, size_t count) { return count == 6 ? GRIM_OK : GRIM_ERROR_INVALID_PARAM; };
            api.create_timer = [](const GrimPluginAPI*, uint64_t, bool, GrimTimerCallback, void*) { return reinterpret_cast<GrimTimerHandle>(1); };
            api.cancel_timer = [](const GrimPluginAPI*, GrimTimerHandle) { return GRIM_OK; };
            api.replace_resources = [](const GrimPluginAPI*, const char*, const char*, size_t) { return GRIM_OK; };
            api.show_notification = [](const GrimPluginAPI*, const char*, const char*) {};
            api.log = [](const GrimPluginAPI*, GrimLogLevel, const char*) {};
            handle = LoadLibraryW(path.c_str());
            require(handle != nullptr, "Could not load native Schedule DLL against test host exports");
            shutdown = reinterpret_cast<void(*)()>(GetProcAddress(handle, "grim_plugin_shutdown"));
            require(shutdown != nullptr, "Missing native shutdown export");
        }
        GrimResult init() {
            const auto callback = reinterpret_cast<GrimResult (*)(const GrimPluginAPI*)>(GetProcAddress(handle, "grim_plugin_init"));
            require(callback != nullptr, "Missing native init export");
            prepareProviderPanelFixture(api, lease, true);
            return callback(&api);
        }
        void publish(const std::string& replacing = {}) {
            require(MMO::ProviderRegistry::instance().publish(
                {MMO::ToolProviderType::Plugin, lease->providerName, lease->owner, {}}, {}, replacing), "DLL provider publish failed");
            UI::activateProviderPanels(lease);
        }
        void close() {
            if (!handle) return;
            UI::retireProviderPanels(lease); shutdown();
            require(lease->nativePanels.empty(), "DLL shutdown did not destroy its panels");
            MMO::ProviderRegistry::instance().retire(lease->providerName, lease->owner);
            FreeLibrary(handle); handle = nullptr;
        }
        ~Module() { close(); }
    };
    try {
        for (const auto* name : {"first.dll", "second.dll", "cancelled.dll", "failed.dll"})
            std::filesystem::copy_file(dllPath, directory / name);
        {
            Module first(directory / "first.dll", "dll-first"); require(first.init() == GRIM_OK, "Native DLL init failed");
            require(!root.getPanelByKey("calendar"), "DLL init attached before commit");
            first.publish(); root.processPendingTasks();
            auto* panel = root.getPanelByKey("calendar"); require(panel != nullptr, "Native DLL Calendar missing");
            const auto runtime = panel->getPanelID();
            panel->setVisible(true);
            OverlayRenderer renderer; require(panel->drawOverlay(renderer), "Native DLL draw dispatch failed");
            Module second(directory / "second.dll", "dll-second"); require(second.init() == GRIM_OK, "Replacement DLL init failed");
            require(UI::validateProviderPanels(second.lease, first.lease), "Native DLL replacement rejected");
            first.close(); second.publish(); root.processPendingTasks();
            require(root.getPanelByKey("calendar") && root.getPanelByKey("calendar")->getPanelID() != runtime, "Replacement did not attach fresh native panel");
        }
        require(!root.getPanelByKey("calendar"), "Unloaded DLL left live panel");
        {
            Module cancelled(directory / "cancelled.dll", "dll-cancelled"); require(cancelled.init() == GRIM_OK, "Cancelled DLL init failed");
            cancelled.publish(); cancelled.close(); root.processPendingTasks();
            require(!root.getPanelByKey("calendar"), "Queued task reached unloaded DLL");
        }
        {
            Module failed(directory / "failed.dll", "dll-failed");
            failed.api.create_timer = [](const GrimPluginAPI*, uint64_t, bool, GrimTimerCallback, void*) -> GrimTimerHandle { return nullptr; };
            require(failed.init() != GRIM_OK, "Failed provider initialized successfully");
            failed.close(); require(!root.getPanelByKey("calendar"), "Failed DLL leaked Calendar");
        }
    } catch (...) { std::filesystem::remove_all(directory); throw; }
    std::filesystem::remove_all(directory);
}
#endif
