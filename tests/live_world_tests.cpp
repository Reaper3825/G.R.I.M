#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "ui/world_lab/live_world_session.hpp"
#include <iostream>

using namespace GRIM::MMO;
using namespace GRIM::WorldLab;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main() try {
    {
        const auto directory = std::filesystem::temp_directory_path() / makeEnvironmentId();
        std::filesystem::create_directories(directory);
        struct Cleanup {
            std::filesystem::path directory;
            ~Cleanup() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
        } cleanup{directory};
        auto& providers = ProviderRegistry::instance();
        const auto file = directory / "live_navigation.json";
        LiveWorldSession session(providers, file);
        auto snapshot = loadJsonFile(file);
        check(snapshot.at("format") == "grim-live-navigation-snapshot" &&
              snapshot.at("rooms").size() == 1 && snapshot.at("hub_id") == "system_hub",
              "Initial navigation snapshot missing hub");
        const ProviderDescriptor provider{ToolProviderType::Plugin, "Navigation Plugin", "first", ""};
        check(providers.publishResourceFixture(provider), "navigation fixture publication");
        session.synchronize();
        snapshot = loadJsonFile(file);
        const auto room = snapshot.at("rooms").at(1);
        check(room.at("provider").at("name") == provider.name &&
              room.at("provider").at("type") == "plugin" && room.at("path") == "/Navigation" &&
              room.at("parent_id") == "system_hub", "Navigation identity/path projection mismatch");
        session.command("go /Navigation");
        check(loadJsonFile(file) == snapshot, "Movement changed topology snapshot");
        // Pin a known timestamp so the no-rewrite assertion does not depend on clock resolution.
        const auto timestamp = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
        std::filesystem::last_write_time(file, timestamp);
        session.synchronize();
        check(std::filesystem::last_write_time(file) == timestamp, "Unchanged snapshot rewritten");
        ResourceDescriptor resource;
        resource.resource_id = "navigation-fixture"; resource.name = "Fixture";
        resource.kind = "ui.panel"; resource.runtime_id = 1;
        resource.state = {{"visible", true}};
        const auto toolRevision = ToolRegistry::instance().version();
        check(providers.publishResource(provider.instance, resource), "navigation resource fixture");
        check(ToolRegistry::instance().version() == toolRevision, "Resource publication changed tool revision");
        session.synchronize();
        check(std::filesystem::last_write_time(file) == timestamp && loadJsonFile(file) == snapshot,
              "Resource update rewrote navigation snapshot");
        auto replacement = provider; replacement.instance = "second";
        providers.retire(provider.name, provider.instance);
        check(providers.publishResourceFixture(replacement), "navigation replacement");
        check(providers.snapshot().resourceCount() == 0, "Replacement kept retired resources");
        check(!providers.publishResource(provider.instance, resource), "Retired provider republished resource");
        check(providers.publishResource(replacement.instance, resource), "Replacement resource fixture");
        providers.removeResource(provider.instance, resource);
        auto staleBinding = resource;
        ++staleBinding.runtime_id;
        providers.removeResource(replacement.instance, staleBinding);
        const auto current = providers.snapshot();
        check(current.providers.size() == 1 && current.providers.front().provider == replacement &&
              current.providers.front().resources == std::vector<ResourceDescriptor>{resource},
              "Stale removal withdrew the current resource");
        session.synchronize();
        snapshot = loadJsonFile(file);
        check(snapshot.at("rooms").at(1).at("provider").at("instance") == "second" &&
              snapshot.at("rooms").at(1).at("room_id") == room.at("room_id") &&
              snapshot.at("session_id") == session.world().document().environmentId,
              "Replacement lost navigation identity");
        std::filesystem::remove(file);
        session.synchronize();
        check(loadJsonFile(file) == snapshot, "Deleted snapshot not restored");
        providers.retire(provider.name, replacement.instance);
        session.synchronize();
        check(loadJsonFile(file).at("rooms").size() == 1 && session.world().location() == "system_hub",
              "Provider unload retained exported room");
        LiveNavigationStore failing(directory); // A directory cannot be replaced by the JSON file.
        failing.synchronize(session.world().navigation());
        check(!failing.error().empty() && std::filesystem::is_directory(directory),
              "Persistence failure was not isolated/reported");
    }
    {
        auto& providers = ProviderRegistry::instance();
        const ProviderDescriptor provider{ToolProviderType::Plugin, "Hardware fixture", "hardware-1", ""};
        check(providers.publishResourceFixture(provider), "hardware inspection provider");
        ResourceDescriptor resource;
        resource.resource_id = "cpu"; resource.name = "CPU";
        resource.kind = "system_info.snapshot"; resource.runtime_id = 1;
        resource.state = {{"format", "grim-system-info-resource"}, {"version", 1}, {"component", "cpu"},
            {"data", {{"cpu_cores", 12}}}, {"capture_time_ns", 123000000000LL},
            {"stale", false}, {"error", nullptr}};
        check(providers.publishResource(provider.instance, resource), "publish CPU fixture");
        LiveWorldSession session(providers);
        session.command("inspect CPU");
        check(session.history().back().feedback.starts_with("Action rejected") &&
              !session.observation().contains("resource_inspection"), "remote resource inspection bypassed room access");
        session.command("go " + session.world().providerPath(providerKey(provider)));
        session.command("inspect CPU");
        check(session.history().back().feedback == "CPU\nLogical cores: 12\nStatus: Current",
              "CPU inspection did not return the agreed readable format");
        check(session.history().back().state.at("resource_inspection") == nlohmann::json(RegisteredResource{resource, provider}),
              "Inspection lost structured state, identity, or capture metadata");
        const auto historyState = session.history().back().state;
        // UI panels publish geometry/state every frame. These refreshes must not
        // append full room snapshots or displace unrelated inspection evidence.
        auto panel = resource;
        panel.resource_id = "panel:fixture"; panel.name = "Fixture panel"; panel.kind = "ui.panel";
        panel.state = {{"visible", true}, {"position", {{"x", 0}, {"y", 0}}}};
        check(providers.publishResource(provider.instance, panel), "publish refresh regression fixture");
        session.synchronize();
        check(session.history().back().action == "Resources changed" &&
              session.history().back().feedback == "Added: Hardware fixture / Fixture panel (panel:fixture)\n",
              "Resource addition repeated the complete room description");
        const auto refreshHistory = session.history().size();
        for (int frame = 1; frame <= 100; ++frame) {
            panel.state["position"]["x"] = frame;
            check(providers.publishResource(provider.instance, panel), "publish animated panel state");
            session.synchronize();
            check(session.history().size() == refreshHistory &&
                  session.observation().at("resource_inspection") == nlohmann::json(RegisteredResource{resource, provider}),
                  "Frame refresh duplicated history or invalidated unrelated inspection");
        }
        check(session.world().resourcesHere().back().state == panel.state,
              "Silent refresh failed to update live resource state");
        const auto stableRevision = providers.resourceVersion();
        for (int frame = 0; frame < 10; ++frame) {
            check(providers.publishResource(provider.instance, panel), "republish unchanged panel state");
            session.synchronize();
        }
        check(session.history().size() == refreshHistory && providers.resourceVersion() == stableRevision,
              "Unchanged publication created history or resource revision churn");
        providers.removeResource(provider.instance, panel); session.synchronize();
        check(session.history().size() == refreshHistory + 1 &&
              session.history().back().feedback == "Removed: Hardware fixture / Fixture panel (panel:fixture)\n",
              "Resource removal was missed or repeated the room snapshot");
        const auto commands = session.observation().at("admissible_commands").get<std::vector<std::string>>();
        check(std::find(commands.begin(), commands.end(), "inspect cpu") != commands.end(),
              "Resource inspect action missing");
        session.command("inspect-resource cpu");
        check(session.history().back().feedback == "CPU\nLogical cores: 12\nStatus: Current", "legacy inspection format differs");
        resource.state["stale"] = true;
        resource.state["error"] = {{"kind", "exception"}, {"message", "Fixture capture failed"}};
        check(providers.publishResource(provider.instance, resource), "publish stale CPU");
        session.synchronize();
        check(!session.observation().contains("resource_inspection"), "Refresh retained an outdated inspection");
        session.command("inspect cpu");
        check(session.history().back().feedback ==
              "CPU\nLogical cores: 12\nStatus: Stale - latest capture failed.\nCapture error: Fixture capture failed",
              "Stale inspection hid readings or capture failure");
        check(session.history().back().state.at("resource_inspection").at("state").at("capture_time_ns") == 123000000000LL &&
              historyState.at("resource_inspection").at("state").at("stale") == false,
              "Inspection changed capture provenance or historical state");
        resource.state["data"] = nullptr; resource.state["capture_time_ns"] = nullptr;
        check(providers.publishResource(provider.instance, resource), "publish unavailable CPU");
        session.command("inspect CPU");
        check(session.history().back().feedback.find("CPU information unavailable") != std::string::npos &&
              session.history().back().feedback.find("Logical cores") == std::string::npos,
              "Unavailable CPU inspection fabricated a core count");
        session.command("look");
        check(!session.observation().contains("resource_inspection"), "Unrelated action retained inspection result");
        resource.state["component"] = "memory"; resource.state["data"] = {{"ram_total_mb", 32768}};
        check(resourceInspectionFeedback({resource, provider}).find("Total RAM: 32768 MB") != std::string::npos, "Memory formatting lost units");
        resource.state["component"] = "system";
        resource.state["data"] = {{"os_name", "Windows"}, {"arch", "x86_64"}, {"wifi_connected", true},
            {"gpu_count", 1}, {"monitor_count", 1}, {"desktop", {{"origin_x", -1920}, {"origin_y", 0}, {"width", 3840}, {"height", 1080}}}};
        check(resourceInspectionFeedback({resource, provider}).find("Wi-Fi: Connected") != std::string::npos &&
              resourceInspectionFeedback({resource, provider}).find("at (-1920, 0)") != std::string::npos, "System formatting lost Wi-Fi or signed geometry");
        resource.state["component"] = "gpu";
        resource.state["data"] = {{"device_index", 0}, {"name", "Fixture GPU"}, {"vram_mb", 8192},
            {"driver_version", ""}, {"has_cuda", true}, {"has_metal", false}, {"has_rocm", false}};
        check(resourceInspectionFeedback({resource, provider}).find("Total VRAM: 8192 MB") != std::string::npos &&
              resourceInspectionFeedback({resource, provider}).find("Driver: Unavailable") != std::string::npos, "GPU formatting lost capacity or unknown driver");
        resource.state["component"] = "monitor";
        resource.state["data"] = {{"monitor_index", 0}, {"x", -1920}, {"y", -200}, {"width", 1920}, {"height", 1080}, {"is_primary", false}};
        check(resourceInspectionFeedback({resource, provider}).find("Position: (-1920, -200)") != std::string::npos &&
              resourceInspectionFeedback({resource, provider}).find("Primary: No") != std::string::npos, "Monitor formatting lost topology");
        // A display name cannot redirect a command to a different object.
        auto duplicate = resource; duplicate.resource_id = "other";
        check(providers.publishResource(provider.instance, duplicate), "publish ambiguous fixture");
        session.command("inspect CPU");
        check(session.history().back().feedback.starts_with("Action rejected: Ambiguous"), "Ambiguous object name was guessed");
        session.command("inspect-resource cpu");
        check(session.observation().at("resource_inspection").at("resource_id") == "cpu", "Explicit object ID did not resolve ambiguity");
        providers.retire(provider.name, provider.instance);
        session.command("inspect cpu");
        check(session.history().back().feedback.starts_with("Action rejected") &&
              !session.observation().contains("resource_inspection"), "Unloaded object remained inspectable");
    }
    {
        LiveWorld world;
        const ProviderDescriptor extra{ToolProviderType::Plugin, "Extra Plugin", "extra-1", ""};
        const ProviderDescriptor ui{ToolProviderType::Plugin, "UI Plugin", "ui-1", ""};
        const ProviderDescriptor collision{ToolProviderType::Plugin, "UI-Plugin", "other-1", ""};
        const ProviderSnapshot extraGroup{extra};
        const ProviderSnapshot uiGroup{.provider = ui};
        const ProviderSnapshot collisionGroup{.provider = collision};
        world.reconcile({0, 0, {extraGroup, uiGroup}});
        world.move("UI");
        check(world.location() == world.roomId(providerKey(ui)) && world.document().find(world.location())->name == "UI",
              "Short provider name did not select/display UI room");
        world.move("/");
        world.reconcile({1, 0, {extraGroup, uiGroup, collisionGroup}});
        check(world.providerPath(providerKey(extra)) == "/Extra" &&
              world.providerPath(providerKey(ui)) == "/UI", "Readable provider paths missing");
        const auto alternate = world.providerPath(providerKey(collision));
        check(alternate != "/UI", "Normalized provider names collided");
        world.move("UI Plugin");
        check(world.location() == world.roomId(providerKey(ui)), "Exact provider name navigation failed");
        world.move("/"); world.move(alternate);
        check(world.location() == world.roomId(providerKey(collision)), "Collision path selected wrong provider");
        world.reconcile({2, 0, {extraGroup, collisionGroup}});
        check(world.providerPath(providerKey(collision)) == alternate, "Unload reassigned another provider path");
        world.reconcile({3, 0, {collisionGroup, uiGroup, extraGroup}});
        check(world.providerPath(providerKey(ui)) == "/UI", "Reload lost provider path");
    }
    std::cout << "Live world navigation and resource tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
