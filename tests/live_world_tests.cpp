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
        const ProviderDescriptor provider{ToolProviderType::Plugin, "Navigation Plugin", "first", ""};
        check(providers.publishResourceFixture(provider), "navigation fixture publication");
        ResourceDescriptor resource;
        resource.resource_id = "navigation-fixture"; resource.name = "Fixture";
        resource.kind = "ui.panel"; resource.runtime_id = 1;
        resource.state = {{"visible", true}};
        check(providers.publishResource(provider.instance, resource), "publish before opening Live");

        LiveWorldSession* live = nullptr;
        int deliveries = 0;
        auto feed = providers.subscribeProviderUpdates({provider}, [&](auto updates) {
            ++deliveries;
            live->applyUpdates(std::move(updates));
        });
        LiveWorldSession session(std::move(feed.initial), file);
        live = &session;
        const auto snapshot = loadJsonFile(file);
        const auto room = snapshot.at("rooms").at(1);
        check(snapshot.at("format") == "grim-live-navigation-snapshot" &&
              snapshot.at("hub_id") == "system_hub" && room.at("provider").at("name") == provider.name &&
              room.at("path") == "/Navigation" && room.at("parent_id") == "system_hub",
              "Startup navigation export mismatch");
        const auto topology = session.document().json();
        session.command("go /Navigation");
        check(session.observation().at("resources").size() == 1, "Resource published before Live was lost");

        const auto timestamp = std::filesystem::file_time_type::clock::now() - std::chrono::hours(1);
        std::filesystem::last_write_time(file, timestamp);
        const auto version = ToolRegistry::instance().version();
        resource.state["visible"] = false;
        check(providers.publishResource(provider.instance, resource), "publish resource state");
        resource.state["frame"] = 1;
        check(providers.publishResource(provider.instance, resource), "publish latest resource state");
        providers.dispatchProviderUpdates();
        providers.dispatchProviderUpdates();
        check(deliveries == 1 && session.providerInRoom(session.location())->resources.front() == resource &&
              ToolRegistry::instance().version() == version, "Resource deliveries did not coalesce to current state");

        auto replacement = provider;
        replacement.instance = "second";
        providers.retire(provider.name, provider.instance);
        check(providers.publishResourceFixture(replacement), "navigation replacement");
        check(providers.publishResource(replacement.instance, resource), "replacement resource fixture");
        providers.dispatchProviderUpdates();
        check(deliveries == 2 && session.observation().at("resources").empty() &&
              session.location() == room.at("room_id").get<std::string>() && session.document().json() == topology,
              "Retirement or a late generation changed startup topology or leaked contents");
        check(std::filesystem::last_write_time(file) == timestamp && loadJsonFile(file) == snapshot,
              "Live updates rewrote the startup navigation export");
        providers.retire(provider.name, replacement.instance);
        LiveWorldSession failing({}, directory);
        check(!failing.navigationPersistenceError().empty() && std::filesystem::is_directory(directory),
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
        LiveWorldSession* live = nullptr;
        auto feed = providers.subscribeProviderUpdates({provider}, [&](auto updates) {
            live->applyUpdates(std::move(updates));
        });
        LiveWorldSession session(std::move(feed.initial));
        live = &session;
        session.command("inspect CPU");
        check(session.history().back().feedback.starts_with("Action rejected") &&
              !session.observation().contains("resource_inspection"), "remote resource inspection bypassed room access");
        session.command("go " + session.providerPath(provider.instance));
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
        providers.dispatchProviderUpdates();
        check(session.history().back().action == "Resources changed" &&
              session.history().back().feedback == "Added: Hardware fixture / Fixture panel (panel:fixture)\n",
              "Resource addition repeated the complete room description");
        const auto refreshHistory = session.history().size();
        for (int frame = 1; frame <= 100; ++frame) {
            panel.state["position"]["x"] = frame;
            check(providers.publishResource(provider.instance, panel), "publish animated panel state");
            providers.dispatchProviderUpdates();
            check(session.history().size() == refreshHistory &&
                  session.observation().at("resource_inspection") == nlohmann::json(RegisteredResource{resource, provider}),
                  "Frame refresh duplicated history or invalidated unrelated inspection");
        }
        check(session.providerInRoom(session.location())->resources.back().state == panel.state,
              "Silent refresh failed to update live resource state");
        const auto stableRevision = providers.resourceVersion();
        for (int frame = 0; frame < 10; ++frame) {
            check(providers.publishResource(provider.instance, panel), "republish unchanged panel state");
            providers.dispatchProviderUpdates();
        }
        check(session.history().size() == refreshHistory && providers.resourceVersion() == stableRevision,
              "Unchanged publication created history or resource revision churn");
        providers.removeResource(provider.instance, panel); providers.dispatchProviderUpdates();
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
        providers.dispatchProviderUpdates();
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
        providers.dispatchProviderUpdates();
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
        providers.dispatchProviderUpdates();
        session.command("inspect CPU");
        check(session.history().back().feedback.starts_with("Action rejected: Ambiguous"), "Ambiguous object name was guessed");
        session.command("inspect-resource cpu");
        check(session.observation().at("resource_inspection").at("resource_id") == "cpu", "Explicit object ID did not resolve ambiguity");
        providers.retire(provider.name, provider.instance);
        providers.dispatchProviderUpdates();
        session.command("inspect cpu");
        check(session.history().back().feedback.starts_with("Action rejected") &&
              !session.observation().contains("resource_inspection"), "Unloaded object remained inspectable");
    }
    {
        const ProviderDescriptor ui{ToolProviderType::Plugin, "UI Plugin", "ui-1", ""};
        const ProviderDescriptor collision{ToolProviderType::Plugin, "UI-Plugin", "other-1", ""};
        ToolDescriptor tool;
        tool.tool_id = "test_tool";
        tool.description = "Original tool description";
        LiveWorldSession session({{ui, {tool}, {}}, {collision}});
        const auto path = session.providerPath(ui.instance);
        check(path != session.providerPath(collision.instance), "Normalized startup paths collided");
        const auto topology = session.document().json();
        session.command("go " + path);
        check(session.document().find(session.location())->name == "UI", "Provider room name not shortened");
        session.command("open tool_catalog");
        auto updated = tool;
        updated.description = "Updated tool description";
        auto other = tool;
        other.tool_id = "other_tool";
        session.applyUpdates({{collision.instance, std::vector{other}}, {"unknown-provider", std::vector{updated}}});
        check(session.observation().at("tools").at(0).at("description") == tool.description,
              "A tool update affected the wrong room");
        session.applyUpdates({{ui.instance, std::vector{updated}}});
        check(session.observation().at("tools").at(0).at("description") == updated.description,
              "Provider tool change did not reach the current room");
        session.applyUpdates({{ui.instance, std::vector<ToolDescriptor>{}}});
        const auto state = session.observation();
        check(state.at("tools").empty() && state.at("tool_catalog").at("tools").size() == 1 &&
              !state.contains("world_revision") && !state.contains("registry_revision") &&
              !state.contains("resource_revision") && session.document().json() == topology,
              "Tool withdrawal changed topology, affected another provider, or retained revision fields");
    }
    std::cout << "Live world navigation and resource tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
