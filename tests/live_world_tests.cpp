#include "ui/world_lab/live_world_session.hpp"
#include "commands/command_result.hpp"
#include <atomic>
#include <thread>
#include <iostream>

using namespace GRIM::MMO;
using namespace GRIM::WorldLab;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
ToolRegistration tool(const ProviderDescriptor& p, const std::string& id, ToolUsage usage = ToolUsage::Model) {
    ToolDescriptor d;
    d.tool_id = d.display_name = id; d.provider_type = p.type; d.provider_name = p.name;
    d.provider_instance = p.instance; d.provider_path = p.path; d.usage_type = usage;
    d.description = "Fixture tool";
    return {d, [](const std::string&) -> CommandResult { throw std::runtime_error("Discovery must never execute tools"); }};
}
int main() try {
    {
        const auto directory = std::filesystem::temp_directory_path() / makeEnvironmentId();
        std::filesystem::create_directories(directory);
        struct Cleanup {
            std::filesystem::path directory;
            ~Cleanup() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
        } cleanup{directory};
        auto& registry = ToolRegistry::instance();
        const auto file = directory / "live_navigation.json";
        LiveWorldSession session(registry, file);
        auto snapshot = loadJsonFile(file);
        check(snapshot.at("format") == "grim-live-navigation-snapshot" &&
              snapshot.at("rooms").size() == 1 && snapshot.at("hub_id") == "system_hub",
              "Initial navigation snapshot missing hub");
        const ProviderDescriptor provider{ToolProviderType::Plugin, "Navigation Plugin", "first", ""};
        check(registry.publishProvider(provider, {}), "navigation fixture publication");
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
        resource.provider_name = provider.name; resource.provider_instance = provider.instance;
        resource.resource_id = "navigation-fixture"; resource.name = "Fixture";
        resource.kind = "ui.panel"; resource.runtime_id = 1;
        resource.state = {{"visible", true}};
        check(registry.publishResource(resource), "navigation resource fixture");
        session.synchronize();
        check(std::filesystem::last_write_time(file) == timestamp && loadJsonFile(file) == snapshot,
              "Resource update rewrote navigation snapshot");
        auto replacement = provider; replacement.instance = "second";
        check(registry.publishProvider(replacement, {}, provider.instance), "navigation replacement");
        session.synchronize();
        snapshot = loadJsonFile(file);
        check(snapshot.at("rooms").at(1).at("provider").at("instance") == "second" &&
              snapshot.at("rooms").at(1).at("room_id") == room.at("room_id") &&
              snapshot.at("session_id") == session.world().document().environmentId,
              "Replacement lost navigation identity");
        std::filesystem::remove(file);
        session.synchronize();
        check(loadJsonFile(file) == snapshot, "Deleted snapshot not restored");
        registry.unregisterInstance(provider.name, replacement.instance);
        session.synchronize();
        check(loadJsonFile(file).at("rooms").size() == 1 && session.world().location() == "system_hub",
              "Provider unload retained exported room");
        LiveNavigationStore failing(directory); // A directory cannot be replaced by the JSON file.
        failing.synchronize(session.world().navigation());
        check(!failing.error().empty() && std::filesystem::is_directory(directory),
              "Persistence failure was not isolated/reported");
    }
    {
        auto& registry = ToolRegistry::instance();
        const ProviderDescriptor provider{ToolProviderType::Plugin, "Hardware fixture", "hardware-1", ""};
        check(registry.publishProvider(provider, {tool(provider, "cpu.tool")}), "hardware inspection provider");
        ResourceDescriptor resource;
        resource.provider_name = provider.name; resource.provider_instance = provider.instance;
        resource.resource_id = "cpu"; resource.name = "CPU";
        resource.kind = "system_info.snapshot"; resource.runtime_id = 1;
        resource.state = {{"format", "grim-system-info-resource"}, {"version", 1}, {"component", "cpu"},
            {"data", {{"cpu_cores", 12}}}, {"capture_time_ns", 123000000000LL},
            {"stale", false}, {"error", nullptr}};
        check(registry.publishResource(resource), "publish CPU fixture");
        LiveWorldSession session(registry);
        session.command("inspect CPU");
        check(session.history().back().feedback.starts_with("Action rejected") &&
              !session.observation().contains("resource_inspection"), "remote resource inspection bypassed room access");
        session.command("go " + session.world().providerPath(providerKey(provider)));
        session.command("inspect CPU");
        check(session.history().back().feedback == "CPU\nLogical cores: 12\nStatus: Current",
              "CPU inspection did not return the agreed readable format");
        check(session.history().back().state.at("resource_inspection") == nlohmann::json(resource),
              "Inspection lost structured state, identity, or capture metadata");
        const auto historyState = session.history().back().state;
        // UI panels publish geometry/state every frame. These refreshes must not
        // append full room snapshots or displace unrelated inspection evidence.
        auto panel = resource;
        panel.resource_id = "panel:fixture"; panel.name = "Fixture panel"; panel.kind = "ui.panel";
        panel.state = {{"visible", true}, {"position", {{"x", 0}, {"y", 0}}}};
        check(registry.publishResource(panel), "publish refresh regression fixture");
        session.synchronize();
        check(session.history().back().action == "Resources changed" &&
              session.history().back().feedback == "Added: Hardware fixture / Fixture panel (panel:fixture)\n",
              "Resource addition repeated the complete room description");
        const auto refreshHistory = session.history().size();
        for (int frame = 1; frame <= 100; ++frame) {
            panel.state["position"]["x"] = frame;
            check(registry.publishResource(panel), "publish animated panel state");
            session.synchronize();
            check(session.history().size() == refreshHistory &&
                  session.observation().at("resource_inspection") == nlohmann::json(resource),
                  "Frame refresh duplicated history or invalidated unrelated inspection");
        }
        check(session.world().registry().resources.back().state == panel.state,
              "Silent refresh failed to update live resource state");
        const auto stableRevision = registry.resourceVersion();
        for (int frame = 0; frame < 10; ++frame) {
            check(registry.publishResource(panel), "republish unchanged panel state");
            session.synchronize();
        }
        check(session.history().size() == refreshHistory && registry.resourceVersion() == stableRevision,
              "Unchanged publication created history or resource revision churn");
        registry.removeResource(panel); session.synchronize();
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
        check(registry.publishResource(resource), "publish stale CPU");
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
        check(registry.publishResource(resource), "publish unavailable CPU");
        session.command("inspect CPU");
        check(session.history().back().feedback.find("CPU information unavailable") != std::string::npos &&
              session.history().back().feedback.find("Logical cores") == std::string::npos,
              "Unavailable CPU inspection fabricated a core count");
        session.command("look");
        check(!session.observation().contains("resource_inspection"), "Unrelated action retained inspection result");
        resource.state["component"] = "memory"; resource.state["data"] = {{"ram_total_mb", 32768}};
        check(resourceInspectionFeedback(resource).find("Total RAM: 32768 MB") != std::string::npos, "Memory formatting lost units");
        resource.state["component"] = "system";
        resource.state["data"] = {{"os_name", "Windows"}, {"arch", "x86_64"}, {"wifi_connected", true},
            {"gpu_count", 1}, {"monitor_count", 1}, {"desktop", {{"origin_x", -1920}, {"origin_y", 0}, {"width", 3840}, {"height", 1080}}}};
        check(resourceInspectionFeedback(resource).find("Wi-Fi: Connected") != std::string::npos &&
              resourceInspectionFeedback(resource).find("at (-1920, 0)") != std::string::npos, "System formatting lost Wi-Fi or signed geometry");
        resource.state["component"] = "gpu";
        resource.state["data"] = {{"device_index", 0}, {"name", "Fixture GPU"}, {"vram_mb", 8192},
            {"driver_version", ""}, {"has_cuda", true}, {"has_metal", false}, {"has_rocm", false}};
        check(resourceInspectionFeedback(resource).find("Total VRAM: 8192 MB") != std::string::npos &&
              resourceInspectionFeedback(resource).find("Driver: Unavailable") != std::string::npos, "GPU formatting lost capacity or unknown driver");
        resource.state["component"] = "monitor";
        resource.state["data"] = {{"monitor_index", 0}, {"x", -1920}, {"y", -200}, {"width", 1920}, {"height", 1080}, {"is_primary", false}};
        check(resourceInspectionFeedback(resource).find("Position: (-1920, -200)") != std::string::npos &&
              resourceInspectionFeedback(resource).find("Primary: No") != std::string::npos, "Monitor formatting lost topology");
        // A display name cannot redirect a command to a different object.
        auto duplicate = resource; duplicate.resource_id = "other";
        check(registry.publishResource(duplicate), "publish ambiguous fixture");
        session.command("inspect CPU");
        check(session.history().back().feedback.starts_with("Action rejected: Ambiguous"), "Ambiguous object name was guessed");
        session.command("inspect-resource cpu");
        check(session.observation().at("resource_inspection").at("resource_id") == "cpu", "Explicit object ID did not resolve ambiguity");
        session.command("inspect cpu.tool");
        check(session.history().back().feedback.find("cpu.tool") != std::string::npos, "Resource route broke tool inspection");
        registry.unregisterInstance(provider.name, provider.instance);
        session.command("inspect cpu");
        check(session.history().back().feedback.starts_with("Action rejected") &&
              !session.observation().contains("resource_inspection"), "Unloaded object remained inspectable");
    }
    {
        LiveWorld world;
        const ProviderDescriptor builtin{ToolProviderType::Builtin, "builtin", "", ""};
        const ProviderDescriptor ui{ToolProviderType::Plugin, "UI Plugin", "ui-1", ""};
        const ProviderDescriptor collision{ToolProviderType::Plugin, "UI-Plugin", "other-1", ""};
        world.reconcile({0, {builtin, ui}, {}});
        world.move("UI");
        check(world.location() == world.roomId(providerKey(ui)) && world.document().find(world.location())->name == "UI",
              "Short provider name did not select/display UI room");
        world.move("/");
        world.reconcile({1, {builtin, ui, collision}, {}});
        check(world.providerPath(providerKey(builtin)) == "/builtin" &&
              world.providerPath(providerKey(ui)) == "/UI", "Readable provider paths missing");
        const auto alternate = world.providerPath(providerKey(collision));
        check(alternate != "/UI", "Normalized provider names collided");
        world.move("UI Plugin");
        check(world.location() == world.roomId(providerKey(ui)), "Exact provider name navigation failed");
        world.move("/"); world.move(alternate);
        check(world.location() == world.roomId(providerKey(collision)), "Collision path selected wrong provider");
        world.reconcile({2, {builtin, collision}, {}});
        check(world.providerPath(providerKey(collision)) == alternate, "Unload reassigned another provider path");
        world.reconcile({3, {collision, ui, builtin}, {}});
        check(world.providerPath(providerKey(ui)) == "/UI", "Reload lost provider path");
    }
    auto& registry = ToolRegistry::instance();
    const ProviderDescriptor alpha{ToolProviderType::Plugin, "Alpha", "alpha-1", "alpha.dll"};
    const ProviderDescriptor beta{ToolProviderType::Plugin, "Beta", "beta-1", "beta.dll"};
    LiveWorldSession session;
    check(session.world().document().entities.size() == 1, "Empty registry must leave only System Hub");
    check(registry.publishProvider(alpha, {tool(alpha, "alpha.read"), tool(alpha, "alpha.debug", ToolUsage::Debug)}), "publish alpha");
    check(registry.publishProvider(beta, {}), "zero-tool provider publication");
    session.synchronize();
    LiveWorldSession preloaded;
    check(preloaded.world().document().entities.size() == 3, "Existing providers did not populate at startup");
    const auto sessionId = session.observation().at("session_id");
    check(session.world().document().entities.size() == 3 && session.world().location() == "system_hub", "providers did not automatically populate");
    check(session.world().registry().providers.size() == 2, "zero-tool provider not discoverable");
    check(session.observation().at("inventory").at(0).at("entity_id") == "tool_catalog", "catalog missing from inventory");
    check(!session.observation().contains("tool_catalog"), "closed catalog exposes contents");
    session.command("inventory");
    check(session.history().back().feedback.find("Tool catalog") != std::string::npos, "inventory command missing catalog");
    session.command("open tool_catalog");
    check(session.observation().at("tool_catalog").at("tools").size() == 1, "catalog model filtering");
    session.command("inspect alpha.read");
    check(!session.history().back().feedback.starts_with("Action rejected"), "inventory catalog inspection from hub failed");
    const auto room = session.world().roomId(providerKey(alpha));
    check(session.world().providerPath(providerKey(alpha)) == "/Alpha", "Provider path does not use its name");
    session.command("go /Alpha");
    check(session.observation().at("inventory").at(0).at("state") == "open", "navigation lost inventory state");
    check(session.world().location() == room && session.world().toolsHere().size() == 1, "navigation/model filtering");
    session.command("inspect alpha.read");
    check(session.history().back().feedback.find("alpha.read") != std::string::npos, "tool inspection");
    session.command("go /Beta");
    check(session.world().location() == room && session.history().back().feedback.starts_with("Action rejected"), "nonadjacent move accepted");
    auto replacement = alpha; replacement.instance = "alpha-2";
    const auto revision = registry.version();
    auto collision = tool(replacement, "alpha.new"); collision.descriptor.aliases = {"beta.tool"};
    check(registry.registerBatch(beta.name, {tool(beta, "beta.tool")}), "late tool publication");
    const auto beforeRejected = registry.snapshot();
    check(!registry.publishProvider(replacement, {collision}, alpha.instance), "collision accepted");
    check(registry.version() == beforeRejected.revision && registry.getTool("alpha.read")->provider_instance == alpha.instance, "failed replacement changed registry");
    check(registry.publishProvider(replacement, {tool(replacement, "alpha.new")}, alpha.instance), "replacement rejected");
    session.synchronize();
    check(session.world().location() == room && session.world().roomId(providerKey(alpha)) == room, "replacement changed room/location");
    check(session.world().toolsHere().at(0).tool_id == "alpha.new", "replacement catalog stale");
    check(session.observation().at("tool_catalog").at("tools").dump().find("alpha.read") == std::string::npos, "inventory retained replaced tool");
    check(registry.invokeModel("alpha.new", "", revision).errorCode == "ERR_TOOL_STALE", "stale revision accepted");
    registry.unregisterInstance(alpha.name, alpha.instance);
    check(registry.getTool("alpha.new").has_value(), "old-generation cleanup removed replacement");
    session.command("inspect alpha.read");
    check(session.history().back().feedback.starts_with("Action rejected"), "stale tool inspect accepted");
    const auto historySize = session.history().size();
    const auto moves = session.observation().at("moves");
    registry.unregisterInstance(alpha.name, replacement.instance); session.synchronize();
    check(session.world().location() == "system_hub" && !session.world().document().find(room), "occupied unload did not return to hub");
    check(session.history().size() > historySize && session.observation().at("session_id") == sessionId && session.observation().at("moves") == moves, "unload reset session/history");
    session.command("go /Alpha");
    check(session.world().location() == "system_hub", "removed room remained navigable");
    check(registry.publishProvider(replacement, {}), "reload zero tools"); session.synchronize();
    check(session.world().connected(providerKey(alpha)), "Reloaded provider did not populate automatically");
    check(session.world().roomId(providerKey(alpha)) == room, "reconnect lost stable identity");
    registry.unregisterInstance(alpha.name, replacement.instance);
    registry.unregisterByProvider(beta.name);
    session.synchronize();
    check(session.world().document().entities.size() == 1, "Unloading all providers must retain System Hub");
    check(registry.snapshot().providers.empty(), "provider cleanup retained zero-tool record");
    check(session.observation().at("tool_catalog").at("tools").empty(), "catalog retained unloaded tools");
    check(session.observation().at("inventory").size() == 1, "provider unload removed carried catalog");
    session.command("close tool_catalog");
    check(!session.observation().contains("tool_catalog"), "closing catalog retained contents");

    // Reader must never see a provider generation paired with another generation's tools.
    auto p = alpha; p.instance = "0";
    check(registry.publishProvider(p, {tool(p, "atomic.tool")}), "atomic fixture");
    std::atomic<bool> finished = false, inconsistent = false;
    std::thread reader([&] {
        while (!finished) {
            const auto s = registry.snapshot();
            if (s.providers.size() != 1 || s.tools.size() != 1 || s.providers[0].instance != s.tools[0].provider_instance) inconsistent = true;
        }
    });
    bool published = true;
    for (int i = 1; i <= 100; ++i) {
        const auto old = p.instance; p.instance = std::to_string(i);
        published &= registry.publishProvider(p, {tool(p, "atomic.tool")}, old);
    }
    finished = true; reader.join();
    check(published && !inconsistent, "mixed provider/tool snapshot");
    registry.unregisterInstance(p.name, p.instance);
    std::cout << "Live world lifecycle, navigation, history and atomic snapshot tests passed\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
