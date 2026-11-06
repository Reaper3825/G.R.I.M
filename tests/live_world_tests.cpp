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
    const auto room = session.world().roomId(providerKey(alpha));
    check(session.world().providerPath(providerKey(alpha)) == "/Alpha", "Provider path does not use its name");
    session.command("go /Alpha");
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
