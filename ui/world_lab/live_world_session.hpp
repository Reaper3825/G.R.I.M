#pragma once
#include "live_world.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "MMO/Core/ToolDescriptorJson.hpp"
#include <sstream>

namespace GRIM::WorldLab {
// A persistent manual live session, independent of compiled TextWorld episodes.
// Discovery/navigation only: no production handler is invoked by this backend.
class LiveWorldSession {
public:
    struct Entry { std::string action, feedback; nlohmann::json state; };
    explicit LiveWorldSession(MMO::ToolRegistry& registry = MMO::ToolRegistry::instance()) : registry_(registry) {
        world_.reconcile(registry_.snapshot());
        record("Start", describe());
    }
    const LiveWorld& world() const { return world_; }
    const std::vector<Entry>& history() const { return history_; }
    void synchronize() {
        if (registry_.version() == world_.registry().revision) return;
        const auto previousTools = observation().at("tools");
        const auto changes = world_.reconcile(registry_.snapshot());
        if (!changes.empty()) record("Providers changed", changes + describe());
        else if (previousTools != observation().at("tools")) record("Tools changed", describe());
    }
    void command(const std::string& command) {
        synchronize();
        try {
            if (command == "look" || command == "tools") record(command, describe());
            else if (command == "back" || command.starts_with("go ")) {
                world_.move(command == "back" ? "/" : command.substr(3));
                ++moves_; record(command, describe());
            } else if (command.starts_with("inspect ")) {
                const auto id = command.substr(8);
                for (const auto& tool : world_.toolsHere()) if (tool.tool_id == id) {
                    record(command, nlohmann::json(tool).dump(2)); return;
                }
                throw std::runtime_error("Tool is unavailable in the current provider room");
            } else throw std::runtime_error("Use look, tools, go /builtin, back, or inspect <tool_id>");
        } catch (const std::exception& error) { record(command, std::string("Action rejected: ") + error.what()); }
    }
    nlohmann::json observation() const {
        auto commands = nlohmann::json::array({"look", "tools"});
        if (world_.location() != "system_hub") commands.push_back("back");
        else for (const auto& p : world_.document().paths)
            if (!p.parent.empty()) commands.push_back("go " + world_.document().roomPath(p.id));
        auto tools = nlohmann::json::array();
        for (const auto& t : world_.toolsHere()) {
            tools.push_back(t); commands.push_back("inspect " + t.tool_id);
        }
        return {{"room_id", world_.location()}, {"location_path", world_.document().roomPath(world_.location())},
            {"session_id", world_.document().environmentId}, {"world_revision", world_.revision()},
            {"registry_revision", world_.registry().revision}, {"moves", moves_},
            {"facts", nlohmann::json::array()}, {"tools", tools}, {"admissible_commands", commands}};
    }
    std::string describe() const {
        std::ostringstream text;
        text << world_.document().find(world_.location())->name << "  " << world_.document().roomPath(world_.location()) << "\n";
        if (world_.location() == "system_hub") {
            text << "Registered providers appear automatically. Choose a room to enter.\n";
            for (const auto& p : world_.document().paths) if (!p.parent.empty())
                text << world_.document().roomPath(p.id) << " - " << world_.document().find(p.id)->name << '\n';
        } else {
            const auto tools = world_.toolsHere();
            text << "Available model tools: " << tools.size() << '\n';
            for (const auto& t : tools) text << t.tool_id << " - " << (t.short_description.empty() ? t.description : t.short_description) << '\n';
            text << "Use inspect <tool_id> for its contract. Use back to return to System Hub.\n";
        }
        return text.str();
    }
private:
    void record(const std::string& action, const std::string& feedback) {
        history_.push_back({action, feedback, observation()});
    }
    MMO::ToolRegistry& registry_;
    LiveWorld world_;
    uint64_t moves_ = 0;
    std::vector<Entry> history_;
};
}
