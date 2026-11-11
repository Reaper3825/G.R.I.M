#pragma once
#include "live_world.hpp"
#include "live_world_navigation_store.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "MMO/Core/Descriptors/ToolDescriptorJson.hpp"
#include "resource_inspection.hpp"
#include <map>
#include <sstream>

namespace GRIM::WorldLab {
// A persistent manual live session, independent of compiled TextWorld episodes.
// Discovery/navigation only: no production handler is invoked by this backend.
class LiveWorldSession {
public:
    struct Entry { std::string action, feedback; nlohmann::json state; };
    explicit LiveWorldSession(MMO::ToolRegistry& registry = MMO::ToolRegistry::instance(),
                              std::filesystem::path navigationPath = {})
        : registry_(registry), navigationStore_(std::move(navigationPath)) {
        world_.reconcile(registry_.snapshot());
        record("Start", describe());
        navigationStore_.synchronize(world_.navigation());
    }
    const LiveWorld& world() const { return world_; }
    const std::vector<Entry>& history() const { return history_; }
    const std::string& navigationPersistenceError() const { return navigationStore_.error(); }
    void synchronize() {
        if (registry_.version() == world_.registry().revision &&
            registry_.resourceVersion() == world_.registry().resource_revision) {
            navigationStore_.synchronize(world_.navigation());
            return;
        }
        const auto previousTools = catalogTools();
        const auto previousResources = world_.resourcesHere();
        const auto changes = world_.reconcile(registry_.snapshot());
        navigationStore_.synchronize(world_.navigation());
        // Frame-level state publication is observation refresh, not a user action.
        // Keep an inspection valid when only unrelated resources changed.
        if (!resourceInspection_.is_null()) {
            bool retained = false;
            if (const auto* group = world_.providerInRoom(world_.location())) {
                for (const auto& resource : group->resources)
                    if (nlohmann::json(MMO::RegisteredResource{resource, group->provider}) == resourceInspection_) {
                        retained = true;
                        break;
                    }
            }
            if (!retained) resourceInspection_ = nullptr;
        }
        if (!changes.empty()) record("Providers changed", changes + describe());
        else if (previousTools != catalogTools()) record("Tools changed", describe());
        else {
            const auto resourceChanges = describeResourceChanges(previousResources, world_.resourcesHere());
            if (!resourceChanges.empty()) record("Resources changed", resourceChanges);
        }
    }
    void command(const std::string& command) {
        synchronize();
        resourceInspection_ = nullptr;
        try {
            if (command == "look" || command == "tools" || command == "resources") record(command, describe());
            else if (command == "inventory") record(command, "You are carrying: Tool catalog (tool_catalog). Use open tool_catalog to read it.");
            else if (command == "open tool_catalog") {
                catalogOpen_ = true; record(command, describeCatalog());
            } else if (command == "close tool_catalog") {
                catalogOpen_ = false; record(command, "Tool catalog closed.");
            }
            else if (command == "back" || command.starts_with("go ")) {
                world_.move(command == "back" ? "/" : command.substr(3));
                ++moves_; record(command, describe());
            } else if (command.starts_with("inspect-resource ")) {
                const auto id = command.substr(17);
                for (const auto& r : world_.resourcesHere()) if (r.resource_id == id) {
                    inspectResource(command, r); return;
                }
                throw std::runtime_error("Resource is unavailable in the current provider room");
            } else if (command.starts_with("inspect ")) {
                const auto id = command.substr(8);
                const auto& resources = world_.resourcesHere();
                const auto tools = inspectableTools();
                const MMO::ResourceDescriptor* selected = nullptr;
                for (const auto& resource : resources)
                    if (folded(resource.resource_id) == folded(id) || folded(resource.name) == folded(id)) {
                        if (selected) throw std::runtime_error("Ambiguous object; use inspect-resource <resource_id>");
                        selected = &resource;
                    }
                const MMO::ToolDescriptor* selectedTool = nullptr;
                for (const auto& tool : tools) if (tool.tool_id == id) selectedTool = &tool;
                if (selected && selectedTool)
                    throw std::runtime_error("Object/tool name collision; use inspect-resource <resource_id> for the object");
                if (selected) { inspectResource(command, *selected); return; }
                if (selectedTool) {
                    record(command, nlohmann::json(*selectedTool).dump(2)); return;
                }
                throw std::runtime_error("Object or tool unavailable here; open tool_catalog to inspect tools across providers");
            } else throw std::runtime_error("Use look, inventory, open tool_catalog, close tool_catalog, tools, resources, go <path>, back, inspect <object name, resource_id, or tool_id>, or inspect-resource <resource_id>");
        } catch (const std::exception& error) { record(command, std::string("Action rejected: ") + error.what()); }
    }
    nlohmann::json observation() const {
        auto commands = nlohmann::json::array({"look", "tools", "resources", "inventory", "open tool_catalog"});
        if (catalogOpen_) commands.push_back("close tool_catalog");
        if (world_.location() != "system_hub") commands.push_back("back");
        else for (const auto& room : world_.navigation().rooms)
            if (!room.parentId.empty()) commands.push_back("go " + room.path);
        auto tools = nlohmann::json::array();
        for (const auto& t : world_.toolsHere()) {
            tools.push_back(t);
        }
        const auto inspectable = inspectableTools();
        for (const auto& t : inspectable) commands.push_back("inspect " + t.tool_id);
        const auto* group = world_.providerInRoom(world_.location());
        auto resources = nlohmann::json::array();
        for (const auto& r : world_.resourcesHere()) {
            resources.push_back(MMO::RegisteredResource{r, group->provider}); commands.push_back("inspect-resource " + r.resource_id);
            bool collision = false;
            for (const auto& tool : inspectable) collision |= tool.tool_id == r.resource_id;
            for (const auto& other : world_.resourcesHere())
                collision |= other.resource_id != r.resource_id &&
                    (folded(other.resource_id) == folded(r.resource_id) || folded(other.name) == folded(r.resource_id));
            if (!collision) commands.push_back("inspect " + r.resource_id);
        }
        auto state = nlohmann::json{{"room_id", world_.location()}, {"location_path", world_.document().roomPath(world_.location())},
            {"session_id", world_.document().environmentId}, {"world_revision", world_.revision()},
            {"registry_revision", world_.registry().revision}, {"moves", moves_},
            {"resource_revision", world_.registry().resource_revision}, {"resources", resources},
            {"facts", nlohmann::json::array()}, {"tools", tools}, {"admissible_commands", commands}};
        state["inventory"] = nlohmann::json::array({{
            {"entity_id", "tool_catalog"}, {"name", "Tool catalog"}, {"kind", "container"},
            {"state", catalogOpen_ ? "open" : "closed"}, {"open_provider", "tool_catalog"}}});
        if (catalogOpen_) state["tool_catalog"] = {{"entity_id", "tool_catalog"}, {"tools", catalogTools()}};
        if (!resourceInspection_.is_null()) state["resource_inspection"] = resourceInspection_;
        return state;
    }
    std::string describe() const {
        std::ostringstream text;
        text << world_.document().find(world_.location())->name << "  " << world_.document().roomPath(world_.location()) << "\n";
        text << "Inventory: Tool catalog. Use open tool_catalog to read it.\n";
        if (world_.location() == "system_hub") {
            text << "Registered providers appear automatically. Choose a room to enter.\n";
            for (const auto& room : world_.navigation().rooms) if (!room.parentId.empty())
                text << room.path << " - " << room.name << '\n';
        } else {
            const auto& resources = world_.resourcesHere();
            text << "Live resources: " << resources.size() << '\n';
            for (const auto& r : resources) {
                text << r.name << " (" << r.resource_id << ")";
                if (r.state.contains("visible")) text << (r.state.at("visible").get<bool>() ? " - visible" : " - hidden");
                text << '\n';
            }
            text << "Use inspect <object name or resource_id> for its readings; inspect-resource <resource_id> selects an object explicitly.\n";
            const auto tools = world_.toolsHere();
            text << "Available model tools: " << tools.size() << '\n';
            for (const auto& t : tools) text << t.tool_id << " - " << (t.short_description.empty() ? t.description : t.short_description) << '\n';
            text << "Use inspect <tool_id> for its contract. Use back to return to System Hub.\n";
        }
        if (catalogOpen_) text << describeCatalog();
        return text.str();
    }
private:
    std::string describeResourceChanges(const std::vector<MMO::ResourceDescriptor>& before,
                                        const std::vector<MMO::ResourceDescriptor>& after) const {
        const auto* group = world_.providerInRoom(world_.location());
        if (!group) return {};
        std::map<std::string, const MMO::ResourceDescriptor*> previous, current;
        for (const auto& resource : before) previous.emplace(resource.resource_id, &resource);
        for (const auto& resource : after) current.emplace(resource.resource_id, &resource);
        std::ostringstream text;
        const auto line = [&](const char* change, const MMO::ResourceDescriptor& resource) {
            text << change << ": " << providerRoomName(group->provider.name) << " / " << resource.name
                 << " (" << resource.resource_id << ")\n";
        };
        for (const auto& [id, resource] : current) {
            const auto old = previous.find(id);
            if (old == previous.end()) line("Added", *resource);
            else {
                const auto& p = *old->second;
                // Frame-level state changes refresh the projection without adding history.
                if (p.runtime_id != resource->runtime_id || p.name != resource->name ||
                    p.kind != resource->kind || p.parent_id != resource->parent_id)
                    line("Updated", *resource);
            }
        }
        for (const auto& [id, resource] : previous)
            if (!current.count(id)) line("Removed", *resource);
        return text.str();
    }
    static std::string folded(std::string value) {
        for (auto& c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        return value;
    }
    void inspectResource(const std::string& command, const MMO::ResourceDescriptor& resource) {
        const auto* group = world_.providerInRoom(world_.location());
        const MMO::RegisteredResource registered{resource, group->provider};
        resourceInspection_ = registered;
        record(command, resourceInspectionFeedback(registered));
    }
    std::vector<MMO::ToolDescriptor> inspectableTools() const {
        return catalogOpen_ ? catalogModelTools() : world_.toolsHere();
    }
    std::vector<MMO::ToolDescriptor> catalogModelTools() const {
        std::vector<MMO::ToolDescriptor> result;
        for (const auto& group : world_.registry().providers)
            for (const auto& tool : group.tools)
                if (tool.usage_type == MMO::ToolUsage::Model) result.push_back(tool);
        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.tool_id < b.tool_id; });
        return result;
    }
    nlohmann::json catalogTools() const {
        auto entries = nlohmann::json::array();
        for (const auto& tool : catalogModelTools())
            entries.push_back({{"tool_id", tool.tool_id},
                {"name", tool.display_name.empty() ? tool.tool_id : tool.display_name},
                {"short_description", tool.short_description.empty() ? tool.description : tool.short_description}});
        return entries;
    }
    std::string describeCatalog() const {
        std::ostringstream text;
        text << "Tool catalog:\n";
        const auto entries = catalogTools();
        for (const auto& t : entries)
            text << t.at("tool_id").get<std::string>() << " - " << t.at("short_description").get<std::string>() << '\n';
        if (entries.empty()) text << "No model tools are currently available.\n";
        text << "Use inspect <tool_id> for its contract.\n";
        return text.str();
    }
    void record(const std::string& action, const std::string& feedback) {
        history_.push_back({action, feedback, observation()});
    }
    MMO::ToolRegistry& registry_;
    LiveNavigationStore navigationStore_;
    LiveWorld world_;
    uint64_t moves_ = 0;
    bool catalogOpen_ = false;
    nlohmann::json resourceInspection_ = nullptr;
    std::vector<Entry> history_;
};
}
