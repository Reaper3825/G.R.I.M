#pragma once
#include "live_world_navigation.hpp"
#include "world_card_layout.hpp"
#include "world_document.hpp"
#include "MMO/Core/Registries/ProviderRegistrySnapshot.hpp"
#include "MMO/Core/Descriptors/ToolDescriptorJson.hpp"
#include "resource_inspection.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <set>
#include <sstream>

namespace GRIM::WorldLab {
inline std::string providerRoomName(std::string name) {
    // Presentation only: never change plugin ownership or generation identity.
    const auto original = name;
    if (name.size() > 7) {
        const auto start = name.size() - 6;
        auto suffix = name.substr(start);
        for (auto& c : suffix) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        if (suffix == "plugin" && (name[start - 1] == ' ' || name[start - 1] == '-' || name[start - 1] == '_')) {
            name.resize(start - 1);
            while (!name.empty() && (name.back() == ' ' || name.back() == '-' || name.back() == '_')) name.pop_back();
        }
    }
    return name.empty() ? original : name;
}

// A persistent manual live session, independent of compiled TextWorld episodes.
// Discovery/navigation only: no production handler is invoked by this backend.
class LiveWorldSession {
public:
    struct Entry { std::string action, feedback; nlohmann::json state; };
    explicit LiveWorldSession(std::vector<MMO::ProviderSnapshot> providers,
                              const std::filesystem::path& navigationPath = {}) {
        buildRooms(std::move(providers));
        record("Start", describe());
        if (!navigationPath.empty()) {
            try {
                if (!navigationPath.parent_path().empty())
                    std::filesystem::create_directories(navigationPath.parent_path());
                saveJsonFile(navigationPath, navigationJson(navigation()));
            } catch (const std::exception& error) {
                navigationPersistenceError_ = error.what();
            }
        }
    }

    const Document& document() const { return document_; }
    const std::string& location() const { return location_; }
    const std::vector<Entry>& history() const { return history_; }
    const std::string& navigationPersistenceError() const { return navigationPersistenceError_; }

    const MMO::ProviderSnapshot* providerInRoom(const std::string& room) const {
        const auto found = providerIdsByRoom_.find(room);
        return found == providerIdsByRoom_.end() ? nullptr : &providers_.at(found->second).contents;
    }
    std::string providerPath(const std::string& providerId) const {
        return document_.roomPath(providers_.at(providerId).roomId);
    }
    std::size_t resourceCount() const {
        std::size_t count = 0;
        for (const auto& [id, provider] : providers_) count += provider.contents.resources.size();
        return count;
    }
    LiveNavigationSnapshot navigation() const {
        LiveNavigationSnapshot result{document_.environmentId, document_.spawn, {}};
        for (const auto& path : document_.paths) {
            LiveNavigationRoom room{path.id, path.parent, document_.find(path.id)->name,
                                    document_.roomPath(path.id), {}};
            if (const auto* provider = providerInRoom(path.id)) room.provider = provider->provider;
            result.rooms.push_back(std::move(room));
        }
        return result;
    }

    void applyUpdates(std::vector<MMO::ProviderUpdate> updates) {
        std::optional<nlohmann::json> previousCatalog;
        std::string resourceChanges;
        for (auto& update : updates) {
            const auto provider = providers_.find(update.providerId);
            if (provider == providers_.end()) continue;
            auto& contents = provider->second.contents;
            if (update.tools) {
                if (!previousCatalog) previousCatalog = catalogTools();
                contents.tools = std::move(*update.tools);
            }
            if (update.resources) {
                if (provider->second.roomId == location_) {
                    resourceChanges += describeResourceChanges(contents.resources, *update.resources);
                    if (!resourceInspection_.is_null() &&
                        std::none_of(update.resources->begin(), update.resources->end(), [&](const auto& resource) {
                            return nlohmann::json(MMO::RegisteredResource{resource, contents.provider}) == resourceInspection_;
                        })) {
                        resourceInspection_ = nullptr;
                    }
                }
                contents.resources = std::move(*update.resources);
            }
        }
        if (previousCatalog && *previousCatalog != catalogTools()) record("Tools changed", describe());
        if (!resourceChanges.empty()) record("Resources changed", resourceChanges);
    }
    void command(const std::string& command) {
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
                move(command == "back" ? "/" : command.substr(3));
                ++moves_; record(command, describe());
            } else if (command.starts_with("inspect-resource ")) {
                const auto id = command.substr(17);
                for (const auto& r : resourcesHere()) if (r.resource_id == id) {
                    inspectResource(command, r); return;
                }
                throw std::runtime_error("Resource is unavailable in the current provider room");
            } else if (command.starts_with("inspect ")) {
                const auto id = command.substr(8);
                const auto& resources = resourcesHere();
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
        if (location_ != "system_hub") commands.push_back("back");
        else for (const auto& room : navigation().rooms)
            if (!room.parentId.empty()) commands.push_back("go " + room.path);
        auto tools = nlohmann::json::array();
        for (const auto& t : toolsHere()) {
            tools.push_back(t);
        }
        const auto inspectable = inspectableTools();
        for (const auto& t : inspectable) commands.push_back("inspect " + t.tool_id);
        const auto* group = providerInRoom(location_);
        auto resources = nlohmann::json::array();
        for (const auto& r : resourcesHere()) {
            resources.push_back(MMO::RegisteredResource{r, group->provider}); commands.push_back("inspect-resource " + r.resource_id);
            bool collision = false;
            for (const auto& tool : inspectable) collision |= tool.tool_id == r.resource_id;
            for (const auto& other : resourcesHere())
                collision |= other.resource_id != r.resource_id &&
                    (folded(other.resource_id) == folded(r.resource_id) || folded(other.name) == folded(r.resource_id));
            if (!collision) commands.push_back("inspect " + r.resource_id);
        }
        auto state = nlohmann::json{{"room_id", location_}, {"location_path", document_.roomPath(location_)},
            {"session_id", document_.environmentId}, {"moves", moves_}, {"resources", resources},
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
        text << document_.find(location_)->name << "  " << document_.roomPath(location_) << "\n";
        text << "Inventory: Tool catalog. Use open tool_catalog to read it.\n";
        if (location_ == "system_hub") {
            text << "Choose a startup provider room to enter.\n";
            for (const auto& room : navigation().rooms) if (!room.parentId.empty())
                text << room.path << " - " << room.name << '\n';
        } else {
            const auto& resources = resourcesHere();
            text << "Live resources: " << resources.size() << '\n';
            for (const auto& r : resources) {
                text << r.name << " (" << r.resource_id << ")";
                if (r.state.contains("visible")) text << (r.state.at("visible").get<bool>() ? " - visible" : " - hidden");
                text << '\n';
            }
            text << "Use inspect <object name or resource_id> for its readings; inspect-resource <resource_id> selects an object explicitly.\n";
            const auto tools = toolsHere();
            text << "Available model tools: " << tools.size() << '\n';
            for (const auto& t : tools) text << t.tool_id << " - " << (t.short_description.empty() ? t.description : t.short_description) << '\n';
            text << "Use inspect <tool_id> for its contract. Use back to return to System Hub.\n";
        }
        if (catalogOpen_) text << describeCatalog();
        return text.str();
    }
private:
    struct ProviderRoom {
        std::string roomId;
        MMO::ProviderSnapshot contents;
    };

    void buildRooms(std::vector<MMO::ProviderSnapshot> providers) {
        if (providers.size() > 9999) throw std::runtime_error("Live world room limit reached");
        std::sort(providers.begin(), providers.end(), [](const auto& a, const auto& b) {
            return a.provider.name < b.provider.name;
        });
        document_.title = "Live System World";
        document_.spawn = "system_hub";
        document_.entities.push_back({"system_hub", "System Hub", "room", "", "The hub connects the startup provider rooms."});
        document_.layout.push_back({"system_hub", 80, 80});
        document_.paths.push_back({"system_hub", "", ""});
        const double count = static_cast<double>(providers.size());
        const double radius = count > 1 ? std::max(160.0, 75 / std::sin(std::numbers::pi / count)) : 80.0;
        std::set<std::string> paths;
        for (std::size_t index = 0; index < providers.size(); ++index) {
            auto& provider = providers[index];
            const auto id = provider.provider.instance;
            if (id.empty() || providers_.contains(id)) throw std::runtime_error("Duplicate or empty startup provider ID");
            const auto room = "provider_" + std::to_string(index + 1);
            const auto name = providerRoomName(provider.provider.name);
            const auto base = roomSegment(name);
            auto segment = base;
            for (std::size_t suffix = 2; !paths.insert(segment).second; ++suffix)
                segment = base + "-" + std::to_string(suffix);
            const double angle = -std::numbers::pi / 2 + 2 * std::numbers::pi * index / count;
            document_.entities.push_back({room, name.substr(0, 256), "room", "", "A live provider room."});
            document_.layout.push_back({room, 80 + static_cast<float>(radius * std::cos(angle)),
                                             80 + static_cast<float>(radius * std::sin(angle))});
            document_.paths.push_back({room, "system_hub", std::move(segment)});
            providerIdsByRoom_.emplace(room, id);
            providers_.emplace(id, ProviderRoom{room, std::move(provider)});
        }
        CardLayout::separate(document_);
        document_.validate();
    }

    static std::string roomSegment(const std::string& name) {
        std::string segment;
        for (unsigned char c : name) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '-') segment += static_cast<char>(c);
            else if (!segment.empty() && segment.back() != '-') segment += '-';
        }
        segment.resize(std::min<std::size_t>(segment.size(), 48));
        while (!segment.empty() && segment.back() == '-') segment.pop_back();
        return segment.empty() ? "provider" : segment;
    }

    void move(const std::string& path) {
        std::string target;
        if (path.starts_with('/')) {
            for (const auto& room : document_.paths)
                if (document_.roomPath(room.id) == path) target = room.id;
        } else {
            for (const auto& [id, provider] : providers_) {
                const auto& name = provider.contents.provider.name;
                if (name != path && providerRoomName(name) != path) continue;
                if (!target.empty()) throw std::runtime_error("Provider name is ambiguous; use its listed room path");
                target = provider.roomId;
            }
        }
        if (target.empty()) throw std::runtime_error("Unknown room");
        if (target == location_) return;
        if (target != "system_hub" && location_ != "system_hub")
            throw std::runtime_error("Return to System Hub before entering another provider room");
        location_ = std::move(target);
    }

    std::vector<MMO::ToolDescriptor> toolsHere() const {
        std::vector<MMO::ToolDescriptor> result;
        if (const auto* group = providerInRoom(location_)) {
            for (const auto& tool : group->tools)
                if (tool.usage_type == MMO::ToolUsage::Model) result.push_back(tool);
        }
        return result;
    }
    const std::vector<MMO::ResourceDescriptor>& resourcesHere() const {
        static const std::vector<MMO::ResourceDescriptor> empty;
        const auto* group = providerInRoom(location_);
        return group ? group->resources : empty;
    }

    std::string describeResourceChanges(const std::vector<MMO::ResourceDescriptor>& before,
                                        const std::vector<MMO::ResourceDescriptor>& after) const {
        const auto* group = providerInRoom(location_);
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
        const auto* group = providerInRoom(location_);
        const MMO::RegisteredResource registered{resource, group->provider};
        resourceInspection_ = registered;
        record(command, resourceInspectionFeedback(registered));
    }
    std::vector<MMO::ToolDescriptor> inspectableTools() const {
        return catalogOpen_ ? catalogModelTools() : toolsHere();
    }
    std::vector<MMO::ToolDescriptor> catalogModelTools() const {
        std::vector<MMO::ToolDescriptor> result;
        for (const auto& [id, provider] : providers_)
            for (const auto& tool : provider.contents.tools)
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
    Document document_;
    std::map<std::string, ProviderRoom> providers_;
    std::map<std::string, std::string> providerIdsByRoom_;
    std::string location_ = "system_hub";
    std::string navigationPersistenceError_;
    uint64_t moves_ = 0;
    bool catalogOpen_ = false;
    nlohmann::json resourceInspection_ = nullptr;
    std::vector<Entry> history_;
};
}
