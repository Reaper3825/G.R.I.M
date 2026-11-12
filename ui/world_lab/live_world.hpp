#pragma once
#include "world_document.hpp"
#include "live_world_navigation.hpp"
#include "world_card_layout.hpp"
#include "MMO/Core/Registries/ProviderRegistrySnapshot.hpp"
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <numbers>

namespace GRIM::WorldLab {
using ProviderKey = std::pair<MMO::ToolProviderType, std::string>;
inline ProviderKey providerKey(const MMO::ProviderDescriptor& p) { return {p.type, p.name}; }
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

// Host-thread-owned topology. Registry snapshots contain data, never native callbacks.
class LiveWorld {
public:
    LiveWorld() {
        document_.title = "Live System World";
        document_.spawn = "system_hub";
        document_.entities.push_back({"system_hub", "System Hub", "room", "",
            "The permanent center of the live world. Registered providers automatically appear as rooms beneath this hub."});
        document_.layout.push_back({"system_hub", 80, 80});
        document_.paths.push_back({"system_hub", "", ""});
        document_.validate();
    }
    const Document& document() const { return document_; }
    const MMO::ProviderRegistrySnapshot& registry() const { return snapshot_; }
    const std::string& location() const { return location_; }
    uint64_t revision() const { return revision_; }
    LiveNavigationSnapshot navigation() const {
        LiveNavigationSnapshot result{document_.environmentId, document_.spawn, {}};
        for (const auto& path : document_.paths) {
            const auto* entity = document_.find(path.id);
            LiveNavigationRoom room{path.id, path.parent, entity->name, document_.roomPath(path.id), {}};
            for (const auto& key : connections_)
                if (roomId(key) == path.id) { room.provider = *provider(key); break; }
            result.rooms.push_back(std::move(room));
        }
        return result;
    }
    bool connected(const ProviderKey& key) const { return connections_.count(key) != 0; }
    std::string roomId(const ProviderKey& key) const {
        const auto it = ids_.find(key);
        return it == ids_.end() ? std::string{} : it->second;
    }
    std::string providerPath(const ProviderKey& key) const { return "/" + segments_.at(key); }
    const MMO::ProviderDescriptor* provider(const ProviderKey& key) const {
        for (const auto& group : snapshot_.providers)
            if (providerKey(group.provider) == key) return &group.provider;
        return nullptr;
    }
    // Commit a complete projection so invalid snapshots cannot leave partial topology.
    std::string reconcile(MMO::ProviderRegistrySnapshot snapshot) {
        auto next = *this;
        auto changes = next.reconcileSnapshot(std::move(snapshot));
        *this = std::move(next);
        return changes;
    }
private:
    void assignPath(const ProviderKey& key) {
        if (segments_.count(key)) return;
        std::string segment;
        for (unsigned char c : providerRoomName(key.second)) {
            if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '_' || c == '-') segment += static_cast<char>(c);
            else if (!segment.empty() && segment.back() != '-') segment += '-';
        }
        while (!segment.empty() && segment.back() == '-') segment.pop_back();
        if (segment.empty()) segment = "provider";
        segment.resize(std::min<size_t>(segment.size(), 48));
        const auto occupied = [&](const std::string& value) {
            return std::any_of(segments_.begin(), segments_.end(), [&](const auto& entry) { return entry.second == value; });
        };
        if (occupied(segment)) {
            // Stable identity-derived suffix, never a registration ordinal. Keep
            // retired names reserved so an old path cannot silently change owner.
            uint64_t hash = 14695981039346656037ULL;
            for (unsigned char c : key.second) { hash ^= c; hash *= 1099511628211ULL; }
            segment += "-plugin-" + std::to_string(hash);
            if (occupied(segment)) throw std::runtime_error("Provider navigation path collision");
        }
        segments_.emplace(key, std::move(segment));
    }
    std::string reconcileSnapshot(MMO::ProviderRegistrySnapshot snapshot) {
        if (snapshot.providers.size() > 9999) throw std::runtime_error("Live world room limit reached");
        std::string changes;
        for (auto it = connections_.begin(); it != connections_.end();) {
            const auto found = std::find_if(snapshot.providers.begin(), snapshot.providers.end(),
                [&](const auto& p) { return providerKey(p.provider) == *it; });
            if (found == snapshot.providers.end()) {
                changes += providerRoomName(it->second) + " disconnected: provider unloaded.\n";
                if (location_ == roomId(*it)) {
                    location_ = "system_hub";
                    changes += "Returned to System Hub.\n";
                }
                it = connections_.erase(it);
            } else {
                const auto* previous = provider(*it);
                if (previous && previous->instance != found->provider.instance)
                    changes += providerRoomName(found->provider.name) + " updated to its replacement generation.\n";
                ++it;
            }
        }
        for (const auto& group : snapshot.providers) {
            const auto& p = group.provider;
            const auto key = providerKey(p);
            if (connections_.insert(key).second) {
                if (!ids_.count(key)) ids_[key] = "provider_" + std::to_string(nextId_++);
                assignPath(key);
                changes += providerRoomName(p.name) + " registered: room added.\n";
            }
        }
        const bool resourcesChanged = snapshot.resource_revision != snapshot_.resource_revision;
        snapshot_ = std::move(snapshot);
        rebuild();
        if (!changes.empty() || resourcesChanged) ++revision_;
        return changes;
    }
public:
    void move(const std::string& path) {
        std::string target;
        if (path.starts_with('/')) {
            for (const auto& p : document_.paths)
                if (document_.roomPath(p.id) == path) target = p.id;
        } else for (const auto& key : connections_) if (key.second == path || providerRoomName(key.second) == path) {
            if (!target.empty()) throw std::runtime_error("Provider name is ambiguous; use its listed room path");
            target = roomId(key);
        }
        if (target.empty()) throw std::runtime_error("That room is no longer connected");
        if (target == location_) return;
        if (target != "system_hub" && location_ != "system_hub")
            throw std::runtime_error("Return to System Hub before entering another provider room");
        location_ = target;
    }
    const MMO::ProviderSnapshot* providerInRoom(const std::string& room) const {
        for (const auto& group : snapshot_.providers)
            if (roomId(providerKey(group.provider)) == room) return &group;
        return nullptr;
    }
    std::vector<MMO::ToolDescriptor> toolsHere() const {
        std::vector<MMO::ToolDescriptor> result;
        if (const auto* group = providerInRoom(location_))
            for (const auto& tool : group->tools)
                if (tool.usage_type == MMO::ToolUsage::Model) result.push_back(tool);
        return result;
    }
    const std::vector<MMO::ResourceDescriptor>& resourcesHere() const {
        static const std::vector<MMO::ResourceDescriptor> empty;
        const auto* group = providerInRoom(location_);
        return group ? group->resources : empty;
    }
private:
    void rebuild() {
        Document next = document_;
        next.entities.resize(1); next.layout.resize(1); next.paths.resize(1);
        const double count = static_cast<double>(connections_.size());
        // Compact ring: one quarter of the original count-scaled radius.
        const double radius = count > 1 ? std::max(160.0, 75 / std::sin(std::numbers::pi / count)) : 80.0;
        const auto hub = next.layout.front();
        size_t index = 0;
        for (const auto& key : connections_) {
            const auto id = roomId(key);
            next.entities.push_back({id, providerRoomName(key.second).substr(0, 256), "room", "",
                "A live provider room. Use tools to inspect its available model tools."});
            const double angle = -std::numbers::pi / 2 + 2 * std::numbers::pi * index / count;
            next.layout.push_back({id, hub.x + static_cast<float>(radius * std::cos(angle)),
                hub.y + static_cast<float>(radius * std::sin(angle))});
            next.paths.push_back({id, "system_hub", segments_.at(key)});
            ++index;
        }
        CardLayout::separate(next); // The live map displays room headers only.
        for (const auto& group : snapshot_.providers) {
            const auto key = providerKey(group.provider);
            for (const auto& resource : group.resources) {
                const auto identity = std::make_pair(key, resource.resource_id);
                auto [it, inserted] = resourceIds_.try_emplace(identity);
                if (inserted) it->second = "resource_" + std::to_string(nextResourceId_++);
                Entity entity{it->second, resource.name, "object", roomId(key), "Live " + resource.kind + " resource"};
                entity.resourceBinding = ResourceBinding{resource.kind, resource.resource_id};
                next.entities.push_back(std::move(entity));
            }
        }
        next.validate();
        document_ = std::move(next);
    }
    Document document_;
    MMO::ProviderRegistrySnapshot snapshot_;
    std::set<ProviderKey> connections_;
    std::map<ProviderKey, std::string> ids_;
    std::map<ProviderKey, std::string> segments_;
    std::map<std::pair<ProviderKey, std::string>, std::string> resourceIds_;
    uint64_t nextResourceId_ = 1;
    std::string location_ = "system_hub";
    uint64_t revision_ = 0, nextId_ = 1;
};
}
