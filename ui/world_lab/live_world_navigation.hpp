#pragma once
#include "MMO/Core/Descriptors/ProviderDescriptor.hpp"
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace GRIM::WorldLab {
// Navigation read model shared by presentation, persistence, and future Goal
// binding. Provider identity is durable; room IDs and paths belong to a session.
struct LiveNavigationRoom {
    std::string roomId, parentId, name, path;
    std::optional<MMO::ProviderDescriptor> provider;
};
struct LiveNavigationSnapshot {
    std::string sessionId, hubId;
    std::vector<LiveNavigationRoom> rooms;
};
inline nlohmann::json navigationJson(const LiveNavigationSnapshot& snapshot) {
    auto rooms = nlohmann::json::array();
    for (const auto& room : snapshot.rooms) {
        nlohmann::json entry = {{"room_id", room.roomId}, {"parent_id", room.parentId},
            {"display_name", room.name}, {"path", room.path}};
        if (room.provider) entry["provider"] = {
            {"type", "plugin"},
            {"name", room.provider->name}, {"instance", room.provider->instance}};
        rooms.push_back(std::move(entry));
    }
    return {{"format", "grim-live-navigation-snapshot"}, {"version", 1},
        {"session_id", snapshot.sessionId}, {"hub_id", snapshot.hubId}, {"rooms", std::move(rooms)}};
}
}
