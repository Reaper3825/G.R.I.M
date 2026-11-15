#pragma once
#include "core/host_exports.hpp"
#include "ui/world_lab/world_goal.hpp"

namespace GRIM::MMO { class ModelLoader; class ResourceSignal; }
namespace GRIM::UI {
// Borrow existing host services; the panel never owns their lifecycle.
GRIM_HOST_API MMO::ModelLoader* modelWorkspaceLoader();
GRIM_HOST_API MMO::ResourceSignal* modelWorkspaceResources();
GRIM_HOST_API nlohmann::json modelWorkspaceConfig();
GRIM_HOST_API std::string modelWorkspaceServerURL();
GRIM_HOST_API WorldLab::GoalAccess modelWorkspaceGoalAccess();
}
