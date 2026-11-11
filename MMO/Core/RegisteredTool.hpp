#pragma once
#include "ProviderDescriptor.hpp"
#include "ToolDescriptor.hpp"

namespace GRIM::MMO {
// Host inspection view copied from a single registry revision.
struct RegisteredTool : ToolDescriptor {
    ProviderDescriptor provider;
};
}
