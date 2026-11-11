#pragma once
#include <cstdint>
#include <string>

namespace GRIM::MMO {
enum class ToolProviderType : uint8_t { Builtin, Plugin };
// Identity is (type, name); instance identifies one loaded generation.
// Providers exist independently of how many tools they expose.
struct ProviderDescriptor {
    ToolProviderType type = ToolProviderType::Plugin;
    std::string name;
    std::string instance;
    std::string path;

    bool operator==(const ProviderDescriptor& other) const {
        return type == other.type && name == other.name &&
               instance == other.instance && path == other.path;
    }
};
}
