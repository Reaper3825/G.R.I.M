#pragma once
#include <string>
#include <vector>
#include <cstdint>
#
namespace GRIM {
    struct CapabilityOutput {
        std::string Type;
        std::string Value;


    };

    struct CapabilityInput {
        std::string Type;
        std::string Value;
        

    };

    struct Capability {
        std::string Name;
        std::string Id;
        std::vector<std::string> Tags;
        std::string Type; // Memory, Tool, Model, Web
        CapabilityOutput Output;
        CapabilityInput Input;
    };
} // namespace GRIM