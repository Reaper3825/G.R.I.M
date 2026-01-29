#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace GRIM {
    struct UserPreference {
        std::string key;
        std::string value;

    };

    struct Personal {
        std::string name;
        std::string description;
        std::unordered_map<std::string, float> attributes;

    };
}