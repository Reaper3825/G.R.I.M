// intent.hpp
#pragma once
#include <string>
#include <unordered_map>

struct Intent {
    std::string name;                       
    float score = 0.0f;                     
    std::unordered_map<std::string, std::string> slots;
    bool matched = false;
};
