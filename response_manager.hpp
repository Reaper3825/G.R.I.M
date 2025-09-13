#pragma once
#include <string>
#include <vector>

namespace ResponseManager {
    // Get a randomized natural-sounding response for a given intent key
    std::string get(const std::string& intent);
}
