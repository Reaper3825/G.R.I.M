// Multi-Model Orchestration (MMO) - Shared MMD Definitions
// Model Metadata (MMD) Header
//======================================================//
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

namespace GRIM {
namespace MMO {
    class MMD {
        ~MMD() = default;
    public:


    struct ModelInfo {
        std::string ID;
        std::string name;
        std::string Version;
        std::string Subject;
        std::string Description;
        std::string model_path;
        std::string Date_Created;
        std::string Date_Modified;
        std::vector<std::string> SubjectTags;
        float Usage_Weight;
    };

    std::vector<std::string> getSubjectTags(std::string RawInput);
    std::vector<std::string> SubjectTags;

    };
    
}// MMO
} // namespace GRIM