#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <unordered_map>

namespace GRIM {
    namespace Organize {

        struct SubjectData {
            std::string ID;
            std::vector<std::string> SubjectTags;
            std::vector<std::string> Path;
        };

        std::unordered_map<std::string, SubjectData> Subject_Maps;

        std::vector<std::string> GetSubjectTags(const std::string& subject_name) {};

        


    }
}