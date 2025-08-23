#include "nlp.hpp"
#include "commands.hpp" // so we can call handleCommand
#include <algorithm>
#include <sstream>

std::string parseNaturalLanguage(const std::string& line, fs::path& currentDir) {
    std::string lowered = line;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), ::tolower);

    if (lowered.find("where am i") != std::string::npos ||
        lowered.find("current directory") != std::string::npos) {
        return handleCommand("pwd", currentDir);
    }

    if (lowered.find("list files") != std::string::npos ||
        lowered.find("show files") != std::string::npos) {
        return handleCommand("list", currentDir);
    }

    if (lowered.find("make folder") != std::string::npos ||
        lowered.find("create folder") != std::string::npos) {
        return handleCommand("mkdir new_folder", currentDir);
    }

    if (lowered.find("delete old") != std::string::npos) {
        return handleCommand("rmolder 30 -r", currentDir);
    }

    if (lowered.find("move") != std::string::npos && lowered.find("to") != std::string::npos) {
        std::istringstream iss(lowered);
        std::string word, src, dst;
        while (iss >> word) {
            if (word == "move" && iss >> src && iss >> word && word == "to" && iss >> dst) {
                return handleCommand("move " + src + " " + dst, currentDir);
            }
        }
    }

    return ""; // no match
}
