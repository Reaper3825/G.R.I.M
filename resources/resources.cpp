#include "resources.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

std::string getResourcePath() {
    // Directory of the running binary
    fs::path exePath = fs::canonical("/proc/self/exe").parent_path();

    // resources is one level up from build/
    fs::path resPath = exePath.parent_path() / "resources";

    return resPath.string();
}

std::string loadTextResource(const std::string& filename, int argc, char** argv) {
    fs::path fullPath = fs::path(getResourcePath()) / filename;
    std::ifstream file(fullPath);
    if (!file.is_open()) {
        std::cerr << "[GRIM] Resource not found: " << filename << "\n";
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history) {
    fs::path resDir = getResourcePath();
    for (auto& p : fs::directory_iterator(resDir)) {
        if (p.path().extension() == ".ttf") {
            return p.path().string();
        }
    }
    history->push("[ERROR] No font found in resources/ or system fonts.", sf::Color::Red);
    return {};
}
