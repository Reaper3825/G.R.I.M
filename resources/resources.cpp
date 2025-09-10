#include "resources.hpp"
#include "console_history.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <SFML/Graphics/Color.hpp>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

std::string getResourcePath() {
#if defined(GRIM_PORTABLE_ONLY)
    // Portable mode: resources live next to executable
    fs::path exePath;
#if defined(_WIN32)
    char buffer[MAX_PATH];
    GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    exePath = fs::path(buffer).parent_path();
#elif defined(__APPLE__)
    char buffer[1024];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        exePath = fs::path(buffer).parent_path();
    }
#else
    exePath = fs::canonical("/proc/self/exe").parent_path();
#endif
    return (exePath / "resources").string();
#else
    // Installed mode: use system data directory set by CMake
    return std::string(GRIM_DATA_DIR) + "/resources";
#endif
}

std::string loadTextResource(const std::string& filename, int argc, char** argv) {
    fs::path resDir = getResourcePath();
    fs::path filePath = resDir / filename;

    if (!fs::exists(filePath)) {
        std::cerr << "[ERROR] Resource file not found: " << filePath << "\n";
        return {};
    }

    std::ifstream in(filePath);
    if (!in) {
        std::cerr << "[ERROR] Failed to open resource file: " << filePath << "\n";
        return {};
    }

    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history) {
    fs::path resDir = getResourcePath();

    if (fs::exists(resDir)) {
        for (auto& p : fs::directory_iterator(resDir)) {
            if (p.path().extension() == ".ttf") {
                return p.path().string();
            }
        }
    }

    // Fallback: try common system fonts
#if defined(_WIN32)
    fs::path winFonts = "C:/Windows/Fonts";
    for (auto& p : fs::directory_iterator(winFonts)) {
        if (p.path().extension() == ".ttf") {
            return p.path().string();
        }
    }
#elif defined(__APPLE__)
    fs::path macFonts = "/System/Library/Fonts/Supplemental";
    for (auto& p : fs::directory_iterator(macFonts)) {
        if (p.path().extension() == ".ttf") {
            return p.path().string();
        }
    }
#else
    fs::path linuxFonts = "/usr/share/fonts";
    if (fs::exists(linuxFonts)) {
        for (auto& p : fs::recursive_directory_iterator(linuxFonts)) {
            if (p.path().extension() == ".ttf") {
                return p.path().string();
            }
        }
    }
#endif

    if (history) {
        history->push("[ERROR] No font found in resources/ or system fonts.", sf::Color::Red);
    } else {
        std::cerr << "[ERROR] No font found in resources/ or system fonts.\n";
    }
    return {};
}
