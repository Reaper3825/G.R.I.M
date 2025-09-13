#include "resources.hpp"
#include "console_history.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

// -------------------------------------------------------------
// Locate resource root (hybrid: build/resources OR ../resources)
// -------------------------------------------------------------
std::string getResourcePath() {
#if defined(GRIM_PORTABLE_ONLY)
    // Portable mode: look next to executable
    fs::path exePath;
  #if defined(_WIN32)
    char buffer[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buffer, MAX_PATH)) {
        exePath = fs::path(buffer).parent_path();
    } else {
        exePath = fs::current_path();
    }
  #elif defined(__APPLE__)
    char buffer[1024];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        exePath = fs::path(buffer).parent_path();
    } else {
        exePath = fs::current_path();
    }
  #else
    exePath = fs::current_path();
  #endif
    fs::path portablePath = exePath / "resources";
    std::cout << "[GRIM] Using portable resource path: " << portablePath << "\n";
    return portablePath.string();
#else
    // System mode: prefer build/resources, fallback to ../resources
    fs::path buildPath   = fs::current_path() / "resources";                // e.g. D:/G.R.I.M/build/resources
    fs::path projectPath = fs::current_path().parent_path() / "resources";  // e.g. D:/G.R.I.M/resources

    if (fs::exists(buildPath)) {
        std::cout << "[GRIM] Using resource path: " << buildPath << "\n";
        return buildPath.string();
    }
    if (fs::exists(projectPath)) {
        std::cout << "[GRIM] Using fallback resource path: " << projectPath << "\n";
        return projectPath.string();
    }

    std::cerr << "[GRIM] WARNING: No resources directory found. Defaulting to: " << buildPath << "\n";
    return buildPath.string();
#endif
}

// -------------------------------------------------------------
// Load text resource from resources/ folder
// -------------------------------------------------------------
std::string loadTextResource(const std::string& filename, int argc, char** argv) {
    (void)argc;
    (void)argv;

    fs::path filePath = fs::path(getResourcePath()) / filename;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[GRIM] Resource not found: " << filename
                  << " (looked in " << filePath.string() << ")\n";
        return {};
    }

    return { std::istreambuf_iterator<char>(file),
             std::istreambuf_iterator<char>() };
}

// -------------------------------------------------------------
// Find any usable font in resources/ (first .ttf or .otf)
// -------------------------------------------------------------
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history) {
    (void)argc;
    (void)argv;

    fs::path resDir = fs::path(getResourcePath());

    if (!fs::exists(resDir)) {
        if (history) {
            history->push("[ERROR] Resource directory missing: " + resDir.string(), sf::Color::Red);
        } else {
            std::cerr << "[ERROR] Resource directory missing: " << resDir << std::endl;
        }
        return {};
    }

    for (auto& p : fs::directory_iterator(resDir)) {
        if (p.is_regular_file()) {
            auto ext = p.path().extension().string();
            if (ext == ".ttf" || ext == ".otf") {
                return p.path().string();
            }
        }
    }

    if (history) {
        history->push("[ERROR] No font found in resources/ or system fonts.", sf::Color::Red);
    } else {
        std::cerr << "[ERROR] No font found in resources/ or system fonts." << std::endl;
    }
    return {};
}
