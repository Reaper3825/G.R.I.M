#include "resources.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#endif

namespace fs = std::filesystem;

std::string getResourcePath() {
#if defined(GRIM_PORTABLE_ONLY)
    // Portable mode: resources live next to the executable
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
    } else {
        exePath = fs::current_path();
    }
  #else
    exePath = fs::current_path();
  #endif
    return (exePath / "resources").string();
#else
    // System mode: resources folder inside project root
    return (fs::current_path() / "resources").string();
#endif
}

std::string loadTextResource(const std::string& filename, int argc, char** argv) {
    (void)argc;
    (void)argv;

    fs::path filePath = fs::path(getResourcePath()) / filename;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "[GRIM] Resource not found: " << filename << std::endl;
        return {};
    }
    return { std::istreambuf_iterator<char>(file),
             std::istreambuf_iterator<char>() };
}

std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history) {
    (void)argc;
    (void)argv;
    (void)history;

    fs::path resDir = fs::path(getResourcePath());

    // Look for common font files in the resources directory
    for (auto& p : fs::directory_iterator(resDir)) {
        if (p.path().extension() == ".ttf" || p.path().extension() == ".otf") {
            return p.path().string();
        }
    }

    std::cerr << "[ERROR] No font found in resources/ or system fonts." << std::endl;
    return {};
}
