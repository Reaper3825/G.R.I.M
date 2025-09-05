#include "resources.hpp"
#include "console_history.hpp"
#include <filesystem>
#include <string>
#include <algorithm>
#include <system_error>
#include <iostream>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

// Generic resource finder (JSON, fonts, etc.)
static fs::path findResource(const std::string& name, int argc, char** argv) {
    std::error_code ec;

    // 1. exe directory
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);

    fs::path local = exeDir / name;
    if (fs::exists(local, ec)) return local;

    fs::path resLocal = exeDir / "resources" / name;
    if (fs::exists(resLocal, ec)) return resLocal;

#ifndef GRIM_PORTABLE_ONLY
    // 2. installed system data dir (set via CMake)
    fs::path system = fs::path(GRIM_DATA_DIR) / name;
    if (fs::exists(system, ec)) return system;
#endif

    return {};
}

// Load a text resource into memory (e.g. JSON)
std::string loadTextResource(const std::string& name, int argc, char** argv) {
    auto path = findResource(name, argc, argv);
    if (path.empty()) {
        std::cerr << "[GRIM] Resource not found: " << name << "\n";
        return {};
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "[GRIM] Failed to open resource: " << path << "\n";
        return {};
    }

    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

// Find any .ttf font in resources/ or system locations
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* history) {
    std::error_code ec;

    // Helper: look in a directory for a .ttf file
    auto has_ttf = [&](const fs::path& dir) -> std::string {
        if (!fs::exists(dir, ec) || !fs::is_directory(dir, ec)) return {};
        for (auto& e : fs::directory_iterator(dir, ec)) {
            if (!e.is_regular_file(ec)) continue;
            auto ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".ttf") return e.path().string();
        }
        return {};
    };

    // 1. exe/resources folder
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
    if (auto s = has_ttf(exeDir / "resources"); !s.empty()) return s;

    // 2. working dir/resources
    if (auto s = has_ttf(fs::current_path(ec) / "resources"); !s.empty()) return s;

#ifndef GRIM_PORTABLE_ONLY
    // 3. system datadir/resources
    fs::path sysRes = fs::path(GRIM_DATA_DIR) / "resources";
    if (auto s = has_ttf(sysRes); !s.empty()) return s;
#endif

#ifdef _WIN32
    // 4. common Windows fonts
    for (const char* f : {
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    }) {
        if (fs::exists(f, ec)) return std::string(f);
    }
#endif

    // Nothing found → warn
    if (history) {
        history->push("[ERROR] No font found in resources/ or system fonts.", sf::Color::Red);
    }
    std::cerr << "[ERROR] No font found in resources/ or system fonts." << std::endl;
    return {};
}
