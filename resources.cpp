#include "resources.hpp"
#include "console_history.hpp"
#include <filesystem>
#include <string>
#include <algorithm>
#include <system_error>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

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

    // Check executable directory /resources
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
    if (auto s = has_ttf(exeDir / "resources"); !s.empty()) return s;

    // Check current working directory /resources
    if (auto s = has_ttf(fs::current_path(ec) / "resources"); !s.empty()) return s;

#ifdef _WIN32
    // Try common Windows fonts
    for (const char* f : {
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    }) {
        if (fs::exists(f, ec)) return std::string(f);
    }
#endif

    // If nothing found → warn
    if (history) {
        history->push("[ERROR] No font found in resources/ or system fonts.", sf::Color::Red);
    }
    std::cerr << "[ERROR] No font found in resources/ or system fonts." << std::endl;
    return {};
}
