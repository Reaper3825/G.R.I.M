#include "resources.hpp"
#include "console_history.hpp"
#include "logger.hpp"

#if defined(_WIN32)
    #include <windows.h>  // GetModuleFileNameA
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#else
    #include <unistd.h>
    #include <limits.h>
#endif

namespace fs = std::filesystem;

// ====================================================-
// Global state definitions
// ====================================================-
nlohmann::json longTermMemory;
nlohmann::json aiConfig;

ConsoleHistory history;
std::vector<Timer> timers;
std::filesystem::path g_currentDir;

// ====================================================-
// Get GRIM root directory
// ====================================================-
std::string getGrimRootDir() {
    namespace fs = std::filesystem;
    
#ifdef GRIM_ROOT_DIR
    fs::path root = fs::path(GRIM_ROOT_DIR);
    if (fs::exists(root)) {
        return root.string();
    }
#endif

    // Fallback: try to find GRIM root by walking up from executable or current directory
    fs::path exeDir;
#if defined(_WIN32)
    char buffer[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buffer, MAX_PATH)) {
        exeDir = fs::path(buffer).parent_path();
    }
#elif defined(__APPLE__)
    char buffer[PATH_MAX];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) == 0) {
        exeDir = fs::path(buffer).parent_path();
    }
#else
    char buffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len > 0) {
        buffer[len] = '\0';
        exeDir = fs::path(buffer).parent_path();
    }
#endif

    // Walk up from exe directory or current path to find GRIM root
    std::vector<fs::path> searchPaths = {exeDir, fs::current_path()};
    
    for (auto& base : searchPaths) {
        if (base.empty()) continue;
        
        fs::path probe = base;
        for (int depth = 0; depth < 10 && !probe.empty(); ++depth) {
            // Check if this is GRIM root (has both "control" and "resources" directories)
            if (fs::exists(probe / "control") && fs::exists(probe / "resources")) {
                return probe.string();
            }
            if (!probe.has_parent_path()) break;
            probe = probe.parent_path();
        }
    }
    
    // Last resort: return current directory
    return fs::current_path().string();
}

// ====================================================-
// Locate resource root (prefer repo/resources over build/resources)
// ====================================================-
std::string getResourcePath() {
    namespace fs = std::filesystem;
    fs::path base = fs::path(GRIM_ROOT_DIR) / "resources";

    if (fs::exists(base)) {
        return base.string();
    }

    // Fallback: try relative to exe or current dir
    fs::path exeDir;
#if defined(_WIN32)
    char buffer[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buffer, MAX_PATH))
        exeDir = fs::path(buffer).parent_path();
#endif

    for (auto& tryPath : {
        exeDir / "resources",
        fs::current_path() / "resources"
    }) {
        if (fs::exists(tryPath)) {
            LOG_PHASE("Resource path set", true);
            LOG_DEBUG("Resources", "Using fallback resource path: " + tryPath.string());
            return tryPath.string();
        }
    }

    LOG_PHASE("Resource path set", true);
    LOG_ERROR("Resources", "No resource directory found (checked canonical, exe, cwd)");
    return fs::current_path().string();
}
#include <filesystem>
#include "logger.hpp"

namespace fs = std::filesystem;

std::vector<std::string> listFiles(const std::string& folderPath) {
    std::vector<std::string> files;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }

    return files;
}

// ====================================================-
// Load text resource from resources/ folder
// ====================================================-
std::string loadTextResource(const std::string& filename, int argc, char** argv) {
    (void)argc;
    (void)argv;

    fs::path filePath = fs::path(getResourcePath()) / filename;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        LOG_ERROR("Resources", "Resource not found: " + filename +
                               " (looked in " + filePath.string() + ")");
        LOG_PHASE("Resource load", false);
        return {};
    }

    LOG_PHASE("Resource load", true);
    LOG_DEBUG("Resources", "Loaded text resource: " + filename);
    return { std::istreambuf_iterator<char>(file),
             std::istreambuf_iterator<char>() };
}

// ====================================================-
// Find any usable font in resources/ (first .ttf or .otf)
// ====================================================-
std::string findAnyFontInResources(int argc, char** argv, ConsoleHistory* historyPtr) {
    (void)argc;
    (void)argv;

    fs::path resDir = fs::path(getResourcePath());

    if (!fs::exists(resDir)) {
        std::string msg = "Resource directory missing: " + resDir.string();
        if (historyPtr) {
            historyPtr->push("[ERROR] " + msg, 0xFF0000FF);
        }
        LOG_ERROR("Resources", msg);
        LOG_PHASE("Font search", false);
        return {};
    }

    // Recursively search for any .ttf or .otf font file
    try {
        for (auto& p : fs::recursive_directory_iterator(resDir, fs::directory_options::skip_permission_denied)) {
            if (p.is_regular_file()) {
                auto ext = p.path().extension().string();
                if (ext == ".ttf" || ext == ".otf") {
                    LOG_PHASE("Font search", true);
                    LOG_DEBUG("Resources", "Found font: " + p.path().string());
                    return p.path().string();
                }
            }
        }
    } catch (...) {}

    // Fallback: check build directory for ui_font.ttf (copied by CMake)
    for (auto& tryPath : {
        fs::current_path() / "ui_font.ttf",
        fs::path(getGrimRootDir()) / "out" / "build" / "ui_font.ttf"
    }) {
        if (fs::exists(tryPath)) {
            LOG_PHASE("Font search", true);
            LOG_DEBUG("Resources", "Found fallback font: " + tryPath.string());
            return tryPath.string();
        }
    }

    std::string errMsg = "No font found in resources/ or system fonts.";
    if (historyPtr) {
        historyPtr->push("[ERROR] " + errMsg, 0xFF0000FF);
    }
    LOG_ERROR("Resources", errMsg);
    LOG_PHASE("Font search", false);
    return {};
}
