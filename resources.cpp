#include "resources.hpp"
#include "console_history.hpp"
#include "logger.hpp"

#if defined(__APPLE__)
    #include <mach-o/dyld.h>
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
// Locate resource root (prefer repo/resources over build/resources)
// ====================================================-
std::string getResourcePath() {
    namespace fs = std::filesystem;
    fs::path base = fs::path(GRIM_ROOT_DIR) / "resources";

    if (fs::exists(base)) {
        LOG_PHASE("Resource path set", true);
        LOG_DEBUG("Resources", "Using canonical resource path: " + base.string());
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

std::string getSafeResourcePath(const std::string& target, PathResolutionMode mode) {
    // Default exclusions for search mode
    static const std::vector<std::string> defaultExclusions = {
        "external", "vcpkg_installed", ".git", "node_modules", 
        "build", "cmake-build-release", "cmake-build-debug", "venv", ".venv"
    };
    
    try {
        fs::path targetPath(target);
        
        // Helper function to check if a path should be excluded
        auto shouldExclude = [](const fs::path& path) -> bool {
            std::string pathStr = path.string();
            for (const auto& excluded : defaultExclusions) {
                if (pathStr.find(excluded) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        
        // Mode 1: Absolute path - validate it exists
        if (mode == PathResolutionMode::Absolute) {
            if (targetPath.is_absolute()) {
                if (fs::exists(targetPath)) {
                    LOG_DEBUG("Resources", "Absolute path exists: " + targetPath.string());
                    return targetPath.string();
                } else {
                    LOG_ERROR("Resources", "Absolute path not found: " + targetPath.string());
                    return "";
                }
            } else {
                LOG_ERROR("Resources", "Path is not absolute: " + target);
                return "";
            }
        }
        
        // Mode 2: Relative path - resolve from GRIM root
        if (mode == PathResolutionMode::Relative) {
            fs::path grimRoot(GRIM_ROOT_DIR);
            fs::path resolvedPath = grimRoot / targetPath;
            
            if (fs::exists(resolvedPath)) {
                LOG_DEBUG("Resources", "Relative path resolved: " + resolvedPath.string());
                return resolvedPath.string();
            } else {
                LOG_ERROR("Resources", "Relative path not found: " + resolvedPath.string());
                return "";
            }
        }
        
        // Mode 3: Search - recursively search in GRIM root with exclusions
        if (mode == PathResolutionMode::Search) {
            fs::path grimRoot(GRIM_ROOT_DIR);
            std::string filename = targetPath.filename().string();
            
            LOG_DEBUG("Resources", "Searching for: " + filename);
            
            try {
                for (const auto& entry : fs::recursive_directory_iterator(
                    grimRoot,
                    fs::directory_options::skip_permission_denied
                )) {
                    try {
                        // Skip excluded directories
                        if (entry.is_directory() && shouldExclude(entry.path())) {
                            continue;
                        }
                        
                        // Check if this is the file we're looking for
                        if (entry.is_regular_file() && 
                            entry.path().filename().string() == filename &&
                            !shouldExclude(entry.path())) {
                            LOG_DEBUG("Resources", "Found file at: " + entry.path().string());
                            return entry.path().string();
                        }
                    } catch (const fs::filesystem_error&) {
                        // Skip files/directories that cause errors
                        continue;
                    }
                }
            } catch (const fs::filesystem_error& e) {
                LOG_ERROR("Resources", std::string("Search error: ") + e.what());
                return "";
            }
            
            LOG_ERROR("Resources", "File not found in search: " + filename);
            return "";
        }
        
        LOG_ERROR("Resources", "Invalid path resolution mode");
        return "";
        
    } catch (const std::exception& e) {
        LOG_ERROR("Resources", std::string("Path resolution error: ") + e.what());
        return "";
    }
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

    for (auto& p : fs::directory_iterator(resDir)) {
        if (p.is_regular_file()) {
            auto ext = p.path().extension().string();
            if (ext == ".ttf" || ext == ".otf") {
                LOG_PHASE("Font search", true);
                LOG_DEBUG("Resources", "Found font: " + p.path().string());
                return p.path().string();
            }
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
