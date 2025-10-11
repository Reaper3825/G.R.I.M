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
            historyPtr->push("[ERROR] " + msg, sf::Color::Red);
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
        historyPtr->push("[ERROR] " + errMsg, sf::Color::Red);
    }
    LOG_ERROR("Resources", errMsg);
    LOG_PHASE("Font search", false);
    return {};
}
