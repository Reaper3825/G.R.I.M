#include "resources.hpp"
#include "console_history.hpp"
#include "logger.hpp"
#include "control/training_paths.hpp"
#include "settings/runtime_ai_config.hpp"
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cctype>
#include <system_error>

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

std::vector<std::filesystem::path> listFiles(const std::filesystem::path& folderPath) {
    std::error_code ec;
    return listFiles(folderPath, ec);
}

std::vector<std::filesystem::path> listFiles(const std::filesystem::path& folderPath, std::error_code& ec) {
    std::vector<std::filesystem::path> files;
    ec.clear();

    const bool exists = fs::exists(folderPath, ec);
    if (ec) {
        return files;
    }
    if (!exists) {
        ec = std::make_error_code(std::errc::no_such_file_or_directory);
        return files;
    }

    const bool isDirectory = fs::is_directory(folderPath, ec);
    if (ec) {
        return files;
    }
    if (!isDirectory) {
        ec = std::make_error_code(std::errc::not_a_directory);
        return files;
    }

    for (fs::directory_iterator it(folderPath, fs::directory_options::skip_permission_denied, ec), end;
         it != end && !ec;
         it.increment(ec)) {
        if (it->is_regular_file(ec)) {
            files.push_back(it->path());
        }
        if (ec) {
            files.clear();
            return files;
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

    // Recursively search for a usable text font. We deliberately DO NOT grab
    // the first .ttf/.otf in directory order: math/symbol fonts (e.g.
    // DejaVuMathTeXGyre.ttf) crash stb_truetype's glyph packer with an access
    // violation, and styled cuts (bold/italic) render poorly as the base UI
    // font. Score every candidate and pick the best upright text font.
    auto scoreFontName = [](std::string name) -> int {
        for (auto& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        // Hard reject: not text fonts / known to break the packer.
        if (name.find("math")     != std::string::npos ||
            name.find("symbol")   != std::string::npos ||
            name.find("emoji")    != std::string::npos ||
            name.find("dingbat")  != std::string::npos ||
            name.find("webding")  != std::string::npos ||
            name.find("wingding") != std::string::npos) {
            return -1;
        }
        int score = 100;
        if (name.find("sans") != std::string::npos) score += 20;  // prefer proportional sans for UI
        if (name.find("bold") != std::string::npos) score -= 8;
        if (name.find("italic") != std::string::npos ||
            name.find("oblique") != std::string::npos) score -= 8;
        if (name.find("condensed") != std::string::npos) score -= 4;
        if (name.find("light") != std::string::npos) score -= 4;
        if (name.find("mono") != std::string::npos) score -= 2;
        if (name.find("serif") != std::string::npos) score -= 2;
        return score;
    };

    try {
        std::string bestPath;
        int bestScore = -1;
        std::string fallbackPath;  // any font at all, preserves old behavior as last resort

        for (auto& p : fs::recursive_directory_iterator(resDir, fs::directory_options::skip_permission_denied)) {
            if (p.is_regular_file()) {
                auto ext = p.path().extension().string();
                if (ext == ".ttf" || ext == ".otf") {
                    if (fallbackPath.empty()) fallbackPath = p.path().string();
                    int score = scoreFontName(p.path().filename().string());
                    if (score > bestScore) {
                        bestScore = score;
                        bestPath = p.path().string();
                    }
                }
            }
        }

        if (bestScore >= 0 && !bestPath.empty()) {
            LOG_PHASE("Font search", true);
            LOG_DEBUG("Resources", "Found font: " + bestPath);
            return bestPath;
        }
        if (!fallbackPath.empty()) {
            LOG_PHASE("Font search", true);
            LOG_DEBUG("Resources", "No preferred text font found; using fallback (may render poorly): " + fallbackPath);
            return fallbackPath;
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

static std::string trimEnvToken(const char* s) {
    if (!s || !*s) return {};
    std::string t(s);
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r' || t.back() == ' ' || t.back() == '\t'))
        t.pop_back();
    size_t i = 0;
    while (i < t.size() && (t[i] == ' ' || t[i] == '\t')) ++i;
    return t.substr(i);
}

std::string resolveHuggingFaceApiToken() {
    if (const char* e = std::getenv("HF_TOKEN")) {
        std::string t = trimEnvToken(e);
        if (!t.empty()) return t;
    }
    if (const char* e = std::getenv("HUGGINGFACE_HUB_TOKEN")) {
        std::string t = trimEnvToken(e);
        if (!t.empty()) return t;
    }
    if (aiConfig.contains("api_keys") && aiConfig["api_keys"].contains("huggingface")) {
        std::string t = aiConfig["api_keys"]["huggingface"].get<std::string>();
        if (!t.empty()) return t;
    }
    return {};
}

std::filesystem::path getGrimAiConfigPath() {
    return GRIM::Training::getAiConfigFilePath();
}

std::filesystem::path getGrimLocalAiConfigPath() {
    return GRIM::Training::getAiConfigLocalFilePath();
}

nlohmann::json loadGrimRuntimeAiConfig() {
    return Settings::loadRuntimeAiConfig();
}

nlohmann::json saveGrimRuntimeAiConfig(const nlohmann::json& pending) {
    return Settings::saveRuntimeAiConfig(pending);
}
