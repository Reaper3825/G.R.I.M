#pragma once

#include <filesystem>
#include <vector>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include "core/grim_platform.h"
#else
#include <limits.h>
#include <unistd.h>
#endif

namespace GRIM::Training {

inline std::filesystem::path resolveResourceRoot() {
    static std::filesystem::path cached;
    static bool initialized = false;
    if (initialized) {
        return cached;
    }
    initialized = true;

    const std::filesystem::path marker = std::filesystem::path("models") / "GRIM-text" / "training";
    std::vector<std::filesystem::path> bases;

#ifdef GRIM_ROOT_DIR
    bases.emplace_back(std::filesystem::path(GRIM_ROOT_DIR));
#endif

    bases.emplace_back(std::filesystem::current_path());

#ifdef _WIN32
    char exeBuffer[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, exeBuffer, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        bases.emplace_back(std::filesystem::path(exeBuffer).parent_path());
    }
#else
    char exeBuffer[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exeBuffer, sizeof(exeBuffer) - 1);
    if (len > 0) {
        exeBuffer[len] = '\0';
        bases.emplace_back(std::filesystem::path(exeBuffer).parent_path());
    }
#endif

    bases.emplace_back(std::filesystem::path(__FILE__).parent_path());

    for (auto base : bases) {
        if (base.empty()) {
            continue;
        }

        std::error_code absErr;
        base = std::filesystem::absolute(base, absErr);
        std::filesystem::path probe = base;

        for (int depth = 0; depth < 10 && !probe.empty(); ++depth) {
            auto resourcesCandidate = probe / "resources";
            std::error_code existsErr;
            if (std::filesystem::exists(resourcesCandidate / marker, existsErr) && !existsErr) {
                cached = resourcesCandidate;
                return cached;
            }
            if (!probe.has_parent_path()) {
                break;
            }
            probe = probe.parent_path();
        }
    }

    cached = std::filesystem::current_path();
    return cached;
}

inline std::filesystem::path resolveGrimRoot() {
    const std::filesystem::path resourceRoot = resolveResourceRoot();
    if (resourceRoot.filename() == "resources") {
        return resourceRoot.parent_path();
    }
    return resourceRoot;
}

inline std::filesystem::path getAiConfigFilePath() {
    return resolveGrimRoot() / "ai_config.json";
}

inline std::filesystem::path getAiConfigLocalFilePath() {
    return resolveGrimRoot() / "ai_config.local.json";
}

inline std::filesystem::path getTrainingStatusFilePath() {
    return resolveResourceRoot() / "models/GRIM-text/training/training_status.fb";
}

// Path resolution modes
enum class PathResolutionMode {
    Absolute,  // Treat as absolute path
    Relative,  // Resolve relative to GRIM root
    Search     // Search for file in GRIM directory tree
};

inline std::string getSafeResourcePath(
    const std::filesystem::path& target, 
    PathResolutionMode mode = PathResolutionMode::Relative,
    const std::vector<std::string>& excludeDirs = {"external", "vcpkg_installed", ".git", "node_modules", "build", "cmake-build-release", "cmake-build-debug"}
) {
    try {
        std::filesystem::path targetPath(target);
        
        // Helper function to check if a path should be excluded
        auto shouldExclude = [&excludeDirs](const std::filesystem::path& path) -> bool {
            std::string pathStr = path.string();
            for (const auto& excluded : excludeDirs) {
                // Check if any part of the path contains the excluded directory name
                if (pathStr.find(excluded) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };
        
        // Mode 1: Absolute path - validate it exists
        if (mode == PathResolutionMode::Absolute) {
            if (targetPath.is_absolute()) {
                if (std::filesystem::exists(targetPath)) {
                    return targetPath.string();
                } else {
                    return "";
                }
            } else {
                return "";
            }
        }
        
        // Mode 2: Relative path - resolve from GRIM root
        if (mode == PathResolutionMode::Relative) {
            // Get GRIM root by going up from resources
            std::filesystem::path grimRoot;
            
#ifdef GRIM_ROOT_DIR
            grimRoot = std::filesystem::path(GRIM_ROOT_DIR);
#else
            // Try to find GRIM root from current executable location
            std::filesystem::path exePath;
#ifdef _WIN32
            char exeBuffer[MAX_PATH];
            DWORD len = GetModuleFileNameA(nullptr, exeBuffer, MAX_PATH);
            if (len > 0 && len < MAX_PATH) {
                exePath = std::filesystem::path(exeBuffer).parent_path();
            }
#else
            char exeBuffer[PATH_MAX];
            ssize_t len = readlink("/proc/self/exe", exeBuffer, sizeof(exeBuffer) - 1);
            if (len > 0) {
                exeBuffer[len] = '\0';
                exePath = std::filesystem::path(exeBuffer).parent_path();
            }
#endif
            
            // Walk up to find GRIM root (look for control folder or resources folder)
            grimRoot = exePath;
            for (int depth = 0; depth < 10 && !grimRoot.empty(); ++depth) {
                if (std::filesystem::exists(grimRoot / "control") || 
                    std::filesystem::exists(grimRoot / "resources")) {
                    break;
                }
                if (!grimRoot.has_parent_path()) {
                    grimRoot = std::filesystem::current_path();
                    break;
                }
                grimRoot = grimRoot.parent_path();
            }
#endif
            
            std::filesystem::path resolvedPath = grimRoot / targetPath;
            
            if (std::filesystem::exists(resolvedPath)) {
                return resolvedPath.string();
            } else {
                return "";
            }
        }
        
        // Mode 3: Search - recursively search in GRIM root
        if (mode == PathResolutionMode::Search) {
            std::filesystem::path grimRoot;
            
#ifdef GRIM_ROOT_DIR
            grimRoot = std::filesystem::path(GRIM_ROOT_DIR);
#else
            std::filesystem::path exePath;
#ifdef _WIN32
            char exeBuffer[MAX_PATH];
            DWORD len = GetModuleFileNameA(nullptr, exeBuffer, MAX_PATH);
            if (len > 0 && len < MAX_PATH) {
                exePath = std::filesystem::path(exeBuffer).parent_path();
            }
#endif
            grimRoot = exePath;
            for (int depth = 0; depth < 10; ++depth) {
                if (std::filesystem::exists(grimRoot / "control") || 
                    std::filesystem::exists(grimRoot / "resources")) {
                    break;
                }
                if (!grimRoot.has_parent_path()) {
                    grimRoot = std::filesystem::current_path();
                    break;
                }
                grimRoot = grimRoot.parent_path();
            }
#endif
            
            std::string filename = targetPath.filename().string();
            
            // Use directory iterator with error handling and exclusion
            try {
                for (const auto& entry : std::filesystem::recursive_directory_iterator(
                    grimRoot,
                    std::filesystem::directory_options::skip_permission_denied
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
                            return entry.path().string();
                        }
                    } catch (const std::filesystem::filesystem_error&) {
                        // Skip files/directories that cause errors
                        continue;
                    }
                }
            } catch (const std::filesystem::filesystem_error&) {
                // If we can't iterate at all, return empty
                return "";
            }
            
            return "";
        }
        
        return "";
        
    } catch (const std::exception&) {
        return "";
    }
}

} // namespace GRIM::Training
