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
#include <windows.h>
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

inline std::filesystem::path getTrainingStatusFilePath() {
    return resolveResourceRoot() / "models/GRIM-text/training/training_status.fb";
}

} // namespace GRIM::Training
