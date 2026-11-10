#pragma once
#include "live_world_navigation.hpp"
#include "world_document.hpp"
#include <chrono>

namespace GRIM::WorldLab {
// Optional persistence consumer. Failed publication never rolls back live state;
// retain the old complete file and retry with the latest snapshot after a delay.
class LiveNavigationStore {
public:
    explicit LiveNavigationStore(std::filesystem::path path = {}) : path_(std::move(path)) {}
    void synchronize(const LiveNavigationSnapshot& snapshot) {
        if (path_.empty()) return;
        const auto now = std::chrono::steady_clock::now();
        if (now < retryAt_) return;
        try {
            auto value = navigationJson(snapshot);
            if (value == published_ && std::filesystem::exists(path_)) return;
            if (!path_.parent_path().empty()) std::filesystem::create_directories(path_.parent_path());
            saveJsonFile(path_, value);
            published_ = std::move(value);
            error_.clear();
        } catch (const std::exception& error) {
            error_ = error.what();
            retryAt_ = now + std::chrono::seconds(5);
        }
    }
    const std::string& error() const { return error_; }
private:
    std::filesystem::path path_;
    nlohmann::json published_;
    std::string error_;
    std::chrono::steady_clock::time_point retryAt_{};
};
}
