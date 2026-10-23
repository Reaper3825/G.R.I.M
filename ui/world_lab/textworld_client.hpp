#pragma once
#include <nlohmann/json.hpp>
#include <memory>
#include <optional>

namespace GRIM::WorldLab {
// Single-flight asynchronous worker. All Docker waits and filesystem transport
// happen off the UI thread. No image builds, pulls, or package installation.
class TextWorldClient {
public:
    TextWorldClient();
    ~TextWorldClient();
    TextWorldClient(const TextWorldClient&) = delete;
    TextWorldClient& operator=(const TextWorldClient&) = delete;
    bool begin(nlohmann::json request);
    bool busy() const;
    std::optional<nlohmann::json> poll();
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
