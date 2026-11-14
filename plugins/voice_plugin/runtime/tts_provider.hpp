#pragma once

#include <string>
#include <vector>

namespace Voice {

enum class TTSProviderState {
    Stopped,
    Starting,
    Ready,
    Failed
};

struct TTSProviderCapabilities {
    bool supports_voice_selection = false;
    bool supports_voice_cloning = false;
    bool supports_language_selection = false;
    bool supports_rate = false;
    bool supports_pitch = false;
    bool supports_emphasis = false;
    bool supports_streaming = false;
    std::vector<std::string> languages;
};

struct TTSSynthesisRequest {
    std::string text;
    std::string category;
    std::string voice_id;
    std::string language;
    double rate = 1.0;
    double pitch = 1.0;
    double emphasis = 0.5;
};

struct TTSSynthesisResult {
    bool success = false;
    std::string audio_path;
    std::string error_code;
    std::string error_message;
};

class ITTSProvider {
public:
    virtual ~ITTSProvider() = default;

    virtual const char* providerId() const noexcept = 0;
    virtual bool initialize() = 0;
    virtual void shutdown() noexcept = 0;
    virtual TTSProviderState state() const noexcept = 0;
    virtual TTSProviderCapabilities capabilities() const = 0;
    virtual TTSSynthesisResult synthesize(const TTSSynthesisRequest& request) = 0;
};

} // namespace Voice
