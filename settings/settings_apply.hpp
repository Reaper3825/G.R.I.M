#pragma once
#include <nlohmann/json.hpp>
#include <functional>
#include <string>
#include <unordered_map>

// Forward declarations — avoid pulling in UI/voice/platform headers
class OverlayRenderer;

namespace Settings {

// Callback the caller provides so settings_apply can set a font
// without depending on UIRoot.
using SetFontFn = std::function<void(const std::string& path, int size)>;

// Callback for blur style (hwnd is opaque void*)
using SetBlurFn = std::function<void(bool enabled, float opacity, int intensity)>;

// Apply runtime side-effects of a config change.
// `config`    – the full merged document (what was just saved).
// `fontMap`   – font-name → path map (caller owns, may be empty).
// `setFont`   – called when a font change is requested.
// `setBlur`   – called when blur settings change.
void applyRuntimeSideEffects(const nlohmann::json& config,
                             const std::unordered_map<std::string, std::string>& fontMap,
                             SetFontFn setFont,
                             SetBlurFn setBlur);

// Scan for speaker embedding .npz files under the resources voice dir.
std::vector<std::string> scanSpeakerEmbeddings();

// Scan for TTF/OTF fonts under the resources dir.
// Populates `outFontMap` (name → absolute path) and returns sorted name list.
std::vector<std::string> scanFonts(std::unordered_map<std::string, std::string>& outFontMap);

} // namespace Settings
