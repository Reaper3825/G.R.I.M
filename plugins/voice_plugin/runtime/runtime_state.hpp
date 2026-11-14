#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <atomic>
namespace VoiceRuntime {
extern nlohmann::json config;
extern std::atomic<bool> cancel;
std::string root();
void reportError(const std::string& error);
void transcript(const std::string& text);
nlohmann::json snapshot();
void reset();
}
inline std::string getGrimRootDir() { return VoiceRuntime::root(); }
inline std::string getResourcePath() { return VoiceRuntime::root() + "/resources"; }
