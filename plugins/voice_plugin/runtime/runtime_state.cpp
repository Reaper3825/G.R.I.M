#include "runtime_state.hpp"
#include <mutex>
#include <deque>
namespace VoiceRuntime {
nlohmann::json config = nlohmann::json::object();
std::atomic<bool> cancel{false};
namespace { std::mutex mutex; std::deque<nlohmann::json> events; uint64_t sequence=0; }
std::string root() { return config.at("root").get<std::string>(); }
void reportError(const std::string& error) {
 std::lock_guard<std::mutex> lock(mutex);
 events.push_back({{"sequence",++sequence},{"type",error == "wake_detected" ? "wake" : "error"},{"message",error}});
 if(events.size()>64) events.pop_front();
}
void transcript(const std::string& text) {
 std::lock_guard<std::mutex> lock(mutex);
 events.push_back({{"sequence",++sequence},{"type","transcript"},{"text",text}});
 if(events.size()>64) events.pop_front();
}
nlohmann::json snapshot() { std::lock_guard<std::mutex> lock(mutex); return {{"sequence",sequence},{"events",events}}; }
void reset() { std::lock_guard<std::mutex> lock(mutex); events.clear(); sequence=0; config=nlohmann::json::object(); cancel=false; }
}
