#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "perception/perception_queries.hpp"
#include "commands/command_result.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

// Controlled host observation services. The real DLL, API permission/exception
// boundary, registry, and result writer are tested without screen/model access.
namespace {
bool available = false;
int availabilityCalls = 0, failureMode = 0;
std::array<int, 3> observationCalls{};
std::array<std::string, 3> observationText{};
std::array<int, 4> lastRegion{};
void failIfRequested() {
    if (failureMode == 1) throw std::runtime_error("fixture perception failure");
    if (failureMode == 2) throw 42;
}
}
namespace GRIM::Perception {
bool isAvailable() { ++availabilityCalls; if (failureMode == 3) throw std::runtime_error("fixture availability failure"); return available; }
std::string analyzeScreen() { ++observationCalls[0]; failIfRequested(); return observationText[0]; }
std::string readText(int x, int y, int width, int height) {
    ++observationCalls[1]; lastRegion = {x, y, width, height}; failIfRequested(); return observationText[1];
}
std::string detectObjects() { ++observationCalls[2]; failIfRequested(); return observationText[2]; }
}

void runPerceptionPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    available = false; availabilityCalls = failureMode = 0; observationCalls = {};
    auto* denied = PluginAPI::createPluginAPI("perception-denied", "", 0);
    check(denied->query_perception(denied, GRIM_PERCEPTION_READ_TEXT) == nullptr && availabilityCalls == 0 && observationCalls == std::array<int, 3>{},
          "Perception permission denial called host observations");
    check(denied->query_perception(nullptr, GRIM_PERCEPTION_READ_TEXT) == nullptr, "Perception accepted invalid API identity");
    PluginAPI::cleanupPluginContext(denied);
    auto* reader = PluginAPI::createPluginAPI("perception-reader", "", GRIM_PERM_SYSTEM);
    check(reader->query_perception(reader, static_cast<GrimPerceptionQuery>(99)) == nullptr && availabilityCalls == 0,
          "Invalid observation query called host perception");
    PluginAPI::cleanupPluginContext(reader);
    const auto folder = fs::temp_directory_path() / ("grim-perception-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create perception fixture directory");
    struct Cleanup {
        fs::path folder;
        ~Cleanup() {
            PluginManager::unloadAll(); failureMode = 0;
            std::error_code error;
            fs::remove(folder / "perception_plugin.dll", error); fs::remove(folder, error);
        }
    } cleanup{folder};
    fs::copy_file(pluginDirectory / "perception_plugin.dll", folder / "perception_plugin.dll");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 3 && availabilityCalls == 0, "Perception DLL load probed/initialized host state or failed registration");
    const char* ids[] = {"perception_what_see", "perception_read_text", "perception_detect_objects"};
    for (const auto* id : ids) {
        const auto d = registry.getTool(id);
        check(d && d->provider_name == "Perception Plugin" && d->provider_type == ToolProviderType::Plugin &&
              d->permission_bits == GRIM_PERM_SYSTEM && d->is_informational && !d->needs_confirmation &&
              d->parameters.empty() && d->examples.size() == 1 && d->preconditions.size() == 1,
              "Perception metadata/provider/policy incorrect");
        const auto unavailableResult = registry.invoke(id, "ignored arguments");
        check(!unavailableResult.success && unavailableResult.message == "[Perception] System not initialized" &&
              unavailableResult.errorCode == "ERR_PERCEPTION_NOT_INITIALIZED" && unavailableResult.category == "error" &&
              unavailableResult.voice == "Perception system not available" && unavailableResult.color.toUInt() == Colors::Red.toUInt() &&
              unavailableResult.structuredOutputJson.empty(), "Unavailable perception result changed");
    }
    check(availabilityCalls == 3 && observationCalls == std::array<int, 3>{}, "Unavailable commands ran observations or cached availability");
    available = true;
    observationText = {"Screen analysis\nSecond line", "OCR text: \"quoted\" [brackets] \\ path", "button at (10, 20)"};
    const auto matches = [&](const CommandResult& r, const std::string& text) {
        check(r.success && r.message == text && r.voice == text && r.errorCode == "ERR_NONE" && r.category == "routine" &&
              r.color.toUInt() == Colors::Cyan.toUInt() && r.structuredOutputJson.empty(), "Perception result text/voice/status changed");
    };
    for (size_t i = 0; i < 3; ++i) {
        for (const auto* args : {"", "monitor 2 region 1 2 3 4"}) {
            const auto beforeCalls = observationCalls; const auto beforeAvailability = availabilityCalls;
            matches(registry.invoke(ids[i], args), observationText[i]);
            auto expectedCalls = beforeCalls; ++expectedCalls[i];
            check(observationCalls == expectedCalls && availabilityCalls == beforeAvailability + 1,
                  "Perception invoked wrong observation, repeated work, or changed availability checks");
        }
    }
    check(lastRegion == std::array<int, 4>{-1, -1, -1, -1}, "OCR defaults changed or ignored command arguments became active");
    for (size_t i = 0; i < 3; ++i) {
        observationText[i] = "";
        matches(registry.invoke(ids[i], ""), "");
        observationText[i] = "[Error] existing host observation failure";
        matches(registry.invoke(ids[i], ""), observationText[i]);
        observationText[i] = "Updated live context";
        matches(registry.invoke(ids[i], ""), observationText[i]);
    }
    observationText[1] = std::string(70000, 'x');
    matches(registry.invoke(ids[1], ""), observationText[1]);
    for (const auto* id : ids) {
        failureMode = 1;
        const auto standard = registry.invoke(id, "");
        check(!standard.success && standard.errorCode == "ERR_CMD_EXCEPTION" &&
              standard.message == std::string("[Error] Exception while running command: ") + id + ": fixture perception failure" &&
              standard.voice.empty() && standard.category.empty() && standard.color.toUInt() == Colors::Red.toUInt(),
              "Standard perception exception response changed");
        failureMode = 2;
        const auto unknown = registry.invoke(id, "");
        check(!unknown.success && unknown.errorCode == "ERR_CMD_EXCEPTION" &&
              unknown.message == std::string("[Error] Exception while running command: ") + id, "Unknown perception exception response changed");
    }
    failureMode = 3;
    const auto unavailableException = registry.invoke(ids[0], "");
    check(unavailableException.errorCode == "ERR_CMD_EXCEPTION" &&
          unavailableException.message == "[Error] Exception while running command: perception_what_see: fixture availability failure",
          "Availability exception response changed");
    failureMode = 0; available = false;
    check(registry.invoke(ids[1], "").errorCode == "ERR_PERCEPTION_NOT_INITIALIZED", "Perception cached stale availability");
    const auto beforeRemoval = observationCalls; const auto beforeAvailability = availabilityCalls;
    fs::remove(folder / "perception_plugin.dll"); PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && observationCalls == beforeRemoval && availabilityCalls == beforeAvailability && !available,
          "Removing perception DLL changed host state or retained registrations");
    for (const auto* id : ids) check(!registry.isRegistered(id), "Removed perception command remains registered");
    std::cout << "Perception plugin availability, routing, literal/large results, exceptions, permissions, and removal tests passed\n";
}
