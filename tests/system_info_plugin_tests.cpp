#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
#include "MMO/Core/HardwareInventoryJson.hpp"
#include "commands/command_result.hpp"
#include "ui/world_lab/live_world_session.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <windows.h>

extern GRIM::MMO::HardwareInventory g_hardwareInventory;

namespace {
GRIM::MMO::HardwareInventory detectedInventory;
int detectionMode = 0;
std::string setting(const char* key) {
    const auto size = GetEnvironmentVariableA(key, nullptr, 0);
    if (!size) return {};
    std::string value(size, '\0'); GetEnvironmentVariableA(key, value.data(), size); value.pop_back(); return value;
}
int detectionCalls() { return std::stoi(setting("GRIM_SYSTEM_INFO_TEST_CALLS")); }
void prepareDetection() {
    const auto json = GRIM::MMO::hardwareInventoryJson(detectedInventory).dump();
    SetEnvironmentVariableA("GRIM_SYSTEM_INFO_TEST_INVENTORY", json.c_str());
    SetEnvironmentVariableA("GRIM_SYSTEM_INFO_TEST_MODE", std::to_string(detectionMode).c_str());
}
struct FixtureEnvironment {
    const char* keys[3] = {"GRIM_SYSTEM_INFO_TEST_INVENTORY", "GRIM_SYSTEM_INFO_TEST_MODE", "GRIM_SYSTEM_INFO_TEST_CALLS"};
    std::string previous[3];
    FixtureEnvironment() {
        for (int i = 0; i < 3; ++i) previous[i] = setting(keys[i]);
        SetEnvironmentVariableA(keys[2], "0");
    }
    ~FixtureEnvironment() {
        for (int i = 0; i < 3; ++i) SetEnvironmentVariableA(keys[i], previous[i].empty() ? nullptr : previous[i].c_str());
    }
};
}

void runSystemInfoPluginTests(const std::filesystem::path& pluginDirectory) {
    FixtureEnvironment environment;
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto resourceFor = [&](const std::string& provider, const std::string& id = "system") {
        for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if (group.provider.name == provider)
                for (const auto& resource : group.resources)
                    if (resource.resource_id == id) return RegisteredResource{resource, group.provider};
        return RegisteredResource{};
    };
    const auto resourcesFor = [&](const std::string& provider) {
        std::vector<RegisteredResource> result;
        for (const auto& group : GRIM::MMO::ProviderRegistry::instance().snapshot().providers)
            if (group.provider.name == provider)
                for (const auto& resource : group.resources) result.push_back({resource, group.provider});
        return result;
    };
    const auto folder = fs::temp_directory_path() / ("grim-system-info-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create system information test folder");
    struct Cleanup {
        fs::path folder;
        HardwareInventory startup = g_hardwareInventory;
        ~Cleanup() {
            PluginManager::unloadAll();
            g_hardwareInventory = startup;
            detectionMode = 0;
            std::error_code error;
            fs::remove(folder / "system_info_plugin.dll", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    detectionMode = 0;
    detectedInventory = {};
    detectedInventory.capture_time = std::chrono::steady_clock::time_point{std::chrono::seconds(123)};
    detectedInventory.os_name = "Fixture OS";
    detectedInventory.arch = "fixture-arch";
    detectedInventory.cpu_cores = 12;
    detectedInventory.ram_total_mb = 32768;
    detectedInventory.gpu_count = 2;
    detectedInventory.gpus = {{0, "Fixture GPU", 8192, "driver-one", true, true, true},
                             {1, "Second GPU", 4096, "driver-two", false, false, false}};
    detectedInventory.monitor_count = 2;
    detectedInventory.virtual_origin_x = -1920;
    detectedInventory.virtual_origin_y = -200;
    detectedInventory.virtual_desktop_w = 3840;
    detectedInventory.virtual_desktop_h = 1280;
    detectedInventory.monitors = {{-1920, -200, 1920, 1080, false}, {0, 0, 1920, 1080, true}};
    detectedInventory.audio_output_device = "Fixture output";
    detectedInventory.audio_input_device = "Fixture input";
    detectedInventory.wifi_connected = true;
    detectedInventory.has_sapi = true;
    detectedInventory.has_say = true;
    detectedInventory.has_piper = true;
    detectedInventory.suggested_whisper_model = "base.en";
    g_hardwareInventory.os_name = "startup cache";
    const auto hardware = hardwareInventoryJson(detectedInventory);
    check(hardwareInventoryJson(hardwareInventoryFromJson(hardware)) == hardware, "Hardware codec lost data");
    fs::copy_file(pluginDirectory / "system_info_plugin.dll", folder / "system_info_plugin.dll");
    const auto beforeLoad = detectionCalls();
    prepareDetection();
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 1 && detectionCalls() == beforeLoad + 1,
          "Plugin commit did not capture hardware exactly once or failed registration");
    const auto initialResource = resourceFor("System Information Plugin");
    const auto initialResources = resourcesFor("System Information Plugin");
    check(initialResource.kind == "system_info.snapshot" && initialResource.runtime_id != 0 &&
          initialResource.state.at("data").at("wifi_connected") == true &&
          initialResource.state.at("data").at("desktop").at("origin_x") == -1920 &&
          initialResource.state.at("data").at("desktop").at("height") == 1280 &&
          initialResources.size() == 7 &&
          resourceFor("System Information Plugin", "hardware").resource_id.empty(),
          "Plugin hardware resource missing or incorrect");
    check(resourceFor("System Information Plugin", "cpu").state.at("data") == nlohmann::json{{"cpu_cores", 12}} &&
          resourceFor("System Information Plugin", "memory").state.at("data") == nlohmann::json{{"ram_total_mb", 32768}} &&
          resourceFor("System Information Plugin", "gpu_1").state.at("data").at("driver_version") == "driver-two" &&
          resourceFor("System Information Plugin", "monitor_0").state.at("data").at("x") == -1920 &&
          resourceFor("System Information Plugin", "monitor_1").state.at("data").at("is_primary") == true,
          "Organized resources omitted CPU, memory, GPU, or monitor data");
    for (const auto& resource : initialResources) {
        check(resource.state.at("format") == "grim-system-info-resource" && resource.state.at("version") == 1 &&
              resource.state.at("capture_time_ns") == 123000000000LL && !resource.state.at("stale").get<bool>() &&
              resource.state.at("error").is_null(),
              "Organized resource capture metadata inconsistent");
        for (const char* removed : {"audio_output_device", "audio_input_device", "has_sapi", "has_say",
                                    "has_piper", "suggested_whisper_model"})
            check(!resource.state.at("data").contains(removed), "Resource exposed removed audio/voice/model fields");
    }
    GRIM::WorldLab::LiveWorldSession live(ProviderRegistry::instance());
    // Room paths are derived from provider names; resolve rather than assume their spelling.
    live.command("go " + live.world().providerPath(GRIM::WorldLab::providerKey(initialResource.provider)));
    const auto beforeInspection = detectionCalls();
    for (const auto& resource : initialResources) {
        live.command("inspect-resource " + resource.resource_id);
        check(live.history().back().state.at("resource_inspection").at("state") == resource.state &&
              live.history().back().feedback.find("Status: Current") != std::string::npos,
              "Live world did not expose organized resource state");
    }
    check(live.observation().at("resources").size() == 7 && detectionCalls() == beforeInspection,
          "Live inspection omitted objects or triggered hardware capture");
    const auto descriptor = registry.getTool("system_info");
    check(descriptor && descriptor->provider.name == "System Information Plugin" && descriptor->provider.type == ToolProviderType::Plugin &&
          descriptor->permission_bits == GRIM_PERM_SYSTEM && descriptor->is_informational && !descriptor->needs_confirmation &&
          descriptor->category == "information" && descriptor->parameters.empty(),
          "System information metadata/policy incorrect");
    const auto invoke = [&](const std::string& args = "") { prepareDetection(); return registry.invoke("system_info", args); };
    const auto matches = [&](const CommandResult& r, const std::string& text) {
        check(r.success && r.message == text && r.errorCode == "ERR_NONE" && r.category == "summary" &&
              r.voice == "System information shown" && r.color.toUInt() == Colors::Cyan.toUInt() && r.structuredOutputJson.empty(),
              "System information result differs from original command");
    };
    const std::string header = "[System Info]\nOS         : Fixture OS (fixture-arch)\nCPU Cores  : 12\nRAM        : 32768 MB\n";
    const auto beforeCommand = detectionCalls();
    matches(invoke(), header + "GPU        : Fixture GPU (2 device(s))\nCUDA       : Supported\nMetal      : Supported\nROCm       : Supported\nSuggested Whisper model: base.en\n");
    check(detectionCalls() == beforeCommand + 1, "system_info did not perform exactly one fresh detection");
    detectedInventory.gpu_count = 0;
    // Preserve gpu_count-based branch even when backend flags remain in the vector.
    matches(invoke("ignored arguments"), header + "GPU        : None detected\nSuggested Whisper model: base.en\n");
    detectedInventory.gpu_count = 1;
    detectedInventory.gpus.clear();
    matches(invoke(), header + "GPU        : Unknown (1 device(s))\nSuggested Whisper model: base.en\n");
    const auto refreshedResource = resourceFor("System Information Plugin");
    check(refreshedResource.provider.instance == initialResource.provider.instance &&
          refreshedResource.runtime_id == initialResource.runtime_id &&
          resourceFor("System Information Plugin", "gpu_0").resource_id.empty() &&
          resourceFor("System Information Plugin", "gpu_1").resource_id.empty() &&
          resourcesFor("System Information Plugin").size() == 5,
          "system_info changed object identity or retained removed GPUs");
    const auto lastGoodResources = resourcesFor("System Information Plugin");
    live.synchronize();
    check(live.observation().at("resources").size() == 5, "Live world retained removed device objects");
    detectionMode = 1;
    const auto failed = invoke();
    check(!failed.success && failed.errorCode == "ERR_CMD_EXCEPTION" &&
          failed.message == "[Error] Exception while running command: system_info: fixture detection failure" &&
          failed.voice.empty() && failed.category.empty() && failed.color.toUInt() == Colors::Red.toUInt(),
          "Standard detection exception behavior changed");
    const auto staleResource = resourceFor("System Information Plugin");
    check(staleResource.state.at("data") == refreshedResource.state.at("data") &&
          staleResource.state.at("stale") == true &&
          staleResource.state.at("error").at("message") == "fixture detection failure",
          "Capture failure discarded last good hardware or concealed staleness");
    for (const auto& previous : lastGoodResources) {
        const auto stale = resourceFor("System Information Plugin", previous.resource_id);
        check(stale.runtime_id == previous.runtime_id && stale.state.at("data") == previous.state.at("data") &&
              stale.state.at("capture_time_ns") == previous.state.at("capture_time_ns") &&
              stale.state.at("stale") == true && stale.state.at("error").at("kind") == "exception",
              "Capture failure lost objects or left inconsistent capture metadata");
    }
    detectionMode = 2;
    const auto unknown = invoke();
    check(!unknown.success && unknown.errorCode == "ERR_CMD_EXCEPTION" &&
          unknown.message == "[Error] Exception while running command: system_info", "Unknown detection exception behavior changed");
    detectionMode = 0;
    detectedInventory.monitors.clear();
    detectedInventory.monitor_count = 0;
    invoke();
    check(resourceFor("System Information Plugin").state.at("stale") == false &&
          resourceFor("System Information Plugin").state.at("error").is_null() &&
          resourcesFor("System Information Plugin").size() == 3 &&
          resourceFor("System Information Plugin", "monitor_0").resource_id.empty(),
          "Successful capture did not clear stale/error state or remove monitors");
    // Re-enumerating the same GPU index restores its key; ordering is not identity.
    detectedInventory.gpus = {{1, "Renamed GPU", 4096, "new-driver", false, false, false}};
    invoke();
    check(resourceFor("System Information Plugin", "gpu_1").state.at("data").at("name") == "Renamed GPU" &&
          resourceFor("System Information Plugin", "gpu_0").resource_id.empty(),
          "GPU identity followed vector position instead of device index");
    detectionMode = 1;
    prepareDetection();
    const auto oldInstance = resourceFor("System Information Plugin").provider.instance;
    fs::last_write_time(folder / "system_info_plugin.dll",
        fs::last_write_time(folder / "system_info_plugin.dll") + std::chrono::seconds(2));
    PluginManager::checkForHotReload();
    const auto replacement = resourceFor("System Information Plugin");
    check(replacement.provider.instance != oldInstance && resourcesFor("System Information Plugin").size() == 3 &&
          replacement.state.at("stale") == true && replacement.state.at("data").is_null() &&
          replacement.state.at("capture_time_ns").is_null(), "Reload reused old hardware state or hid initial failure");
    detectionMode = 0;
    check(invoke().success && !resourceFor("System Information Plugin").state.at("stale").get<bool>() &&
          resourceFor("System Information Plugin", "gpu_1").state.at("data").at("name") == "Renamed GPU",
          "Replacement did not recover its own hardware state");
    const auto beforeRemoval = detectionCalls();
    fs::remove(folder / "system_info_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("system_info") && detectionCalls() == beforeRemoval &&
          g_hardwareInventory.os_name == "startup cache" && resourcesFor("System Information Plugin").empty(),
          "Plugin removal retained tool/resource or mutated hardware state");
    std::cout << "System information plugin topology, permissions, exact results, detection errors, and removal tests passed\n";
}
