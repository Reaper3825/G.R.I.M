#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "MMO/Core/HardwareInventoryJson.hpp"
#include "commands/command_result.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

extern GRIM::MMO::HardwareInventory g_hardwareInventory;
namespace {
GRIM::MMO::HardwareInventory detectedInventory;
int detectionCalls = 0, detectionMode = 0;
}
namespace GRIM::MMO {
HardwareInventory detectHardware() {
    ++detectionCalls;
    if (detectionMode == 1) throw std::runtime_error("fixture detection failure");
    if (detectionMode == 2) throw 42;
    return detectedInventory;
}
}

void runSystemInfoPluginTests(const std::filesystem::path& pluginDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    const auto folder = fs::temp_directory_path() / ("grim-system-info-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create system information test folder");
    struct Cleanup {
        fs::path folder;
        HardwareInventory startup = g_hardwareInventory;
        GrimPluginAPI* api = nullptr;
        ~Cleanup() {
            PluginManager::unloadAll();
            if (api) PluginAPI::cleanupPluginContext(api);
            g_hardwareInventory = startup;
            detectionMode = 0;
            std::error_code error;
            fs::remove(folder / "system_info_plugin.dll", error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    detectionCalls = detectionMode = 0;
    detectedInventory = {};
    auto* denied = PluginAPI::createPluginAPI("hardware-denied", "", 0);
    check(denied->get_hardware_snapshot(denied) == nullptr && detectionCalls == 0 &&
          denied->get_hardware_snapshot(nullptr) == nullptr, "Hardware permission/identity guard failed");
    PluginAPI::cleanupPluginContext(denied);
    cleanup.api = PluginAPI::createPluginAPI("hardware-reader", "", GRIM_PERM_SYSTEM);
    auto* reader = cleanup.api;
    const auto snapshot = [&] {
        const char* text = reader->get_hardware_snapshot(reader);
        check(text != nullptr, "Hardware snapshot failed");
        const std::string copy(text);
        reader->string_free(reader, text);
        auto value = nlohmann::json::parse(copy);
        check(value.at("format") == "grim-hardware-snapshot" && value.at("version") == 1, "Hardware snapshot schema incorrect");
        return value;
    };
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
    const auto full = snapshot();
    const auto& hardware = full.at("hardware");
    check(full.at("error").is_null() && hardware.at("os_name") == "Fixture OS" &&
          hardware.at("capture_time_ns") == 123000000000LL && hardware.at("gpus")[1].at("driver_version") == "driver-two" &&
          hardware.at("monitors")[0].at("x") == -1920 && hardware.at("audio_input_device") == "Fixture input" &&
          hardware.at("virtual_origin_y") == -200 && hardware.at("wifi_connected") == true &&
          hardware.at("has_sapi") == true && hardware.at("has_say") == true && hardware.at("has_piper") == true,
          "Hardware snapshot omitted topology, provenance, or capability fields");
    const auto decoded = hardwareInventoryFromJson(hardware);
    check(hardwareInventoryJson(decoded) == hardware && decoded.capture_time == detectedInventory.capture_time,
          "Hardware snapshot codec lost data");
    check(g_hardwareInventory.os_name == "startup cache", "Fresh snapshot mutated startup inventory");

    fs::copy_file(pluginDirectory / "system_info_plugin.dll", folder / "system_info_plugin.dll");
    const auto beforeLoad = detectionCalls;
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 1 && detectionCalls == beforeLoad, "Plugin load detected hardware or failed registration");
    const auto descriptor = registry.getTool("system_info");
    check(descriptor && descriptor->provider_name == "System Information Plugin" && descriptor->provider_type == ToolProviderType::Plugin &&
          descriptor->permission_bits == GRIM_PERM_SYSTEM && descriptor->is_informational && !descriptor->needs_confirmation &&
          descriptor->category == "information" && descriptor->parameters.empty() && descriptor->examples.size() == 1,
          "System information metadata/policy incorrect");
    const auto invoke = [&](const std::string& args = "") { return registry.invoke("system_info", args); };
    const auto matches = [&](const CommandResult& r, const std::string& text) {
        check(r.success && r.message == text && r.errorCode == "ERR_NONE" && r.category == "summary" &&
              r.voice == "System information shown" && r.color.toUInt() == Colors::Cyan.toUInt() && r.structuredOutputJson.empty(),
              "System information result differs from original command");
    };
    const std::string header = "[System Info]\nOS         : Fixture OS (fixture-arch)\nCPU Cores  : 12\nRAM        : 32768 MB\n";
    const auto beforeCommand = detectionCalls;
    matches(invoke(), header + "GPU        : Fixture GPU (2 device(s))\nCUDA       : Supported\nMetal      : Supported\nROCm       : Supported\nSuggested Whisper model: base.en\n");
    check(detectionCalls == beforeCommand + 1, "system_info did not perform exactly one fresh detection");
    detectedInventory.gpu_count = 0;
    // Preserve gpu_count-based branch even when backend flags remain in the vector.
    matches(invoke("ignored arguments"), header + "GPU        : None detected\nSuggested Whisper model: base.en\n");
    detectedInventory.gpu_count = 1;
    detectedInventory.gpus.clear();
    matches(invoke(), header + "GPU        : Unknown (1 device(s))\nSuggested Whisper model: base.en\n");
    detectedInventory.os_name = "Updated OS";
    check(snapshot().at("hardware").at("os_name") == "Updated OS" && hardware.at("os_name") == "Fixture OS",
          "Hardware snapshots were stale or shared mutable storage");
    detectionMode = 1;
    const auto failed = invoke();
    check(!failed.success && failed.errorCode == "ERR_CMD_EXCEPTION" &&
          failed.message == "[Error] Exception while running command: system_info: fixture detection failure" &&
          failed.voice.empty() && failed.category.empty() && failed.color.toUInt() == Colors::Red.toUInt(),
          "Standard detection exception behavior changed");
    detectionMode = 2;
    const auto unknown = invoke();
    check(!unknown.success && unknown.errorCode == "ERR_CMD_EXCEPTION" &&
          unknown.message == "[Error] Exception while running command: system_info", "Unknown detection exception behavior changed");
    detectionMode = 0;
    const auto beforeRemoval = detectionCalls;
    fs::remove(folder / "system_info_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("system_info") && detectionCalls == beforeRemoval &&
          g_hardwareInventory.os_name == "startup cache", "Plugin removal retained tool or mutated hardware state");
    std::cout << "System information plugin topology, permissions, exact results, detection errors, and removal tests passed\n";
}
