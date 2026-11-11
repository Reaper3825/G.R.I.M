#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "MMO/Core/HardwareInventoryJson.hpp"
#include "commands/command_result.hpp"
#include "ui/world_lab/live_world_session.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

extern GRIM::MMO::HardwareInventory g_hardwareInventory;
// v3.19 consumes a reserved slot; prior service offsets and total ABI size stay fixed.
static_assert(sizeof(GrimPluginAPI) == offsetof(GrimPluginAPI, application_service) + 14 * sizeof(void*));
static_assert(offsetof(GrimPluginAPI, register_hardware_resource) ==
              offsetof(GrimPluginAPI, application_service) + sizeof(void*));
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
    const auto resourceFor = [&](const std::string& provider, const std::string& id = "system") {
        for (const auto& group : registry.snapshot().providers)
            if (group.provider.name == provider)
                for (const auto& resource : group.resources)
                    if (resource.resource_id == id) return RegisteredResource{resource, group.provider};
        return RegisteredResource{};
    };
    const auto resourcesFor = [&](const std::string& provider) {
        std::vector<RegisteredResource> result;
        for (const auto& group : registry.snapshot().providers)
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
    check(denied->register_hardware_resource(denied) == GRIM_ERROR_PERMISSION_DENIED &&
          denied->register_hardware_resource(nullptr) == GRIM_ERROR_INVALID_PARAM,
          "Hardware resource registration bypassed identity/permission guard");
    PluginAPI::cleanupPluginContext(denied);
    auto* deniedStage = PluginAPI::createPluginAPI("hardware-denied-stage", "", 0, true);
    check(deniedStage->register_hardware_resource(deniedStage) == GRIM_ERROR_PERMISSION_DENIED &&
          !PluginAPI::commitPluginTools(deniedStage) && detectionCalls == 0,
          "Denied hardware declaration did not reject staged provider");
    PluginAPI::cleanupPluginContext(deniedStage);
    auto* abandoned = PluginAPI::createPluginAPI("hardware-abandoned", "", GRIM_PERM_SYSTEM, true);
    check(abandoned->register_hardware_resource(abandoned) == GRIM_OK && detectionCalls == 0 &&
          resourceFor("hardware-abandoned").resource_id.empty(), "Staging captured or published hardware");
    PluginAPI::cleanupPluginContext(abandoned);
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

    // Host-only provider replacement: old contexts cannot refresh a replacement's resource.
    auto* first = PluginAPI::createPluginAPI("hardware-fixture", "", GRIM_PERM_SYSTEM, true);
    struct ContextCleanup {
        const GrimPluginAPI* first;
        const GrimPluginAPI* second = nullptr;
        ~ContextCleanup() {
            if (second) { PluginAPI::beginPluginShutdown(second); PluginAPI::cleanupPluginContext(second); }
            PluginAPI::beginPluginShutdown(first); PluginAPI::cleanupPluginContext(first);
        }
    } contexts{first};
    detectionMode = 1;
    check(first->register_hardware_resource(first) == GRIM_OK &&
          first->register_hardware_resource(first) == GRIM_ERROR_ALREADY_EXISTS &&
          PluginAPI::commitPluginTools(first), "Cannot commit hardware resource provider");
    check(resourcesFor("hardware-fixture").size() == 3 &&
          resourceFor("hardware-fixture").state.at("data").is_null() &&
          resourceFor("hardware-fixture").state.at("capture_time_ns").is_null() &&
          resourceFor("hardware-fixture").state.at("stale") == true,
          "Initial capture failure fabricated hardware or omitted stale state");
    detectionMode = 0;
    const auto* recovered = first->get_hardware_snapshot(first);
    check(recovered != nullptr, "Initial hardware failure did not recover");
    first->string_free(first, recovered);
    const auto firstResource = resourceFor("hardware-fixture");
    check(resourcesFor("hardware-fixture").size() == 7 &&
          firstResource.state.at("data").at("os_name") == "Fixture OS" && !firstResource.state.at("stale").get<bool>() &&
          first->register_hardware_resource(first) == GRIM_ERROR_UNSUPPORTED,
          "Hardware initial state or initialization-only registration incorrect");
    contexts.second = PluginAPI::createPluginAPI("hardware-fixture", "", GRIM_PERM_SYSTEM, true, first);
    check(contexts.second->register_hardware_resource(contexts.second) == GRIM_OK &&
          PluginAPI::commitPluginTools(contexts.second, first), "Cannot replace hardware provider");
    const auto secondResource = resourceFor("hardware-fixture");
    const auto afterReplacement = detectionCalls;
    check(secondResource.provider.instance != firstResource.provider.instance &&
          first->get_hardware_snapshot(first) == nullptr && detectionCalls == afterReplacement &&
          resourceFor("hardware-fixture") == secondResource, "Retired provider refreshed replacement hardware");
    PluginAPI::beginPluginShutdown(first);
    check(resourceFor("hardware-fixture") == secondResource, "Old shutdown removed replacement hardware");
    PluginAPI::beginPluginShutdown(contexts.second);
    check(resourcesFor("hardware-fixture").empty(), "Shutdown retained hardware resources");

    fs::copy_file(pluginDirectory / "system_info_plugin.dll", folder / "system_info_plugin.dll");
    const auto beforeLoad = detectionCalls;
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 1 && detectionCalls == beforeLoad + 1,
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
    GRIM::WorldLab::LiveWorldSession live(registry);
    // Room paths are derived from provider names; resolve rather than assume their spelling.
    live.command("go " + live.world().providerPath(GRIM::WorldLab::providerKey(initialResource.provider)));
    const auto beforeInspection = detectionCalls;
    for (const auto& resource : initialResources) {
        live.command("inspect-resource " + resource.resource_id);
        check(live.history().back().state.at("resource_inspection").at("state") == resource.state &&
              live.history().back().feedback.find("Status: Current") != std::string::npos,
              "Live world did not expose organized resource state");
    }
    check(live.observation().at("resources").size() == 7 && detectionCalls == beforeInspection,
          "Live inspection omitted objects or triggered hardware capture");
    const auto descriptor = registry.getTool("system_info");
    check(descriptor && descriptor->provider.name == "System Information Plugin" && descriptor->provider.type == ToolProviderType::Plugin &&
          descriptor->permission_bits == GRIM_PERM_SYSTEM && descriptor->is_informational && !descriptor->needs_confirmation &&
          descriptor->category == "information" && descriptor->parameters.empty(),
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
    detectedInventory.os_name = "Updated OS";
    check(snapshot().at("hardware").at("os_name") == "Updated OS" && hardware.at("os_name") == "Fixture OS",
          "Hardware snapshots were stale or shared mutable storage");
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
    const auto beforeRemoval = detectionCalls;
    fs::remove(folder / "system_info_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline && !registry.isRegistered("system_info") && detectionCalls == beforeRemoval &&
          g_hardwareInventory.os_name == "startup cache" && resourcesFor("System Information Plugin").empty(),
          "Plugin removal retained tool/resource or mutated hardware state");
    std::cout << "System information plugin topology, permissions, exact results, detection errors, and removal tests passed\n";
}
