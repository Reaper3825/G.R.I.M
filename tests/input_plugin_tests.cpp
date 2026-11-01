#include "core/plugin_api_impl.hpp"
#include "core/plugin_manager.hpp"
#include "perception/input_command_service.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/commands_core.hpp"
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
CommandResult fixture;
GrimInputOperation observedOperation = GRIM_INPUT_MOVE_MOUSE;
std::string observedArguments;
int calls = 0, exceptionMode = 0;
std::mutex gateMutex;
std::condition_variable gateChanged;
bool block = false, entered = false, released = false;
nlohmann::json take(const GrimPluginAPI* api, const char* raw) {
    check(raw != nullptr, "Input service returned null");
    struct Release { const GrimPluginAPI* api; const char* raw; ~Release() { api->string_free(api, raw); } } release{api, raw};
    return nlohmann::json::parse(raw);
}
void same(const CommandResult& actual, const CommandResult& expected) {
    check(actual.success == expected.success && actual.message == expected.message && actual.errorCode == expected.errorCode &&
          actual.category == expected.category && actual.voice == expected.voice &&
          actual.structuredOutputJson == expected.structuredOutputJson &&
          actual.color.r == expected.color.r && actual.color.g == expected.color.g && actual.color.b == expected.color.b,
          "Input plugin changed a complete host result");
}
}

namespace GRIM::Perception {
// The loader/API/registry/plugin are real; all native input is replaced with this sink.
CommandResult ExecuteInputCommandService(GrimInputOperation operation, const std::string& arguments) {
    ++calls; observedOperation = operation; observedArguments = arguments;
    if (block) {
        std::unique_lock lock(gateMutex); entered = true; gateChanged.notify_all();
        gateChanged.wait(lock, [] { return released; });
    }
    if (exceptionMode == 1) throw std::runtime_error("fixture input exception");
    if (exceptionMode == 2) throw 42;
    return fixture;
}
}

void runInputPluginTests(const std::filesystem::path& directory) {
    static_assert(sizeof(GrimPluginAPI) == offsetof(GrimPluginAPI, input_service) + 19 * sizeof(void*));
    auto* denied = PluginAPI::createPluginAPI("input-denied", "", GRIM_PERM_UI | GRIM_PERM_SYSTEM);
    check(!denied->input_service(denied, GRIM_INPUT_CLICK, "", 0), "UI permission unexpectedly authorized native input");
    PluginAPI::cleanupPluginContext(denied);
    auto* inputOnly = PluginAPI::createPluginAPI("input-only", "", GRIM_PERM_INPUT);
    check(!inputOnly->input_service(inputOnly, GRIM_INPUT_CLICK_ON, "Submit", 6), "Targeting did not require perception permission");
    fixture = {true, "sink", "ERR_NONE", "routine", "sink voice", Colors::Green};
    take(inputOnly, inputOnly->input_service(inputOnly, GRIM_INPUT_CLICK, nullptr, 0));
    check(observedArguments.empty(), "Empty raw input was changed");
    check(!inputOnly->input_service(inputOnly, GRIM_INPUT_TYPE, nullptr, 1) &&
          !inputOnly->input_service(inputOnly, static_cast<GrimInputOperation>(999), "", 0) &&
          !inputOnly->input_service(inputOnly, GRIM_INPUT_TYPE, "x", 65537), "Invalid input API arguments accepted");
    PluginAPI::cleanupPluginContext(inputOnly);
    auto* staging = PluginAPI::createPluginAPI("input-staging", "", GRIM_PERM_INPUT | GRIM_PERM_SYSTEM, true);
    const auto before = calls;
    check(!staging->input_service(staging, GRIM_INPUT_CLICK, "left", 4) && calls == before,
          "Plugin initialization was able to execute input");
    PluginAPI::cleanupPluginContext(staging);
    if (directory.empty()) return;

    namespace fs = std::filesystem;
    const auto source = fs::absolute(directory) / "input_plugin.dll";
    const auto folder = fs::absolute(directory).parent_path() / "input-plugin-test-deployment";
    fs::create_directories(folder);
    fs::copy_file(source, folder / "input_plugin.dll", fs::copy_options::overwrite_existing);
    auto& registry = GRIM::MMO::ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 5, "Input DLL did not register five tools");
    const char* ids[] = {"move_mouse", "click", "type", "key", "click_on"};
    const char* arguments[] = {"-1200, 300 trailing", "", "  raw text\n", "ctrl + c", "object save-button"};
    for (int i = 0; i < 5; ++i) {
        const auto descriptor = registry.getTool(ids[i]);
        check(descriptor && descriptor->provider_name == "Input Plugin" && descriptor->provider_type == GRIM::MMO::ToolProviderType::Plugin &&
              descriptor->needs_confirmation && !descriptor->is_informational && descriptor->argument_format == "raw_text" &&
              descriptor->permission_bits == (GRIM_PERM_INPUT | (i == 4 ? GRIM_PERM_SYSTEM : 0)) &&
              descriptor->parameters.size() == 1 && !descriptor->examples.empty(), "Input metadata/policy incomplete");
        fixture = {true, "complete result " + std::to_string(i), "ERR_NONE", "routine", "voice", Colors::Cyan};
        fixture.structuredOutputJson = R"({"input_sent":true,"application_response_verified":false,"desktop_x":-123})";
        same(dispatchCommand(ids[i], arguments[i]), fixture);
        check(static_cast<int>(observedOperation) == i && observedArguments == arguments[i], "Input operation/raw arguments changed");
        fixture = {false, "ambiguous/stale/dispatch failure", "ERR_INPUT_TARGET_AMBIGUOUS", "error", "failed", Colors::Red};
        same(dispatchCommand(ids[i], arguments[i]), fixture);
    }
    fixture.message.assign(70000, 'x');
    same(dispatchCommand("type", "long result"), fixture);
    const std::string bytes("a\0b", 3);
    dispatchCommand("type", bytes);
    check(observedArguments == bytes, "Length-delimited arguments were truncated at NUL");
    exceptionMode = 1;
    check(dispatchCommand("click", "left").errorCode == "ERR_CMD_EXCEPTION", "Standard host exception changed dispatch policy");
    exceptionMode = 2;
    check(dispatchCommand("click", "left").errorCode == "ERR_CMD_EXCEPTION", "Unknown host exception changed dispatch policy");
    exceptionMode = 0;

    // In-flight host input must finish before its plugin/API can retire.
    fixture = {true, "finished", "ERR_NONE", "routine", "", Colors::Green};
    block = true; entered = released = false;
    auto invoking = std::async(std::launch::async, [] { return dispatchCommand("click", "left"); });
    {
        std::unique_lock lock(gateMutex);
        check(gateChanged.wait_for(lock, std::chrono::seconds(5), [] { return entered; }), "Blocked input never entered host service");
    }
    auto unloading = std::async(std::launch::async, [] { PluginManager::unloadAll(); });
    const bool draining = unloading.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout;
    { std::lock_guard lock(gateMutex); released = true; }
    gateChanged.notify_all();
    check(invoking.get().success, "In-flight input result was lost");
    unloading.get(); block = false;
    check(draining && registry.toolCount() == baseline, "Input plugin did not drain/unregister on unload");
    PluginManager::initialize(folder.string());
    check(registry.toolCount() == baseline + 5, "Input plugin did not reload from file");
    fs::remove(folder / "input_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline, "Deleted input plugin retained registrations");
    PluginManager::unloadAll();
    std::cout << "Input plugin results, permissions, metadata, exceptions, active-call draining and file-removal tests passed\n";
}
