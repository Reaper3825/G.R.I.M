#include "core/plugin_api_impl.hpp"
#include "core/plugin_manager.hpp"
#include "core/input/input_service.hpp"
#include "MMO/Core/Registries/ToolRegistry.hpp"
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
bool available = true;
nlohmann::json primitiveRequest;
int primitiveEffects = 0;
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

namespace GRIM::Input {
bool ExecuteInputPrimitive(const nlohmann::json& request) {
    primitiveRequest = request;
    if (request.at("op") == "available") return available;
    ++primitiveEffects;
    if (block) {
        std::unique_lock lock(gateMutex); entered = true; gateChanged.notify_all();
        gateChanged.wait(lock, [] { return released; });
    }
    if (exceptionMode == 1) throw std::runtime_error("fixture input exception");
    if (exceptionMode == 2) throw 42;
    return available;
}

}

void runInputPluginTests(const std::filesystem::path& directory) {

    auto* denied = PluginAPI::createPluginAPI("input-denied", "", GRIM_PERM_UI | GRIM_PERM_SYSTEM);
    check(!denied->input_primitive(denied, "{\"op\":\"available\"}", 18), "Primitive bypassed INPUT permission");
    PluginAPI::cleanupPluginContext(denied);
    auto* staging = PluginAPI::createPluginAPI("input-staging", "", GRIM_PERM_INPUT | GRIM_PERM_SYSTEM, true);
    check(!staging->input_primitive(staging, "{\"op\":\"available\"}", 18), "Staging provider used primitive");
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
    check(registry.toolCount() == baseline + 4, "Input DLL did not register four tools");
    const char* ids[] = {"move_mouse", "click", "type", "key", "click_on"};
    for (int i = 0; i < 4; ++i) {
        const auto descriptor = registry.getTool(ids[i]);
        check(descriptor && descriptor->provider.name == "Input Plugin" && descriptor->provider.type == GRIM::MMO::ToolProviderType::Plugin &&
              descriptor->needs_confirmation && !descriptor->is_informational && descriptor->argument_format == "raw_text" &&
              descriptor->permission_bits == (GRIM_PERM_INPUT | (i == 4 ? GRIM_PERM_SYSTEM : 0)) &&
              descriptor->parameters.size() == 1, "Input metadata/policy incomplete");
    }
    const auto oldCalls = calls;
    same(dispatchCommand("move_mouse", "-1200, 300 trailing"),
         {true, "Mouse moved to (-1200, 300)", "ERR_NONE", "routine", "Mouse moved", Colors::Green});
    check(primitiveRequest == nlohmann::json{{"op","move"},{"x",-1200},{"y",300}}, "Coordinates were not parsed in plugin");
    same(dispatchCommand("click", ""), {true,"Clicked left mouse button","ERR_NONE","routine","Clicked",Colors::Green});
    check(primitiveRequest.at("button") == "left", "Default button changed");
    same(dispatchCommand("type", "  raw text\n"), {true,"Typed:   raw text\n","ERR_NONE","routine","Text typed",Colors::Green});
    check(primitiveRequest.at("text") == "  raw text\n", "Raw text changed");
    same(dispatchCommand("key", "ctrl + c"), {true,"Pressed key combo: ctrl + c","ERR_NONE","routine","Key combo pressed",Colors::Green});
    check(primitiveRequest.at("keys") == nlohmann::json::array({"ctrl","c"}), "Combo parsing changed");
    dispatchCommand("key", "enter");
    check(primitiveRequest == nlohmann::json{{"op","key"},{"key","enter"}}, "Single key changed");
    const auto effects = primitiveEffects;
    for (const auto& entry : {std::pair{"move_mouse","bad"}, {"type",""}, {"key",""}})
        check(dispatchCommand(entry.first, entry.second).errorCode == "ERR_INVALID_ARGS", "Invalid input reached primitive");
    check(primitiveEffects == effects && calls == oldCalls, "Basic tools used legacy command service or invalid input caused effects");
    available = false;
    check(dispatchCommand("move_mouse", "bad").errorCode == "ERR_INPUT_NOT_INITIALIZED", "Availability precedence changed");
    available = true;
    const std::string bytes("a\0b", 3);
    dispatchCommand("type", bytes);
    check(primitiveRequest.at("text").get<std::string>() == bytes, "Embedded NUL was truncated");
    check(dispatchCommand("type", std::string(65537, 'x')).errorCode == "ERR_INPUT_SERVICE", "Raw size limit lost");
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
    check(registry.toolCount() == baseline + 4, "Input plugin did not reload from file");
    fs::remove(folder / "input_plugin.dll");
    PluginManager::checkForHotReload();
    check(registry.toolCount() == baseline, "Deleted input plugin retained registrations");
    PluginManager::unloadAll();
    std::cout << "Input plugin results, permissions, metadata, exceptions, active-call draining and file-removal tests passed\n";
}
