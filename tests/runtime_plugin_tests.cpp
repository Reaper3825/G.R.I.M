// Real DLLs, timer queue, memory rotation, storage, and persistence in a private fixture.
#include "core/plugin_manager.hpp"
#include "core/plugin_api_impl.hpp"
#include "MMO/Core/ToolRegistry.hpp"
#include "commands/commands_timers.hpp"
#include "memory/unified_memory.hpp"
#include "memory/memory_buffer_rotation.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

GRIM::UnifiedMemoryStorage g_memoryStorage;
std::vector<Timer> timers;

void runRuntimePluginTests(const std::filesystem::path& timerDirectory, const std::filesystem::path& memoryDirectory) {
    namespace fs = std::filesystem;
    using namespace GRIM;
    using namespace GRIM::MMO;
    const auto check = [](bool ok, const char* message) { if (!ok) throw std::runtime_error(message); };
    auto& registry = ToolRegistry::instance();
    const auto baseline = registry.toolCount();
    auto& rotation = MemoryBufferRotation::instance();
    timers.clear(); rotation.clear();
    const auto folder = fs::temp_directory_path() / ("grim-runtime-plugins-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(folder), "Cannot create runtime plugin fixture");
    struct Cleanup {
        fs::path folder;
        GrimPluginAPI* api = nullptr;
        ~Cleanup() {
            PluginManager::unloadAll();
            if (api) PluginAPI::cleanupPluginContext(api);
            g_memoryStorage.shutdown();
            MemoryBufferRotation::instance().clear(); timers.clear();
            std::error_code error;
            for (const auto* name : {"timers_plugin.dll", "memory_plugin.dll", "records.fb", "records.json", "records.idx",
                                     "records.fb.tmp", "records.json.tmp", "records.idx.tmp"}) fs::remove(folder / name, error);
            fs::remove(folder, error);
        }
    } cleanup{folder};
    auto* denied = PluginAPI::createPluginAPI("runtime-denied", "", 0);
    check(denied->schedule_user_timer(denied, 1, "fixture") == nullptr && timers.empty(), "Timer permission bypass");
    check(denied->stage_manual_memory(denied, "fixture") == nullptr && rotation.stats().preproc_count == 0 &&
          denied->search_memory(denied, "fixture") == nullptr && denied->flush_memory_record_if_present(denied, 1) == nullptr,
          "Memory permission bypass");
    check(denied->schedule_user_timer(nullptr, 1, "fixture") == nullptr && denied->stage_manual_memory(nullptr, "fixture") == nullptr,
          "Invalid runtime API identity accepted");
    PluginAPI::cleanupPluginContext(denied);
    cleanup.api = PluginAPI::createPluginAPI("runtime-reader", "", GRIM_PERM_SYSTEM | GRIM_PERM_MEMORY);
    auto* api = cleanup.api;
    const auto value = [&](const char* text) {
        check(text != nullptr, "Runtime API failed");
        const std::string copy(text); api->string_free(api, text);
        const auto reply = nlohmann::json::parse(copy);
        check(reply.at("format") == "grim-runtime-service" && reply.at("version") == 1 && reply.at("error").is_null(), "Runtime reply schema/error incorrect");
        return reply.at("value");
    };
    check(api->schedule_user_timer(api, 0, "fixture") == nullptr && api->schedule_user_timer(api, 1, nullptr) == nullptr &&
          api->stage_manual_memory(api, "") == nullptr && api->search_memory(api, nullptr) == nullptr, "Invalid runtime parameters accepted");
    if (!timerDirectory.empty()) fs::copy_file(timerDirectory / "timers_plugin.dll", folder / "timers_plugin.dll");
    if (!memoryDirectory.empty()) {
        g_memoryStorage.initialize((folder / "records.fb").string());
        fs::copy_file(memoryDirectory / "memory_plugin.dll", folder / "memory_plugin.dll");
    }
    PluginManager::initialize(folder.string());
    const auto expectedCount = baseline + (timerDirectory.empty() ? 0 : 1) + (memoryDirectory.empty() ? 0 : 3);
    check(registry.toolCount() == expectedCount && timers.empty() && rotation.stats().preproc_count == 0, "Runtime plugin load changed state or registration count");
    const auto matches = [&](const CommandResult& r, bool ok, const std::string& message, const char* code,
                             const char* category, const std::string& voice, Color color) {
        check(r.success == ok && r.message == message && r.errorCode == code && r.category == category &&
              r.voice == voice && r.color.toUInt() == color.toUInt() && r.structuredOutputJson.empty(), "Runtime command result differs from original");
    };
    if (!timerDirectory.empty()) {
        const auto d = registry.getTool("set_timer");
        check(d && d->provider_name == "Timers Plugin" && d->permission_bits == GRIM_PERM_SYSTEM && !d->is_informational && d->needs_confirmation,
              "Timer metadata/policy incorrect");
        for (const auto* input : {"", "invalid", "99999999999999999999999"})
            matches(registry.invoke("set_timer", input), false, "[Timer][Error] Invalid number of seconds.", "ERR_TIMER_INVALID", "error", "Invalid timer duration", Colors::Red);
        for (const auto* input : {"0", "-3"})
            matches(registry.invoke("set_timer", input), false, "[Timer][Error] Duration must be positive.", "ERR_TIMER_NONPOSITIVE", "error", "Non-positive timer duration", Colors::Red);
        check(timers.empty(), "Invalid timer input mutated queue");
        const auto before = std::chrono::steady_clock::now();
        matches(registry.invoke("set_timer", " 5suffix"), true, "[Timer] Set for 5 seconds.", "ERR_NONE", "routine", "Timer set", Colors::Green);
        const auto after = std::chrono::steady_clock::now();
        check(timers.size() == 1 && timers[0].message == "Timer expired after 5s" && timers[0].expiry >= before + std::chrono::seconds(5) &&
              timers[0].expiry <= after + std::chrono::seconds(5), "Host timer scheduling or prefix parsing changed");
        fs::remove(folder / "timers_plugin.dll"); PluginManager::checkForHotReload();
        check(!registry.isRegistered("set_timer") && timers.size() == 1, "Timer DLL unload lost pending timer");
        check(checkExpiredTimers().empty(), "Unexpired timer delivered early");
        timers[0].expiry = std::chrono::steady_clock::now() - std::chrono::seconds(1);
        const auto expired = checkExpiredTimers();
        check(expired.size() == 1 && timers.empty() && checkExpiredTimers().empty(), "Host expiry did not consume timer exactly once");
        matches(expired[0], true, "[Timer] Timer expired after 5s", "ERR_NONE", "routine", "Timer expired", Colors::Yellow);
    }
    if (!memoryDirectory.empty()) {
        for (const auto* id : {"remember", "recall", "forget"}) {
            const auto d = registry.getTool(id); const bool recall = std::string(id) == "recall";
            check(d && d->provider_name == "Memory Plugin" && d->permission_bits == GRIM_PERM_MEMORY && d->is_informational == recall && d->needs_confirmation != recall,
                  "Memory metadata/policy incorrect");
        }
        matches(registry.invoke("remember", ""), false, "configured:ERR_MEMORY_MISSING_INPUT", "ERR_MEMORY_MISSING_INPUT", "error", "Missing memory input", Colors::Red);
        for (const auto* id : {"recall", "forget"}) {
            matches(registry.invoke(id, ""), false, "configured:ERR_MEMORY_MISSING_KEY", "ERR_MEMORY_MISSING_KEY", "error", "Missing memory key", Colors::Red);
            matches(registry.invoke(id, "not-present"), false, "configured:ERR_MEMORY_KEY_NOT_FOUND: not-present", "ERR_MEMORY_KEY_NOT_FOUND", "error", "Memory key not found", Colors::Red);
        }
        const std::vector<std::string> inputs = {"my color is blue", "place = home", "fact words", "single", "  spaced text  "};
        for (const auto& text : inputs)
            matches(registry.invoke("remember", text), true, "[Memory] Remembered: " + text, "ERR_NONE", "routine", "Remembered that " + text, Colors::Green);
        check(rotation.stats().preproc_count == inputs.size() && g_memoryStorage.search("blue").empty(), "Remember bypassed preprocessing or wrote directly to storage");
        rotation.mergeToWorking();
        const auto staged = rotation.workingSnapshot();
        check(staged.size() == inputs.size(), "Staged memory count incorrect");
        for (size_t i = 0; i < staged.size(); ++i) {
            const auto& obj = staged[i];
            check(obj.id != 0 && obj.timestamp != 0 && obj.domain == MemoryDomain::USER && obj.type == TypeTag::STRING &&
                  obj.context == ContextType::CONVERSATION && obj.confidence == 0.98f && obj.raw == inputs[i] &&
                  obj.normalized == "remember " + inputs[i] && obj.tags == std::vector<std::string>{"manual", "remember"},
                  "Manual memory record fields changed");
        }
        rotation.syncToLongTerm(g_memoryStorage); g_memoryStorage.flush();
        matches(registry.invoke("recall", "blue"), true, "[Memory] my color is blue", "ERR_NONE", "summary", "Recalled memory for blue", Colors::Cyan);
        matches(registry.invoke("recall", "BLUE"), false, "configured:ERR_MEMORY_KEY_NOT_FOUND: BLUE", "ERR_MEMORY_KEY_NOT_FOUND", "error", "Memory key not found", Colors::Red);
        UnifiedMemoryObject recent;
        recent.id = UnifiedMemoryObject::generateID(); recent.raw = "recent blue fact"; recent.normalized = recent.raw;
        g_memoryStorage.storeShortTerm(recent);
        matches(registry.invoke("recall", "blue"), true, "[Memory] recent blue fact", "ERR_NONE", "summary", "Recalled memory for blue", Colors::Cyan);
        const auto hits = value(api->search_memory(api, "blue"));
        check(hits.size() == 2 && hits[0].at("raw") == recent.raw && hits[0].at("id") == recent.id &&
              hits[1].at("id") == staged[0].id, "Memory search result order/data changed");
        check(!value(api->flush_memory_record_if_present(api, 0)).get<bool>(), "Missing record reported present");
        matches(registry.invoke("forget", "blue"), true, "[Memory] Forgotten entries for: blue", "ERR_NONE", "routine", "Forgotten blue", Colors::Green);
        check(g_memoryStorage.getById(staged[0].id).has_value() && g_memoryStorage.getById(recent.id).has_value(), "Migration changed legacy forget into deletion");
        matches(registry.invoke("remember", "pending stage"), true, "[Memory] Remembered: pending stage", "ERR_NONE", "routine", "Remembered that pending stage", Colors::Green);
        fs::remove(folder / "memory_plugin.dll"); PluginManager::checkForHotReload();
        check(!registry.isRegistered("remember") && !registry.isRegistered("recall") && !registry.isRegistered("forget") &&
              g_memoryStorage.search("blue").size() == 2 && rotation.stats().preproc_count == 1,
              "Memory plugin unload changed storage/staging or retained commands");
        g_memoryStorage.shutdown();
        UnifiedMemoryStorage reloaded; reloaded.initialize((folder / "records.fb").string());
        check(reloaded.search("blue").size() == 1, "Memory state did not persist to fixture store");
        reloaded.shutdown();
    }
    check(registry.toolCount() == baseline, "Runtime plugin cleanup left registrations");
    std::cout << "Timer and memory plugins: real host queue, rotation, storage, persistence, compatibility, and unload tests passed\n";
}
