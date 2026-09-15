#include "memory/unified_memory.hpp"
#include "logger.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// The focused test links the storage without the full host logger.
void logDebug(const std::string&, const std::string&) {}
void logTrace(const std::string&, const std::string&) {}
void logError(const std::string& tag, const std::string& message) {
    std::cerr << '[' << tag << "] " << message << '\n';
}

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

GRIM::UnifiedMemoryObject MakeRecord(uint64_t id) {
    GRIM::UnifiedMemoryObject obj;
    obj.id = id;
    obj.timestamp = static_cast<uint64_t>(std::time(nullptr));
    obj.domain = GRIM::MemoryDomain::FIELD;
    obj.type = GRIM::TypeTag::STRING;
    obj.context = GRIM::ContextType::CONVERSATION;
    obj.modality = GRIM::Modality::VISION;
    obj.raw = "async record " + std::to_string(id);
    obj.normalized = obj.raw;
    obj.tags = {"test", "async"};
    return obj;
}

size_t JsonRecordCount(const std::filesystem::path& path) {
    std::ifstream input(path);
    Require(input.is_open(), "JSON snapshot was not readable");
    nlohmann::json json;
    input >> json;
    Require(json.contains("memories"), "JSON snapshot has no memories array");
    return json["memories"].size();
}

} // anonymous namespace

int main() {
    namespace fs = std::filesystem;
    const fs::path test_dir = fs::temp_directory_path()
        / ("grim_unified_memory_async_"
           + std::to_string(
               std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(test_dir);
    const fs::path store_path = test_dir / "memories.fb";
    const fs::path json_path = test_dir / "memories.json";

    try {
        constexpr uint64_t kRecordCount = 200;
        {
            GRIM::UnifiedMemoryStorage storage;
            storage.initialize(store_path.string());

            // Concurrent callers exercise the short critical section. IDs are
            // deterministic because generateID() is outside this test's scope.
            std::vector<std::thread> producers;
            for (uint64_t worker = 0; worker < 4; ++worker) {
                producers.emplace_back([&, worker]() {
                    for (uint64_t i = worker; i < kRecordCount; i += 4) {
                        storage.storeLongTerm(MakeRecord(i + 1));
                    }
                });
            }
            for (auto& producer : producers) producer.join();

            // Explicit flush is the durability boundary: it bypasses the
            // debounce and returns only after both atomic files are published.
            storage.flush();
            Require(fs::exists(store_path), "FlatBuffer snapshot was not written");
            Require(fs::exists(json_path), "JSON snapshot was not written");
            Require(JsonRecordCount(json_path) == kRecordCount,
                    "JSON snapshot did not contain the complete batch");
            Require(storage.getStats().long_term_count == kRecordCount,
                    "in-memory record count is incorrect");
            storage.shutdown();
        }

        // Startup prefers FlatBuffer, so verify the authoritative file contains
        // the same complete batch as the JSON compatibility snapshot.
        {
            GRIM::UnifiedMemoryStorage reloaded;
            reloaded.initialize(store_path.string());
            Require(reloaded.getStats().long_term_count == kRecordCount,
                    "FlatBuffer reload did not recover the complete batch");
            reloaded.shutdown();
        }

        fs::remove_all(test_dir);
        std::cout << "unified_memory_async_tests: PASS\n";
        return 0;
    } catch (...) {
        std::error_code ec;
        fs::remove_all(test_dir, ec);
        throw;
    }
}
