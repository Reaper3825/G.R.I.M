#include "core/plugin_tool_result.hpp"
#include "memory_provider.hpp"
#include "memory/atomic_writer.hpp"
#include "plugins/plugin_logging.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unordered_set>

namespace {
using GRIM::UnifiedMemoryObject;
using GRIM::UnifiedMemoryStorage;
using nlohmann::json;

struct MemoryState {
    UnifiedMemoryStorage storage;
    std::filesystem::path transientPath;
    std::vector<UnifiedMemoryObject> pending;
    std::mutex mutex;

    explicit MemoryState(const std::filesystem::path& directory)
        : transientPath(directory / "transient.json") {
        storage.initialize((directory / "records.fb").string());
        if (!std::filesystem::exists(transientPath)) return;

        std::ifstream input(transientPath, std::ios::binary);
        const auto snapshot = json::parse(input);
        if (snapshot.at("version") != 1)
            throw std::runtime_error("Unsupported memory transient snapshot");
        for (const auto& value : snapshot.at("pending")) {
            auto record = decode(value);
            // A previous run may have committed this record before replacing the snapshot.
            if (!storage.getById(record.id)) pending.push_back(std::move(record));
        }
        for (const auto& value : snapshot.at("short_term"))
            storage.storeShortTerm(decode(value));
    }

    static UnifiedMemoryObject decode(const json& value) {
        auto record = UnifiedMemoryObject::fromJSON(value.dump());
        record.embedding = value.at("embedding").get<decltype(record.embedding)>();
        return record;
    }

    static json encode(const std::vector<UnifiedMemoryObject>& records) {
        auto values = json::array();
        for (const auto& record : records) {
            auto value = json::parse(record.toJSON());
            value["embedding"] = record.embedding;
            values.push_back(std::move(value));
        }
        return values;
    }

    void saveTransient() {
        const json snapshot{
            {"version", 1}, {"pending", encode(pending)},
            {"short_term", encode(storage.shortTermSnapshot())}
        };
        GRIM::AtomicWriter::writeString(transientPath.string(), snapshot.dump());
    }

    void commitPending() {
        for (const auto& record : pending) {
            if (!storage.getById(record.id)) storage.storeLongTerm(record);
        }
        pending.clear();
    }
};

std::unique_ptr<MemoryState> state;

json retrieve(const std::string& prompt, const std::string& category, int limit) {
    std::lock_guard lock(state->mutex);
    state->commitPending();

    auto hits = state->storage.search(prompt, limit);
    auto breadcrumbs = json::array({{{"query", prompt}, {"results_found", hits.size()}}});
    if (!category.empty()) {
        auto tagged = state->storage.getByTag(category);
        breadcrumbs.push_back({{"query", "tag:" + category}, {"results_found", tagged.size()}});
        std::unordered_set<uint64_t> seen;
        for (const auto& hit : hits) seen.insert(hit.id);
        for (auto& hit : tagged) {
            if (seen.insert(hit.id).second) hits.push_back(std::move(hit));
        }
    }
    if (hits.size() > static_cast<size_t>(limit)) {
        std::partial_sort(hits.begin(), hits.begin() + limit, hits.end(),
            [](const auto& a, const auto& b) { return a.confidence > b.confidence; });
        hits.resize(limit);
    }

    auto memories = json::array();
    for (const auto& hit : hits) {
        memories.push_back({{"id", hit.id}, {"raw", hit.raw}, {"normalized", hit.normalized},
            {"confidence", hit.confidence}, {"score", hit.confidence}, {"tags", hit.tags}});
    }
    return {{"memories", std::move(memories)}, {"retrieval_breadcrumbs", std::move(breadcrumbs)}};
}

GrimResult retrieveTool(const GrimToolCall& request, GrimToolResultWriter* writer) {
    const auto args = json::parse(request.arguments, request.arguments + request.arguments_size, nullptr, false);
    if (!args.is_object() || !args.contains("prompt") || !args["prompt"].is_string() ||
        !args.contains("nlp_category") || !args["nlp_category"].is_string() ||
        !args.contains("limit") || !args["limit"].is_number_integer() ||
        args["limit"] < 1 || args["limit"] > 100) return GRIM_ERROR_INVALID_PARAM;

    const auto payload = retrieve(args["prompt"].get<std::string>(),
        args["nlp_category"].get<std::string>(), args["limit"].get<int>()).dump();
    return grim::completeToolResult(writer, true, nullptr, "ERR_NONE", nullptr, 0, nullptr, payload.c_str());
}
}  // namespace

namespace memoryProvider {
void stageManual(const std::string& raw) {
    std::lock_guard lock(state->mutex);
    UnifiedMemoryObject record(GRIM::MemoryDomain::USER, GRIM::TypeTag::STRING,
        GRIM::ContextType::CONVERSATION, raw, 0.98f);
    record.normalized = "remember " + raw;
    record.tags = {"manual", "remember"};
    state->pending.push_back(std::move(record));
}

std::vector<UnifiedMemoryObject> search(const std::string& query) {
    std::lock_guard lock(state->mutex);
    return state->storage.search(query);
}
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_init(const GrimPluginAPI* table, GrimPluginHandle instance) {
    const grim::PluginHost api{table, instance};
    if (!api || (api->api_version >> 16) != GRIM_API_VERSION_MAJOR || api->api_version < GRIM_API_VERSION ||
        !api->get_plugin_path || !api->string_free || !api->get_user_error_message)
        return GRIM_ERROR_API_VERSION_MISMATCH;

    grim::runtime::host = api;
    grim::plugin::setLoggingAPI(api);
    try {
        const std::filesystem::path pluginPath = grim::runtime::take(api->get_plugin_path(instance));
        if (pluginPath.empty()) return GRIM_ERROR_INVALID_PARAM;
        state = std::make_unique<MemoryState>(pluginPath.parent_path() / "memory_plugin_data");
        return GRIM_OK;
    } catch (const std::exception& error) {
        LOG_ERROR("Memory", error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT GrimResult grim_plugin_dispatch_tool(
    const char* tool_id, const GrimToolCall* request, GrimToolResultWriter* writer) {
    if (!tool_id || !request || !request->arguments || !writer || !writer->complete)
        return GRIM_ERROR_INVALID_PARAM;
    if (!state) return GRIM_ERROR_PLUGIN_NOT_LOADED;
    const std::string_view id(tool_id);
    try {
        if (id == "memory.retrieve") return retrieveTool(*request, writer);
        const std::string arguments(request->arguments, request->arguments_size);
        if (id == "remember") return grim::completeToolResult(writer, cmdRemember(arguments));
        if (id == "recall") return grim::completeToolResult(writer, cmdRecall(arguments));
        return GRIM_ERROR_NOT_FOUND;
    } catch (const std::exception& error) {
        LOG_ERROR("Memory", error.what());
        return GRIM_ERROR_UNKNOWN;
    }
}

extern "C" PLUGIN_EXPORT void grim_plugin_shutdown() {
    if (state) {
        try {
            state->storage.flush();
            state->saveTransient();
        } catch (const std::exception& error) {
            LOG_ERROR("Memory", error.what());
        }
        state.reset();
    }
    grim::plugin::setLoggingAPI({});
    grim::runtime::host = {};
}
