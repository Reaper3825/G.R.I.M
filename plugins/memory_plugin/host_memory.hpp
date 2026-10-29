#pragma once
#include "plugins/runtime_service_client.hpp"
#include <vector>

namespace hostMemory {
struct Hit { uint64_t id; std::string raw; };
inline void stageManual(const std::string& raw) {
    grim::runtime::response(grim::runtime::host->stage_manual_memory(grim::runtime::host, raw.c_str()));
}
inline auto search(const std::string& query) {
    const auto value = grim::runtime::response(grim::runtime::host->search_memory(grim::runtime::host, query.c_str()));
    std::vector<Hit> hits;
    for (const auto& hit : value) hits.push_back({hit.at("id").get<uint64_t>(), hit.at("raw").get<std::string>()});
    return hits;
}
inline bool flushIfPresent(uint64_t id) {
    return grim::runtime::response(grim::runtime::host->flush_memory_record_if_present(grim::runtime::host, id)).get<bool>();
}
inline std::string errorMessage(const std::string& code) {
    return grim::runtime::take(grim::runtime::host->get_user_error_message(grim::runtime::host, code.c_str()));
}
}
CommandResult cmdRemember(const std::string&);
CommandResult cmdRecall(const std::string&);
CommandResult cmdForget(const std::string&);
