#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM::CommandMemory {
struct Hit { uint64_t id; std::string raw; };
void stageManual(const std::string& raw);
std::vector<Hit> search(const std::string& query);
// Existing forget behavior: check presence and flush. Does not delete a record.
bool flushIfPresent(uint64_t id);
}
