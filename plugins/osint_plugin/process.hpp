#pragma once
#include <atomic>
#include <string>
#include <vector>
namespace osintProvider {
inline std::atomic<bool> cancelled{false};
// Shell-free child execution. Timeout/cancellation terminates the child tree.
int runProcess(const std::vector<std::string>& arguments, int timeoutSeconds);
}
