#include "shell_open.hpp"
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

int64_t shellOpenTarget(const std::string& target) {
#ifdef _WIN32
    return static_cast<int64_t>(reinterpret_cast<intptr_t>(
        ShellExecuteA(nullptr, "open", target.c_str(), nullptr, nullptr, SW_SHOWNORMAL)));
#else
    (void)target;
    return 0;
#endif
}
