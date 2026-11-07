#include "application_catalog.hpp"
#include "aliases.hpp"
#include <filesystem>
#include <cctype>

namespace GRIM::Applications {
std::vector<Backend> systemBackends() {
    return {nativeBackend(), {"legacy-aliases", [] {
        std::vector<Application> result;
#ifdef _WIN32
        auto aliases = aliases::getAll();
        // Deterministic precedence if multiple names point to exactly the same file.
        std::map<std::string, std::string> ordered(aliases.begin(), aliases.end());
        for (const auto& [name, target] : ordered) {
            const auto path = std::filesystem::path(std::u8string(target.begin(), target.end()));
            std::error_code error;
            if (!path.is_absolute() || !std::filesystem::is_regular_file(path, error)) continue;
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            // An alias can also point at documents or commands; only executable/shortcut files are applications.
            if (ext != ".exe" && ext != ".lnk") continue;
            const auto normalized = path.lexically_normal().u8string();
            std::string id(reinterpret_cast<const char*>(normalized.data()), normalized.size());
            result.push_back({"windows.file", id, name, "file", target, ""});
        }
#endif
        return result;
    }}};
}
}
