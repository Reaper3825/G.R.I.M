#include "application_catalog.hpp"
#include <filesystem>
#include <cctype>

namespace GRIM::Applications {
std::vector<Backend> systemBackends(AliasSource aliases) {
    return {nativeBackend(), portableBackend(), {"user-aliases", [aliases = std::move(aliases)] {
        std::vector<Application> result;
        const auto ordered = aliases();
#ifdef _WIN32
        // User-defined names have their own identities even when targets match.
        for (const auto& [name, target] : ordered) {
            const auto path = std::filesystem::path(std::u8string(target.begin(),target.end()));
            std::error_code error;
            if (!path.is_absolute() || !std::filesystem::is_regular_file(path, error)) continue;
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            // An alias can also point at documents or commands; only executable/shortcut files are applications.
            if (ext != ".exe" && ext != ".lnk" && ext != ".bat" && ext != ".cmd") continue;
            result.push_back({"user.alias", name, name, "file", target, ""});
        }
#endif
        return result;
    }}};
}
}
