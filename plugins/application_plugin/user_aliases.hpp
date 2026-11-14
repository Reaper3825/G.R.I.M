#pragma once
#include <filesystem>
#include <map>
#include <mutex>
#include <string>

namespace GRIM::Applications {
// Read-only during load/refresh: never rewrites a user's legacy database.
class UserAliases {
public:
    explicit UserAliases(std::filesystem::path directory) : directory_(std::move(directory)) {}
    std::map<std::string,std::string> reload();
    std::string resolve(const std::string& name) const;
    static std::filesystem::path defaultDirectory();
private:
    std::filesystem::path directory_;
    mutable std::mutex mutex_;
    std::map<std::string,std::string> entries_;
};
}
