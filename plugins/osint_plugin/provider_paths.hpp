#pragma once
#include <filesystem>
#include <string>
#include <stdexcept>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cctype>
namespace osintProvider {
inline std::filesystem::path root;
inline std::recursive_mutex storageMutex;
inline void initializePaths() { root = std::filesystem::current_path(); }
inline std::filesystem::path cache() { return root / "cache" / "osint"; }
inline std::filesystem::path scripts() { return root / "plugins" / "osint_plugin" / "python"; }
struct TemporaryFile {
    std::filesystem::path path;
    explicit TemporaryFile(const std::string& prefix) {
        static std::atomic<unsigned long long> sequence{0};
        std::filesystem::create_directories(cache());
        path = cache() / (prefix + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence++) + ".json");
    }
    ~TemporaryFile() { std::error_code error; std::filesystem::remove(path,error); }
    TemporaryFile(const TemporaryFile&) = delete;
};
inline void validateTarget(const std::string& value) {
    if (value.empty() || value.size()>128 || value=="." || value==".." || value.front()=='-' || value.back()=='.')
        throw std::invalid_argument("Invalid OSINT target");
    auto base = value.substr(0,value.find('.'));
    std::transform(base.begin(),base.end(),base.begin(),[](unsigned char c){return static_cast<char>(std::toupper(c));});
    if (base=="CON" || base=="PRN" || base=="AUX" || base=="NUL" ||
        (base.size()==4 && (base.substr(0,3)=="COM" || base.substr(0,3)=="LPT") && base[3]>='1' && base[3]<='9'))
        throw std::invalid_argument("Reserved OSINT target filename");
    for (unsigned char c:value) if (!((c>='a' && c<='z') || (c>='A' && c<='Z') || (c>='0' && c<='9') || c=='_' || c=='-' || c=='.' || c=='@' || c=='+'))
        throw std::invalid_argument("OSINT targets must contain only letters, digits, _, -, ., @ or +");
}
}
