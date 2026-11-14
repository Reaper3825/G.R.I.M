#include "application_catalog.hpp"
#include <filesystem>
#include <cctype>
namespace GRIM::Applications {
std::vector<Application> scanPortableDirectories(const std::vector<std::filesystem::path>& roots) {
    std::vector<Application> result;
    auto scan=[&](const std::filesystem::path& path) {
        for(const auto& file:std::filesystem::directory_iterator(path,std::filesystem::directory_options::skip_permission_denied)) {
            if(!file.is_regular_file())continue;
            auto ext=file.path().extension().string();
            for(auto& c:ext)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if(ext!=".exe" && ext!=".bat" && ext!=".cmd")continue;
            const auto bytes=std::filesystem::absolute(file.path()).lexically_normal().u8string();
            const std::string target(bytes.begin(),bytes.end());
            const auto name=file.path().stem().u8string();
            result.push_back({"windows.file",target,std::string(name.begin(),name.end()),"file",target,""});
            if(result.size()>10000)throw std::runtime_error("Portable application limit exceeded");
        }
    };
    for(const auto& root:roots) {
        if(!std::filesystem::exists(root))continue;
        // Preserve the old scanner's one-subdirectory depth; include direct portable files too.
        scan(root);
        for(const auto& entry:std::filesystem::directory_iterator(root,std::filesystem::directory_options::skip_permission_denied))
            if(entry.is_directory() && !entry.is_symlink())scan(entry.path());
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.nativeId<b.nativeId;});
    return result;
}
}
