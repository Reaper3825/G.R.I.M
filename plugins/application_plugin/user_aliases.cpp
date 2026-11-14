#include "user_aliases.hpp"
#include "aliases_generated.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdlib>
#include <cctype>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#endif

namespace GRIM::Applications {
namespace {
std::string normalized(std::string name) {
    const auto first=name.find_first_not_of(" \t\r\n");
    if(first==std::string::npos)return {};
    name=name.substr(first,name.find_last_not_of(" \t\r\n")-first+1);
    for(auto& c:name)if(c>='A' && c<='Z')c=static_cast<char>(c-'A'+'a');
    return name;
}
void insert(std::map<std::string,std::string>& entries,const std::string& name,const std::string& target) {
    auto key=normalized(name);
    if(key.empty() || key.size()>256 || target.empty() || target.size()>8192 ||
       key.find_first_of("\r\n\0",0,3)!=std::string::npos || target.find_first_of("\r\n\0",0,3)!=std::string::npos)
        throw std::runtime_error("Invalid user application alias");
    // Validate UTF-8 before a value reaches the catalog or native launcher.
    (void)nlohmann::json::array({key,target}).dump();
    if(!entries.emplace(key,target).second)throw std::runtime_error("Duplicate normalized user alias: "+key);
    if(entries.size()>10000)throw std::runtime_error("User alias limit exceeded");
}
std::vector<uint8_t> read(const std::filesystem::path& path) {
    const auto size=std::filesystem::file_size(path);
    if(size>16*1024*1024)throw std::runtime_error("Alias file exceeds 16 MiB");
    std::ifstream stream(path,std::ios::binary);
    if(!stream)throw std::runtime_error("Cannot read application aliases");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(size));
    if(stream.gcount()!=static_cast<std::streamsize>(size) || stream.peek()!=std::char_traits<char>::eof() || stream.bad())
        throw std::runtime_error("Alias file changed or failed while reading");
    return bytes;
}
}
std::filesystem::path UserAliases::defaultDirectory() {
    if(const auto* configured=std::getenv("GRIM_APPLICATION_CONFIG_DIR")) {
        const std::string utf8(configured);
        auto path=std::filesystem::path(std::u8string(utf8.begin(),utf8.end()));
        if(!path.is_absolute())throw std::invalid_argument("GRIM_APPLICATION_CONFIG_DIR must be absolute");
        return path;
    }
#ifdef GRIM_APPLICATION_ROOT
    auto resources=std::filesystem::path(GRIM_APPLICATION_ROOT)/"resources";
    if(std::filesystem::is_directory(resources))return resources;
#endif
#ifdef _WIN32
    std::wstring executable(32768,L'\0');
    const auto length=GetModuleFileNameW(nullptr,executable.data(),static_cast<DWORD>(executable.size()));
    if(length && length<executable.size()) {
        executable.resize(length);
        auto resources=std::filesystem::path(executable).parent_path()/"resources";
        if(std::filesystem::is_directory(resources))return resources;
    }
#endif
    return std::filesystem::current_path()/"resources";
}
std::map<std::string,std::string> UserAliases::reload() {
    std::lock_guard lock(mutex_);
    std::map<std::string,std::string> next;
    const auto jsonPath=directory_/"application_aliases.json";
    const auto legacyPath=directory_/"app_aliases.fb";
    if(std::filesystem::exists(jsonPath)) {
        const auto bytes=read(jsonPath);
        const auto document=nlohmann::json::parse(bytes);
        if(!document.is_object() || document.size()!=2 || document.at("version")!=1 || !document.at("aliases").is_object())
            throw std::runtime_error("Expected application alias configuration {version:1,aliases:{name:target}}");
        for(const auto& [name,target]:document.at("aliases").items())insert(next,name,target.get<std::string>());
    } else if(std::filesystem::exists(legacyPath)) {
        const auto bytes=read(legacyPath);
        flatbuffers::Verifier verifier(bytes.data(),bytes.size());
        if(!AliasesSchema::VerifyAliasesBuffer(verifier))throw std::runtime_error("Invalid legacy application alias database");
        const auto* source=AliasesSchema::GetAliases(bytes.data());
        if(source->user())for(const auto* entry:*source->user()) {
            if(!entry || !entry->key() || !entry->path())throw std::runtime_error("Invalid legacy user alias");
            insert(next,entry->key()->str(),entry->path()->str());
        }
        // Deliberately ignore auto_: automatic discovery has no persisted alias cache.
    }
    entries_=next;
    return next;
}
std::string UserAliases::resolve(const std::string& name) const {
    std::lock_guard lock(mutex_);
    const auto it=entries_.find(normalized(name));
    return it==entries_.end()?name:it->second;
}
}
