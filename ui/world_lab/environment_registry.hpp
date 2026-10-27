#pragma once
#include "world_document.hpp"
#include <algorithm>
#include <set>

namespace GRIM::WorldLab {
struct EnvironmentEntry {
    std::string id, title;
    std::filesystem::path path;
};

// Persistent identity/location index. Loading and lookup never rewrite worlds.
class EnvironmentRegistry {
public:
    explicit EnvironmentRegistry(std::filesystem::path directory):directory_(std::move(directory)) {}
    std::filesystem::path file() const { return directory_/"environment_registry.json"; }
    std::vector<EnvironmentEntry> entries() const {
        if(!std::filesystem::exists(file()))return {};
        const auto value=loadJsonFile(file());
        if(value.at("format")!="grim-environments" || value.at("version")!=1 || !value.at("environments").is_array())
            throw std::runtime_error("Unsupported environment registry");
        std::vector<EnvironmentEntry> result;std::set<std::string> ids;
        for(const auto& entry:value.at("environments")) {
            auto id=entry.at("id").get<std::string>();
            auto path=std::filesystem::u8path(entry.at("path").get<std::string>());
            if(id.empty() || !ids.insert(id).second || !path.is_absolute())throw std::runtime_error("Invalid environment registry entry");
            result.push_back({id,entry.at("title").get<std::string>(),path});
        }
        return result;
    }
    EnvironmentEntry resolve(const std::string& id) const {
        if(id.empty())throw std::runtime_error("Course has no environment assigned");
        for(auto entry:entries())if(entry.id==id) {
            if(!std::filesystem::is_regular_file(entry.path))throw std::runtime_error("Environment file missing for "+id+"; open its new location in World Lab or reassign the course");
            const auto raw=loadJsonFile(entry.path);
            if(!raw.contains("environment_id") || raw.at("environment_id")!=id)
                throw std::runtime_error("Environment ID mismatch for "+id);
            entry.title=Document::parse(raw).title;return entry;
        }
        throw std::runtime_error("Environment not registered: "+id);
    }
    Document load(const std::string& id) const {
        const auto entry=resolve(id);auto document=Document::parse(loadJsonFile(entry.path));
        if(document.environmentId!=id)throw std::runtime_error("Environment changed during resolution: "+id);
        return document;
    }
    void checkDestination(const Document& document,const std::filesystem::path& path) const {
        document.validate();auto values=entries();merge(values,{document.environmentId,document.title,normalized(path)});
        if(std::filesystem::exists(path)) {
            const auto raw=loadJsonFile(path);
            (void)Document::parse(raw); // Never overwrite another kind of JSON file.
            // Legacy files may receive their first ID on explicit save.
            if(raw.contains("environment_id") && raw.at("environment_id")!=document.environmentId)
                throw std::runtime_error("Destination belongs to another environment; choose a new file path");
        }
    }
    void registerFile(const std::filesystem::path& path) const {
        const auto raw=loadJsonFile(path);
        if(!raw.contains("environment_id"))throw std::runtime_error("Save this legacy world in World Lab to persist its environment ID first");
        const auto document=Document::parse(raw);auto values=entries();
        merge(values,{document.environmentId,document.title,normalized(path)});write(values);
    }
    // Discover saved IDs without migrating legacy files. Batch validation avoids
    // partially changing the index on malformed files or duplicate identities.
    void indexDirectory() const {
        auto values=entries();const auto before=valuesJson(values);
        if(std::filesystem::exists(directory_))for(const auto& entry:std::filesystem::directory_iterator(directory_)) {
            if(!entry.is_regular_file() || !text(entry.path().filename()).ends_with(".grimworld.json"))continue;
            const auto raw=loadJsonFile(entry.path());
            if(!raw.contains("environment_id"))continue;
            const auto document=Document::parse(raw);
            merge(values,{document.environmentId,document.title,normalized(entry.path())});
        }
        if(valuesJson(values)!=before)write(values);
    }
private:
    static std::string text(const std::filesystem::path& path) {
        const auto value=path.u8string();return {reinterpret_cast<const char*>(value.data()),value.size()};
    }
    static std::filesystem::path normalized(const std::filesystem::path& path) {
        if(path.empty())throw std::runtime_error("Enter an environment file path");
        return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
    }
    static void merge(std::vector<EnvironmentEntry>& values,EnvironmentEntry entry) {
        auto match=values.end();
        for(auto it=values.begin();it!=values.end();++it) {
            auto& previous=*it;
            const bool samePath=previous.path==entry.path ||
                (std::filesystem::exists(previous.path) && std::filesystem::exists(entry.path) && std::filesystem::equivalent(previous.path,entry.path));
            if(previous.id==entry.id) {
                if(!samePath && std::filesystem::exists(previous.path))
                    throw std::runtime_error("Duplicate environment ID "+entry.id+"; use Save copy for an independent world");
                match=it;
            }
            else if(samePath)throw std::runtime_error("Environment path already registered to another ID");
        }
        if(match!=values.end())*match=std::move(entry);
        else values.push_back(std::move(entry));
    }
    static nlohmann::json valuesJson(const std::vector<EnvironmentEntry>& values) {
        auto result=nlohmann::json::array();
        for(const auto& entry:values)result.push_back({{"id",entry.id},{"title",entry.title},{"path",text(entry.path)}});
        return result;
    }
    void write(const std::vector<EnvironmentEntry>& values) const {
        std::filesystem::create_directories(directory_);
        saveJsonFile(file(),{{"format","grim-environments"},{"version",1},{"environments",valuesJson(values)}});
    }
    std::filesystem::path directory_;
};
}
