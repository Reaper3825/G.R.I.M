#pragma once
#include "world_document.hpp"
#include "resources/models/GRIM-text/Shared/Goal/GoalContract.hpp"
#include <sstream>
#include <iomanip>

namespace GRIM::WorldLab {
struct GoalAccess {
    std::function<std::vector<nlohmann::json>(const std::string&)> list;
    std::function<void(const nlohmann::json&)> import;
};
inline std::string recordKey(const std::string& text) {
    uint64_t hash=14695981039346656037ull;
    for(unsigned char c:text){hash^=c;hash*=1099511628211ull;}
    std::ostringstream out;out<<std::hex<<hash;return out.str();
}
inline bool hasLegacyChallenge(const Document& d) {
    return !d.objective.empty() || !d.winConditions.empty() || !d.failConditions.empty() || !d.solution.empty();
}
inline nlohmann::json legacyGoal(const Document& d) {
    ConceptBlockGoal goal;goal.target_state=d.objective.empty()?"Complete the imported challenge.":d.objective;
    const auto describe=[&](const Condition& c){
        const auto name=[&](const std::string& id){if(auto* e=d.find(id))return e->name;for(const auto& p:d.passages)if(p.id==id)return p.doorName;return id;};
        return c.predicate+" "+name(c.subject)+(c.target.empty()?"":" / "+name(c.target));
    };
    for(const auto& c:d.winConditions)goal.success_criteria.push_back({describe(c),EvidenceCondition{c.predicate,c.subject,c.target}});
    for(const auto& c:d.failConditions)goal.constraints.push_back({"Avoid: "+describe(c),EvidenceCondition{c.predicate,c.subject,c.target}});
    return {{"source_id","world-legacy-"+recordKey(d.json().dump())},{"name",d.title+" (imported Goal)"},{"goal",goalToJson(goal)}};
}
inline nlohmann::json readEpisodeRecord(const std::filesystem::path& path) {
    auto value=loadJsonFile(path);
    if(value.at("format")!="grim-episode" || value.at("version")!=1)throw std::runtime_error("Unsupported episode record");
    (void)goalFromJson(value.at("selection").at("goal"));
    if(!value.at("commands").is_array() || value.at("commands").size()>1000)throw std::runtime_error("Invalid recorded commands");
    for(const auto& command:value.at("commands")) {
        const auto text=command.get<std::string>();
        if(text.empty() || text.size()>512 || text.find_first_of("\r\n\0",0,3)!=std::string::npos)throw std::runtime_error("Invalid recorded command");
    }
    return value;
}
} // namespace GRIM::WorldLab
