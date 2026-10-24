#include "world_actions.hpp"
#include "textworld_action_catalog.generated.hpp"
#include <algorithm>
#include <stdexcept>

namespace GRIM::WorldLab {
const nlohmann::json& actionCatalog() {
    static const auto catalog=nlohmann::json::parse(kTextWorldActionCatalog);
    return catalog;
}
bool actionTypeMatches(const std::string& type,const std::string& kind) {
    const bool portable=kind=="object" || kind=="key" || kind=="food";
    if(type=="t")return portable || kind=="container" || kind=="supporter" || kind=="door";
    if(type=="o")return portable;
    return (type=="r" && kind=="room") || (type=="k" && kind=="key") ||
        (type=="f" && kind=="food") || (type=="c" && kind=="container") ||
        (type=="s" && kind=="supporter") || (type=="d" && kind=="door");
}
std::vector<std::string> actionCandidates(const Document& d,const std::string& type) {
    std::vector<std::string> result;
    for(const auto& e:d.entities)if(actionTypeMatches(type,e.kind))result.push_back(e.id);
    for(const auto& p:d.passages)if(!p.doorName.empty() && actionTypeMatches(type,"door"))result.push_back(p.id);
    return result;
}
std::string actionEntityName(const Document& d,const std::string& id) {
    if(const auto* e=d.find(id))return e->name;
    for(const auto& p:d.passages)if(p.id==id && !p.doorName.empty())return p.doorName;
    throw std::runtime_error("Action references a missing entity; select its parameters again");
}
bool actionRelevant(const Document& d,const nlohmann::json& action,const std::string& selection) {
    if(selection.empty())return true;
    const auto id=action.at("id").get<std::string>();
    const auto* e=d.find(selection);
    if(e && e->kind=="room")return id=="look" || id=="inventory" || id.starts_with("go/");
    for(const auto& p:d.passages)if(p.id==selection && id.starts_with("go/"))
        return id=="go/"+p.direction || id=="go/"+opposite(p.direction);
    for(const auto& parameter:action.at("parameters")) {
        const auto candidates=actionCandidates(d,parameter.at("type"));
        if(std::find(candidates.begin(),candidates.end(),selection)!=candidates.end())return true;
    }
    return false;
}
std::string composeAction(const Document& d,const nlohmann::json& action,const std::vector<std::string>& bindings) {
    const auto& parameters=action.at("parameters");
    if(bindings.size()!=parameters.size())throw std::runtime_error("Choose all action parameters");
    // Substitute only template tokens, never tokens appearing inside entity names.
    const auto format=action.at("command").get<std::string>();std::string result;
    for(size_t i=0;i<format.size();) {
        if(format[i]!='{'){result+=format[i++];continue;}
        const auto end=format.find('}',i);if(end==std::string::npos)throw std::runtime_error("Invalid command template");
        const auto token=format.substr(i+1,end-i-1);bool found=false;
        for(size_t p=0;p<parameters.size();++p)if(parameters[p].at("name")==token) {
            const auto choices=actionCandidates(d,parameters[p].at("type"));
            if(std::find(choices.begin(),choices.end(),bindings[p])==choices.end())throw std::runtime_error("Choose a valid "+token+" parameter");
            result+=actionEntityName(d,bindings[p]);found=true;break;
        }
        if(!found)throw std::runtime_error("Unknown command parameter");i=end+1;
    }
    if(result.size()>512 || result.find_first_of("\r\n")!=std::string::npos || result.find('\0')!=std::string::npos)
        throw std::runtime_error("Command must be one line, at most 512 bytes");
    return result;
}
}
