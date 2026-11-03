#pragma once
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace GRIM::Environment {
// Project interpreter prose without its terminal chrome. Raw host responses
// remain intact. Never apply this cleanup to authored prompts or Goal text.
inline std::string cleanEnvironmentObservation(const std::string& source,
                                             const std::string& location_path = {}) {
    const auto trim=[](const std::string& value){
        const auto begin=value.find_first_not_of(" \t\r\n");
        return begin==std::string::npos?std::string{}:value.substr(begin,value.find_last_not_of(" \t\r\n")-begin+1);
    };
    static const std::regex room(R"(^-= .+ =-$)");
    static const std::regex chrome(R"(^>?[ \t]*-= .+ =-([ \t]*-?[0-9]+[ \t]*/[ \t]*[0-9]+)?$)");
    std::vector<std::string> lines;std::istringstream stream(source);std::string line;
    while(std::getline(stream,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back();
        lines.push_back(line);
    }
    std::size_t bannerEnd=0,first=0;
    while(first<lines.size() && trim(lines[first]).empty())++first;
    // Recognize the pinned TextWorld startup logo, rather than removing
    // arbitrary ASCII diagrams from tool feedback.
    if(first<lines.size() && trim(lines[first])=="________  ________  __    __  ________"){
        for(auto i=first+1;i<lines.size();++i){
            const auto text=trim(lines[i]);
            if(std::regex_match(text,room)){bannerEnd=i;break;}
            if(text.find_first_not_of(" _\\|/$><-")!=std::string::npos)break;
        }
    }
    std::string result;bool blank=false;
    for(std::size_t i=0;i<lines.size();++i){
        if(bannerEnd && i<bannerEnd)continue;
        const auto text=trim(lines[i]);
        if(text==">" || std::regex_match(text,chrome) ||
           (!location_path.empty() && text=="LOCATION: "+location_path))continue;
        if(text.empty()){blank=!result.empty();continue;}
        if(!result.empty())result+=blank?"\n\n":"\n";
        auto end=lines[i].find_last_not_of(" \t\r");
        result+=lines[i].substr(0,end+1);blank=false;
    }
    return result;
}
} // namespace GRIM::Environment
