#include "world_asset.hpp"
#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <stdexcept>

namespace GRIM::WorldLab {
namespace {
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
std::string lower(std::string s){for(auto& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));return s;}
}
Document Asset::preview() const {
    need(!name.empty() && name.size()<=256,"Asset name must contain 1-256 bytes");
    need(!entities.empty() && entities.size()<10000,"Asset must contain 1-9999 entities");
    Document d;d.title=name;d.entities=entities;
    const auto* top=d.find(root);
    need(top && top->parent.empty(),"Asset root is missing or has an external parent");
    for(const auto& e:entities) {
        need(e.id==root || (e.kind!="room" && !e.parent.empty()),"Asset must have exactly one root");
        std::set<std::string> visited;
        auto* node=&e;
        while(node && node->id!=root){need(visited.insert(node->id).second,"Asset containment cycle");node=d.find(node->parent);}
        need(node!=nullptr,"Asset contains an external reference");
    }
    if(top->kind=="room") {d.spawn=root;d.layout.push_back({root,80,80});}
    else {
        std::string staging="asset-preview";
        while(d.find(staging))staging+="-";
        d.find(root)->parent=staging;
        d.entities.push_back({staging,"Object preview","room","","Preview space; not saved with the object."});
        d.spawn=staging;d.layout.push_back({staging,80,80});
    }
    d.validate();return d;
}
nlohmann::json Asset::json() const {
    preview();
    nlohmann::json j={{"format","grim-world-asset"},{"version",1},{"name",name},{"root",root},{"entities",nlohmann::json::array()}};
    for(const auto& e:entities)j["entities"].push_back({{"id",e.id},{"name",e.name},{"kind",e.kind},{"parent",e.parent},{"description",e.description},{"state",e.state}});
    return j;
}
Asset Asset::parse(const nlohmann::json& j) {
    need(j.at("format")=="grim-world-asset" && j.at("version")==1,"Unsupported asset format/version");
    Asset a;a.name=j.at("name");a.root=j.at("root");
    need(j.at("entities").is_array(),"Expected asset entities array");
    for(const auto& e:j.at("entities"))a.entities.push_back({e.at("id"),e.at("name"),e.at("kind"),e.at("parent"),e.at("description"),e.at("state")});
    a.preview();return a;
}
Asset Asset::capture(const Document& d,const std::string& root,const std::string& name) {
    d.validate();need(d.find(root)!=nullptr,"Select a room or object to save as an asset");
    Asset a;a.name=name;a.root=root;
    for(const auto& e:d.entities) {
        for(auto* node=&e;node;node=d.find(node->parent))if(node->id==root){a.entities.push_back(e);break;}
    }
    for(auto& e:a.entities)if(e.id==root)e.parent.clear();
    a.preview();return a;
}
Asset Asset::create(const std::string& kind) {
    Document d=Document::initial();
    auto root=kind=="room"?d.spawn:d.addObject(d.spawn,kind);
    d.find(root)->name="New "+kind;
    return capture(d,root,"New "+kind);
}
std::string Asset::insert(Document& destination,const std::string& parent,float x,float y) const {
    const auto source=preview();
    Document d=destination; // Failed placement cannot partially alter a world.
    std::map<std::string,std::string> ids;
    std::set<std::string> names;
    for(const auto& e:d.entities)names.insert(lower(e.name));
    for(const auto& p:d.passages)if(!p.doorName.empty())names.insert(lower(p.doorName));
    while(ids.size()<entities.size()) {
        bool progress=false;
        for(const auto& e:entities) {
            if(ids.count(e.id) || (e.id!=root && !ids.count(e.parent)))continue;
            const auto id=e.kind=="room"?d.addRoom(x,y):d.addObject(e.id==root?parent:ids.at(e.parent),e.kind);
            auto* inserted=d.find(id);const auto location=inserted->parent;
            *inserted=e;inserted->id=id;inserted->parent=location;
            std::string unique=e.name;
            for(unsigned n=2;names.count(lower(unique));++n) {
                auto stem=e.name;
                const auto suffix=" "+std::to_string(n);
                if(stem.size()+suffix.size()>80) {
                    auto end=80-suffix.size();
                    while(end>0 && (static_cast<unsigned char>(stem[end])&0xc0)==0x80)--end;
                    stem.resize(end);
                }
                unique=stem+suffix;
            }
            inserted->name=unique;names.insert(lower(unique));ids[e.id]=id;progress=true;
        }
        need(progress,"Cannot resolve asset contents");
    }
    d.validate();destination=std::move(d);return ids.at(root);
}
std::vector<AssetEntry> listAssets(const std::filesystem::path& directory) {
    std::vector<AssetEntry> result;
    for(const auto* folder:{"rooms","objects"}) {
        const auto path=directory/folder;
        if(!std::filesystem::exists(path))continue;
        for(const auto& entry:std::filesystem::directory_iterator(path)) {
            if(!entry.is_regular_file() || !lower(entry.path().filename().string()).ends_with(".grimasset.json"))continue;
            try {auto a=Asset::parse(loadJsonFile(entry.path()));result.push_back({entry.path(),std::string(folder)+" / "+a.name});}
            catch(const std::exception&) {result.push_back({entry.path(),std::string(folder)+" / [invalid] "+entry.path().filename().string()});}
        }
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.name<b.name;});return result;
}
std::filesystem::path newAssetPath(const std::filesystem::path& directory,const Asset& a) {
    auto d=a.preview();
    const auto folder=directory/(d.find(a.root)->kind=="room"?"rooms":"objects");
    std::filesystem::create_directories(folder);
    std::string stem;
    for(unsigned char c:a.name) {if(stem.size()>=48)break;stem+=std::isalnum(c) && c<128?static_cast<char>(std::tolower(c)):'_';}
    stem="asset_"+stem; // Also avoids Windows reserved device filenames.
    auto path=folder/(stem+".grimasset.json");
    for(unsigned n=2;std::filesystem::exists(path);++n)path=folder/(stem+"_"+std::to_string(n)+".grimasset.json");
    return path;
}
}
