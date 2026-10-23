#include "world_document.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <set>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace GRIM::WorldLab {
namespace {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
bool room(const Entity* e) { return e && e->kind == "room"; }
bool holder(const Entity* e) { return e && (e->kind == "room" || e->kind == "container" || e->kind == "supporter"); }
std::string allocate(Document& d) {
    for (;;) {
        require(d.nextId < 1000000000, "Entity ID limit reached");
        auto id = "e" + std::to_string(d.nextId++);
        if (!d.find(id) && std::none_of(d.passages.begin(), d.passages.end(), [&](const auto& p){ return p.id == id; })) return id;
    }
}
}
std::string opposite(const std::string& d) {
    if (d == "north") return "south";
    if (d == "south") return "north";
    if (d == "east") return "west";
    if (d == "west") return "east";
    throw std::runtime_error("Unknown compass direction: " + d);
}
const Entity* Document::find(const std::string& id) const {
    auto it = std::find_if(entities.begin(), entities.end(), [&](const auto& e){return e.id == id;});
    return it == entities.end() ? nullptr : &*it;
}
Entity* Document::find(const std::string& id) { return const_cast<Entity*>(static_cast<const Document*>(this)->find(id)); }
const RoomLayout* Document::position(const std::string& id) const {
    auto it = std::find_if(layout.begin(), layout.end(), [&](const auto& p){return p.id == id;});
    return it == layout.end() ? nullptr : &*it;
}
std::string Document::addRoom(float x, float y) {
    auto id = allocate(*this);
    entities.push_back({id, "Room " + std::to_string(layout.size()+1), "room", "", ""});
    layout.push_back({id,x,y});
    if (spawn.empty()) spawn = id;
    return id;
}
std::string Document::addObject(const std::string& parent, const std::string& kind) {
    require(holder(find(parent)), "Select a room, container, or supporter first");
    require(kind == "object" || kind == "key" || kind == "container" || kind == "supporter" || kind == "food", "Unsupported object kind");
    auto id = allocate(*this);
    entities.push_back({id, "New " + kind, kind, parent, ""});
    return id;
}
std::string Document::connect(const std::string& from, const std::string& to, const std::string& direction) {
    require(room(find(from)) && room(find(to)) && from != to, "Connect two different rooms");
    const auto reverse = opposite(direction);
    for (const auto& p : passages) {
        require(!((p.from == from && p.direction == direction) || (p.to == from && opposite(p.direction) == direction)), "That exit is already connected");
        require(!((p.from == to && p.direction == reverse) || (p.to == to && opposite(p.direction) == reverse)), "The destination exit is already connected");
        require(!((p.from == from && p.to == to) || (p.from == to && p.to == from)), "These rooms are already connected");
    }
    auto id = allocate(*this); passages.push_back({id,from,to,direction}); return id;
}
void Document::moveRoom(const std::string& id, float x, float y) {
    for (auto& p : layout) if (p.id == id) { p.x=x; p.y=y; return; }
    throw std::runtime_error("Room no longer exists");
}
void Document::erase(const std::string& id) {
    std::set<std::string> removed{id};
    bool changed;
    do { changed=false; for (const auto& e : entities) if (removed.count(e.parent)) changed |= removed.insert(e.id).second; } while(changed);
    entities.erase(std::remove_if(entities.begin(),entities.end(),[&](const auto& e){return removed.count(e.id);}),entities.end());
    layout.erase(std::remove_if(layout.begin(),layout.end(),[&](const auto& p){return removed.count(p.id);}),layout.end());
    for (const auto& p : passages) if (removed.count(p.from) || removed.count(p.to)) removed.insert(p.id);
    passages.erase(std::remove_if(passages.begin(),passages.end(),[&](const auto& p){return removed.count(p.id)||removed.count(p.from)||removed.count(p.to);}),passages.end());
    for (auto& p : passages) if (removed.count(p.key)) { p.key.clear(); if(p.doorState=="locked")p.doorState="closed"; }
    for (auto* conditions : {&winConditions, &failConditions}) {
        conditions->erase(std::remove_if(conditions->begin(), conditions->end(), [&](const auto& c) {
            return removed.count(c.subject) || removed.count(c.target);
        }), conditions->end());
    }
    if (removed.count(spawn)) spawn = layout.empty() ? "" : layout.front().id;
}
void Document::validate() const {
    require(!title.empty() && title.size() <= 256, "World title must contain 1-256 bytes");
    require(entities.size() <= 10000 && passages.size() <= 20000, "World is too large");
    require(nextId > 0 && nextId < 1000000000, "Invalid next ID");
    std::set<std::string> ids, positions;
    std::set<std::pair<std::string,std::string>> exits, pairs;
    for (const auto& e : entities) {
        require(!e.id.empty() && e.id.size() <= 128 && ids.insert(e.id).second, "Duplicate or invalid entity ID");
        require(!e.name.empty() && e.name.size() <= 256 && e.description.size() <= 16384, "Invalid name or description: " + e.id);
        require(e.kind=="room" || e.kind=="object" || e.kind=="key" || e.kind=="container" || e.kind=="supporter" || e.kind=="food", "Unsupported kind: " + e.kind);
        require(e.state=="open" || e.state=="closed", "Invalid container state");
        if(e.kind=="room") require(e.parent.empty(), "Rooms cannot have a parent");
        else {
            require(holder(find(e.parent)), "Object has no valid location: " + e.name);
            std::set<std::string> chain{e.id};
            auto p = find(e.parent);
            while(p) { require(chain.insert(p->id).second, "Containment cycle: " + e.name); p=find(p->parent); }
        }
    }
    for(const auto& p:layout) {
        require(room(find(p.id)) && positions.insert(p.id).second, "Invalid or duplicate room layout");
        require(std::isfinite(p.x) && std::isfinite(p.y) && std::abs(p.x)<=1e6f && std::abs(p.y)<=1e6f, "Room position is out of range");
    }
    for(const auto& e:entities) if(e.kind=="room") require(positions.count(e.id)!=0, "Room has no layout");
    require(layout.empty() ? spawn.empty() : room(find(spawn)), "Player start must be a room");
    for(const auto& p:passages) {
        require(!p.id.empty() && p.id.size()<=128 && ids.insert(p.id).second, "Duplicate or invalid passage ID");
        require(room(find(p.from)) && room(find(p.to)) && p.from!=p.to, "Invalid passage endpoints");
        require(exits.insert({p.from,p.direction}).second && exits.insert({p.to,opposite(p.direction)}).second, "Multiple passages use the same exit");
        require(pairs.insert({std::min(p.from,p.to),std::max(p.from,p.to)}).second, "Duplicate room connection");
        require(p.doorName.size()<=80, "Door name exceeds 80 bytes");
        require(p.doorState=="open" || p.doorState=="closed" || p.doorState=="locked", "Invalid door state");
        require(p.key.empty() || (find(p.key) && find(p.key)->kind=="key"), "Door must reference a key");
        require(p.doorName.empty() ? (p.key.empty() && p.doorState=="open") : (p.doorState!="locked" || !p.key.empty()), "Locked doors need a name and matching key");
    }
    require(objective.size()<=4096 && winConditions.size()<=64 && failConditions.size()<=64, "Challenge is too large");
    for(const auto* conditions : {&winConditions,&failConditions})for(const auto& c:*conditions)validateCondition(c);
    require(solution.size()<=1000,"Solution exceeds 1000 commands");
    for(const auto& command:solution)require(!command.empty() && command.size()<=512 && command.find_first_of("\r\n\0",0,3)==std::string::npos,"Invalid solution command");
}
void Document::validateCondition(const Condition& c) const {
    const auto* e=find(c.subject);
    const bool portable=e && (e->kind=="object" || e->kind=="key" || e->kind=="food");
    const bool door=std::any_of(passages.begin(),passages.end(),[&](const auto& p){return p.id==c.subject && !p.doorName.empty();});
    const auto* target=find(c.target);
    bool valid=false;
    if(c.predicate=="player_at")valid=room(e) && c.target.empty();
    else if(c.predicate=="carrying")valid=portable && c.target.empty();
    else if(c.predicate=="eaten")valid=e && e->kind=="food" && c.target.empty();
    else if(c.predicate=="open" || c.predicate=="closed" || c.predicate=="locked")valid=(door || (e && e->kind=="container")) && c.target.empty();
    else if(c.predicate=="at")valid=portable && room(target);
    else if(c.predicate=="in")valid=portable && target && target->kind=="container";
    else if(c.predicate=="on")valid=portable && target && target->kind=="supporter";
    require(valid,"Invalid "+c.predicate+" condition for "+c.subject);
}
nlohmann::json Document::json() const {
    nlohmann::json j={{"format","grim-world"},{"version",2},{"title",title},{"spawn",spawn},{"next_id",nextId},
        {"entities",nlohmann::json::array()},{"passages",nlohmann::json::array()},{"editor",{{"rooms",nlohmann::json::array()}}}};
    for(const auto& e:entities) j["entities"].push_back({{"id",e.id},{"name",e.name},{"kind",e.kind},{"parent",e.parent},{"description",e.description},{"state",e.state}});
    for(const auto& p:passages) j["passages"].push_back({{"id",p.id},{"from",p.from},{"to",p.to},{"direction",p.direction},{"door_name",p.doorName},{"door_state",p.doorState},{"key",p.key}});
    for(const auto& p:layout) j["editor"]["rooms"].push_back({{"id",p.id},{"x",p.x},{"y",p.y}});
    j["objective"]=objective;j["solution"]=solution;
    j["win_conditions"]=nlohmann::json::array();j["fail_conditions"]=nlohmann::json::array();
    for(const auto& c:winConditions)j["win_conditions"].push_back({{"predicate",c.predicate},{"subject",c.subject},{"target",c.target}});
    for(const auto& c:failConditions)j["fail_conditions"].push_back({{"predicate",c.predicate},{"subject",c.subject},{"target",c.target}});
    return j;
}
nlohmann::json Document::gameJson() const { auto j=json();j.erase("editor");j.erase("solution");j.erase("next_id");return j; }
Document Document::parse(const nlohmann::json& j) {
    require(j.at("format")=="grim-world" && (j.at("version")==1 || j.at("version")==2), "Unsupported world file format/version");
    Document d;
    d.title=j.at("title").get<std::string>(); d.spawn=j.at("spawn").get<std::string>();
    require(j.at("next_id").is_number_integer(),"Invalid next ID");
    const auto next=j.at("next_id").get<int64_t>(); require(next>0 && next<1000000000,"Invalid next ID"); d.nextId=static_cast<unsigned>(next);
    require(j.at("entities").is_array() && j.at("passages").is_array() && j.at("editor").at("rooms").is_array(), "Expected world arrays");
    for(const auto& e:j.at("entities")) d.entities.push_back({e.at("id"),e.at("name"),e.at("kind"),e.at("parent"),e.at("description"),e.at("state")});
    for(const auto& p:j.at("passages")) d.passages.push_back({p.at("id"),p.at("from"),p.at("to"),p.at("direction"),p.value("door_name",std::string{}),p.value("door_state",std::string{"open"}),p.value("key",std::string{})});
    if(j.at("version")==2) {
        require(j.at("solution").is_array() && j.at("win_conditions").is_array() && j.at("fail_conditions").is_array(),"Expected challenge arrays");
        d.objective=j.at("objective");d.solution=j.at("solution").get<std::vector<std::string>>();
        for(const auto& c:j.at("win_conditions"))d.winConditions.push_back({c.at("predicate"),c.at("subject"),c.at("target")});
        for(const auto& c:j.at("fail_conditions"))d.failConditions.push_back({c.at("predicate"),c.at("subject"),c.at("target")});
    }
    for(const auto& p:j.at("editor").at("rooms")) d.layout.push_back({p.at("id"),p.at("x"),p.at("y")});
    d.validate(); return d;
}
Document Document::initial() { Document d; auto id=d.addRoom(80,80); d.find(id)->name="Starting room"; return d; }
Editor::Editor():current_(Document::initial()) {}
void Editor::reset(const Document& document) {document.validate();current_=document;undo_.clear();redo_.clear();saved_=current_.json();}
bool Editor::change(const std::function<void(Document&)>& operation) {
    auto candidate=current_; operation(candidate); candidate.validate();
    if(candidate.json()==current_.json()) return false;
    undo_.push_back(current_); if(undo_.size()>100) undo_.erase(undo_.begin());
    current_=std::move(candidate); redo_.clear(); return true;
}
bool Editor::undo() { if(undo_.empty()) return false; redo_.push_back(current_); current_=std::move(undo_.back()); undo_.pop_back(); return true; }
bool Editor::redo() { if(redo_.empty()) return false; undo_.push_back(current_); current_=std::move(redo_.back()); redo_.pop_back(); return true; }
bool Editor::dirty() const { return current_.json()!=saved_; }
void Editor::save(const std::filesystem::path& path) {
    current_.validate(); const auto serialized=current_.json();
    saveJsonFile(path,serialized);saved_=serialized;
}
void saveJsonFile(const std::filesystem::path& path,const nlohmann::json& serialized) {
    require(!path.empty(),"Enter a world file path");
    // Write alongside the destination, then replace atomically. Never truncate the old save.
    // An exclusively created staging directory avoids clobbering an unrelated
    // .tmp file or another editor's in-progress save.
    auto staging=path;
    staging += ".saving-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    require(std::filesystem::create_directory(staging),"Cannot create save staging directory");
    const auto temp=staging/"world.json";
    struct Cleanup {
        std::filesystem::path file, directory;
        ~Cleanup(){std::error_code ec;std::filesystem::remove(file,ec);std::filesystem::remove(directory,ec);}
    } cleanup{temp,staging};
    { std::ofstream out(temp,std::ios::binary|std::ios::trunc);
      if (!out) {
          const auto utf8 = temp.u8string();
          throw std::runtime_error("Cannot open save file: " +
              std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size()));
      }
      out << serialized.dump(2) << '\n'; out.close(); require(bool(out),"Failed to write world file"); }
#ifdef _WIN32
    if(!MoveFileExW(temp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) {
        const auto error=GetLastError();
        throw std::runtime_error("Cannot replace world file (Windows error " + std::to_string(error) + ")");
    }
#else
    std::filesystem::rename(temp,path);
#endif
}
nlohmann::json loadJsonFile(const std::filesystem::path& path) {
    require(std::filesystem::file_size(path)<=16*1024*1024,"World file exceeds 16 MB");
    std::ifstream in(path,std::ios::binary); require(bool(in),"Cannot open world file");
    nlohmann::json j; in >> j; in >> std::ws;
    require(in.eof(),"Unexpected trailing data in world file");
    return j;
}
void Editor::load(const std::filesystem::path& path) {
    auto candidate=Document::parse(loadJsonFile(path));
    change([&](Document& d){d=candidate;}); saved_=candidate.json();
}
void Editor::newWorld() { change([](Document& d){d=Document::initial();}); saved_=nullptr; }
}
