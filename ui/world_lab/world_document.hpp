#pragma once
#include <nlohmann/json.hpp>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace GRIM::WorldLab {
std::string makeEnvironmentId();
struct Entity {
    std::string id, name, kind, parent, description;
    std::string state = "open"; // Containers: open or closed. Other kinds ignore this.
};
struct RoomLayout { std::string id; float x = 0, y = 0; };
struct RoomPath { std::string id, parent, segment; };
struct Passage {
    std::string id, from, to, direction;
    std::string doorName, doorState = "open", key;
};
struct Condition { std::string predicate, subject, target; };
struct Document {
    std::string environmentId = makeEnvironmentId();
    std::string title = "Untitled world", spawn;
    unsigned nextId = 1;
    std::vector<Entity> entities;
    std::vector<RoomLayout> layout;
    std::vector<Passage> passages;
    // Separate from object containment. Empty means legacy compass navigation.
    std::vector<RoomPath> paths;
    std::string objective;
    std::vector<Condition> winConditions, failConditions;
    std::vector<std::string> solution;

    const Entity* find(const std::string& id) const;
    Entity* find(const std::string& id);
    const RoomLayout* position(const std::string& id) const;
    std::string addRoom(float x, float y);
    std::string addObject(const std::string& parent, const std::string& kind);
    std::string connect(const std::string& from, const std::string& to, const std::string& direction);
    void erase(const std::string& id); // Removes contents and incident passages together.
    void moveRoom(const std::string& id, float x, float y);
    void enablePathNavigation(const std::string& root);
    std::string roomPath(const std::string& id) const;
    void setRoomPath(const std::string& id, const std::string& path);
    void validate() const;
    void validateCondition(const Condition&) const;
    // The compiled game excludes editor positions and recorded solution metadata.
    nlohmann::json gameJson() const;
    nlohmann::json json() const;
    static Document parse(const nlohmann::json& value);
    static Document initial();
};
void saveJsonFile(const std::filesystem::path&, const nlohmann::json&);
nlohmann::json loadJsonFile(const std::filesystem::path&);
std::string opposite(const std::string& direction);

// Snapshot transactions keep multi-entity edits atomic. UI gestures commit once.
class Editor {
public:
    Editor();
    void reset(const Document& document);
    const Document& document() const { return current_; }
    bool change(const std::function<void(Document&)>& operation);
    bool undo();
    bool redo();
    bool canUndo() const { return !undo_.empty(); }
    bool canRedo() const { return !redo_.empty(); }
    bool dirty() const;
    void markUnsaved() { saved_ = nullptr; }
    void save(const std::filesystem::path& path);
    void load(const std::filesystem::path& path);
    void newWorld();
    void copyWorld(); // Independent environment identity; contents are preserved.
private:
    Document current_;
    std::vector<Document> undo_, redo_;
    nlohmann::json saved_;
};
}
