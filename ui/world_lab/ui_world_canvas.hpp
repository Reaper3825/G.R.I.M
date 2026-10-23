#pragma once
#include "ui/primitives/ui_canvas.hpp"
#include "world_document.hpp"
#include <functional>
#include <map>

class UIWorldCanvas : public UICanvas {
public:
    explicit UIWorldCanvas(GRIM::WorldLab::Editor& editor) : editor_(editor) {}
    std::string selected;
    bool selectionOnly = false; // Asset preview: select contents without changing topology.
    std::function<bool(const std::string&)> onSelect;
    std::function<void(const std::string&)> onError;
    std::function<void()> onChanged;
    void cancelInput() override;
    void fitWorld();
    void setPlayState(const nlohmann::json& state);
    bool wantsFocus() const override { return true; }
protected:
    bool updateCanvasContent(const InputState&, float) override;
    void drawCanvasContent(OverlayRenderer&) override;
private:
    struct Row { std::string id; int depth; };
    std::vector<Row> rows(const std::string& room) const;
    CanvasRect roomRect(const std::string& id) const;
    Vec2 port(const std::string& room, const std::string& direction) const;
    std::string hitRoom(Vec2 p) const;
    std::string hitEntity(Vec2 p) const;
    bool select(std::string id);
    GRIM::WorldLab::Editor& editor_;
    std::string drag_, direction_;
    Vec2 start_{}, pointer_{};
    bool moved_ = false;
    std::string playRoom_;
    std::vector<std::string> changedEntities_;
    nlohmann::json playFacts_;
    std::map<std::string,std::string> playStates_;
};
