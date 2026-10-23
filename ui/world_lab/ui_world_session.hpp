#pragma once
#include "world_document.hpp"
#include "textworld_client.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_scrollbox.hpp"
#include <functional>

class UIWorldSession {
public:
    explicit UIWorldSession(GRIM::WorldLab::Editor&);
    void setChallenge(bool value);
    void update(const InputState&,float,const PanelRect&);
    void draw(OverlayRenderer&,const PanelRect&);
    void hide();
    void refresh();
    bool apply();
    bool hasPendingEdits() const { return objective_.getText()!=editor_.document().objective; }
    bool busy() const { return client_.busy(); }
    const std::string& status() const { return message_; }
    std::function<void(const std::string&)> onErrorEntity;
    std::function<void(const nlohmann::json&)> onState;
private:
    void layout(const PanelRect&);
    void conditionOptions();
    void addCondition(bool failure);
    void request(const std::string& operation);
    void receive();
    bool currentGame() const;
    void report(const std::function<void()>&);
    std::vector<Widget*> controls();
    GRIM::WorldLab::Editor& editor_;
    GRIM::WorldLab::TextWorldClient client_;
    bool challenge_=true,recording_=false,live_=false,done_=false;
    std::string operation_,sentCommand_,details_,message_="Define a challenge, then Compile & Play.";
    nlohmann::json compiled_,sentDocument_,lastDocument_;
    UIInputBox objective_,command_;
    UIScrollBox scroll_;
    PanelRect area_{},transcript_{};
    std::unique_ptr<UIDropdown> predicate_,subject_,target_,conditions_,history_;
    std::vector<std::string> subjects_,targets_,recorded_;
    std::vector<std::unique_ptr<UIButton>> challengeButtons_,playButtons_;
    struct Entry { std::string command,feedback; nlohmann::json state; };
    std::vector<Entry> historyEntries_;
    int historyIndex_=-1;
};
