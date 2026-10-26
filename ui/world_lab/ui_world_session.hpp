#pragma once
#include "world_goal.hpp"
#include "textworld_client.hpp"
#include "ui_world_actions.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_textarea.hpp"

class UIWorldSession {
public:
    explicit UIWorldSession(GRIM::WorldLab::Editor&, std::filesystem::path recordsDirectory = {});
    void setGoalAccess(GRIM::WorldLab::GoalAccess access) { access_=std::move(access); }
    void selectGoal(nlohmann::json selection);
    void reloadGoals();
    void importLegacyGoal();
    void setChallenge(bool value);
    void update(const InputState&,float,const PanelRect&);
    void draw(OverlayRenderer&,const PanelRect&);
    void hide();
    void refresh();
    void prepareCommand(const std::string& command) { command_.setText(command); }
    const std::string& preparedCommand() const { return command_.getText(); }
    bool apply() { return true; }
    bool hasPendingEdits() const { return false; }
    bool busy() const { return client_.busy(); }
    const std::string& status() const { return message_; }
    const nlohmann::json& selectedGoal() const { return selected_; }
    std::function<void(const std::string&)> onErrorEntity;
    std::function<void(const nlohmann::json&)> onState;
private:
    void layout(const PanelRect&);
    void request(const std::string& operation);
    void receive();
    bool currentGame() const;
    bool currentEpisode() const;
    void report(const std::function<void()>&);
    std::vector<Widget*> controls();
    std::filesystem::path recordPath() const;
    void saveWitness(const nlohmann::json&);
    GRIM::WorldLab::Editor& editor_;
    GRIM::WorldLab::GoalAccess access_;
    std::filesystem::path recordsDirectory_;
    UIWorldActions actions_;
    GRIM::WorldLab::TextWorldClient client_;
    bool challenge_=true,recording_=false,live_=false,done_=false;
    std::string operation_,sentCommand_,details_,message_="Select a ConceptBlock Goal, then compile/start.";
    nlohmann::json compiled_,sentDocument_,lastDocument_,selected_,active_,sentGoal_,episode_;
    UIInputBox command_,goalSearch_;
    UITextArea transcriptText_{"", "", {}};
    PanelRect area_{},transcript_{};
    std::unique_ptr<UIDropdown> goals_,history_,available_;
    std::vector<nlohmann::json> choices_;
    std::vector<std::string> availableCommands_,recorded_;
    bool catalogCompatible_=true,composerVisible_=false;
    std::vector<std::unique_ptr<UIButton>> challengeButtons_,playButtons_;
    struct Entry { std::string command,feedback; nlohmann::json state; };
    std::vector<Entry> historyEntries_;
    int historyIndex_=-1;
};
