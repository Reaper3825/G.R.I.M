#pragma once
#include "world_goal.hpp"
#include "resources/models/GRIM-text/Shared/Environment/EnvironmentInput.hpp"
#include "textworld_client.hpp"
#include "ui_world_actions.hpp"
#include "ui/primitives/ui_panel.hpp"
#include "ui/primitives/ui_button.hpp"
#include "ui/primitives/ui_inputbox.hpp"
#include "ui/primitives/ui_dropdown.hpp"
#include "ui/primitives/ui_textarea.hpp"
#include <optional>

class UIWorldSession {
public:
    explicit UIWorldSession(GRIM::WorldLab::Editor&, std::filesystem::path recordsDirectory = {});
    void setGoalAccess(GRIM::WorldLab::GoalAccess access) { access_=std::move(access); }
    void selectGoal(nlohmann::json selection);
    void reloadGoals();
    void reloadRegistry();
    void selectCurriculum(int index);
    void selectCourse(int index);
    void moveConceptBlock(int delta);
    int conceptBlockIndex() const { return blockIndex_; }
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
    const std::optional<GRIM::Environment::EnvironmentInferenceInput>& initialModelInput() const { return initialModelInput_; }
    // Manual preview boundaries are byte offsets, not sampled model token positions.
    struct HistoryEntry {
        std::string command, observation, continuationStatus;
        nlohmann::json state, receipt;
        std::size_t trajectoryEnd = 0;
    };
    const std::vector<HistoryEntry>& historyEntries() const { return historyEntries_; }
    int historyIndex() const { return historyIndex_; }
    void selectHistory(int index);
    std::string historyModelText(std::size_t index) const { return trajectory_.substr(0, historyEntries_.at(index).trajectoryEnd); }
    std::string initialModelText() const { return initialModelInput_ && !historyEntries_.empty()?historyModelText(0):std::string{}; }
    std::string continuationStatus() const;
    std::string historyHostDetails(std::size_t index) const;
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
    void selectCourseBlock(int index);
    void refreshGoalChoices();
    void clearHistory();
    void appendHistory(HistoryEntry entry);
    GRIM::WorldLab::Editor& editor_;
    GRIM::WorldLab::GoalAccess access_;
    std::filesystem::path recordsDirectory_;
    UIWorldActions actions_;
    GRIM::WorldLab::TextWorldClient client_;
    bool challenge_=true,recording_=false,live_=false,done_=false;
    std::string operation_,sentCommand_,details_,message_="Select a ConceptBlock Goal, then compile/start.";
    nlohmann::json compiled_,sentDocument_,lastDocument_,selected_,active_,sentGoal_,episode_;
    std::optional<GRIM::Environment::EnvironmentInferenceInput> initialModelInput_;
    UIInputBox command_,goalSearch_;
    UITextArea transcriptText_{"", "", {}};
    PanelRect area_{},transcript_{};
    std::unique_ptr<UIDropdown> goals_,history_,available_;
    std::unique_ptr<UIButton> historyView_;
    bool hostDetailsVisible_=false;
    std::unique_ptr<UIDropdown> curriculums_,courses_;
    std::unique_ptr<UIButton> previousBlock_,nextBlock_;
    nlohmann::json registry_;
    std::vector<nlohmann::json> courseChoices_;
    std::vector<std::string> courseBlockIds_;
    int curriculumIndex_=-1,courseIndex_=-1,blockIndex_=-1;
    std::string blockName_;
    std::vector<nlohmann::json> choices_;
    std::size_t searchChoiceCount_=1;
    std::vector<std::string> availableCommands_,recorded_;
    bool catalogCompatible_=true,composerVisible_=false;
    std::vector<std::unique_ptr<UIButton>> challengeButtons_,playButtons_;
    std::string trajectory_,pendingExecute_,continuationStatus_="not_started";
    std::vector<HistoryEntry> historyEntries_;
    int historyIndex_=-1;
};
