#pragma once
#include "primitives/ui_panel.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "primitives/ui_grid.hpp"
#include "core/scheduling_api.h"
#include <chrono>
#include <array>
class UICalendarPanel : public UIPanel {
public:
    UICalendarPanel();
    ~UICalendarPanel() override;
    UICalendarPanel(const UICalendarPanel&) = delete;
    UICalendarPanel& operator=(const UICalendarPanel&) = delete;
    void update(const InputState&, float) override;
    bool drawOverlay(OverlayRenderer&) override;
private:
    struct Entry { std::string id,title,start,end; uint64_t revision; };
    GrimScheduleProviderV1 provider{};
    std::vector<Entry> entries;
    size_t selected=0;
    float refreshElapsed=0;
    std::chrono::year_month month;
    std::chrono::sys_days selectedDay;
    std::shared_ptr<UIGrid> dayGrid;
    std::array<std::shared_ptr<UIButton>,42> days;
    std::array<int,42> counts{};
    std::shared_ptr<UIButton> previousMonth,nextMonth,todayButton;
    void changeMonth(int offset);
    std::chrono::sys_days gridStart() const;
    void chooseDay(std::chrono::sys_days day);
    std::string titleText,startText,endText,status;
    std::shared_ptr<UIInputBox> titleBox,startBox,endBox;
    std::shared_ptr<UIButton> addButton,deleteButton,previousButton,nextButton;
    void refresh();
    void add();
    void layout();
    std::shared_ptr<UIScrollBox> scroll;
    float gridX=0,gridY=0,gridWidth=0,rowHeight=0,editorX=0,editorY=0,editorWidth=0;
};
