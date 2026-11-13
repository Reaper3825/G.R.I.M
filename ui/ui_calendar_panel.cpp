#include "ui_calendar_panel.hpp"
#include "calendar_provider_client.hpp"
#include "core/calendar_time.hpp"
#include "overlay_renderer.hpp"
#include <limits>
#include <random>
#include <sstream>

UICalendarPanel::UICalendarPanel() : UIPanel("Calendar",true,"calendar") {
    position={240,180}; size={1160,600}; setVisible(false);
    scroll=std::make_shared<UIScrollBox>();
    addChild(scroll);
    selectedDay=std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
    const std::chrono::year_month_day date{selectedDay}; month=date.year()/date.month();
    previousMonth=std::make_shared<UIButton>("< Month",[this]{changeMonth(-1);});
    nextMonth=std::make_shared<UIButton>("Month >",[this]{changeMonth(1);});
    todayButton=std::make_shared<UIButton>("Today",[this]{
        auto today=std::chrono::floor<std::chrono::days>(std::chrono::system_clock::now());
        const std::chrono::year_month_day date{today};month=date.year()/date.month();chooseDay(today);
    });
    addChild(previousMonth);addChild(nextMonth);addChild(todayButton);
    dayGrid=std::make_shared<UIGrid>(7);
    dayGrid->setSpacing(10,10);dayGrid->setFitRows(true);addChild(dayGrid);
    for(size_t i=0;i<days.size();++i) {
        days[i]=std::make_shared<UIButton>("",[this,i]{chooseDay(gridStart()+std::chrono::days{int(i)});});
        dayGrid->addWidget(days[i]);
    }
    titleBox=std::make_shared<UIInputBox>(&titleText); titleBox->setPlaceholder("Event title");
    startBox=std::make_shared<UIInputBox>(&startText); startBox->setPlaceholder("YYYY-MM-DD HH:MM");
    endBox=std::make_shared<UIInputBox>(&endText); endBox->setPlaceholder("YYYY-MM-DD HH:MM");
    addButton=std::make_shared<UIButton>("Add event",[this]{add();});
    previousButton=std::make_shared<UIButton>("Previous",[this]{if(selected) --selected;});
    nextButton=std::make_shared<UIButton>("Next",[this]{if(selected+1<entries.size()) ++selected;});
    deleteButton=std::make_shared<UIButton>("Delete event",[this]{
        if(entries.empty()) return;
        const auto e=entries[selected];
        try {
            GRIM::UI::callSchedule("cancel",{{"id",e.id},{"revision",e.revision}});
            status="Event deleted."; refresh();
        } catch(const std::exception& error) {status=error.what();}

    });
    addChild(titleBox); addChild(startBox); addChild(endBox);
    addChild(addButton); addChild(previousButton); addChild(nextButton); addChild(deleteButton);
    chooseDay(selectedDay);
}
UICalendarPanel::~UICalendarPanel() = default;
std::chrono::sys_days UICalendarPanel::gridStart() const {
    return grim::scheduling::calendarGridStart(month);
}
void UICalendarPanel::changeMonth(int offset) {
    const auto next=month+std::chrono::months{offset};
    if(int(next.year())<1 || int(next.year())>9999) return;
    month=next;chooseDay(std::chrono::sys_days{month/1});
}
void UICalendarPanel::chooseDay(std::chrono::sys_days day) {
    const auto year=int(std::chrono::year_month_day{day}.year());
    if(year<1 || year>9999) return;
    selectedDay=day;selected=0;
    const auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(day.time_since_epoch()).count();
    startBox->setText(grim::scheduling::formatCalendarTime(ms+9*3600000));
    endBox->setText(grim::scheduling::formatCalendarTime(ms+10*3600000));
    refresh();
}
void UICalendarPanel::refresh() {
    try {
        std::vector<Entry> nextEntries; std::array<int,42> nextCounts{};
        const auto grid=std::chrono::duration_cast<std::chrono::milliseconds>(gridStart().time_since_epoch()).count();
        const auto day=std::chrono::duration_cast<std::chrono::milliseconds>(selectedDay.time_since_epoch()).count();
        size_t offset=0;
        for(;;) {
            const auto page=GRIM::UI::callSchedule("list",{{"kind","event"},{"from_utc_ms",grid},
                {"until_utc_ms",grid+42LL*86400000},{"limit",200},{"offset",offset}});
            const auto& rows=page.at("items");
            for(const auto& e:rows) {
                const auto start=e.at("start_utc_ms").get<int64_t>(), end=e.at("end_utc_ms").get<int64_t>();
                for(size_t i=0;i<42;++i) {
                    const auto cell=grid+int64_t(i)*86400000;
                    if(start<cell+86400000 && end>cell) ++nextCounts[i];
                }
                if(start<day+86400000 && end>day) nextEntries.push_back({e.at("id"),e.at("title"),
                    grim::scheduling::formatCalendarTime(start),grim::scheduling::formatCalendarTime(end),e.at("revision")});
            }
            offset+=rows.size();
            if(rows.empty() || offset>=page.at("total").get<size_t>()) break;
        }
        entries=std::move(nextEntries);counts=nextCounts;
        selected=entries.empty()?0:(std::min)(selected,entries.size()-1);
    } catch(const std::exception& error) {entries.clear();counts={};selected=0;status=error.what();}
}
void UICalendarPanel::add() {
    int64_t start,end;
    if(titleText.find_first_not_of(" \t\r\n")==std::string::npos || titleText.size()>120) {status="Enter a title (1-120 characters).";return;}
    if(!grim::scheduling::parseCalendarTime(startText,start) || !grim::scheduling::parseCalendarTime(endText,end) || end<=start) {
        status="Use valid UTC dates; end must be after start.";return;
    }
    // Random identifier remains independent of process lifetime and deleted records.
    std::random_device random;
    std::ostringstream key;key<<"calendar-"<<std::hex;
    for(int i=0;i<4;++i) key<<random()<<'-';
    const auto id=key.str();
    try { GRIM::UI::callSchedule("create",{{"id",id},{"kind","event"},{"title",titleText},
        {"start_utc_ms",start},{"end_utc_ms",end}}); }
    catch(const std::exception& error) {status=error.what();return;}
    titleBox->clear(); status="Event saved.";
    selectedDay=std::chrono::floor<std::chrono::days>(std::chrono::sys_time<std::chrono::milliseconds>{std::chrono::milliseconds{start}});
    const std::chrono::year_month_day date{selectedDay};month=date.year()/date.month();
    selected=0;refresh();
}
void UICalendarPanel::layout() {
    const auto c=getContentRect();
    const float width=(std::max)(1.0f,c.size.x-40);
    const bool wide=width>=900;
    gridWidth=wide ? width*0.46f : width;
    const float gridHeight=wide ? (std::max)(360.0f,c.size.y-130) : 360.0f;
    rowHeight=(gridHeight-100)/6;
    scroll->setPosition(c.origin.x,c.origin.y);scroll->setSize(c.size.x,c.size.y);
    scroll->setContentHeight(wide ? (std::max)(gridHeight+70,490.0f) : gridHeight+70+490);
    gridX=c.origin.x+12;gridY=c.origin.y+8-scroll->getScrollOffset();
    editorX=wide ? gridX+gridWidth+24 : gridX;
    editorY=wide ? gridY : gridY+gridHeight+70;
    editorWidth=wide ? width-gridWidth-24 : width;
    const float navWidth=(std::min)(100.0f,(gridWidth-24)/3);
    previousMonth->setPosition(gridX,gridY+32);previousMonth->setSize(navWidth,28);
    todayButton->setPosition(gridX+(gridWidth-navWidth)/2,gridY+32);todayButton->setSize(navWidth,28);
    nextMonth->setPosition(gridX+gridWidth-navWidth,gridY+32);nextMonth->setSize(navWidth,28);
    dayGrid->setPosition(gridX,gridY+100);
    dayGrid->setSize(gridWidth,6*rowHeight-10);dayGrid->layout();
    const float cell=gridWidth/7;
    for(size_t i=0;i<42;++i) {
        const auto day=gridStart()+std::chrono::days{int(i)};
        const std::chrono::year_month_day date{day};
        days[i]->setText(std::to_string(unsigned(date.day()))+(counts[i] && cell>=64?" ("+std::to_string(counts[i])+")":""));
        const auto base=day==selectedDay ? UITheme::Colors::Primary : UITheme::Colors::WidgetBg;
        days[i]->setColors(base,UITheme::Colors::WidgetBgHover,UITheme::Colors::WidgetBgActive);
    }
    titleBox->setPosition(editorX,editorY+52);titleBox->setSize(editorWidth,28);
    startBox->setPosition(editorX,editorY+108);startBox->setSize(editorWidth,28);
    endBox->setPosition(editorX,editorY+164);endBox->setSize(editorWidth,28);
    addButton->setPosition(editorX,editorY+210);addButton->setSize(120,28);
    const float buttonWidth=(editorWidth-24)/3;
    previousButton->setPosition(editorX,editorY+426);previousButton->setSize(buttonWidth,28);
    nextButton->setPosition(editorX+buttonWidth+12,editorY+426);nextButton->setSize(buttonWidth,28);
    deleteButton->setPosition(editorX+2*(buttonWidth+12),editorY+426);deleteButton->setSize(buttonWidth,28);
    deleteButton->setText(editorWidth<400?"Delete":"Delete event");
}
void UICalendarPanel::update(const InputState& input,float dt) {
    if(!isVisible()) return;
    refreshElapsed+=dt; if(refreshElapsed>=0.5f){refreshElapsed=0;refresh();}
    layout(); UIPanel::update(input,dt);
}
bool UICalendarPanel::drawOverlay(OverlayRenderer& renderer) {
    if(!UIPanel::drawOverlay(renderer)) return false;
    layout();
    const auto c=getContentRect();
    scroll->drawOverlay(renderer,position);
    const char* names[]={"January","February","March","April","May","June","July","August","September","October","November","December"};
    renderer.pushClipRect({gridX,c.origin.y},{gridWidth,c.size.y});
    renderer.drawText({gridX,gridY},std::string(names[unsigned(month.month())-1])+" "+std::to_string(int(month.year())),UITheme::Colors::TextPrimary);
    const char* weekdays[]={"Mon","Tue","Wed","Thu","Fri","Sat","Sun"};
    for(int i=0;i<7;++i) renderer.drawText({gridX+i*gridWidth/7,gridY+76},weekdays[i],UITheme::Colors::TextSecondary);
    previousMonth->drawOverlay(renderer,position);nextMonth->drawOverlay(renderer,position);todayButton->drawOverlay(renderer,position);
    dayGrid->drawOverlay(renderer,position);
    const auto dateText=grim::scheduling::formatCalendarTime(std::chrono::duration_cast<std::chrono::milliseconds>(selectedDay.time_since_epoch()).count()).substr(0,10);
    renderer.drawText({gridX,gridY+110+6*rowHeight},"Selected: "+dateText+" UTC",UITheme::Colors::TextPrimary);
    renderer.popClipRect();
    renderer.pushClipRect({editorX,c.origin.y},{editorWidth,c.size.y});
    auto text=[&](float offset,const std::string& value){renderer.drawText({editorX,editorY+offset},value,UITheme::Colors::TextPrimary);};
    text(0,"Saved local calendar | UTC");text(30,"Title");text(86,"Start (UTC)");text(142,"End (UTC)");
    text(254,status);
    if(entries.empty()) text(296,"No events on this day.");
    else {const auto& e=entries[selected];text(290,"Event "+std::to_string(selected+1)+" of "+std::to_string(entries.size()));
        text(316,e.title);text(348,"Start: "+e.start+" UTC");text(374,"End: "+e.end+" UTC");}
    titleBox->drawOverlay(renderer,position);startBox->drawOverlay(renderer,position);endBox->drawOverlay(renderer,position);
    addButton->drawOverlay(renderer,position);previousButton->drawOverlay(renderer,position);
    nextButton->drawOverlay(renderer,position);deleteButton->drawOverlay(renderer,position);
    renderer.popClipRect();renderer.popClipRect();return true;
}
