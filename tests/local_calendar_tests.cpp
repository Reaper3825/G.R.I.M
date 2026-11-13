#include "plugins/schedule_plugin/local_calendar.hpp"
#include "core/calendar_time.hpp"
#include <cassert>
#include <vector>
#include <string>
static int32_t collect(void* p, const GrimScheduleEvent* e) {
    static_cast<std::vector<std::string>*>(p)->push_back(e->id); return 0;
}
int main() {
    using namespace std::chrono;
    assert(grim::scheduling::calendarGridStart(year{2026}/November)==sys_days{year{2026}/October/26});
    assert(grim::scheduling::calendarGridStart(year{2024}/February)==sys_days{year{2024}/January/29});
    assert(grim::scheduling::calendarGridStart(year{2026}/June)==sys_days{year{2026}/June/1});
    int64_t time=0;
    assert(grim::scheduling::parseCalendarTime("2024-02-29 23:59",time));
    assert(grim::scheduling::formatCalendarTime(time)=="2024-02-29 23:59");
    assert(!grim::scheduling::parseCalendarTime("2025-02-29 10:00",time));
    assert(!grim::scheduling::parseCalendarTime("2026-11-02 24:00",time));
    assert(!grim::scheduling::parseCalendarTime("2026-11-02 12:60",time));
    assert(!grim::scheduling::parseCalendarTime("2026-11-02 12:00x",time));
    assert(grim::scheduling::parseCalendarTime("1969-12-31 23:59",time) && time==-60000);
    assert(grim::scheduling::formatCalendarTime(time)=="1969-12-31 23:59");
    GrimScheduleProviderV1 p{};
    p.abi_version=1; p.struct_size=sizeof(p);
    assert(grim::scheduling::LocalCalendar::create(2,&p)==GRIM_SCHEDULE_UNSUPPORTED);
    assert(!p.context);
    assert(grim::scheduling::LocalCalendar::create(1,&p)==0);
    assert(p.capabilities==GRIM_SCHEDULE_CALENDAR && !p.put_alarm);
    assert(p.authorize(p.context,GRIM_SCHEDULE_ALARMS)==GRIM_SCHEDULE_UNSUPPORTED);
    GrimScheduleEvent e{"b","local","Meeting","","Etc/UTC",100,200,0};
    uint64_t rev=0;
    assert(p.put_event(p.context,&e,&rev)==0 && rev);
    assert(p.put_event(p.context,&e,&rev)==GRIM_SCHEDULE_CONFLICT);
    e.id="a"; assert(p.put_event(p.context,&e,&rev)==0);
    std::vector<std::string> ids;
    assert(p.query(p.context,"local",100,200,collect,&ids)==0);
    assert((ids==std::vector<std::string>{"a","b"}));
    ids.clear(); assert(p.query(p.context,"local",200,300,collect,&ids)==0 && ids.empty());
    assert(p.remove_event(p.context,"a",rev+1)==GRIM_SCHEDULE_CONFLICT);
    assert(p.remove_event(p.context,"a",rev)==0);
    auto old=rev; assert(p.put_event(p.context,&e,&rev)==0 && rev>old);
    e.revision=old; assert(p.put_event(p.context,&e,&rev)==GRIM_SCHEDULE_CONFLICT);
    e.timezone="America/New_York";
    assert(p.put_event(p.context,&e,&rev)==GRIM_SCHEDULE_UNSUPPORTED);
    p.destroy(p.context);
    namespace fs=std::filesystem;
    const auto root=fs::temp_directory_path()/("grim-calendar-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto file=root/"events.json";
    auto open=[&](GrimScheduleProviderV1& table){table={};table.abi_version=1;table.struct_size=sizeof(table);return grim::scheduling::LocalCalendar::create(1,&table,file);};
    assert(open(p)==0 && (p.capabilities & GRIM_SCHEDULE_PERSISTENT));
    GrimScheduleProviderV1 second{};
    assert(open(second)==GRIM_SCHEDULE_FAILED && !second.context);
    e={"saved","local","Persistent meeting","Notes","Etc/UTC",100,200,0};
    assert(p.put_event(p.context,&e,&rev)==0);
    const auto savedRevision=rev;
    p.destroy(p.context);
    assert(open(p)==0);
    ids.clear();assert(p.query(p.context,"local",0,300,collect,&ids)==0);
    assert((ids==std::vector<std::string>{"saved"}));
    // Failure to stage a write must not alter memory or revision output.
    fs::create_directory(root/"events.json.tmp");
    e.id="failed";rev=777;
    assert(p.put_event(p.context,&e,&rev)==GRIM_SCHEDULE_FAILED && rev==777);
    assert(p.remove_event(p.context,"saved",savedRevision)==GRIM_SCHEDULE_FAILED);
    ids.clear();p.query(p.context,"local",0,300,collect,&ids);
    assert((ids==std::vector<std::string>{"saved"}));
    fs::remove(root/"events.json.tmp");
    assert(p.remove_event(p.context,"saved",savedRevision)==0);
    p.destroy(p.context);assert(open(p)==0);
    ids.clear();p.query(p.context,"local",0,300,collect,&ids);assert(ids.empty());
    e.id="saved";assert(p.put_event(p.context,&e,&rev)==0 && rev>savedRevision);
    p.destroy(p.context);
    {std::ofstream bad(file);bad<<"{broken";}
    assert(open(p)==GRIM_SCHEDULE_FAILED && !p.context);
    {std::ifstream input(file);std::string content;input>>content;assert(content=="{broken");}
    fs::remove_all(root);
}
