#pragma once
#include <chrono>
#include <string>
#include <cstdio>
namespace grim::scheduling {
inline std::chrono::sys_days calendarGridStart(std::chrono::year_month month) {
    const std::chrono::sys_days first{month/1};
    return first-std::chrono::days{std::chrono::weekday{first}.iso_encoding()-1};
}
// Strict UTC civil minute input. No platform-dependent local-time conversion.
inline bool parseCalendarTime(const std::string& text, int64_t& result) {
    if (text.size()!=16 || text[4]!='-' || text[7]!='-' || text[10]!=' ' || text[13]!=':') return false;
    for (size_t i=0;i<text.size();++i)
        if (i!=4 && i!=7 && i!=10 && i!=13 && (text[i]<'0' || text[i]>'9')) return false;
    const int y=std::stoi(text.substr(0,4)), m=std::stoi(text.substr(5,2)), d=std::stoi(text.substr(8,2));
    const int h=std::stoi(text.substr(11,2)), minute=std::stoi(text.substr(14,2));
    const std::chrono::year_month_day date{std::chrono::year{y},std::chrono::month{unsigned(m)},std::chrono::day{unsigned(d)}};
    if (y<1 || !date.ok() || h>23 || minute>59) return false;
    result=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::sys_days{date}.time_since_epoch()+std::chrono::hours{h}+std::chrono::minutes{minute}).count();
    return true;
}
inline std::string formatCalendarTime(int64_t ms) {
    using namespace std::chrono;
    const sys_time<milliseconds> time{milliseconds{ms}};
    const auto date=floor<days>(time);
    const year_month_day ymd{date};
    const hh_mm_ss clock{time-date};
    char buffer[32];
    std::snprintf(buffer,sizeof(buffer),"%04d-%02u-%02u %02lld:%02lld",int(ymd.year()),unsigned(ymd.month()),unsigned(ymd.day()),
        static_cast<long long>(clock.hours().count()),static_cast<long long>(clock.minutes().count()));
    return buffer;
}
}
