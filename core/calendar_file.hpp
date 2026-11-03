#pragma once
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace grim::scheduling {
inline std::filesystem::path calendarStorePath() {
#ifdef _WIN32
    const DWORD length=GetEnvironmentVariableW(L"LOCALAPPDATA",nullptr,0);
    if(length) {
        std::wstring value(length,L'\0');
        const auto copied=GetEnvironmentVariableW(L"LOCALAPPDATA",value.data(),length);
        if(copied && copied<length) {value.resize(copied);return std::filesystem::path(value)/"GRIM"/"calendar"/"events.json";}
    }
#elif defined(__APPLE__)
    if (const auto p=std::getenv("HOME")) return std::filesystem::path(p)/"Library"/"Application Support"/"GRIM"/"calendar"/"events.json";
#else
    if (const auto p=std::getenv("XDG_DATA_HOME"); p && std::filesystem::path(p).is_absolute()) return std::filesystem::path(p)/"grim"/"calendar"/"events.json";
    if (const auto p=std::getenv("HOME")) return std::filesystem::path(p)/".local"/"share"/"grim"/"calendar"/"events.json";
#endif
    throw std::runtime_error("Cannot locate calendar data directory");
}
// Exclusive process-lifetime writer lease. OS releases it on abnormal exit.
class CalendarFile {
    std::filesystem::path path;
#ifdef _WIN32
    HANDLE lock=INVALID_HANDLE_VALUE;
#else
    int lock=-1;
#endif
public:
    explicit CalendarFile(std::filesystem::path p):path(std::move(p)) {
        std::filesystem::create_directories(path.parent_path());
        auto lockPath=path; lockPath += ".lock";
#ifdef _WIN32
        lock=CreateFileW(lockPath.c_str(),GENERIC_READ|GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(lock==INVALID_HANDLE_VALUE) throw std::runtime_error("Calendar is locked or inaccessible");
#else
        lock=::open(lockPath.c_str(),O_CREAT|O_RDWR,0600);
        if(lock<0) throw std::runtime_error("Calendar lock inaccessible");
        if(flock(lock,LOCK_EX|LOCK_NB)!=0) {::close(lock);lock=-1;throw std::runtime_error("Calendar is locked");}
#endif
    }
    CalendarFile(const CalendarFile&)=delete;
    ~CalendarFile() {
#ifdef _WIN32
        if(lock!=INVALID_HANDLE_VALUE) CloseHandle(lock);
#else
        if(lock>=0) ::close(lock);
#endif
    }
    const std::filesystem::path& location() const {return path;}
    void write(const std::string& data) {
        auto tmp=path; tmp += ".tmp";
        {std::ofstream out(tmp,std::ios::binary|std::ios::trunc); out.exceptions(std::ios::failbit|std::ios::badbit); out<<data;out.close();}
#ifdef _WIN32
        if(!MoveFileExW(tmp.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) throw std::runtime_error("Cannot commit calendar");
#else
        std::filesystem::rename(tmp,path);
#endif
    }
};
}
