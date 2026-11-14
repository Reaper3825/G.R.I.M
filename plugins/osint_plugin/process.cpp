#include "process.hpp"
#include "provider_paths.hpp"
#include <chrono>
#include <thread>
#include <stdexcept>
#ifdef _WIN32
#include <windows.h>
namespace {
std::wstring wide(const std::string& s) {
 int size=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),static_cast<int>(s.size()),nullptr,0);
 if (!size) throw std::runtime_error("Invalid process argument encoding");
 std::wstring result(size,0); MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),static_cast<int>(s.size()),result.data(),size); return result;
}
std::wstring quote(const std::string& text) {
 std::wstring result=L"\""; size_t slashes=0;
 for(wchar_t c:wide(text)) {
  if(c==L'\\') { ++slashes; continue; }
  result.append(c==L'"'?slashes*2+1:slashes,L'\\'); slashes=0; result+=c;
 }
 result.append(slashes*2,L'\\'); return result+L"\"";
}
struct Handle { HANDLE value=nullptr; ~Handle(){ if(value) CloseHandle(value); } };
}
#else
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <cerrno>
#endif
int osintProvider::runProcess(const std::vector<std::string>& arguments,int timeoutSeconds) {
 if(arguments.empty() || cancelled) return -1;
 const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(timeoutSeconds);
#ifdef _WIN32
 std::wstring command;
 for(const auto& arg:arguments) { if(!command.empty()) command+=L' '; command+=quote(arg); }
 Handle job{CreateJobObjectW(nullptr,nullptr)};
 JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{}; limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
 if(!job.value || !SetInformationJobObject(job.value,JobObjectExtendedLimitInformation,&limits,sizeof(limits))) return -1;
 STARTUPINFOW startup{}; startup.cb=sizeof(startup); PROCESS_INFORMATION process{};
 if(!CreateProcessW(nullptr,command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW|CREATE_SUSPENDED,nullptr,root.c_str(),&startup,&process)) return -1;
 Handle child{process.hProcess}, thread{process.hThread};
 if(!AssignProcessToJobObject(job.value,child.value)) { TerminateProcess(child.value,1); WaitForSingleObject(child.value,INFINITE); return -1; }
 if(ResumeThread(thread.value)==static_cast<DWORD>(-1)) { TerminateJobObject(job.value,1); WaitForSingleObject(child.value,INFINITE); return -1; }
 for(;;) {
  const auto status=WaitForSingleObject(child.value,50);
  if(status==WAIT_OBJECT_0) { DWORD code=1; GetExitCodeProcess(child.value,&code); return static_cast<int>(code); }
  if(status==WAIT_FAILED || cancelled || std::chrono::steady_clock::now()>=deadline) { TerminateJobObject(job.value,1); WaitForSingleObject(child.value,INFINITE); return -1; }
 }
#else
 std::vector<char*> argv; for(const auto& arg:arguments) argv.push_back(const_cast<char*>(arg.c_str())); argv.push_back(nullptr);
 const auto directory=root.string();
 pid_t child=fork(); if(child<0) return -1;
 if(child==0) { setpgid(0,0); if(chdir(directory.c_str())!=0) _exit(127); execvp(argv[0],argv.data()); _exit(127); }
 setpgid(child,child);
 for(;;) {
  int status=0; auto done=waitpid(child,&status,WNOHANG);
  if(done==child) { kill(-child,SIGKILL); return WIFEXITED(status)?WEXITSTATUS(status):-1; }
  if((done<0 && errno!=EINTR) || cancelled || std::chrono::steady_clock::now()>=deadline) { kill(-child,SIGKILL); while(waitpid(child,&status,0)<0 && errno==EINTR) {} return -1; }
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
 }
#endif
}
