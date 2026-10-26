#include "textworld_client.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <stdexcept>
#include <cstdlib>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
extern char** environ;
#endif

namespace GRIM::WorldLab {
namespace fs = std::filesystem;
using namespace std::chrono_literals;
namespace {
nlohmann::json failure(const std::string& message) { return {{"ok",false},{"error",message},{"transport_error",true}}; }
fs::path workerDirectory() {
#ifdef _WIN32
    wchar_t executable[32768];const auto length=GetModuleFileNameW(nullptr,executable,32768);
    fs::path executableDirectory=length?fs::path(executable).parent_path():fs::current_path();
#else
#if defined(__APPLE__)
    uint32_t size=0;
    _NSGetExecutablePath(nullptr,&size);
    std::vector<char> executable(size);
    if(_NSGetExecutablePath(executable.data(),&size)!=0)throw std::runtime_error("Cannot resolve the GRIM executable path");
    fs::path executableDirectory=fs::canonical(executable.data()).parent_path();
#else
    fs::path executableDirectory=fs::canonical("/proc/self/exe").parent_path();
#endif
#endif
    for(auto root:{fs::current_path(),executableDirectory}) {
        for(int depth=0;depth<10 && !root.empty();++depth,root=root.parent_path()) {
            auto candidate=root/"scripts/textworld";
            if(fs::exists(candidate/"world_worker.py"))return fs::canonical(candidate);
        }
    }
    throw std::runtime_error("Cannot find scripts/textworld/world_worker.py from the application or working directory.");
}
#ifdef _WIN32
std::wstring quote(const std::wstring& value) {
    std::wstring out=L"\"";size_t slashes=0;
    for(wchar_t c:value) {
        if(c==L'\\'){++slashes;continue;}
        out.append(c==L'"'?slashes*2+1:slashes,L'\\');slashes=0;out+=c;
    }
    out.append(slashes*2,L'\\');return out+L'"';
}
fs::path dockerPath() {
    wchar_t found[32768];
    if(SearchPathW(nullptr,L"docker.exe",nullptr,32768,found,nullptr))return found;
    wchar_t* programFiles=nullptr;size_t count=0;
    _wdupenv_s(&programFiles,&count,L"ProgramFiles");
    fs::path root=programFiles?programFiles:L"C:\\Program Files";std::free(programFiles);
    auto path=root/L"Docker/Docker/resources/bin/docker.exe";
    if(fs::exists(path))return path;
    throw std::runtime_error("Docker Desktop was not found. Start Docker with Linux containers and use the existing TextWorld image.");
}
// Restrict inherited handles to the explicitly supplied log and NUL streams.
HANDLE launch(const fs::path& executable,const std::vector<std::wstring>& arguments,const fs::path& log) {
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES),nullptr,TRUE};
    HANDLE output=CreateFileW(log.c_str(),FILE_APPEND_DATA,FILE_SHARE_READ|FILE_SHARE_WRITE,&security,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
    HANDLE input=CreateFileW(L"NUL",GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE,&security,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(output==INVALID_HANDLE_VALUE || input==INVALID_HANDLE_VALUE) {
        if(output!=INVALID_HANDLE_VALUE)CloseHandle(output);if(input!=INVALID_HANDLE_VALUE)CloseHandle(input);
        throw std::runtime_error("Cannot open TextWorld process streams");
    }
    STARTUPINFOEXW startup{};startup.StartupInfo.cb=sizeof(startup);
    startup.StartupInfo.dwFlags=STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput=input;startup.StartupInfo.hStdOutput=output;startup.StartupInfo.hStdError=output;
    SIZE_T bytes=0;InitializeProcThreadAttributeList(nullptr,1,0,&bytes);
    std::vector<unsigned char> attributes(bytes);
    startup.lpAttributeList=reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    bool initialized=InitializeProcThreadAttributeList(startup.lpAttributeList,1,0,&bytes)!=FALSE;
    HANDLE inherited[]={input,output};
    bool configured=initialized && UpdateProcThreadAttribute(startup.lpAttributeList,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,inherited,sizeof(inherited),nullptr,nullptr);
    std::wstring command=quote(executable.wstring());for(const auto& arg:arguments)command+=L" "+quote(arg);
    PROCESS_INFORMATION process{};
    bool created=configured && CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_NO_WINDOW|EXTENDED_STARTUPINFO_PRESENT,nullptr,nullptr,&startup.StartupInfo,&process);
    const auto error=GetLastError();
    if(initialized)DeleteProcThreadAttributeList(startup.lpAttributeList);
    CloseHandle(input);CloseHandle(output);
    if(!created)throw std::runtime_error("Cannot launch Docker (Windows error "+std::to_string(error)+")");
    CloseHandle(process.hThread);return process.hProcess;
}
#else
fs::path dockerPath() {
    const char* path=std::getenv("PATH");
    if(path) {
        std::string paths(path);
        for(size_t begin=0;begin<=paths.size();) {
            const auto end=paths.find(':',begin);
            const auto directory=paths.substr(begin,end==std::string::npos?std::string::npos:end-begin);
            const auto candidate=fs::path(directory.empty()?".":directory)/"docker";
            if(access(candidate.c_str(),X_OK)==0)return fs::canonical(candidate);
            if(end==std::string::npos)break;
            begin=end+1;
        }
    }
#if defined(__APPLE__)
    const fs::path desktopDocker="/Applications/Docker.app/Contents/Resources/bin/docker";
    if(access(desktopDocker.c_str(),X_OK)==0)return desktopDocker;
#endif
    throw std::runtime_error("Docker was not found. Install and start Docker Desktop, then build the existing grim-textworld:1.7.0 image.");
}
pid_t launch(const fs::path& executable,const std::vector<std::string>& arguments,const fs::path& log) {
    const int output=open(log.c_str(),O_WRONLY|O_CREAT|O_APPEND,0600);
    const int input=open("/dev/null",O_RDONLY);
    if(output<0 || input<0) {
        if(output>=0)close(output);
        if(input>=0)close(input);
        throw std::runtime_error("Cannot open TextWorld process streams");
    }

    posix_spawn_file_actions_t actions;
    if(posix_spawn_file_actions_init(&actions)!=0) {
        close(input);close(output);
        throw std::runtime_error("Cannot initialize TextWorld process actions");
    }
    posix_spawn_file_actions_adddup2(&actions,input,STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions,output,STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions,output,STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions,input);
    posix_spawn_file_actions_addclose(&actions,output);

    std::vector<std::string> storage{executable.string()};
    storage.insert(storage.end(),arguments.begin(),arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size()+1);
    for(auto& argument:storage)argv.push_back(argument.data());
    argv.push_back(nullptr);

    pid_t process=-1;
    const int error=posix_spawn(&process,executable.c_str(),&actions,nullptr,argv.data(),environ);
    posix_spawn_file_actions_destroy(&actions);
    close(input);close(output);
    if(error!=0)throw std::runtime_error("Cannot launch Docker (POSIX error "+std::to_string(error)+")");
    return process;
}
#endif
}
struct TextWorldClient::Impl {
    std::future<nlohmann::json> pending;
    std::atomic<bool> cancelled{false};
    unsigned sequence=0;
    fs::path directory,docker;
    bool directoryOwned=false;
#ifdef _WIN32
    std::wstring container;
    HANDLE process=nullptr;
#else
    std::string container;
    pid_t process=-1;
#endif
    std::string diagnostics() const {
        std::ifstream in(directory/"worker.log",std::ios::binary);
        std::string text;
        if(in){in.seekg(0,std::ios::end);auto size=in.tellg();in.seekg(size>4096?size-std::streamoff(4096):std::streampos(0));text.assign(std::istreambuf_iterator<char>(in),{});}
        return text;
    }
    void shutdown() noexcept {
        try {
#ifdef _WIN32
            if(directoryOwned) {std::ofstream stop(directory/"stop");}
            if(process && WaitForSingleObject(process,500)==WAIT_TIMEOUT) {
                try {
                    HANDLE remove=launch(docker,{L"rm",L"-f",container},directory/"worker.log");
                    if(WaitForSingleObject(remove,5000)==WAIT_TIMEOUT)TerminateProcess(remove,1);
                    CloseHandle(remove);
                }catch(...){/* Still close the CLI and leave the worker's stop marker. */}
                if(WaitForSingleObject(process,1000)==WAIT_TIMEOUT)TerminateProcess(process,1);
            }
            if(process){CloseHandle(process);process=nullptr;}
#else
            if(directoryOwned) {std::ofstream stop(directory/"stop");}
            if(process>0) {
                int status=0;
                if(waitpid(process,&status,WNOHANG)==0) {
                    try {
                        const pid_t remove=launch(docker,{"rm","-f",container},directory/"worker.log");
                        const auto deadline=std::chrono::steady_clock::now()+5s;
                        while(waitpid(remove,&status,WNOHANG)==0 && std::chrono::steady_clock::now()<deadline)
                            std::this_thread::sleep_for(25ms);
                        if(waitpid(remove,&status,WNOHANG)==0) {
                            kill(remove,SIGKILL);
                            waitpid(remove,&status,0);
                        }
                    }catch(...){/* Still stop the worker and leave its stop marker. */}
                    if(waitpid(process,&status,WNOHANG)==0) {
                        kill(process,SIGTERM);
                        const auto deadline=std::chrono::steady_clock::now()+1s;
                        while(waitpid(process,&status,WNOHANG)==0 && std::chrono::steady_clock::now()<deadline)
                            std::this_thread::sleep_for(25ms);
                        if(waitpid(process,&status,WNOHANG)==0) {
                            kill(process,SIGKILL);
                            waitpid(process,&status,0);
                        }
                    }
                }
                process=-1;
            }
#endif
            // Remove only this client's exclusively created temporary session.
            if(directoryOwned && fs::equivalent(directory.parent_path(),fs::temp_directory_path()) &&
               directory.filename().wstring().find(L"grim-world-session-")==0) {
                std::error_code error;fs::remove_all(directory,error);
            }
            directory.clear();directoryOwned=false;
        }catch(...){/* Destruction/cancellation must not throw. */}
    }
    void start() {
#ifdef _WIN32
        if(process) {
            if(WaitForSingleObject(process,0)==WAIT_TIMEOUT)return;
            const auto reason=diagnostics();shutdown();
            throw std::runtime_error("TextWorld worker exited; compile again. "+reason);
        }
        docker=dockerPath();const auto source=workerDirectory();
        container=L"grim-world-session-"+std::to_wstring(GetCurrentProcessId())+L"-"+std::to_wstring(std::chrono::steady_clock::now().time_since_epoch().count());
        directory=fs::temp_directory_path()/container;
        if(!fs::create_directory(directory))throw std::runtime_error("Cannot create TextWorld session directory");
        directoryOwned=true;
        // --mount CSV syntax cannot represent an unescaped comma in these paths.
        if(source.wstring().find(L',')!=std::wstring::npos || directory.wstring().find(L',')!=std::wstring::npos)
            throw std::runtime_error("TextWorld mount paths cannot contain commas");
        process=launch(docker,{L"run",L"--rm",L"--pull",L"never",L"--network",L"none",L"--name",container,
            L"--mount",L"type=bind,source="+source.wstring()+L",target=/worker,readonly",
            L"--mount",L"type=bind,source="+directory.wstring()+L",target=/session",
            L"grim-textworld:1.7.0",L"python",L"-u",L"/worker/world_worker.py",L"--session",L"/session"},directory/"worker.log");
#else
        if(process>0) {
            int status=0;
            if(waitpid(process,&status,WNOHANG)==0)return;
            const auto reason=diagnostics();shutdown();
            throw std::runtime_error("TextWorld worker exited; compile again. "+reason);
        }
        docker=dockerPath();const auto source=workerDirectory();
        container="grim-world-session-"+std::to_string(getpid())+"-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        directory=fs::temp_directory_path()/container;
        if(!fs::create_directory(directory))throw std::runtime_error("Cannot create TextWorld session directory");
        directoryOwned=true;
        if(source.string().find(',')!=std::string::npos || directory.string().find(',')!=std::string::npos)
            throw std::runtime_error("TextWorld mount paths cannot contain commas");
        const pid_t probe=launch(docker,{"info","--format","{{.ServerVersion}}"},directory/"worker.log");
        int probeStatus=0;
        const auto probeDeadline=std::chrono::steady_clock::now()+10s;
        while(waitpid(probe,&probeStatus,WNOHANG)==0 && std::chrono::steady_clock::now()<probeDeadline)
            std::this_thread::sleep_for(25ms);
        if(waitpid(probe,&probeStatus,WNOHANG)==0) {
            kill(probe,SIGKILL);
            waitpid(probe,&probeStatus,0);
            throw std::runtime_error("Docker daemon did not respond within 10 seconds. Start Docker Desktop and wait until it reports that Docker is running.");
        }
        if(!WIFEXITED(probeStatus) || WEXITSTATUS(probeStatus)!=0)
            throw std::runtime_error("Docker daemon is unavailable. Start Docker Desktop and wait until it reports that Docker is running. "+diagnostics());
        process=launch(docker,{"run","--rm","--pull","never","--network","none","--name",container,
            "--mount","type=bind,source="+source.string()+",target=/worker,readonly",
            "--mount","type=bind,source="+directory.string()+",target=/session",
            "grim-textworld:1.7.0","python","-u","/worker/world_worker.py","--session","/session"},directory/"worker.log");
#endif
    }
    nlohmann::json execute(nlohmann::json request) {
        try {
            start();
            const auto requestId=++sequence;
            request["id"]=requestId;request["protocol"]=2;
            const auto body=request.dump();if(body.size()>16*1024*1024)throw std::runtime_error("World request exceeds 16 MB");
            {std::ofstream out(directory/"request.tmp",std::ios::binary);out<<body;out.close();if(!out)throw std::runtime_error("Cannot write TextWorld request");}
            fs::rename(directory/"request.tmp",directory/"request.json");
            const auto deadline=std::chrono::steady_clock::now()+120s;
            for(;;) {
                if(cancelled)throw std::runtime_error("TextWorld request cancelled");
                if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("TextWorld request timed out after 120 seconds");
                const auto response=directory/"response.json";
                if(fs::exists(response)) {
                    if(fs::file_size(response)>16*1024*1024)throw std::runtime_error("TextWorld response exceeds 16 MB");
                    std::ifstream in(response,std::ios::binary);nlohmann::json result;in>>result;in.close();fs::remove(response);
                    if(result.at("protocol")!=2 || result.at("id")!=requestId)throw std::runtime_error("TextWorld response ID/protocol mismatch");
                    return result;
                }
#ifdef _WIN32
                if(WaitForSingleObject(process,0)!=WAIT_TIMEOUT)throw std::runtime_error("TextWorld worker stopped. "+diagnostics());
#else
                int status=0;
                if(waitpid(process,&status,WNOHANG)!=0) {
                    process=-1;
                    throw std::runtime_error("TextWorld worker stopped. "+diagnostics());
                }
#endif
                std::this_thread::sleep_for(30ms);
            }
        }catch(const std::exception& error){const auto message=std::string(error.what());shutdown();return failure(message);}
    }
};
TextWorldClient::TextWorldClient():impl_(std::make_unique<Impl>()){}
TextWorldClient::~TextWorldClient(){impl_->cancelled=true;if(impl_->pending.valid())impl_->pending.wait();impl_->shutdown();}
bool TextWorldClient::busy() const{return impl_->pending.valid();}
bool TextWorldClient::begin(nlohmann::json request) {
    if(busy())return false;
    impl_->cancelled=false;
    impl_->pending=std::async(std::launch::async,[this,request=std::move(request)]()mutable{return impl_->execute(std::move(request));});return true;
}
std::optional<nlohmann::json> TextWorldClient::poll() {
    if(!busy() || impl_->pending.wait_for(0ms)!=std::future_status::ready)return {};
    return impl_->pending.get();
}
void TextWorldClient::stop() {
    impl_->cancelled=true;
    if(!busy())impl_->pending=std::async(std::launch::async,[this]{impl_->shutdown();return nlohmann::json{{"ok",true},{"result",{{"stopped",true}}}};});
}
}
