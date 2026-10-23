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
#endif

namespace GRIM::WorldLab {
namespace fs = std::filesystem;
using namespace std::chrono_literals;
namespace {
nlohmann::json failure(const std::string& message) { return {{"ok",false},{"error",message},{"transport_error",true}}; }
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
fs::path workerDirectory() {
    wchar_t executable[32768];const auto length=GetModuleFileNameW(nullptr,executable,32768);
    for(auto root:{fs::current_path(),length?fs::path(executable).parent_path():fs::current_path()}) {
        for(int depth=0;depth<10 && !root.empty();++depth,root=root.parent_path()) {
            auto candidate=root/L"scripts/textworld";
            if(fs::exists(candidate/L"world_worker.py"))return fs::canonical(candidate);
        }
    }
    throw std::runtime_error("Cannot find scripts/textworld/world_worker.py from the application or working directory.");
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
#endif
}
struct TextWorldClient::Impl {
    std::future<nlohmann::json> pending;
    std::atomic<bool> cancelled{false};
    unsigned sequence=0;
    fs::path directory,docker;
    bool directoryOwned=false;
    std::wstring container;
#ifdef _WIN32
    HANDLE process=nullptr;
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
        throw std::runtime_error("The World Lab Docker client currently supports Windows hosts.");
#endif
    }
    nlohmann::json execute(nlohmann::json request) {
        try {
            start();
            const auto requestId=++sequence;
            request["id"]=requestId;request["protocol"]=1;
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
                    if(result.at("protocol")!=1 || result.at("id")!=requestId)throw std::runtime_error("TextWorld response ID/protocol mismatch");
                    return result;
                }
#ifdef _WIN32
                if(WaitForSingleObject(process,0)!=WAIT_TIMEOUT)throw std::runtime_error("TextWorld worker stopped. "+diagnostics());
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
