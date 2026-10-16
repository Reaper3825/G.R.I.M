#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

// The bridge owns only the worker it creates. On Windows the job also cleans
// up the worker when MMO forcibly terminates the bridge.
class GrimTextWorkerProcess {
public:
    GrimTextWorkerProcess() = default;
    GrimTextWorkerProcess(const GrimTextWorkerProcess&) = delete;
    GrimTextWorkerProcess& operator=(const GrimTextWorkerProcess&) = delete;
    ~GrimTextWorkerProcess() {
#ifdef _WIN32
        if (job_) CloseHandle(job_);
        if (process_) {
            WaitForSingleObject(process_, 5000);
            CloseHandle(process_);
        }
#else
        if (pid_ > 0) {
            kill(pid_, SIGKILL);
            while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {}
        }
#endif
    }

    void start(const std::filesystem::path& executable, int port,
               const std::filesystem::path& log_path) {
        if (!std::filesystem::is_regular_file(executable)) {
            throw std::runtime_error("Inference worker executable not found: " + executable.string());
        }
        std::filesystem::create_directories(log_path.parent_path());
#ifdef _WIN32
        job_ = CreateJobObjectW(nullptr, nullptr);
        if (!job_) throw std::runtime_error("Cannot create inference worker job");
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
            throw std::runtime_error("Cannot configure inference worker job");
        }
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE log = CreateFileW(log_path.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                 &security, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (log == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open worker log: " + log_path.string());
        HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                   &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (input == INVALID_HANDLE_VALUE) {
            CloseHandle(log);
            throw std::runtime_error("Cannot open inference worker input");
        }
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = input;
        startup.hStdOutput = startup.hStdError = log;
        PROCESS_INFORMATION child{};
        std::wstring command = L"\"" + executable.wstring() +
            L"\" --inference --inference-worker-port " + std::to_wstring(port);
        const BOOL created = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, nullptr, &startup, &child);
        const DWORD error = GetLastError();
        CloseHandle(input);
        CloseHandle(log);
        if (!created) throw std::runtime_error("Cannot start inference worker: Windows error " + std::to_string(error));
        if (!AssignProcessToJobObject(job_, child.hProcess) || ResumeThread(child.hThread) == DWORD(-1)) {
            TerminateProcess(child.hProcess, 1);
            CloseHandle(child.hThread);
            CloseHandle(child.hProcess);
            throw std::runtime_error("Cannot attach/start inference worker in bridge job");
        }
        CloseHandle(child.hThread);
        process_ = child.hProcess;
#else
        posix_spawn_file_actions_t actions;
        int error = posix_spawn_file_actions_init(&actions);
        if (error) throw std::runtime_error("Cannot initialize inference worker spawn");
        error = posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log_path.c_str(),
                                                O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (!error) error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);
        if (!error) error = posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
        std::string path = executable.string(), port_text = std::to_string(port);
        char* args[] = {path.data(), const_cast<char*>("--inference"),
                       const_cast<char*>("--inference-worker-port"), port_text.data(), nullptr};
        if (!error) error = posix_spawn(&pid_, path.c_str(), &actions, nullptr, args, environ);
        posix_spawn_file_actions_destroy(&actions);
        if (error) {
            pid_ = -1;
            throw std::runtime_error("Cannot start inference worker: error " + std::to_string(error));
        }
#endif
    }

    // Called under the bridge's readiness mutex (waitpid reaps exited children).
    bool running() {
#ifdef _WIN32
        DWORD code = 0;
        return process_ && GetExitCodeProcess(process_, &code) && code == STILL_ACTIVE;
#else
        if (pid_ <= 0) return false;
        int status = 0;
        const auto result = waitpid(pid_, &status, WNOHANG);
        if (result == 0 || (result < 0 && errno == EINTR)) return true;
        pid_ = -1;
        return false;
#endif
    }

private:
#ifdef _WIN32
    HANDLE process_ = nullptr;
    HANDLE job_ = nullptr;
#else
    pid_t pid_ = -1;
#endif
};
