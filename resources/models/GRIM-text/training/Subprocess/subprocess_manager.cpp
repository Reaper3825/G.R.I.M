#include "subprocess_manager.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <sys/wait.h>
  #include <unistd.h>
  #include <spawn.h>
  extern char** environ;
  #if defined(__APPLE__)
    #include <mach-o/dyld.h>
  #endif
#endif

namespace fs = std::filesystem;

namespace GRIMText {
namespace Subprocess {

namespace {

// Get the absolute path of the running executable. Throws on failure.
fs::path current_executable_path() {
#if defined(_WIN32)
    char buffer[MAX_PATH];
    DWORD len = GetModuleFileNameA(nullptr, buffer, MAX_PATH);
    if (len == 0 || len >= MAX_PATH) {
        throw std::runtime_error(
            "subprocess_manager: GetModuleFileNameA failed (err=" +
            std::to_string(static_cast<unsigned long>(GetLastError())) + ")");
    }
    return fs::path(buffer);
#elif defined(__APPLE__)
    char buffer[4096];
    uint32_t size = sizeof(buffer);
    if (_NSGetExecutablePath(buffer, &size) != 0) {
        throw std::runtime_error(
            "subprocess_manager: _NSGetExecutablePath failed (need " +
            std::to_string(size) + " bytes)");
    }
    std::error_code ec;
    fs::path canonical = fs::canonical(fs::path(buffer), ec);
    if (ec) {
        throw std::runtime_error(
            "subprocess_manager: fs::canonical failed for executable path: " +
            ec.message());
    }
    return canonical;
#else
    char buffer[4096];
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len <= 0) {
        throw std::runtime_error(
            "subprocess_manager: readlink(/proc/self/exe) failed");
    }
    buffer[len] = '\0';
    return fs::path(buffer);
#endif
}

// Spawn `executable_path` with the given arguments, wait for exit, return exit code.
// Throws on spawn failure.
int spawn_blocking(const std::string& executable_path,
                   const std::vector<std::string>& arguments,
                   const std::string& subprocess_name) {
#if defined(_WIN32)
    // Build a properly quoted command line (Windows CommandLineToArgvW rules).
    auto quote = [](const std::string& s) -> std::string {
        if (!s.empty() && s.find_first_of(" \t\"") == std::string::npos) {
            return s;
        }
        std::string out = "\"";
        for (size_t i = 0; i < s.size(); ++i) {
            std::size_t backslashes = 0;
            while (i < s.size() && s[i] == '\\') { ++backslashes; ++i; }
            if (i == s.size()) {
                out.append(backslashes * 2, '\\');
                break;
            } else if (s[i] == '"') {
                out.append(backslashes * 2 + 1, '\\');
                out.push_back('"');
            } else {
                out.append(backslashes, '\\');
                out.push_back(s[i]);
            }
        }
        out.push_back('"');
        return out;
    };

    std::string cmd = quote(executable_path);
    for (const auto& a : arguments) {
        cmd.push_back(' ');
        cmd.append(quote(a));
    }

    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};

    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(
        executable_path.c_str(),
        mutable_cmd.data(),
        nullptr, nullptr, TRUE, 0, nullptr, nullptr,
        &si, &pi);
    if (!ok) {
        throw std::runtime_error(
            "subprocess_manager: CreateProcessA failed for subprocess '" +
            subprocess_name + "' (exe=" + executable_path +
            ", err=" + std::to_string(static_cast<unsigned long>(GetLastError())) + ")");
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 0;
    if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        throw std::runtime_error(
            "subprocess_manager: GetExitCodeProcess failed for subprocess '" +
            subprocess_name + "'");
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return static_cast<int>(exit_code);
#else
    // posix_spawn requires a non-const argv[]. Build it.
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.push_back(executable_path);
    for (const auto& a : arguments) storage.push_back(a);

    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& s : storage) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);

    pid_t pid = 0;
    int rc = posix_spawn(&pid, executable_path.c_str(),
                         nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
        throw std::runtime_error(
            "subprocess_manager: posix_spawn failed for subprocess '" +
            subprocess_name + "' (exe=" + executable_path +
            ", errno=" + std::to_string(rc) + ")");
    }

    int status = 0;
    while (true) {
        pid_t w = waitpid(pid, &status, 0);
        if (w == -1) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "subprocess_manager: waitpid failed for subprocess '" +
                subprocess_name + "' (errno=" + std::to_string(errno) + ")");
        }
        break;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        throw std::runtime_error(
            "subprocess_manager: subprocess '" + subprocess_name +
            "' terminated by signal " + std::to_string(WTERMSIG(status)));
    }
    throw std::runtime_error(
        "subprocess_manager: subprocess '" + subprocess_name +
        "' exited abnormally (raw status=" + std::to_string(status) + ")");
#endif
}

subprocess_result parse_status_file(const std::string& path,
                                    const std::string& subprocess_name) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error(
            "subprocess_manager: subprocess '" + subprocess_name +
            "' did not produce status file at: " + path);
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "subprocess_manager: subprocess '" + subprocess_name +
            "' wrote malformed status JSON at " + path + ": " + e.what());
    }

    if (!j.is_object() || !j.contains("outcome") || !j["outcome"].is_string()) {
        throw std::runtime_error(
            "subprocess_manager: subprocess '" + subprocess_name +
            "' status file missing required string field 'outcome' at " + path);
    }

    subprocess_result result;
    result.subprocess_name = subprocess_name;

    const std::string outcome_str = j["outcome"].get<std::string>();
    if (outcome_str == "success") {
        // ok_proceed by default; the manager will rewrite this to ok_one_off
        // upstream if the caller's request asked for one-off mode.
        result.outcome = subprocess_outcome::ok_proceed;

        if (!j.contains("vocab_path") || !j["vocab_path"].is_string() ||
            !j.contains("training_data_path") || !j["training_data_path"].is_string() ||
            !j.contains("vocab_size") || !j["vocab_size"].is_number_unsigned()) {
            throw std::runtime_error(
                "subprocess_manager: subprocess '" + subprocess_name +
                "' reported success but status file is missing required fields "
                "(vocab_path, training_data_path, vocab_size) at " + path);
        }
        result.vocab_path = j["vocab_path"].get<std::string>();
        result.training_data_path = j["training_data_path"].get<std::string>();
        result.vocab_size = j["vocab_size"].get<std::uint32_t>();
        return result;
    }

    if (outcome_str == "error") {
        result.outcome = subprocess_outcome::error;
        if (!j.contains("error_message") || !j["error_message"].is_string()) {
            throw std::runtime_error(
                "subprocess_manager: subprocess '" + subprocess_name +
                "' reported error but no 'error_message' field in status file at " +
                path);
        }
        result.error_message = j["error_message"].get<std::string>();
        if (result.error_message.empty()) {
            throw std::runtime_error(
                "subprocess_manager: subprocess '" + subprocess_name +
                "' reported error with empty error_message at " + path);
        }
        return result;
    }

    throw std::runtime_error(
        "subprocess_manager: subprocess '" + subprocess_name +
        "' reported unknown outcome '" + outcome_str +
        "' (must be 'success' or 'error') at " + path);
}

} // namespace

std::string resolve_sibling_executable(const std::string& base_name) {
    if (base_name.empty()) {
        throw std::runtime_error(
            "subprocess_manager::resolve_sibling_executable: base_name is empty");
    }
    fs::path exe_dir = current_executable_path().parent_path();

    fs::path bare = exe_dir / base_name;
    if (fs::exists(bare)) {
        return fs::absolute(bare).string();
    }
#if defined(_WIN32)
    fs::path with_exe = exe_dir / (base_name + ".exe");
    if (fs::exists(with_exe)) {
        return fs::absolute(with_exe).string();
    }
#endif
    throw std::runtime_error(
        "subprocess_manager::resolve_sibling_executable: cannot find '" +
        base_name + "' next to current executable in " + exe_dir.string());
}

subprocess_result spawn_and_wait(const subprocess_request& req) {
    if (req.name.empty()) {
        throw std::runtime_error("subprocess_manager: subprocess_request.name is empty");
    }
    if (req.executable_path.empty()) {
        throw std::runtime_error(
            "subprocess_manager: subprocess_request.executable_path is empty for '" +
            req.name + "'");
    }
    if (req.status_file_path.empty()) {
        throw std::runtime_error(
            "subprocess_manager: subprocess_request.status_file_path is empty for '" +
            req.name + "'");
    }
    if (!fs::exists(req.executable_path)) {
        throw std::runtime_error(
            "subprocess_manager: executable does not exist for subprocess '" +
            req.name + "': " + req.executable_path);
    }

    // Ensure parent directory exists and remove any stale status file so we
    // never read a previous run's outcome by accident.
    fs::path status_path(req.status_file_path);
    fs::path status_parent = status_path.parent_path();
    if (!status_parent.empty()) {
        std::error_code ec;
        fs::create_directories(status_parent, ec);
        if (ec) {
            throw std::runtime_error(
                "subprocess_manager: failed to create status directory " +
                status_parent.string() + " for subprocess '" + req.name +
                "': " + ec.message());
        }
    }
    std::error_code rm_ec;
    fs::remove(status_path, rm_ec); // ignore: it's fine if it didn't exist

    int exit_code = spawn_blocking(req.executable_path, req.arguments, req.name);

    // Whether exit_code is zero or not, the contract is: the child MUST have
    // written a status file. If it didn't, that's an infrastructure bug and we
    // throw with the captured exit code so the cause is unambiguous.
    if (!fs::exists(status_path)) {
        throw std::runtime_error(
            "subprocess_manager: subprocess '" + req.name +
            "' exited with code " + std::to_string(exit_code) +
            " but did not write a status file at " + req.status_file_path);
    }

    subprocess_result result = parse_status_file(req.status_file_path, req.name);

    // Cross-check: a non-zero exit MUST correspond to an error outcome. If the
    // child claimed success but exited non-zero, we treat that as an error and
    // surface it precisely (Rule 20: never silently accept).
    if (exit_code != 0 && result.outcome != subprocess_outcome::error) {
        throw std::runtime_error(
            "subprocess_manager: subprocess '" + req.name +
            "' exited with non-zero code " + std::to_string(exit_code) +
            " but status file reported success");
    }

    return result;
}

} // namespace Subprocess
} // namespace GRIMText
