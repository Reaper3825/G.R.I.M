#if !defined(_WIN32) && !defined(__APPLE__)
#include "application_catalog.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <set>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
extern char** environ;

namespace GRIM::Applications {
namespace {
std::string env(const char* key) { const auto* value = std::getenv(key); return value ? value : ""; }
std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> values; std::istringstream in(text); std::string value;
    while (std::getline(in, value, delimiter)) if (!value.empty()) values.push_back(value);
    return values;
}
bool executable(const std::string& path) {
    if (path.find('/') != std::string::npos) return access(path.c_str(), X_OK) == 0;
    for (const auto& dir : split(env("PATH"), ':'))
        if (access((dir + "/" + path).c_str(), X_OK) == 0) return true;
    return false;
}
std::string unescape(const std::string& value) {
    std::string out;
    for (size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 1 < value.size()) {
            const auto c = value[++i];
            if (c == 's') out += ' ';
            else if (c == 'n') out += '\n';
            else if (c == 't') out += '\t';
            else if (c == 'r') out += '\r';
            else out += c;
        } else out += value[i];
    }
    return out;
}
}
Backend nativeBackend() {
    return {"linux-desktop-entries", [] {
        namespace fs = std::filesystem;
        std::vector<std::string> roots;
        auto home = env("XDG_DATA_HOME");
        if (home.empty() && !env("HOME").empty()) home = env("HOME") + "/.local/share";
        if (!home.empty()) roots.push_back(home);
        auto dirs = env("XDG_DATA_DIRS");
        for (const auto& dir : split(dirs.empty() ? "/usr/local/share:/usr/share" : dirs, ':')) roots.push_back(dir);
        if (!env("HOME").empty()) roots.push_back(env("HOME") + "/.local/share/flatpak/exports/share");
        roots.push_back("/var/lib/flatpak/exports/share"); roots.push_back("/var/lib/snapd/desktop");
        std::set<std::string> seen;
        const auto desktops = split(env("XDG_CURRENT_DESKTOP"), ':');
        std::vector<Application> apps;
        for (const auto& root : roots) {
            if (!fs::path(root).is_absolute()) continue;
            const auto dir = fs::path(root) / "applications";
            if (!fs::exists(dir)) continue;
            std::vector<fs::path> files;
            for (const auto& entry : fs::recursive_directory_iterator(dir))
                if (entry.is_regular_file() && entry.path().extension() == ".desktop") {
                    files.push_back(entry.path());
                    if (files.size() > 10000) throw std::runtime_error("Desktop entry scan limit exceeded");
                }
            std::sort(files.begin(), files.end());
            for (const auto& path : files) {
                auto id = path.lexically_relative(dir).generic_string();
                std::replace(id.begin(), id.end(), '/', '-');
                // Hidden entries also mask lower-priority system entries.
                if (!seen.insert(id).second) continue;
                if (fs::file_size(path) > 1024 * 1024) throw std::runtime_error("Oversized desktop entry");
                std::ifstream input(path);
                if (!input) throw std::runtime_error("Cannot read desktop entry: " + path.string());
                std::map<std::string, std::string> fields;
                bool section = false; std::string line;
                while (std::getline(input, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty() || line[0] == '#') continue;
                    if (line[0] == '[') { section = line == "[Desktop Entry]"; continue; }
                    const auto equal = line.find('=');
                    if (section && equal != std::string::npos) fields[line.substr(0, equal)] = unescape(line.substr(equal + 1));
                }
                if (fields["Type"] != "Application" || fields["Hidden"] == "true" || fields["NoDisplay"] == "true" || fields["Name"].empty()) continue;
                const auto matches = [&](const std::string& key) {
                    for (const auto& name : split(fields[key], ';'))
                        if (std::find(desktops.begin(), desktops.end(), name) != desktops.end()) return true;
                    return false;
                };
                if ((!fields["OnlyShowIn"].empty() && !matches("OnlyShowIn")) || matches("NotShowIn")) continue;
                if (!fields["TryExec"].empty() && !executable(fields["TryExec"])) continue;
                if (fields["Exec"].empty() && fields["DBusActivatable"] != "true") continue;
                apps.push_back({"linux.desktop", id, fields["Name"], "desktop-entry", path.string(), fields["Icon"]});
            }
        }
        return apps;
    }};
}
Backend portableBackend() { return {"portable-applications", [] { return std::vector<Application>{}; }}; }
void launchNative(const Application& app) {
    if (app.launchKind != "desktop-entry") throw std::invalid_argument("Unsupported Linux launch kind");
    // Delegate Exec field codes, quoting, Terminal, and D-Bus activation to the desktop launcher.
    // No shell or hand-built command line interprets application metadata.
    char command[] = "gio", operation[] = "launch";
    char* args[] = {command, operation, const_cast<char*>(app.launchTarget.c_str()), nullptr};
    pid_t process;
    const int error = posix_spawnp(&process, command, nullptr, nullptr, args, environ);
    if (error) throw std::runtime_error("Could not start gio launcher (must be installed): " + std::to_string(error));
    int status = 0; pid_t waited;
    do { waited = waitpid(process, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status)) throw std::runtime_error("Desktop launcher rejected application");
}
}
#endif
