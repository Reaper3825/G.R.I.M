// main.cpp
#include <SFML/Graphics.hpp>
#include <iostream>
#include <deque>
#include <string>
#include <sstream>
#include <vector>
#include <filesystem>
#include <system_error>
#include "NLP.hpp"
#include "ai.hpp"  // <-- bring back AI

namespace fs = std::filesystem;

static const float kWindowW = 480.f;
static const float kWindowH = 800.f;
static const size_t kMaxHistory = 200;

class ConsoleHistory {
public:
    void push(const std::string& line) {
        if (lines.size() >= kMaxHistory) lines.pop_front();
        lines.push_back(line);
    }
    std::string joined() const {
        std::ostringstream oss;
        for (auto& s : lines) oss << s << '\n';
        return oss.str();
    }
private:
    std::deque<std::string> lines;
};

static std::string findFontPath(int argc, char** argv) {
    std::error_code ec;
    fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
    std::vector<fs::path> candidates = {
        exeDir / "DejaVuSans.ttf",
        exeDir / "arial.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeui.ttf"
    };
    for (auto& p : candidates) {
        if (fs::exists(p, ec) && !fs::is_directory(p, ec)) {
            auto sz = fs::file_size(p, ec);
            if (!ec && sz > 1024) return p.string();
        }
    }
    return {};
}

static fs::path resolvePath(const fs::path& currentDir, const std::string& userPath) {
    fs::path p(userPath);
    std::error_code ec;
    if (p.is_absolute()) return fs::weakly_canonical(p, ec);
    return fs::weakly_canonical(currentDir / p, ec);
}
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}
static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out; std::istringstream iss(line); std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

int main(int argc, char** argv) {
    // --- NLP load (exe dir first, then cwd) ---
    NLP nlp;
    {
        std::error_code ec;
        fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
        fs::path rulesPathExe = exeDir / "nlp_rules.json";
        fs::path rulesPathCwd = "nlp_rules.json";
        std::string err;
        bool ok = fs::exists(rulesPathExe, ec) ? nlp.load_rules(rulesPathExe.string(), &err)
                                               : nlp.load_rules(rulesPathCwd.string(), &err);
        if (!ok) std::cerr << "[NLP] Failed to load rules: " << err << "\n";
        else {
            std::cerr << "[NLP] Loaded rules. Intents:";
            for (auto& s : nlp.list_intents()) std::cerr << " " << s;
            std::cerr << "\n";
        }
    }

    // --- Window ---
    sf::RenderWindow window(sf::VideoMode((unsigned)kWindowW, (unsigned)kWindowH), "GRIM");
    window.setVerticalSyncEnabled(true);

    // --- Font ---
    sf::Font font;
    const std::string fontPath = findFontPath(argc, argv);
    if (fontPath.empty() || !font.loadFromFile(fontPath)) {
        std::cerr << "[WARN] Could not load a font. Place a TTF (e.g., arial.ttf) next to the exe.\n";
    } else {
        std::cerr << "[INFO] Using font: " << fontPath << "\n";
    }

    // --- UI ---
    const float inputHeight = 34.f;
    sf::RectangleShape inputBar({kWindowW, inputHeight});
    inputBar.setFillColor(sf::Color(30, 30, 35));
    inputBar.setPosition(0.f, kWindowH - inputHeight);

    sf::Text inputText, historyText;
    if (font.getInfo().family != "") {
        inputText.setFont(font);
        inputText.setCharacterSize(18);
        inputText.setFillColor(sf::Color::White);
        inputText.setPosition(8.f, kWindowH - inputHeight + 6.f);

        historyText.setFont(font);
        historyText.setCharacterSize(18);
        historyText.setFillColor(sf::Color::White);
        historyText.setPosition(8.f, 8.f);
    }

    ConsoleHistory history;
    std::string buffer;
    fs::path currentDir = fs::current_path();

    auto addHistory = [&](const std::string& line){
        history.push(line);
        std::cout << line << "\n";
    };

    addHistory("GRIM is ready. Type a command and press Enter. Type 'quit' to exit.");

    sf::Clock caretClock; bool caretVisible = true;

    while (window.isOpen()) {
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) window.close();
            if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) window.close();

            if (e.type == sf::Event::TextEntered) {
                if (e.text.unicode == 8) { // backspace
                    if (!buffer.empty()) buffer.pop_back();
                } else if (e.text.unicode == 13 || e.text.unicode == 10) { // Enter
                    std::string line = trim(buffer); buffer.clear();

                    if (line == "quit" || line == "exit") { window.close(); break; }
                    if (line.empty()) { addHistory("> "); continue; }

                    addHistory("> " + line);

                    // --- commands ---
                    auto args = split(line);
                    const std::string cmd = args.empty() ? "" : args[0];

                    if (cmd == "help") {
                        addHistory(
                            "Commands:\n"
                            "  help                      - show this help\n"
                            "  pwd                       - print current directory\n"
                            "  cd <path>                 - change directory\n"
                            "  list                      - list items in current directory\n"
                            "  mkdir <path>              - create directory\n"
                            "  rm <path>                 - remove file or empty directory\n"
                            "  reloadnlp                 - reload nlp_rules.json\n"
                            "  ai <prompt>               - ask local model via Ollama\n"
                            "  (natural language also works: 'open notepad', 'search cats')"
                        );
                        continue;
                    }
                    if (cmd == "pwd") { addHistory(currentDir.string()); continue; }
                    if (cmd == "cd") {
                        if (args.size() < 2) { addHistory("Error: cd requires a path."); continue; }
                        fs::path target = resolvePath(currentDir, args[1]);
                        std::error_code ec;
                        if (!fs::exists(target, ec)) { addHistory("Error: path does not exist."); continue; }
                        if (!fs::is_directory(target, ec)) { addHistory("Error: not a directory."); continue; }
                        currentDir = target;
                        addHistory("Directory changed to: " + currentDir.string());
                        continue;
                    }
                    if (cmd == "list") {
                        std::error_code ec;
                        if (!fs::exists(currentDir, ec) || !fs::is_directory(currentDir, ec)) {
                            addHistory("Error: current directory invalid."); continue;
                        }
                        std::ostringstream out; out << "Listing: " << currentDir.string() << "\n";
                        for (const auto& entry : fs::directory_iterator(currentDir, ec)) {
                            if (ec) break;
                            bool isDir = entry.is_directory(ec);
                            out << (isDir ? "[D] " : "    ") << entry.path().filename().string() << "\n";
                        }
                        addHistory(out.str());
                        continue;
                    }
                    if (cmd == "mkdir") {
                        if (args.size() < 2) { addHistory("Error: mkdir requires <path>."); continue; }
                        fs::path p = resolvePath(currentDir, args[1]);
                        std::error_code ec; fs::create_directories(p, ec);
                        if (ec) addHistory("Error creating directory: " + ec.message());
                        else addHistory("Directory created.");
                        continue;
                    }
                    if (cmd == "rm") {
                        if (args.size() < 2) { addHistory("Error: rm requires <path>."); continue; }
                        fs::path p = resolvePath(currentDir, args[1]);
                        std::error_code ec;
                        if (!fs::exists(p, ec)) { addHistory("Error: path does not exist."); continue; }
                        bool ok = fs::remove(p, ec);
                        if (ec) addHistory("Error removing: " + ec.message());
                        else if (!ok) addHistory("Error: could not remove (directory may not be empty).");
                        else addHistory("Removed.");
                        continue;
                    }
                    if (cmd == "reloadnlp") {
                        std::error_code ec;
                        fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
                        fs::path rulesPathExe = exeDir / "nlp_rules.json";
                        fs::path rulesPathCwd = "nlp_rules.json";
                        std::string err;
                        bool ok = fs::exists(rulesPathExe, ec) ? nlp.load_rules(rulesPathExe.string(), &err)
                                                               : nlp.load_rules(rulesPathCwd.string(), &err);
                        if (!ok) addHistory("Failed to reload nlp_rules.json: " + err);
                        else addHistory("NLP rules reloaded.");
                        continue;
                    }
                    if (cmd == "ai") {
                        // Everything after "ai " is the prompt
                        std::string query = (line.size() > 3) ? trim(line.substr(3)) : "";
                        if (query.empty()) { addHistory("Usage: ai <your question>"); continue; }
                        try {
                            std::string answer = callAI(query);
                            if (answer.empty()) answer = "[AI] (empty response)";
                            // split into lines for nicer display
                            std::istringstream iss(answer);
                            std::string l;
                            while (std::getline(iss, l)) addHistory(l);
                        } catch (const std::exception& ex) {
                            addHistory(std::string("[AI] Error: ") + ex.what());
                        }
                        continue;
                    }

                    // --- NLP fallback ---
                    Intent intent = nlp.parse(line);
                    if (!intent.matched) {
                        addHistory("[NLP] No intent matched.");
                    } else {
                        std::ostringstream oss;
                        oss << "[NLP] intent=" << intent.name << " score=" << intent.score;
                        addHistory(oss.str());
                        for (const auto& kv : intent.slots) addHistory("  " + kv.first + " = " + kv.second);

                        // Example routing placeholders
                        if (intent.name == "open_app") {
                            auto it = intent.slots.find("app");
                            if (it != intent.slots.end()) addHistory("Would open app: " + it->second);
                        } else if (intent.name == "search_web") {
                            auto it = intent.slots.find("query");
                            if (it != intent.slots.end()) addHistory("Would search web for: " + it->second);
                        } else if (intent.name == "set_timer") {
                            auto it = intent.slots.find("minutes");
                            if (it != intent.slots.end()) addHistory("Would set timer for " + it->second + " minute(s).");
                        }
                    }
                } else {
                    if (e.text.unicode >= 32 && e.text.unicode < 127) {
                        buffer.push_back(static_cast<char>(e.text.unicode));
                    }
                }
            }
        }

        if (caretClock.getElapsedTime().asSeconds() > 0.5f) {
            caretVisible = !caretVisible;
            caretClock.restart();
        }

        window.clear(sf::Color(18, 18, 22));

        if (font.getInfo().family != "") {
            historyText.setString(history.joined());
            window.draw(historyText);
        }

        window.draw(inputBar);
        if (font.getInfo().family != "") {
            std::string toShow = buffer;
            if (caretVisible) toShow.push_back('_');
            inputText.setString(toShow);
            window.draw(inputText);
        }

        window.display();
    }

    return 0;
}
