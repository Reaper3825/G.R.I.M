#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

// ---------- Helpers ----------
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
static fs::path resolvePath(const fs::path& currentDir, const std::string& userPath) {
    fs::path p(userPath);
    if (p.is_absolute()) return fs::weakly_canonical(p);
    return fs::weakly_canonical(currentDir / p);
}

// ---------- Command handler ----------
static std::string handleCommand(const std::string& raw, fs::path& currentDir) {
    std::string line = trim(raw);
    if (line.empty()) return "";
    auto args = split(line);
    const std::string cmd = args[0];

    if (cmd == "help") {
        return
            "Commands:\n"
            "  help                      - show this help\n"
            "  pwd                       - print current directory\n"
            "  cd <path>                 - change directory\n"
            "  list                      - list items in current directory\n"
            "  move <src> <dst>          - move/rename file or directory\n"
            "  copy <src> <dst>          - copy file or directory (recursive)\n"
            "  mkdir <path>              - create directory (including parents)\n"
            "  rm <path>                 - remove file or empty directory\n"
            "  rmolder <days> [-r] [-n]  - remove files older than <days> (last write time)\n"
            "                               -r recurse, -n dry-run\n";
    }
    if (cmd == "pwd") { return currentDir.string(); }

    if (cmd == "cd") {
        if (args.size() < 2) return "Error: cd requires a path.";
        fs::path target = resolvePath(currentDir, args[1]);
        std::error_code ec;
        if (!fs::exists(target, ec)) return "Error: path does not exist.";
        if (!fs::is_directory(target, ec)) return "Error: not a directory.";
        currentDir = target;
        return std::string("Directory changed to: ") + currentDir.string();
    }

    if (cmd == "list") {
        std::error_code ec;
        if (!fs::exists(currentDir, ec) || !fs::is_directory(currentDir, ec))
            return "Error: current directory invalid.";
        std::ostringstream out; out << "Listing: " << currentDir.string() << "\n";
        for (const auto& entry : fs::directory_iterator(currentDir, ec)) {
            if (ec) break;
            bool isDir = entry.is_directory(ec);
            out << (isDir ? "[D] " : "    ") << entry.path().filename().string() << "\n";
        }
        return out.str();
    }

    if (cmd == "move") {
        if (args.size() < 3) return "Error: move requires <src> <dst>.";
        fs::path src = resolvePath(currentDir, args[1]);
        fs::path dst = resolvePath(currentDir, args[2]);
        std::error_code ec;
        if (!fs::exists(src, ec)) return "Error: source does not exist.";
        if (fs::exists(dst, ec) && fs::is_directory(dst, ec)) dst = dst / src.filename();
        fs::create_directories(dst.parent_path(), ec);
        ec.clear(); fs::rename(src, dst, ec);
        if (ec) return std::string("Error moving: ") + ec.message();
        return "Moved.";
    }

    if (cmd == "copy") {
        if (args.size() < 3) return "Error: copy requires <src> <dst>.";
        fs::path src = resolvePath(currentDir, args[1]);
        fs::path dst = resolvePath(currentDir, args[2]);
        std::error_code ec;
        if (!fs::exists(src, ec)) return "Error: source does not exist.";
        if (fs::is_directory(src, ec)) {
            fs::create_directories(dst, ec); ec.clear();
            fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        } else {
            fs::create_directories(dst.parent_path(), ec); ec.clear();
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
        }
        if (ec) return std::string("Error copying: ") + ec.message();
        return "Copied.";
    }

    if (cmd == "mkdir") {
        if (args.size() < 2) return "Error: mkdir requires <path>.";
        fs::path p = resolvePath(currentDir, args[1]);
        std::error_code ec; fs::create_directories(p, ec);
        if (ec) return std::string("Error creating directory: ") + ec.message();
        return "Directory created.";
    }

    if (cmd == "rm") {
        if (args.size() < 2) return "Error: rm requires <path>.";
        fs::path p = resolvePath(currentDir, args[1]);
        std::error_code ec;
        if (!fs::exists(p, ec)) return "Error: path does not exist.";
        bool ok = fs::remove(p, ec); // files or empty dirs only
        if (ec) return std::string("Error removing: ") + ec.message();
        if (!ok) return "Error: could not remove (directory may not be empty).";
        return "Removed.";
    }

    if (cmd == "rmolder") {
        if (args.size() < 2) return "Error: rmolder requires <days>.";
        int days = 0; try { days = std::stoi(args[1]); } catch (...) { return "Error: <days> must be an integer."; }
        if (days < 0) return "Error: <days> must be >= 0.";
        bool recursive = false, dryRun = false;
        for (size_t i = 2; i < args.size(); ++i) {
            if (args[i] == "-r") recursive = true;
            else if (args[i] == "-n") dryRun = true;
            else return "Error: unknown flag '" + args[i] + "'. Use -r or -n.";
        }
        auto now_ft = fs::file_time_type::clock::now();
        auto cutoff_ft = now_ft - std::chrono::hours(24LL * days);

        std::error_code ec; std::size_t checked=0, matched=0, removed=0, failed=0;
        std::ostringstream out;
        out << "Scanning " << (recursive ? "recursively " : "") << "for files older than "
            << days << " day(s) in: " << currentDir.string() << "\n";
        if (!fs::exists(currentDir, ec) || !fs::is_directory(currentDir, ec)) return "Error: current directory invalid.";

        auto process = [&](const fs::directory_entry& entry) {
            if (entry.is_regular_file(ec)) {
                ++checked; auto ftime = entry.last_write_time(ec);
                if (!ec && ftime < cutoff_ft) {
                    ++matched; out << (dryRun ? "[DRY] " : "") << "old: " << entry.path().filename().string() << "\n";
                    if (!dryRun) { fs::remove(entry.path(), ec);
                        if (ec) { ++failed; out << "  -> remove failed: " << ec.message() << "\n"; }
                        else { ++removed; }
                    }
                }
            }
        };
        if (recursive) { for (const auto& e : fs::recursive_directory_iterator(currentDir, ec)) { if (ec) break; process(e);} }
        else           { for (const auto& e : fs::directory_iterator(currentDir, ec))          { if (ec) break; process(e);} }

        out << "Checked: " << checked << ", Matched: " << matched;
        if (!dryRun) out << ", Removed: " << removed << ", Failed: " << failed;
        return out.str();
    }

    return "Error: unknown command. Type 'help' for options.";
}

int main() {
    // Window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});
    window.setKeyRepeatEnabled(true); // so holding Backspace repeats

    // Font
    sf::Font font;
    if (!font.loadFromFile("resources/DejaVuSans.ttf")) {
        std::cerr << "Could not load font\n";
        return -1;
    }

    // Config
    const int maxMessages = 15;

    // Title
    sf::Text label("G.R.I.M", font, 32);
    { sf::FloatRect b = label.getLocalBounds(); label.setOrigin(b.width/2.f, b.height/2.f); }
    label.setFillColor(sf::Color::Black);
    label.setPosition(800.f/2.f, 50.f);

    // Chat box (input area)
    sf::RectangleShape chatBox(sf::Vector2f(760, 40));
    chatBox.setFillColor(sf::Color(200, 200, 200));
    chatBox.setPosition(20.f, 540.f);

    // Input text
    std::string userInput;
    sf::Text chatText("", font, 20);
    chatText.setFillColor(sf::Color::Black);
    chatText.setPosition(25.f, 545.f);

    // Blinking caret
    sf::Clock caretClock;
    const float caretBlinkPeriod = 0.5f; // seconds
    sf::RectangleShape caret(sf::Vector2f(2.f, 20.f)); // width, height (character size)
    caret.setFillColor(sf::Color::Black);

    // Chat history + scrolling
    std::vector<std::string> chatHistory;
    int scrollOffset = 0; // 0 = newest bottom, positive = scrolled up older messages

    // File manager state
    fs::path currentDir = fs::current_path();
    chatHistory.push_back("System: Type 'help' to see commands.");
    chatHistory.push_back(std::string("System: cwd = ") + currentDir.string());

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) window.close();
                // Optional Delete key support (cursor is at end; behaves like backspace)
                if (event.key.code == sf::Keyboard::Delete) {
                    if (!userInput.empty()) {
                        userInput.pop_back();
                        chatText.setString(userInput);
                    }
                }
                // Scroll with PageUp/PageDown
                if (event.key.code == sf::Keyboard::PageUp) {
                    scrollOffset = std::min<int>( (int)chatHistory.size()-1, scrollOffset + 3 );
                }
                if (event.key.code == sf::Keyboard::PageDown) {
                    scrollOffset = std::max(0, scrollOffset - 3);
                }
            }

            // Mouse wheel scroll
            if (event.type == sf::Event::MouseWheelScrolled) {
                if (event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                    int delta = (event.mouseWheelScroll.delta > 0 ? 3 : -3);
                    scrollOffset = std::clamp(scrollOffset + (delta>0?1:-1)*3, 0,
                                              std::max(0, (int)chatHistory.size()-1));
                }
            }

            // Text entry
            if (event.type == sf::Event::TextEntered) {
                if (event.text.unicode == '\b') {
                    if (!userInput.empty()) userInput.pop_back();
                    chatText.setString(userInput);
                } else if (event.text.unicode == '\r') {
                    std::string line = trim(userInput);
                    if (!line.empty()) {
                        chatHistory.push_back(std::string("You: ") + line);
                        std::string reply = handleCommand(line, currentDir);
                        if (!reply.empty()) {
                            std::istringstream iss(reply); std::string each;
                            while (std::getline(iss, each)) {
                                chatHistory.push_back(std::string("System: ") + each);
                            }
                        }
                        // auto-jump to latest when a new message appears
                        scrollOffset = 0;
                    }
                    userInput.clear();
                    chatText.setString(userInput);
                } else if (event.text.unicode < 128 && event.text.unicode >= 32) {
                    userInput += static_cast<char>(event.text.unicode);
                    chatText.setString(userInput);
                }
            }
        }

        // ----- DRAW -----
        window.clear(sf::Color(225, 225, 225));
        window.draw(label);

        // Draw chat history (with scroll offset)
        int total = (int)chatHistory.size();
        int visible = maxMessages;
        int startIndex = std::max(0, total - visible - scrollOffset);
        int endIndex = std::max(0, total - 1 - scrollOffset);

        float y = 520.f; // start above input box (draw bottom-up)
        for (int i = endIndex; i >= startIndex; --i) {
            sf::Text msg(chatHistory[i], font, 20);
            msg.setFillColor(sf::Color::Black);
            msg.setPosition(25.f, y);
            window.draw(msg);
            y -= 25.f;
        }

        // Input box + text
        window.draw(chatBox);
        window.draw(chatText);

        // Caret position (blinking at end of text)
        // Place caret right after the last character:
        sf::Vector2f caretPos = chatText.findCharacterPos( static_cast<std::size_t>(userInput.size()) );
        // Align vertically with text baseline
        caret.setPosition(caretPos.x, chatText.getPosition().y);
        caret.setSize(sf::Vector2f(2.f, (float)chatText.getCharacterSize()));

        // Blink
        bool showCaret = (std::fmod(caretClock.getElapsedTime().asSeconds(), caretBlinkPeriod*2.f) < caretBlinkPeriod);
        if (showCaret) window.draw(caret);

        window.display();
    }
    return 0;
}
