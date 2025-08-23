#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

// Helpers
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

// Resolve a user path relative to currentDir
static fs::path resolvePath(const fs::path& currentDir, const std::string& userPath) {
    fs::path p(userPath);
    if (p.is_absolute()) return fs::weakly_canonical(p);
    return fs::weakly_canonical(currentDir / p);
}

// Command handler: returns a multi-line string response
static std::string handleCommand(const std::string& raw, fs::path& currentDir) {
    std::string line = trim(raw);
    if (line.empty()) return "";

    auto args = split(line);
    const std::string cmd = args[0];

    // help
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
            "  rm <path>                 - remove file or empty directory\n";
    }

    // pwd
    if (cmd == "pwd") {
        return currentDir.string();
    }

    // cd
    if (cmd == "cd") {
        if (args.size() < 2) return "Error: cd requires a path.";
        fs::path target = resolvePath(currentDir, args[1]);
        std::error_code ec;
        if (!fs::exists(target, ec)) return "Error: path does not exist.";
        if (!fs::is_directory(target, ec)) return "Error: not a directory.";
        currentDir = target;
        return std::string("Directory changed to: ") + currentDir.string();
    }

    // list
    if (cmd == "list") {
        std::error_code ec;
        if (!fs::exists(currentDir, ec) || !fs::is_directory(currentDir, ec))
            return "Error: current directory invalid.";
        std::ostringstream out;
        out << "Listing: " << currentDir.string() << "\n";
        for (const auto& entry : fs::directory_iterator(currentDir, ec)) {
            if (ec) break;
            const bool isDir = entry.is_directory(ec);
            out << (isDir ? "[D] " : "    ") << entry.path().filename().string() << "\n";
        }
        return out.str();
    }

    // move
    if (cmd == "move") {
        if (args.size() < 3) return "Error: move requires <src> <dst>.";
        fs::path src = resolvePath(currentDir, args[1]);
        fs::path dst = resolvePath(currentDir, args[2]);
        std::error_code ec;

        if (!fs::exists(src, ec)) return "Error: source does not exist.";

        // If dst is an existing dir, move into it using src's filename
        if (fs::exists(dst, ec) && fs::is_directory(dst, ec)) {
            dst = dst / src.filename();
        }

        fs::create_directories(dst.parent_path(), ec); // best-effort
        ec.clear();
        fs::rename(src, dst, ec);
        if (ec) return std::string("Error moving: ") + ec.message();
        return "Moved.";
    }

    // copy
    if (cmd == "copy") {
        if (args.size() < 3) return "Error: copy requires <src> <dst>.";
        fs::path src = resolvePath(currentDir, args[1]);
        fs::path dst = resolvePath(currentDir, args[2]);
        std::error_code ec;

        if (!fs::exists(src, ec)) return "Error: source does not exist.";

        if (fs::is_directory(src, ec)) {
            fs::create_directories(dst, ec);
            ec.clear();
            fs::copy(src, dst, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
        } else {
            fs::create_directories(dst.parent_path(), ec);
            ec.clear();
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
        }
        if (ec) return std::string("Error copying: ") + ec.message();
        return "Copied.";
    }

    // mkdir
    if (cmd == "mkdir") {
        if (args.size() < 2) return "Error: mkdir requires <path>.";
        fs::path p = resolvePath(currentDir, args[1]);
        std::error_code ec;
        fs::create_directories(p, ec);
        if (ec) return std::string("Error creating directory: ") + ec.message();
        return "Directory created.";
    }

    // rm
    if (cmd == "rm") {
        if (args.size() < 2) return "Error: rm requires <path>.";
        fs::path p = resolvePath(currentDir, args[1]);
        std::error_code ec;

        if (!fs::exists(p, ec)) return "Error: path does not exist.";
        // remove() removes files or empty dirs; it will fail on non-empty dirs (safer).
        bool ok = fs::remove(p, ec);
        if (ec) return std::string("Error removing: ") + ec.message();
        if (!ok) return "Error: could not remove (directory may not be empty).";
        return "Removed.";
    }

    return "Error: unknown command. Type 'help' for options.";
}

int main() {
    // Window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});

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
    {
        sf::FloatRect b = label.getLocalBounds();
        label.setOrigin(b.width / 2.f, b.height / 2.f);
    }
    label.setFillColor(sf::Color::Black);
    label.setPosition(800.f / 2.f, 50.f);

    // Chat box
    sf::RectangleShape chatBox(sf::Vector2f(760, 40));
    chatBox.setFillColor(sf::Color(200, 200, 200));
    chatBox.setPosition(20, 540);

    // User input line
    std::string userInput;
    sf::Text chatText("", font, 20);
    chatText.setFillColor(sf::Color::Black);
    chatText.setPosition(25.f, 545.f);

    // Chat history/messages
    std::vector<std::string> chatHistory;

    // File manager state: current working directory (starts as process cwd)
    fs::path currentDir = fs::current_path();

    // Show initial info
    chatHistory.push_back(std::string("System: Type 'help' to see commands."));
    chatHistory.push_back(std::string("System: cwd = ") + currentDir.string());

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape) window.close();

            // Text entry
            if (event.type == sf::Event::TextEntered) {
                if (event.text.unicode == '\b') {
                    if (!userInput.empty()) userInput.pop_back();
                } else if (event.text.unicode == '\r') {
                    std::string line = trim(userInput);
                    if (!line.empty()) {
                        // Push user's message
                        chatHistory.push_back(std::string("You: ") + line);

                        // Handle command, push system reply (could be multi-line)
                        std::string reply = handleCommand(line, currentDir);
                        if (!reply.empty()) {
                            std::istringstream iss(reply);
                            std::string each;
                            while (std::getline(iss, each)) {
                                chatHistory.push_back(std::string("System: ") + each);
                            }
                        }
                    }
                    userInput.clear();
                    chatText.setString(userInput);
                } else if (event.text.unicode < 128 && event.text.unicode >= 32) {
                    userInput += static_cast<char>(event.text.unicode);
                    chatText.setString(userInput);
                }
            }
        }

        // DRAW
        window.clear(sf::Color(225, 225, 225));
        window.draw(label);

        // Draw most recent messages (maxMessages), bottom-up above chat box
        int startIndex = (chatHistory.size() > maxMessages)
                         ? static_cast<int>(chatHistory.size()) - maxMessages
                         : 0;
        float y = 520.f;
        for (int i = static_cast<int>(chatHistory.size()) - 1; i >= startIndex; --i) {
            sf::Text msg(chatHistory[i], font, 20);
            msg.setFillColor(sf::Color::Black);
            msg.setPosition(25.f, y);
            window.draw(msg);
            y -= 25.f;
        }

        window.draw(chatBox);
        window.draw(chatText);
        window.display();
    }
    return 0;
}

