// main.cpp
#include <SFML/Graphics.hpp>
#include <iostream>
#include <deque>
#include <string>
#include <sstream>
#include <filesystem>
#include "NLP.hpp"

namespace fs = std::filesystem;

static const float kWindowW = 480.f;
static const float kWindowH = 800.f;
static const size_t kMaxHistory = 100; // lines kept in memory

// Simple text history buffer
class ConsoleHistory {
public:
    void push(const std::string& line) {
        if (lines.size() >= kMaxHistory) lines.pop_front();
        lines.push_back(line);
    }
    const std::deque<std::string>& data() const { return lines; }
private:
    std::deque<std::string> lines;
};

int main() {
    // --- Load NLP rules
    NLP nlp;
    std::string err;
    std::string rulesPath = "nlp_rules.json";
    if (!nlp.load_rules(rulesPath, &err)) {
        std::cerr << "[NLP] Failed to load rules from " << rulesPath << " : " << err << "\n";
        // Not fatal; you can still run the UI
    } else {
        std::cerr << "[NLP] Loaded rules. Intents:";
        for (auto& s : nlp.list_intents()) std::cerr << " " << s;
        std::cerr << "\n";
    }

    // --- Create window
    sf::RenderWindow window(sf::VideoMode((unsigned)kWindowW, (unsigned)kWindowH), "GRIM");
    window.setVerticalSyncEnabled(true);

    // --- Load font
    sf::Font font;
    std::string fontPath = "DejaVuSans.ttf"; // put this file next to the exe or project root
    if (!font.loadFromFile(fontPath)) {
        std::cerr << "[WARN] Could not load font: " << fontPath << "\n"
                  << "       Text will not render. Place a TTF named DejaVuSans.ttf next to the exe.\n";
    }

    // --- UI geometry
    const float inputHeight = 34.f;
    sf::RectangleShape inputBar({kWindowW, inputHeight});
    inputBar.setFillColor(sf::Color(30, 30, 35));
    inputBar.setPosition(0.f, kWindowH - inputHeight);

    sf::Text inputText;
    inputText.setFont(font);
    inputText.setCharacterSize(18);
    inputText.setFillColor(sf::Color::White);
    inputText.setPosition(8.f, kWindowH - inputHeight + 6.f);

    sf::Text historyText;
    historyText.setFont(font);
    historyText.setCharacterSize(18);
    historyText.setFillColor(sf::Color::White);
    historyText.setPosition(8.f, 8.f);

    ConsoleHistory history;
    std::string buffer;

    auto addHistory = [&](const std::string& line){
        history.push(line);
        std::cout << line << "\n";
    };

    addHistory("GRIM is ready. Type a command and press Enter. Type 'quit' to exit.");
    if (!err.empty()) addHistory(std::string("[NLP] Rules load error: ") + err);

    // caret blink
    sf::Clock caretClock;
    bool caretVisible = true;

    while (window.isOpen()) {
        // --- handle events
        sf::Event e;
        while (window.pollEvent(e)) {
            if (e.type == sf::Event::Closed) window.close();

            if (e.type == sf::Event::KeyPressed) {
                if (e.key.code == sf::Keyboard::Escape) window.close();
            }

            if (e.type == sf::Event::TextEntered) {
                // handle basic text input
                if (e.text.unicode == 8) { // backspace
                    if (!buffer.empty()) buffer.pop_back();
                } else if (e.text.unicode == 13 || e.text.unicode == 10) { // Enter
                    std::string line = buffer;
                    buffer.clear();

                    if (line == "quit" || line == "exit") {
                        window.close();
                        break;
                    }

                    if (line.empty()) {
                        addHistory("> ");
                        continue;
                    }

                    addHistory("> " + line);

                    // parse with NLP
                    Intent intent = nlp.parse(line);
                    if (!intent.matched) {
                        addHistory("[NLP] No intent matched.");
                    } else {
                        std::ostringstream oss;
                        oss << "[NLP] intent=" << intent.name << " score=" << intent.score;
                        addHistory(oss.str());
                        for (const auto& kv : intent.slots) {
                            addHistory("  " + kv.first + " = " + kv.second);
                        }

                        // Route to your actions here:
                        // e.g., if (intent.name == "open_app") { launch(intent.slots["app"]); }
                    }
                } else {
                    // filter control chars
                    if (e.text.unicode >= 32 && e.text.unicode < 127) {
                        buffer.push_back(static_cast<char>(e.text.unicode));
                    }
                }
            }
        }

        // update caret
        if (caretClock.getElapsedTime().asSeconds() > 0.5f) {
            caretVisible = !caretVisible;
            caretClock.restart();
        }

        // render
        window.clear(sf::Color(18, 18, 22));

        // draw history
        if (font.getInfo().family != "") {
            std::ostringstream hist;
            for (const auto& line : history.data()) hist << line << "\n";
            historyText.setString(hist.str());
            window.draw(historyText);
        }

        // draw input bar + text
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
