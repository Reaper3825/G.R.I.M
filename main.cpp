#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "commands.hpp"
#include "nlp.hpp" // so you can use parseNaturalLanguage()

namespace fs = std::filesystem;

// ---------------- Global Vars ----------------
int   historyYDis   = 30;
unsigned int WindowWidth  = 400;
unsigned int WindowHeight = 800;

// Wrap text into multiple lines that fit within maxWidth
std::string wrapText(const std::string& str, const sf::Font& font, unsigned int charSize, float maxWidth) {
    std::istringstream iss(str);
    std::string word, line, output;
    sf::Text temp("", font, charSize);

    while (iss >> word) {
        std::string testLine = line + (line.empty() ? "" : " ") + word;
        temp.setString(testLine);
        if (temp.getLocalBounds().width > maxWidth) {
            if (!line.empty()) output += line + "\n";
            line = word;
        } else {
            line = testLine;
        }
    }
    if (!line.empty()) output += line;
    return output;
}

int main() {
    // ---------- Window ----------
    sf::RenderWindow window(
        sf::VideoMode(WindowWidth, WindowHeight),
        "G.R.I.M",
        sf::Style::Resize | sf::Style::Close | sf::Style::Titlebar
    );
    window.setPosition({500, 500});
    window.setKeyRepeatEnabled(true);

    // ---------- Font ----------
    sf::Font font;
    if (!font.loadFromFile("resources/DejaVuSans.ttf")) {
        std::cerr << "Could not load font\n";
        return -1;
    }

    const int maxMessages = 15;

    // ---------- Title ----------
    sf::Text label("G.R.I.M", font, 30);
    label.setFillColor(sf::Color::Black);

    // ---------- Chat box ----------
    sf::RectangleShape chatBox;
    chatBox.setFillColor(sf::Color(200, 200, 200));

    // ---------- Input text + caret ----------
    std::string userInput;
    sf::Text chatText("", font, 20);
    chatText.setFillColor(sf::Color::Black);

    sf::Clock caretClock;
    const float caretBlinkPeriod = 0.5f;
    sf::RectangleShape caret(sf::Vector2f(2.f, 20.f));
    caret.setFillColor(sf::Color::Black);

    // ---------- Histories ----------
    std::vector<std::string> chatHistory;
    int scrollOffset = 0;
    std::vector<std::string> commandHistory;
    int historyIndex = -1;

    // ---------- File manager state ----------
    fs::path currentDir = fs::current_path();
    chatHistory.push_back("Grim: Type 'help' to see commands.");
    chatHistory.push_back("Grim: cwd = " + currentDir.string());

    // ---------------- Main loop ----------------
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            // Handle window resize
            if (event.type == sf::Event::Resized) {
                sf::FloatRect visibleArea(0.f, 0.f,
                    static_cast<float>(event.size.width),
                    static_cast<float>(event.size.height));
                window.setView(sf::View(visibleArea));
            }

            // ---------- Key press events ----------
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code == sf::Keyboard::Escape) window.close();

                if (event.key.code == sf::Keyboard::Up) {
                    if (!commandHistory.empty()) {
                        if (historyIndex == -1) historyIndex = (int)commandHistory.size();
                        if (historyIndex > 0) historyIndex--;
                        userInput = commandHistory[historyIndex];
                        chatText.setString(userInput);
                    }
                }
                if (event.key.code == sf::Keyboard::Down) {
                    if (!commandHistory.empty() && historyIndex != -1) {
                        if (historyIndex < (int)commandHistory.size() - 1) {
                            historyIndex++;
                            userInput = commandHistory[historyIndex];
                        } else {
                            historyIndex = -1;
                            userInput.clear();
                        }
                        chatText.setString(userInput);
                    }
                }

                if (event.key.code == sf::Keyboard::PageUp) {
                    scrollOffset = std::min<int>(std::max(0, (int)chatHistory.size() - 1),
                                                 scrollOffset + 3);
                }
                if (event.key.code == sf::Keyboard::PageDown) {
                    scrollOffset = std::max(0, scrollOffset - 3);
                }
            }

            // ---------- Mouse wheel scroll ----------
            if (event.type == sf::Event::MouseWheelScrolled &&
                event.mouseWheelScroll.wheel == sf::Mouse::VerticalWheel) {
                int step = (event.mouseWheelScroll.delta > 0) ? 3 : -3;
                int maxOffset = std::max(0, (int)chatHistory.size() - 1);
                scrollOffset = std::clamp(scrollOffset - step, 0, maxOffset);
            }

            // ---------- Text input ----------
            if (event.type == sf::Event::TextEntered) {
                if (event.text.unicode == 8) { // backspace
                    if (!userInput.empty()) userInput.pop_back();
                    chatText.setString(userInput);

                } else if (event.text.unicode == '\r') { // enter
                    std::string line = userInput;
                    if (!line.empty()) {
                        chatHistory.push_back("You: " + line);
                        commandHistory.push_back(line);
                        historyIndex = -1;

                        std::string reply = parseNaturalLanguage(line, currentDir);
                        if (reply.empty()) {
                            reply = handleCommand(line, currentDir);
                        }

                        if (!reply.empty()) {
                            std::istringstream iss(reply);
                            std::string each;
                            while (std::getline(iss, each)) {
                                chatHistory.push_back("Grim: " + each);
                            }
                        }
                        scrollOffset = 0;
                    }
                    userInput.clear();
                    chatText.setString(userInput);

                } else if (event.text.unicode >= 32 && event.text.unicode < 128) {
                    userInput += static_cast<char>(event.text.unicode);
                    chatText.setString(userInput);
                }
            }
        }

        // ---------- Layout ----------
        sf::Vector2u winSize = window.getSize();

        sf::FloatRect b = label.getLocalBounds();
        label.setOrigin(b.width / 2.f, b.height / 2.f);
        label.setPosition((float)winSize.x / 2.f, 30.f);

        chatBox.setSize(sf::Vector2f(std::max(0u, winSize.x - 40u), 40.f));
        chatBox.setPosition(20.f, (float)winSize.y - 60.f);

        chatText.setPosition(chatBox.getPosition().x + 5.f,
                             chatBox.getPosition().y + 5.f);
        caret.setSize(sf::Vector2f(2.f, (float)chatText.getCharacterSize()));

        // ---------- DRAW ----------
        window.clear(sf::Color(225, 225, 225));
        window.draw(label);

        // Draw chat history
        int total = (int)chatHistory.size();
        int startIndex = std::max(0, total - maxMessages - scrollOffset);
        int endIndex   = std::max(0, total - 1 - scrollOffset);

        float y = chatBox.getPosition().y - (float)historyYDis;
        float chatAreaWidth = (float)winSize.x - 60.f;

        for (int i = endIndex; i >= startIndex; --i) {
            std::string wrapped = wrapText(chatHistory[i], font, 20, chatAreaWidth);
            std::istringstream iss(wrapped);
            std::string line;
            std::vector<std::string> lines;
            while (std::getline(iss, line)) {
                lines.push_back(line);
            }
            for (int j = (int)lines.size() - 1; j >= 0; --j) {
                sf::Text msg(lines[j], font, 20);
                msg.setFillColor(sf::Color::Black);
                msg.setPosition(25.f, y);
                window.draw(msg);
                y -= 25.f;
            }
        }

        window.draw(chatBox);
        window.draw(chatText);

        // Caret blink
        sf::Vector2f caretPos = chatText.findCharacterPos(userInput.size());
        caret.setPosition(caretPos.x, chatText.getPosition().y);
        bool showCaret = (std::fmod(caretClock.getElapsedTime().asSeconds(),
                                    caretBlinkPeriod * 2.f) < caretBlinkPeriod);
        if (showCaret) window.draw(caret);

        window.display();
    }

    return 0;
}
