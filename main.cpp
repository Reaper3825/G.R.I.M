#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;

std::string a = "G.R.I.M"; // Title

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});

    sf::Font font;
    if (!font.loadFromFile("resources/DejaVuSans.ttf")) {
        std::cerr << "Could not load font\n";
        return -1;
    }

    // ---- Config ----
    const int maxMessages = 15;

    // ---- Title text ----
    sf::Text label(a, font, 32);
    label.setFillColor(sf::Color::Black);
    {
        sf::FloatRect b = label.getLocalBounds();
        label.setOrigin(b.width / 2.f, b.height / 2.f);
    }
    label.setPosition(800.f / 2.f, 100.f);

    // ---- Chat box ----
    sf::RectangleShape chatBox(sf::Vector2f(760, 40));
    chatBox.setFillColor(sf::Color(200, 200, 200));
    chatBox.setPosition(20, 540);

    // ---- User input state ----
    std::string userInput;
    sf::Text chatText("", font, 20);
    chatText.setFillColor(sf::Color::Black);
    chatText.setPosition(25, 545);

    // ---- Chat history ----
    std::vector<std::string> chatHistory;

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
                    // Submit current input to history
                    if (!userInput.empty()) {
                        chatHistory.push_back(userInput);
                        std::cout << "Message: " << userInput << std::endl;
                        userInput.clear();
                    }
                } else if (event.text.unicode < 128 && event.text.unicode >= 32) {
                    // append printable ASCII
                    userInput += static_cast<char>(event.text.unicode);
                }
                chatText.setString(userInput);
            }
        }

        // ---- DRAW ----
        window.clear(sf::Color(225, 225, 225));
        window.draw(label);

        // Draw chat history (last maxMessages), stacked upward above chat box
        int startIndex = (chatHistory.size() > maxMessages)
                         ? static_cast<int>(chatHistory.size()) - maxMessages
                         : 0;
        float y = 520.f; // just above chat box
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
}
