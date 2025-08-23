#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

// var declarations
std::string a = "G.R.I.M"; //Title

int main() {
    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});

    sf::Font font;
    if (!font.loadFromFile("resources/DejaVuSans.ttf")) {
        std::cerr << "Could not load font\n";
        return -1;
    }

    // --- Title text ---
    sf::Text label(a, font, 32);
    label.setFillColor(sf::Color::Black);
    sf::FloatRect bounds = label.getLocalBounds();
    label.setOrigin(bounds.width / 2.f, bounds.height / 2.f);
    label.setPosition(800.f / 2.f, 100.f);

    // --- Chat box background ---
    sf::RectangleShape chatBox(sf::Vector2f(760, 40));  // width, height
    chatBox.setFillColor(sf::Color(200, 200, 200));
    chatBox.setPosition(20, 540);  // bottom of window

    // --- User input text ---
    std::string userInput;
    sf::Text chatText;
    chatText.setFont(font);
    chatText.setCharacterSize(20);
    chatText.setFillColor(sf::Color::Black);
    chatText.setPosition(25, 545);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();

            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape) window.close();

            // --- Text entry handling ---
            if (event.type == sf::Event::TextEntered) {
                if (event.text.unicode == '\b') {
                    // backspace
                    if (!userInput.empty())
                        userInput.pop_back();
                } else if (event.text.unicode == '\r') {
                    // enter key (command finished)
                    std::cout << "Command entered: " << userInput << std::endl;
                    userInput.clear(); // reset input after "sending"
                } else if (event.text.unicode < 128) {
                    // normal ASCII character
                    userInput += static_cast<char>(event.text.unicode);
                }
                chatText.setString(userInput);
            }
        }

        window.clear(sf::Color(225, 225, 225));
        window.draw(label);     // Title
        window.draw(chatBox);   // Chat box background
        window.draw(chatText);  // Chat box text
        window.display();
    }
}
