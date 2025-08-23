#include <iostream>
#include <SFML/Graphics.hpp>
#include <string>  
// var declarations
std::string a = "initialized";

int main() {
    std::cout << "Opertion: " << a << std::endl;

    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window"); // Window size
    window.setPosition({100,100});
     sf::Font font;
   
     // Load a font from file
     if (!font.loadFromFile("arial.ttf")) {
        std::cerr << "Could not load font\n";
        return -1;
    }

      sf::Text label(a, font, 32);   //use string variable directly
    label.setFillColor(sf::Color::Black);

       // Center text in the window
    sf::FloatRect bounds = label.getLocalBounds();
    label.setOrigin(bounds.width / 2.f, bounds.height / 2.f);
    label.setPosition(800.f/2.f, 600.f/2.f);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape) window.close();
        }


        window.clear(sf::Color(225, 225, 225));   // Background color
        window.draw(label); // Draw the text
        window.display();
    }
}
