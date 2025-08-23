#include <iostream>
#include <SFML/Graphics.hpp>

int a = 6;

int main() {
    std::cout << "Value of a: " << a << std::endl;

    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape) window.close();
        }

        // comment movement out for now to avoid it zipping off-screen
        // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) circle.move(2.f, 0.f);

        window.clear(sf::Color(127, 30, 30));   // Background color

        window.display();
    }
}
