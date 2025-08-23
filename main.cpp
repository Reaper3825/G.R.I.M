#include <iostream
#include <SFML/Graphics.hpp>
int a = 6;

int main() {
    std::cout << "Value of a: " << a << std::endl;

    std::cout << "Before window\n";                 // 👈
    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    std::cout << "After window\n";                  // 👈

    window.setPosition({100,100});                  // 👈 avoids off‑screen weirdness

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }
        window.clear();
        window.display();
    }
}
