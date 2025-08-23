#include <iostream>
#include <SFML/Graphics.hpp>

int a = 6;

int main() {
    std::cout << "Value of a: " << a << std::endl;

    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});

    sf::CircleShape circle(50.f);              // radius 50
    circle.setPointCount(100);                 // smoother circle
    circle.setFillColor(sf::Color::Green);
    circle.setOutlineColor(sf::Color::Red);    // outline to make it pop
    circle.setOutlineThickness(3.f);
    circle.setPosition(200.f, 200.f);          // top-left of circle’s bounding box

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::KeyPressed &&
                event.key.code == sf::Keyboard::Escape) window.close();
        }

        // comment movement out for now to avoid it zipping off-screen
        // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) circle.move(2.f, 0.f);

        window.clear(sf::Color(30, 30, 30));   // dark gray background
        window.draw(circle);
        window.display();
    }
}
