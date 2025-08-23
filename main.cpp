#include <iostream>
#include <SFML/Graphics.hpp>

int a = 6;

int main() {
    std::cout << "Value of a: " << a << std::endl;

    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");
    window.setPosition({100,100});

    // Create a green circle
    sf::CircleShape circle(50.f);         // radius 50
    circle.setFillColor(sf::Color::Green);
    circle.setPosition(200.f, 200.f);     // starting position

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            // Close if Escape is pressed
            if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::Escape)
                window.close();
        }

        // Move circle with arrow keys
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up))
            circle.move(0.f, -2.f);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down))
            circle.move(0.f, 2.f);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
            circle.move(-2.f, 0.f);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
            circle.move(2.f, 0.f);

        // --- DRAW SECTION ---
        window.clear();
        window.draw(circle);   // <--- draw your shapes/sprites here
        window.display();
    }
}
