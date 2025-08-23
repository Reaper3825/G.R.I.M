/*
A Wadkins
8/20/25
*/
// Start
#include <iostream>
#include <SFML/Graphics.hpp>
using namespace std;

// Variable declarations
int a = 6;

int main() {
    // Print the variable to console
    cout << "Value of a: " << a << endl;

    // Create SFML window
    sf::RenderWindow window(sf::VideoMode(800, 600), "Blank Window");

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.display();
    }

    return 0;
}
