#include "ui_events.hpp"
#include "commands.hpp"
#include <iostream>

bool processEvents(
    sf::RenderWindow& window,
    std::string& buffer,
    std::filesystem::path& currentDir,
    std::vector<Timer>& timers,
    nlohmann::json& longTermMemory,
    NLP& nlp,
    ConsoleHistory& history
) {
    sf::Event e;
    while (window.pollEvent(e)) {
        // --- Window close ---
        if (e.type == sf::Event::Closed) {
            window.close();
            return false;
        }

        // --- Escape key closes window ---
        if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
            window.close();
            return false;
        }

        // --- Window resize ---
        if (e.type == sf::Event::Resized) {
            sf::FloatRect visibleArea(0, 0, e.size.width, e.size.height);
            window.setView(sf::View(visibleArea));
        }

        // --- Text entry ---
        if (e.type == sf::Event::TextEntered) {
            if (e.text.unicode == 8) { // backspace
                if (!buffer.empty()) buffer.pop_back();
            }
            else if (e.text.unicode == 13 || e.text.unicode == 10) { // enter
                std::string line = buffer;
                buffer.clear();

                if (line == "quit" || line == "exit") {
                    window.close();
                    return false;
                }

                if (!line.empty()) {
                    history.push("> " + line, sf::Color(150,255,150));
                } else {
                    history.push("> ");
                    continue;
                }

                Intent intent = nlp.parse(line);
                handleCommand(intent, buffer, currentDir, timers, longTermMemory, nlp, history);
            }
            else if (e.text.unicode >= 32 && e.text.unicode < 127) { // printable ASCII
                buffer.push_back(static_cast<char>(e.text.unicode));
            }
        }
    }
    return true;
}
