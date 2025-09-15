
#include <SFML/Graphics.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "console_history.hpp"
#include "commands/commands_core.hpp"   // 👈 FIXED
#include "nlp.hpp"


// -------------------------------------------------------------
// Handle UI events (keyboard input, dispatch commands, etc.)
// -------------------------------------------------------------
bool processEvents(sf::RenderWindow& window,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& timers,
                   nlohmann::json& longTermMemory,
                   ConsoleHistory& history)
{
    sf::Event event;
    while (window.pollEvent(event)) {
        // Handle quit
        if (event.type == sf::Event::Closed) {
            window.close();
            return false;
        }

        // Handle text input
        if (event.type == sf::Event::TextEntered) {
            if (event.text.unicode == '\r' || event.text.unicode == '\n') {
                std::string input = buffer;
                buffer.clear();

                if (!input.empty()) {
                    history.push("> " + input, sf::Color::Green);

                    Intent intent = g_nlp.parse(input);
                    if (intent.matched) {
                        handleCommand(intent.name);
                    } else {
                        history.push("[NLP] No intent matched for input: '" + input +
                                     "' (rules loaded=" + std::to_string(g_nlp.rule_count()) + ")",
                                     sf::Color::Red);
                    }
                }
            } else if (event.text.unicode == 8) { // Backspace
                if (!buffer.empty()) buffer.pop_back();
            } else if (event.text.unicode < 128) { // ASCII
                buffer.push_back(static_cast<char>(event.text.unicode));
            }
        }

        if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::Escape) {
                window.close();
                return false;
            }
        }
    }

    return true;
}
