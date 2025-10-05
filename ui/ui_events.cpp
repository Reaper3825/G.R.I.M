#include <SFML/Config.hpp>
#include "console_history.hpp"
#include "commands/commands_core.hpp"   // CommandResult + handleCommand
#include "nlp/nlp.hpp"

// -------------------------------------------------------------
// Handle UI events (SFML 3 style: is<T>() + getIf<T>())
// -------------------------------------------------------------
bool processEvents(sf::RenderWindow& window,
                   std::string& buffer,
                   std::filesystem::path& currentDir,
                   std::vector<Timer>& uiTimers,
                   nlohmann::json& longTermMemory,
                   ConsoleHistory& uiHistory)
{
    while (auto evOpt = window.pollEvent()) {
        const sf::Event& ev = *evOpt;

        // Handle quit
        if (ev.is<sf::Event::Closed>()) {
            window.close();
            return false;
        }

        // Handle text input
        if (const auto* text = ev.getIf<sf::Event::TextEntered>()) {
            if (text->unicode == '\r' || text->unicode == '\n') {
                std::string input = buffer;
                buffer.clear();

                if (!input.empty()) {
                    handleCommand(input);
                    uiHistory.push(input, sf::Color::Green);
                }
            } else if (text->unicode == 8) { // Backspace
                if (!buffer.empty()) buffer.pop_back();
            } else if (text->unicode < 128) { // ASCII
                buffer.push_back(static_cast<char>(text->unicode));
            }
        }

        // Handle key presses
        if (const auto* key = ev.getIf<sf::Event::KeyPressed>()) {
            if (key->code == sf::Keyboard::Key::Escape) {
                window.close();
                return false;
            }
        }
    }

    return true;
}
