#include "ui_events.hpp"
#include "commands.hpp"
#include "ui_helpers.hpp"   // for g_ui_textbox + ui_set_textbox
#include <iostream>

// g_ui_textbox is declared in main.cpp, extern in ui_helpers.hpp
extern std::string g_ui_textbox;

bool processEvents(
    sf::RenderWindow& window,
    std::string& /*unused*/,  // buffer is now global, ignore this param
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
                if (!g_ui_textbox.empty()) {
                    g_ui_textbox.pop_back();
                }
            }
            else if (e.text.unicode == 13 || e.text.unicode == 10) { // enter
                std::string line = g_ui_textbox;
                g_ui_textbox.clear();

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
                handleCommand(intent, g_ui_textbox, currentDir, timers, longTermMemory, nlp, history);
            }
            else if (e.text.unicode >= 32 && e.text.unicode < 127) { // printable ASCII
                g_ui_textbox.push_back(static_cast<char>(e.text.unicode));
            }
        }
    }
    return true;
}
#include "ui_events.hpp"
#include "commands.hpp"
#include "ui_helpers.hpp"   // for g_ui_textbox + ui_set_textbox
#include <iostream>

// g_ui_textbox is declared in main.cpp, extern in ui_helpers.hpp
extern std::string g_ui_textbox;

bool processEvents(
    sf::RenderWindow& window,
    std::string& /*unused*/,  // buffer is now global, ignore this param
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
                if (!g_ui_textbox.empty()) {
                    g_ui_textbox.pop_back();
                }
            }
            else if (e.text.unicode == 13 || e.text.unicode == 10) { // enter
                std::string line = g_ui_textbox;
                g_ui_textbox.clear();

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
                handleCommand(intent, g_ui_textbox, currentDir, timers, longTermMemory, nlp, history);
            }
            else if (e.text.unicode >= 32 && e.text.unicode < 127) { // printable ASCII
                g_ui_textbox.push_back(static_cast<char>(e.text.unicode));
            }
        }
    }
    return true;
}
