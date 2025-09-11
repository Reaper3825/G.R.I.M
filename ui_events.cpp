#include "ui_events.hpp"
#include "commands.hpp"
#include "ui_helpers.hpp"   // for g_ui_textbox + ui_set_textbox
#include <iostream>

// g_ui_textbox is declared in main.cpp, extern in ui_helpers.hpp
extern std::string g_ui_textbox;

bool processEvents(
    sf::RenderWindow& window,
    std::string& /*unused*/,  // buffer is global now
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
            else if (e.text.unicode == 13 || e.text.unicode == 10) { // enter pressed
                std::string line = g_ui_textbox; // copy current buffer
                g_ui_textbox.clear();            // clear for new input

                if (line == "quit" || line == "exit") {
                    history.push("> " + line, sf::Color(150,255,150));
                    window.close();
                    return false;
                }

                if (!line.empty()) {
                    history.push("> " + line, sf::Color(150,255,150));
                    Intent intent = nlp.parse(line);
                    if (intent.matched) {
                        handleCommand(intent, g_ui_textbox, currentDir, timers, longTermMemory, nlp, history);
                    } else {
                        history.push("[WARN] No intent matched for: " + line, sf::Color::Red);
                        std::cout << "[DEBUG][Command] No intent matched for input: '"
                                  << line << "' (rules loaded=" << nlp.rule_count() << ")\n";
                    }
                } else {
                    history.push("> ");
                }
            }
            else if (e.text.unicode >= 32 && e.text.unicode < 127) { // printable ASCII
                g_ui_textbox.push_back(static_cast<char>(e.text.unicode));
            }
        }
    }
    return true;
}
