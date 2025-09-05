#include <SFML/Graphics.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <system_error>

#include "NLP.hpp"
#include "ai.hpp"
#include "aliases.hpp"
#include "synonyms.hpp"
#include "commands.hpp"
#include "console_history.hpp"
#include "timer.hpp"
#include "resources.hpp"
#include "ui_config.hpp"
#include "ui_helpers.hpp"
#include "ui_draw.hpp"
#include "ui_events.hpp"

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <psapi.h>
#endif

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    // --- Load persistent memory ---
    loadMemory();

    // --- NLP setup ---
    NLP nlp;
    {
        std::string rulesText = loadTextResource("nlp_rules.json", argc, argv);
        if (!rulesText.empty()) {
            std::string err;
            if (!nlp.load_rules_from_string(rulesText, &err)) {
                std::cerr << "[NLP] Failed to parse rules: " << err << "\n";
            }
        } else {
            std::cerr << "[NLP] Could not load nlp_rules.json\n";
        }
    }

    // --- Load synonyms & aliases ---
    {
        std::string synText = loadTextResource("synonyms.json", argc, argv);
        if (!synText.empty()) loadSynonymsFromString(synText);

        std::string aliasText = loadTextResource("app_aliases.json", argc, argv);
        if (!aliasText.empty()) loadAliasesFromString(aliasText);
    }

    // --- Create window ---
    sf::RenderWindow window(sf::VideoMode(600, 900), "GRIM", sf::Style::Default);
    window.setVerticalSyncEnabled(true);

    // --- History ---
    ConsoleHistory history;
    auto addHistory = [&](const std::string& s, sf::Color c = sf::Color::White) { 
        history.push(s, c); 
        std::cout << s << "\n"; 
    };

    // --- Font ---
    sf::Font font;
    std::string fontPath = findAnyFontInResources(argc, argv, &history);
    if (!fontPath.empty()) {
        if (!font.loadFromFile(fontPath)) {
            addHistory("[ERROR] Failed to load font from " + fontPath, sf::Color::Red);
        } else {
            std::cout << "[INFO] Font loaded: " << fontPath << "\n";
        }
    }

    // --- Initial state ---
    std::string buffer;
    fs::path currentDir = fs::current_path();
    addHistory("GRIM is ready. Type 'help' for commands. Type 'quit' to exit.", sf::Color(160,200,255));

    sf::Clock caretClock;
    bool caretVisible = true;
    float scrollOffsetLines = 0.f;
    std::vector<Timer> timers;

    // ----------------- Main loop -----------------
    while (window.isOpen()) {
        // --- Process events ---
        if (!processEvents(window, buffer, currentDir, timers, longTermMemory, nlp, history))
            break;

        // --- Timers ---
        for (auto& t : timers) {
            if (!t.done && t.clock.getElapsedTime().asSeconds() >= t.seconds) {
                addHistory("[Timer] Timer finished!", sf::Color(255,200,0));
                t.done = true;
            }
        }

        // --- Caret blink ---
        caretVisible = updateCaretBlink(caretClock, caretVisible);

        // --- Draw UI ---
        drawUI(window, font, history, buffer, caretVisible, scrollOffsetLines);
    }

    return 0;
}
