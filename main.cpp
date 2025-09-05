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

// ----------------- main -----------------
int main(int argc, char** argv) {
    // --- Load persistent memory ---
    loadMemory();

    // --- NLP setup ---
    NLP nlp;
    {
        std::error_code ec;
        fs::path exeDir = (argc > 0) ? fs::path(argv[0]).parent_path() : fs::current_path(ec);
        fs::path rulesExe = exeDir / "nlp_rules.json";
        fs::path rulesCwd = "nlp_rules.json";
        std::string err;
        bool ok = fs::exists(rulesExe, ec)
            ? nlp.load_rules(rulesExe.string(), &err)
            : nlp.load_rules(rulesCwd.string(), &err);
        if (!ok) std::cerr << "[NLP] Failed to load rules: " << err << "\n";
    }

    // --- Load synonyms & aliases ---
    loadSynonyms("synonyms.json");
    loadAliases("app_aliases.json");

    // --- Create window ---
    sf::RenderWindow window(sf::VideoMode(600, 900), "GRIM", sf::Style::Default);
    window.setVerticalSyncEnabled(true);

    // --- History (must exist before font discovery) ---
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
        if (!processEvents(window, buffer, currentDir, timers, nlp, history)) break;

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
