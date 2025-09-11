#include <SFML/Graphics.hpp> 
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <system_error>

#include "nlp.hpp"
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
#include "bootstrap.hpp"
#include "voice.hpp"
#include "voice_stream.hpp"   // <-- new include

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#include <psapi.h>
#endif

namespace fs = std::filesystem;

// Global textbox buffer (used by keyboard + voice stream)
std::string g_ui_textbox;

int main(int argc, char** argv) {
    // --- Load persistent memory ---
    loadMemory();
    std::cout << "[DEBUG] Memory loaded successfully\n";

    // --- Load AI config ---
    loadAIConfig("ai_config.json");
    std::cout << "[DEBUG] AI config loaded from ai_config.json\n";

    // --- Bootstrap check ---
    runBootstrapChecks(argc, argv);
    std::cout << "[DEBUG] Bootstrap checks complete\n";

    // --- Warm up AI model ---
warmupAI();

    // --- NLP setup ---
    NLP nlp;  // <-- declare once here, survives for whole main()

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
    std::cout << "[DEBUG] NLP rules loaded at startup\n";

    // --- Load synonyms & aliases ---
    {
        std::string synText = loadTextResource("synonyms.json", argc, argv);
        if (!synText.empty()) {
            loadSynonymsFromString(synText);
            std::cout << "[DEBUG] Synonyms loaded\n";
        }

        std::string aliasText = loadTextResource("app_aliases.json", argc, argv);
        if (!aliasText.empty()) {
            loadAliasesFromString(aliasText);
            std::cout << "[DEBUG] Aliases loaded\n";
        }
    }

    // --- Whisper context (shared for voice) ---
    if (!initWhisper()) {
        std::cerr << "[ERROR] Failed to initialize Whisper. Voice features will be disabled." << std::endl;
    } else {
        std::cout << "[DEBUG] Whisper context initialized\n";
    }

    // --- Create window ---
    sf::RenderWindow window(sf::VideoMode(600, 900), "GRIM", sf::Style::Default);
    window.setVerticalSyncEnabled(true);
    std::cout << "[DEBUG] Window created (600x900)\n";

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
            std::cout << "[DEBUG] Font loaded: " << fontPath << "\n";
        }
    } else {
        std::cerr << "[ERROR] No font found!\n";
    }

    // --- Initial state ---
    fs::path currentDir = fs::current_path();
    addHistory("GRIM is ready. Type 'help' for commands. Type 'quit' to exit.", sf::Color(160,200,255));
    std::cout << "[DEBUG] GRIM startup complete, entering main loop\n";

    sf::Clock caretClock;
    bool caretVisible = true;
    float scrollOffsetLines = 0.f;
    std::vector<Timer> timers;

    // --- Load LongTerm memory ---
    nlohmann::json longTermMemory;

    // ----------------- Main loop -----------------
    while (window.isOpen()) {
        // --- Process events ---
        if (!processEvents(window, g_ui_textbox, currentDir, timers, longTermMemory, nlp, history))
            break;

        // --- Handle commands ---
        if (!g_ui_textbox.empty() && g_ui_textbox.back() == '\n') {
            std::string cmd = g_ui_textbox;
            g_ui_textbox.clear();

            // trim newline
            if (!cmd.empty() && cmd.back() == '\n') cmd.pop_back();

            if (cmd == "quit") {
                addHistory("[INFO] Quitting...", sf::Color::Yellow);
                break;
            }

            // Voice demo special case
            if (cmd == "voice") {
                addHistory("[Voice] Starting 5-second recording...", sf::Color(0, 255, 128));

                std::string transcript = runVoiceDemo("external/whisper.cpp/models/ggml-small.bin");
                if (!transcript.empty()) {
                    addHistory("[Voice] Heard: " + transcript, sf::Color::Yellow);

                    Intent spokenIntent = nlp.parse(transcript);
                    if (spokenIntent.matched) {
                        handleCommand(spokenIntent, g_ui_textbox, currentDir, timers, longTermMemory, nlp, history);
                    } else {
                        addHistory("[Voice] Sorry, I didn’t understand that.", sf::Color::Red);
                    }
                } else {
                    addHistory("[Voice] No speech detected.", sf::Color::Red);
                }
            }

            // Voice stream special case
            else if (cmd == "voice_stream") {
                addHistory("[VoiceStream] Starting live microphone stream...", sf::Color(0, 200, 255));
                runVoiceStream(g_whisperCtx, &history, timers, longTermMemory, nlp);
                addHistory("[VoiceStream] Stopped.", sf::Color(0, 200, 255));
            }

            // Normal text commands
            else {
                Intent intent = nlp.parse(cmd);
                if (intent.matched) {
                    handleCommand(intent, g_ui_textbox, currentDir, timers, longTermMemory, nlp, history);
                } else {
                    addHistory("[WARN] No intent matched for: " + cmd, sf::Color::Red);
                    std::cout << "[DEBUG][Command] No intent matched for input: '" 
                              << cmd << "' (rules loaded=" << nlp.rule_count() << ")\n";
                }
            }
        }

        // --- Timers ---
        for (auto& t : timers) {
            if (!t.done && t.clock.getElapsedTime().asSeconds() >= t.seconds) {
                addHistory("[Timer] Timer finished!", sf::Color(255,200,0));
                t.done = true;
            }
        }

        // --- Caret blink ---
        caretVisible = updateCaretBlink(caretClock, caretVisible);

        // --- Wrap history for current window width ---
        sf::Text meas;
        meas.setFont(font);
        meas.setCharacterSize(kFontSize); // from ui_config.hpp
        history.ensureWrapped(
            static_cast<float>(window.getSize().x) - (2 * kSidePad), 
            meas
        );

        // --- Draw UI ---
        drawUI(window, font, history, g_ui_textbox, caretVisible, scrollOffsetLines);
    }

    if (g_whisperCtx) {
        whisper_free(g_whisperCtx);
    }

    return 0;
}
