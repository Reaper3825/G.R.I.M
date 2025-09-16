#pragma once
#include <SFML/Graphics.hpp>
#include <string>
#include <filesystem>
#include <vector>
#include "nlp.hpp"
#include "console_history.hpp"
#include "timer.hpp"
#include <nlohmann/json_fwd.hpp>



// Process all SFML events, update buffer/history, and handle commands.
// Returns false if the window should close.
bool processEvents(
    sf::RenderWindow& window,
    std::string& buffer,
    std::filesystem::path& currentDir,
    std::vector<Timer>& timers,
    nlohmann::json& longTermMemory, 
    ConsoleHistory& history
);
