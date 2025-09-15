#include "commands/commands_system.hpp"
#include "system_detect.hpp"
#include "console_history.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>

// Externals
extern ConsoleHistory history;

CommandResult cmdSystemInfo(const std::string& arg) {
    std::cout << "[DEBUG][Command] Dispatch: system_info\n";

    SystemInfo sys = detectSystem();

    history.push("[System Info]", sf::Color::Cyan);
    history.push("OS         : " + sys.osName + " (" + sys.arch + ")", sf::Color::Yellow);
    history.push("CPU Cores  : " + std::to_string(sys.cpuCores), sf::Color::Yellow);
    history.push("RAM        : " + std::to_string(sys.ramMB) + " MB", sf::Color::Green);

    if (sys.hasGPU) {
        std::string gpuLine = sys.gpuName + " (" + std::to_string(sys.gpuCount) + " device(s))";
        history.push("GPU        : " + gpuLine, sf::Color(180, 255, 180));

        if (sys.hasCUDA)  history.push("CUDA       : Supported", sf::Color::Green);
        if (sys.hasMetal) history.push("Metal      : Supported", sf::Color::Green);
        if (sys.hasROCm)  history.push("ROCm       : Supported", sf::Color::Green);
    } else {
        history.push("GPU        : None detected", sf::Color::Red);
    }

    history.push("Suggested Whisper model: " + sys.suggestedModel, sf::Color::Cyan);

    // ✅ Return a proper CommandResult struct
    return {
        "System info displayed successfully.", // message
        true,                                  // success flag
        sf::Color::Green                       // color for console output
    };
}
