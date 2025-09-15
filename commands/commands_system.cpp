#include "commands/commands_system.hpp"
#include "system_detect.hpp"
#include "resources.hpp"       // 🔹 for history
#include "error_manager.hpp"

#include <SFML/Graphics.hpp>
#include <iostream>

// Externals
extern ConsoleHistory history;

CommandResult cmdSystemInfo([[maybe_unused]] const std::string& arg) {
    std::cout << "[DEBUG][Command] Dispatch: system_info\n";

    SystemInfo sys = detectSystem();

    std::ostringstream output;
    output << "[System Info]\n";
    output << "OS         : " << sys.osName << " (" << sys.arch << ")\n";
    output << "CPU Cores  : " << sys.cpuCores << "\n";
    output << "RAM        : " << sys.ramMB << " MB\n";

    if (sys.hasGPU) {
        std::string gpuLine = sys.gpuName + " (" + std::to_string(sys.gpuCount) + " device(s))";
        output << "GPU        : " << gpuLine << "\n";

        if (sys.hasCUDA)  output << "CUDA       : Supported\n";
        if (sys.hasMetal) output << "Metal      : Supported\n";
        if (sys.hasROCm)  output << "ROCm       : Supported\n";
    } else {
        output << "GPU        : None detected\n";
    }

    output << "Suggested Whisper model: " << sys.suggestedModel << "\n";

    return {
        output.str(),     // message (multi-line string)
        true,             // success flag
        sf::Color::Cyan,  // display color
        ""                // errorCode (empty = no error)
    };
}
