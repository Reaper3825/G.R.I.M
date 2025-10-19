#include "commands/commands_system.hpp"
#include "system_detect.hpp"
#include "resources.hpp"
#include "error_manager.hpp"
#include "logger.hpp"

#include <SFML/Graphics.hpp>
#include <sstream>

// Externals
extern ConsoleHistory history;

CommandResult cmdSystemInfo([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: system_info");

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
        output.str(),
        true,
        sf::Color::Cyan,
        "ERR_NONE",
        "System information shown",
        "summary"
    };
}
