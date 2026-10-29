#include "host_hardware.hpp"
#include "plugins/plugin_logging.hpp"
#include <sstream>

CommandResult cmdSystemInfo([[maybe_unused]] const std::string& arg) {
    LOG_DEBUG("Command", "Dispatch: system_info");

    auto inv = hostHardware::detectHardware();

    std::ostringstream output;
    output << "[System Info]\n";
    output << "OS         : " << inv.os_name << " (" << inv.arch << ")\n";
    output << "CPU Cores  : " << inv.cpu_cores << "\n";
    output << "RAM        : " << inv.ram_total_mb << " MB\n";

    if (inv.hasGPU()) {
        std::string gpuName = inv.gpus.empty() ? "Unknown" : inv.gpus[0].name;
        std::string gpuLine = gpuName + " (" + std::to_string(inv.gpu_count) + " device(s))";
        output << "GPU        : " << gpuLine << "\n";

        if (inv.hasCUDA())  output << "CUDA       : Supported\n";
        if (inv.hasMetal()) output << "Metal      : Supported\n";
        if (inv.hasROCm())  output << "ROCm       : Supported\n";
    } else {
        output << "GPU        : None detected\n";
    }

    output << "Suggested Whisper model: " << inv.suggested_whisper_model << "\n";

    return {
        true,                               // success
        output.str(),                       // message
        "ERR_NONE",                         // errorCode
        "summary",                          // category
        "System information shown",         // voice
        Colors::Cyan                        // color
    };
}
