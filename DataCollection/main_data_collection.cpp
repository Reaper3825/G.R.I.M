#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include "pipeline/pipeline_orchestrator.hpp"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

static void printUsage() {
    std::cerr << "Usage: grim_data_collection <mode>\n"
              << "Modes:\n"
              << "  full           Full pipeline: collect -> ingest -> verify -> dedup -> preprocess -> tag -> write\n"
              << "  collect        Collect data from web sources only\n"
              << "  verify         Run verification on existing data\n"
              << "  merge          Ingest -> verify -> dedup -> preprocess -> tag -> write\n"
              << "  merge-rebuild  Same as merge with force rebuild (ignores dedup state)\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string modeStr = argv[1];
    if (modeStr == "--help" || modeStr == "-h") {
        printUsage();
        return 0;
    }

    GRIM::Pipeline::PipelineMode mode;
    if      (modeStr == "full")          mode = GRIM::Pipeline::PipelineMode::Full;
    else if (modeStr == "collect")       mode = GRIM::Pipeline::PipelineMode::CollectOnly;
    else if (modeStr == "verify")        mode = GRIM::Pipeline::PipelineMode::VerifyOnly;
    else if (modeStr == "merge")         mode = GRIM::Pipeline::PipelineMode::MergeOnly;
    else if (modeStr == "merge-rebuild") mode = GRIM::Pipeline::PipelineMode::MergeRebuild;
    else {
        std::cerr << "ERROR: Unknown mode '" << modeStr << "'\n\n";
        printUsage();
        return 1;
    }

    GRIM::Pipeline::PipelineOrchestrator orchestrator;

    orchestrator.setProgressCallback([](float progress, const std::string& message) {
        std::cout << "\r  [" << static_cast<int>(progress) << "%] " << message << std::flush;
    });

    std::cout << "=== GRIM Data Pipeline (" << modeStr << ") ===\n\n";
    orchestrator.startPipeline(mode);

    // Poll until completion
    while (true) {
        auto status = orchestrator.getStatus();
        if (!status.isRunning) {
            if (status.state == GRIM::Pipeline::PipelineState::Error) {
                std::cerr << "\n\nPipeline failed: " << status.message << "\n";
                return 1;
            }
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "\n\nDone.\n";
    return 0;
}
