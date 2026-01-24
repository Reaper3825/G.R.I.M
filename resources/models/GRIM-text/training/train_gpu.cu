//======================================================//
//  GRIM-text GPU Training Orchestrator
//  Three-phase training architecture
//  
//  Phases:
//  - Phase 1: Startup (config, model init, data loading)
//  - Phase 2: Training loop (epochs, batches, optimization)
//  - Phase 3: Cleanup (final save, status, resources)
//  
//  This file orchestrates the three phases for easier
//  debugging and maintainability. Each phase is isolated
//  in its own compilation unit for:
//  - Faster incremental builds
//  - Easier debugging and breakpoints
//  - Clear data flow contracts between phases
//  
//  Author: Austin Wadkins
//  Date: December 2025
//  Version: 3.0.0 - Three-Phase Architecture
//======================================================//

#include "Phases/Phase1_Startup.hpp"
#include "Phases/Phase2_TrainingLoop.hpp"
#include "Phases/Phase3_Cleanup.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"

#include <iostream>
#include <sstream>
#include <exception>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleWarning;
using GRIM::Logging::EmitModuleError;

namespace {

void printBanner() {
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "╔════════════════════════════════════════════════════════╗", 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "║          GRIM-text Training v3.0.0                     ║", 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "║          Three-Phase Architecture                       ║", 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "╚════════════════════════════════════════════════════════╝", 0);
}

void printPhaseHeader(int phase, const char* description) {
    // Emit the phase header using EmitModuleInfo directly.
    EmitModuleInfo(ModuleId::TrainingOrchestrator,
        "┌─────────────────────────────────────────────────────────┐", 0);

    // Build the centered phase line without intermediate ostringstream
    std::string line = "│  PHASE ";
    line += std::to_string(phase);
    line += ": ";
    line += description;
    // Pad to fixed width (47 chars for description area)
    const int target_width = 47;
    int desc_len = static_cast<int>(std::strlen(description));
    int pad = target_width - desc_len;
    if (pad < 0) pad = 0;
    line.append(pad, ' ');
    line += "│";
    EmitModuleInfo(ModuleId::TrainingOrchestrator, line, 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator,
        "└─────────────────────────────────────────────────────────┘", 0);
}

} // anonymous namespace

// Import autograd verbose flag
extern bool g_autograd_verbose;

//======================================================//
//  Main Entry Point
//======================================================//

int main(int argc, char** argv) {
    // Initialize LogRecorder for modular logging
    GRIM::Logging::SetDefaultModuleLogLevel(GRIM::Logging::ModuleLogLevel::Info);
    
    // Check for --autograd-verbose flag to enable detailed autograd tracing
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--autograd-verbose") {
            g_autograd_verbose = true;
            fprintf(stderr, "[train_gpu] Autograd verbose logging ENABLED\n");
        }
    }

    printBanner();
    
    int exit_code = 0;
    
    try {
        //==================================================
        // PHASE 1: STARTUP
        //==================================================
        printPhaseHeader(1, "Startup");
        
        auto ctx = GRIMText::Training::executePhase1(argc, argv);
        
        if (!ctx || !ctx->model) {
            EmitModuleError(ModuleId::TrainingOrchestrator, 
                "Phase 1 failed: model not initialized", 0);
            return 1;
        }
        
        {
            std::ostringstream oss;
            oss << "[Phase 1] ✓ Complete | Model: " << ctx->config.paths.output_model_path
                << " | Train: " << ctx->data.train_views.size()
                << " | Val: " << ctx->data.val_views.size()
                << " | Vocab: " << ctx->tokenizer.vocabSize();
            EmitModuleInfo(ModuleId::TrainingOrchestrator, oss.str(), 0);
        }
        
        //==================================================
        // PHASE 2: TRAINING LOOP
        //==================================================
        printPhaseHeader(2, "Training Loop");
        
        bool training_success = GRIMText::Training::executePhase2(*ctx);
        
        if (training_success) {
            // Use project logger directly (avoid temporary ostringstream)
            std::string msg = "[Phase 2] ✓ Complete | Steps: ";
            msg += std::to_string(ctx->global_step);
            msg += " | Best val loss: ";
            msg += std::to_string(ctx->best_val_loss);
            if (ctx->auto_stop_triggered) {
                msg += " | Auto-stopped: ";
                msg += ctx->auto_stop_reason;
                msg += " (epoch ";
                msg += std::to_string(ctx->auto_stop_epoch);
                msg += ")";
            }
            EmitModuleInfo(ModuleId::TrainingOrchestrator, msg, ctx->global_step);
        } else {
            EmitModuleError(ModuleId::TrainingOrchestrator,
                "[Phase 2] ✗ Training failed", ctx->global_step);
            exit_code = 1;
        }
        
        //==================================================
        // PHASE 3: CLEANUP
        //==================================================
        printPhaseHeader(3, "Cleanup");
        
        auto cleanup_result = GRIMText::Training::executePhase3(*ctx);
        
        if (cleanup_result.success) {
            std::ostringstream oss;
            oss << "[Phase 3] ✓ Complete";
            if (!cleanup_result.final_model_path.empty()) {
                oss << " | Final model: " << cleanup_result.final_model_path;
            }
            oss << " | Duration: " 
                << GRIMText::Training::Internal::formatDuration(
                       cleanup_result.summary.total_duration_seconds);
            EmitModuleInfo(ModuleId::TrainingOrchestrator, oss.str(), ctx->global_step);
        } else {
            std::ostringstream oss;
            oss << "[Phase 3] ✗ Cleanup failed: " << cleanup_result.error_message;
            EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), ctx->global_step);
            exit_code = 1;
        }
        
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "[FATAL] Unhandled exception: " << e.what();
        EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), 0);
        exit_code = 1;
    } catch (...) {
        EmitModuleError(ModuleId::TrainingOrchestrator, "[FATAL] Unknown exception", 0);
        exit_code = 1;
    }
    
    //==================================================
    // FINAL STATUS
    //==================================================
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "════════════════════════════════════════════════════════════", 0);
    if (exit_code == 0) {
        EmitModuleInfo(ModuleId::TrainingOrchestrator, 
            "  TRAINING COMPLETED SUCCESSFULLY", 0);
    } else {
        std::ostringstream oss;
        oss << "  TRAINING FAILED (exit code " << exit_code << ")";
        EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), 0);
    }
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "════════════════════════════════════════════════════════════", 0);
    
    return exit_code;
}
