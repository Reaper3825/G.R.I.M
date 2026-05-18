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

#include <sstream>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

using GRIM::Logging::ModuleId;
using GRIM::Logging::EmitModuleInfo;
using GRIM::Logging::EmitModuleError;

namespace {

void printBanner() {
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "╔════════════════════════════════════════════════════════╗", 0);
    EmitModuleInfo(ModuleId::TrainingOrchestrator, 
        "║          GRIM-text Training v3.0.0                     ║", 0);
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

#ifdef _WIN32
// Issue #142b: Windows SEH handler to catch CUDA device errors that bypass C++ exceptions.
// With default /EHsc, CUDA access violations trigger SEH that C++ catch(...) cannot catch.
// This handler ensures we get a log message instead of silent exit.
static LONG WINAPI GrimSEHHandler(EXCEPTION_POINTERS* ep) {
    DWORD code = ep ? ep->ExceptionRecord->ExceptionCode : 0;
    void* addr = ep ? ep->ExceptionRecord->ExceptionAddress : nullptr;
    
    // Log to stderr (most reliable path during crash)
    fprintf(stderr, "\n[FATAL-SEH] Unhandled structured exception!\n");
    fprintf(stderr, "[FATAL-SEH] Exception code: 0x%08lX\n", code);
    fprintf(stderr, "[FATAL-SEH] Exception address: %p\n", addr);
    
    switch (code) {
        case EXCEPTION_ACCESS_VIOLATION:
            fprintf(stderr, "[FATAL-SEH] EXCEPTION_ACCESS_VIOLATION — likely CUDA device error or buffer overflow\n");
            if (ep && ep->ExceptionRecord->NumberParameters >= 2) {
                fprintf(stderr, "[FATAL-SEH] %s address: 0x%p\n",
                    ep->ExceptionRecord->ExceptionInformation[0] == 0 ? "Read from" : "Write to",
                    (void*)ep->ExceptionRecord->ExceptionInformation[1]);
            }
            break;
        case EXCEPTION_STACK_OVERFLOW:
            fprintf(stderr, "[FATAL-SEH] EXCEPTION_STACK_OVERFLOW\n");
            break;
        case 0xC0000409:  // STATUS_STACK_BUFFER_OVERRUN (fast-fail)
            fprintf(stderr, "[FATAL-SEH] STATUS_STACK_BUFFER_OVERRUN — buffer overrun detected by /GS\n");
            break;
        default:
            fprintf(stderr, "[FATAL-SEH] Unknown exception code\n");
            break;
    }
    
    // Check CUDA state
    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "[FATAL-SEH] Last CUDA error: %s (code=%d)\n",
            cudaGetErrorString(cuda_err), static_cast<int>(cuda_err));
    }
    
    // Also try to log via module system (may fail if state is corrupted)
    EmitModuleError(ModuleId::TrainingOrchestrator,
        std::string("[FATAL-SEH] Exception code=0x") +
        ([](DWORD c) { char buf[16]; snprintf(buf, sizeof(buf), "%08lX", c); return std::string(buf); })(code) +
        " addr=" + ([](void* a) { char buf[24]; snprintf(buf, sizeof(buf), "%p", a); return std::string(buf); })(addr), 0);
    
    fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;  // Let default handler terminate
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    // Install SEH handler FIRST — before any CUDA calls.
    // Without this, CUDA device errors (illegal memory access, buffer overflow)
    // cause silent process exit because /EHsc C++ catch(...) cannot catch SEH.
    SetUnhandledExceptionFilter(GrimSEHHandler);
#endif
    
    // Check for --autograd-verbose flag to enable detailed autograd tracing
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--autograd-verbose") {
            g_autograd_verbose = true;
            EmitModuleInfo(ModuleId::TrainingOrchestrator, "Autograd verbose logging ENABLED", 0);
        }
    }

    printBanner();
    
    int exit_code = 0;
    
    try {
        //==================================================
        // PHASE 1: STARTUP
        //==================================================
        printPhaseHeader(1, "Startup");
        
        auto phase1 = GRIMText::Training::executePhase1(argc, argv);

        if (phase1.outcome == GRIMText::Training::Phase1Outcome::tokenizer_only_complete) {
            return 0;
        }

        auto ctx = std::move(phase1.context);
        
        if (!ctx || !ctx->model || !ctx->tokenizer) {
            EmitModuleError(ModuleId::TrainingOrchestrator, 
                "Phase 1 failed: model or tokenizer not initialized", 0);
            return 1;
        }
        
        {
            std::ostringstream oss;
            oss << "[Phase 1] ✓ Complete | Model: " << ctx->config.paths.output_model_path
                << " | Train: " << ctx->data.train_views.size()
                << " | Val: " << ctx->data.val_views.size()
                << " | Vocab: " << ctx->data_info.actual_vocab_size;
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
        // Rule 20: Print to stderr BEFORE any module logging.
        // The TrainingContext (and its logger) is already destroyed by stack unwinding,
        // so EmitModuleError alone loses the message.
        fprintf(stderr, "\n[FATAL] Unhandled exception: %s\n", e.what());
        fflush(stderr);
        std::ostringstream oss;
        oss << "[FATAL] Unhandled exception: " << e.what();
        EmitModuleError(ModuleId::TrainingOrchestrator, oss.str(), 0);
        exit_code = 1;
    } catch (...) {
        fprintf(stderr, "\n[FATAL] Unknown exception (not std::exception)\n");
        fflush(stderr);
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
